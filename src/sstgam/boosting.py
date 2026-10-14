"""Smooth tensor-product GAM backend for longitudinal features."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from scipy.linalg import block_diag
from sklearn.model_selection import KFold, StratifiedKFold

from ._training import (
    FAMILIES,
    bootstrap_bags,
    fast_gain,
    make_validation_split,
    pair_cells,
    percentile_band,
    quantile_edges,
)
from .bases import basis_design, make_basis_params, second_diff_matrix_np
from .normalization import fit_quantile_tables, robust_center_scale

_NORMALIZATION_CLIP = 4.0
_RIDGE = 1e-6
_MAX_NEWTON_ITERATIONS = 50
_NEWTON_TOLERANCE = 1e-9
_CHUNK_ROWS = 1 << 16
# penalty="auto" starts its search at ~100 x (coefficients per minority-class
# example) -- the information per coefficient is what sets the right penalty.
# ponytail: a starting guess loosely calibrated on two datasets; the search
# steps away from it, so a poor guess costs extra fits, not a worse penalty.
_AUTO_PENALTY_SCALE = 100.0
_AUTO_PENALTY_STEP = float(np.sqrt(10.0))  # initial bracket around the start
_AUTO_PENALTY_WALK = 10.0  # step while the best value is at the bracket's edge
_AUTO_PENALTY_BOUNDS = (1e-4, 1e6)
# Below this many minority-class examples in one held-out split, a single
# split is too noisy to rank penalties (with a few dozen held-out events the
# chosen penalty swung ~100x between resamples), so the search
# cross-validates instead: every row then contributes an out-of-fold loss.
_AUTO_PENALTY_MIN_HELDOUT_EVENTS = 500
_AUTO_PENALTY_CV_FOLDS = 5
# penalty="reml": generalised Fellner-Schall iterations, convergence on the log smoothing parameters, bounds.
_REML_MAX_ITERATIONS = 60
_REML_TOLERANCE = 1e-3
_REML_BOUNDS = (1e-6, 1e8)


def _row_chunks(n_rows: int):
    return (slice(start, min(start + _CHUNK_ROWS, n_rows)) for start in range(0, n_rows, _CHUNK_ROWS))


@dataclass
class _Term:
    """One smooth term of the joint design: its columns, centring basis and penalty pieces.

    ``design`` holds the term's columns in centred coordinates ``theta``; ``basis`` maps ``theta`` to
    the uncentred coefficients (``basis @ theta``). ``pieces`` are the penalty matrices in uncentred
    coordinates, one smoothing parameter each; ``group`` names the shared strength ("temporal" or
    "static") the term uses when the pieces are not tuned separately. A pair term also keeps its two
    axis bases in ``factors`` so its coefficients unpack to a ``(p_u, p_w)`` matrix.
    """

    kind: str
    index: int
    design: np.ndarray
    basis: np.ndarray
    pieces: list[tuple[str, np.ndarray]]
    group: str
    factors: tuple[np.ndarray, np.ndarray] | None = None

    def penalty(self, strength: float) -> np.ndarray:
        """Centred penalty with one strength on every piece, plus the tiny ridge every block gets."""
        total = sum(piece for _, piece in self.pieces)
        return self.basis.T @ (strength * total + _RIDGE * np.eye(self.basis.shape[0])) @ self.basis

    def centred_pieces(self) -> list[np.ndarray]:
        """Each penalty piece in the term's centred coordinates (``basis.T @ piece @ basis``)."""
        return [self.basis.T @ piece @ self.basis for _, piece in self.pieces]

    def penalty_with(self, strengths) -> np.ndarray:
        """Centred penalty with its own strength per piece (REML), plus the tiny ridge every block gets."""
        total = sum(s * piece for s, (_, piece) in zip(strengths, self.pieces))
        return self.basis.T @ (total + _RIDGE * np.eye(self.basis.shape[0])) @ self.basis

    def uncenter(self, theta: np.ndarray) -> np.ndarray:
        if self.factors is None:
            return self.basis @ theta
        u_z, w_z = self.factors
        return u_z @ theta.reshape(u_z.shape[1], -1) @ w_z.T


class SharedShapeTemporalGAMBoost:
    """Penalized GAM over smooth value-time feature surfaces, fitted by P-IRLS.

    ``family`` is ``"binomial"`` (logit link), ``"gaussian"`` (identity) or
    ``"poisson"`` (log). The fit maximises ``sum_i w_i loglik_i - 1/2 beta' P beta`` jointly over
    the intercept and every surface and static coefficient block, with full
    penalized Newton steps (step-halved so each one improves the objective)
    until the objective stops changing. The model has few coefficients, so
    each step is one small dense solve; it typically converges in under 20.

    Every term is centred: each surface averages zero over the observed
    training cells and each static effect over the training rows, so the
    level lives in the intercept and an unobserved cell contributes the same
    as an average observed one.

    ``temporal_penalty``/``static_penalty`` set to ``"auto"`` share one
    penalty chosen by held-out loss (``_select_penalty``).

    ``outer_bags`` bootstrap fits are averaged at the end, as in EBM.
    """

    def __init__(
        self,
        n_value_bases: int = 8,
        n_time_bases: int = 5,
        n_static_bases: int = 8,
        basis: str = "rbf",
        value_transform: str = "quantile",
        temporal_penalty: float | str = "auto",
        static_penalty: float | str = "auto",
        outer_bags: int = 4,
        val_fraction: float = 0.15,
        random_state: int = 42,
        family: str = "binomial",
        n_pair_bases: int = 6,
        n_pair_bins: int = 16,
        penalty_sharing: str = "per_term",
    ) -> None:
        self.family = family
        self.penalty_sharing = penalty_sharing
        self.n_pair_bases = n_pair_bases
        self.n_pair_bins = n_pair_bins
        self.n_value_bases = n_value_bases
        self.n_time_bases = n_time_bases
        self.n_static_bases = n_static_bases
        self.basis = basis
        self.value_transform = value_transform
        self.temporal_penalty = temporal_penalty
        self.static_penalty = static_penalty
        self.outer_bags = outer_bags
        self.val_fraction = val_fraction
        self.random_state = random_state
        self.clip = _NORMALIZATION_CLIP

    @staticmethod
    def _surface_pieces(n_first: int, n_second: int) -> list[tuple[str, np.ndarray]]:
        """Penalty pieces of a tensor-product surface: roughness along each axis, and overall size."""
        first_difference = second_diff_matrix_np(n_first)
        second_difference = second_diff_matrix_np(n_second)
        return [("value", np.kron(first_difference.T @ first_difference, np.eye(n_second))),
                ("time", np.kron(np.eye(n_first), second_difference.T @ second_difference)),
                ("shrink", np.eye(n_first * n_second))]

    @staticmethod
    def _static_pieces(n_bases: int, shrink: bool = False) -> list[tuple[str, np.ndarray]]:
        """Roughness of a static effect; under REML also its overall size, which penalises the straight-line part."""
        difference = second_diff_matrix_np(n_bases)
        pieces = [("value", difference.T @ difference)]
        return pieces + [("shrink", np.eye(n_bases))] if shrink else pieces

    @staticmethod
    def _centering_basis(mean_row: np.ndarray) -> np.ndarray:
        """Orthonormal basis of ``{c : mean_row @ c = 0}`` -- coefficients whose term averages to zero."""
        q, _ = np.linalg.qr(mean_row[:, None], mode="complete")
        return q[:, 1:]

    def normalize_values(self, feature_index: int, values) -> np.ndarray:
        """Map raw values of one temporal feature into the basis coordinate ``[-clip, clip]``."""
        values = np.asarray(values, dtype=np.float32)
        if self.value_transform == "quantile":
            source_values, source_ranks = self._quantile_tables[feature_index]
            normalized = np.interp(values, source_values, source_ranks) * self.clip
        else:
            normalized = (values - self._feature_center[feature_index]) / self._feature_scale[feature_index]
        return np.clip(normalized, -self.clip, self.clip).astype(np.float32)

    def normalize_time(self, time_values) -> np.ndarray:
        """Map raw time coordinates onto the ``[0, 1]`` time-basis coordinate."""
        return (np.asarray(time_values, dtype=np.float32) - self._time_min) / self._time_range

    def normalize_static(self, column_index: int, values) -> np.ndarray:
        """Map raw values of one static covariate into ``[-clip, clip]`` with ``value_transform``.

        Missing values map to the centre: the median rank or the training median.
        """
        values = np.asarray(values, dtype=np.float32)
        finite = np.isfinite(values)
        if self.value_transform == "quantile":
            source_values, source_ranks = self._static_tables[column_index]
            normalized = np.where(finite, np.interp(values, source_values, source_ranks), 0.0) * self.clip
        else:
            mean, scale = self._static_mean[column_index], self._static_scale[column_index]
            normalized = (np.where(finite, values, mean) - mean) / scale
        return np.clip(normalized, -self.clip, self.clip).astype(np.float32)

    def static_effect(self, column_index: int, values, return_interval: bool = False):
        """Effect of raw values of one static covariate on the linear predictor."""
        static_basis = basis_design(self.basis, self.normalize_static(column_index, values),
                                    self._static_basis_a, self._static_basis_b)
        if not return_interval:
            return static_basis @ self._static_coefficients[column_index]
        return percentile_band(np.stack([static_basis @ bag[column_index]
                                         for bag in self._bagged_static_coefficients]))

    def _build_designs(self, values, observed, static_values):
        values = np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0.0)
        observed = np.asarray(observed, dtype=np.float32)
        n_samples, n_features, _ = values.shape

        temporal_design = np.empty(
            (n_samples, n_features, self.n_value_bases * self.n_time_bases), dtype=np.float32
        )
        for feature_index in range(n_features):
            value_basis = basis_design(self.basis, self.normalize_values(feature_index, values[:, feature_index, :]),
                                       self._value_basis_a, self._value_basis_b)
            masked_basis = value_basis * observed[:, feature_index, :, None]
            temporal_design[:, feature_index, :] = np.einsum(
                "ndp,dq->npq", masked_basis, self._time_basis
            ).reshape(n_samples, -1)

        if static_values.shape[1] == 0:
            return temporal_design, None
        static_values = np.asarray(static_values, dtype=np.float32)
        static_design = np.stack([
            basis_design(self.basis, self.normalize_static(column_index, static_values[:, column_index]),
                         self._static_basis_a, self._static_basis_b)
            for column_index in range(static_values.shape[1])
        ], axis=1)
        return temporal_design, static_design

    def _penalized_newton(self, design, target, sample_weight, penalty, initial=None):
        """Maximise the penalized weighted log-likelihood of ``family`` by P-IRLS.

        ``design`` is ``(n, p)`` with an intercept column first. Starts from
        ``initial`` if given (a warm start), else from the mean-outcome intercept.
        Returns the coefficients and the number of Newton iterations taken.
        """
        family = FAMILIES[self.family]
        n_rows = design.shape[0]
        if initial is not None:
            beta = np.array(initial, dtype=np.float64)
        else:
            beta = np.zeros(design.shape[1])
            beta[0] = family.link(np.average(target, weights=sample_weight))

        def linear_predictor(coefficients):
            return np.concatenate([design[rows].astype(np.float64) @ coefficients
                                   for rows in _row_chunks(n_rows)])

        def objective(eta, coefficients):
            negative_loglik = np.sum(sample_weight * family.loss(eta, target))
            return negative_loglik + 0.5 * coefficients @ penalty @ coefficients

        eta = linear_predictor(beta)
        current = objective(eta, beta)
        for iteration in range(1, _MAX_NEWTON_ITERATIONS + 1):
            mean = family.mean(eta)
            residual = sample_weight * (target - mean)
            curvature = sample_weight * family.variance(mean)
            gradient, hessian = -penalty @ beta, penalty.copy()
            for rows in _row_chunks(n_rows):
                chunk = design[rows].astype(np.float64)
                gradient += chunk.T @ residual[rows]
                hessian += chunk.T @ (chunk * curvature[rows, None])
            step = np.linalg.solve(hessian, gradient)

            scale = 1.0
            while True:
                candidate = beta + scale * step
                candidate_eta = linear_predictor(candidate)
                candidate_objective = objective(candidate_eta, candidate)
                if candidate_objective <= current or scale < 1e-4:
                    break
                scale /= 2.0
            converged = abs(current - candidate_objective) <= _NEWTON_TOLERANCE * abs(candidate_objective)
            beta, eta, current = candidate, candidate_eta, candidate_objective
            if converged:
                break
        return beta, iteration

    def _select_penalty(self, design, target, sample_weight, penalty_for):
        """Choose the shared ``"auto"`` penalty by held-out weighted loss (deviance).

        Scores a ``val_fraction`` split when it would hold at least
        ``_AUTO_PENALTY_MIN_HELDOUT_EVENTS`` minority-class examples (rows, for
        a non-binomial family), else the out-of-fold rows of a
        ``_AUTO_PENALTY_CV_FOLDS``-fold split, stratified for binomial.
        Starts at ~100 x coefficients per minority-class example, tries
        that value and one ``sqrt(10)`` step either side, then steps outward
        by 10x while the edge value is clearly better than its neighbour --
        by more than one standard error of their paired per-row loss
        difference, so the tolerance shrinks with more validation data.
        Picks the largest penalty not clearly worse than the best (the
        one-standard-error rule). Each fit warm-starts from the nearest one
        already done. Returns the chosen penalty and its coefficients (a warm
        start for the final fit).
        """
        family = FAMILIES[self.family]
        binomial = self.family == "binomial"
        positives = target.sum()
        # ponytail: for gaussian/poisson every row counts as one "event"; fine as a starting guess.
        minority = max(min(positives, target.size - positives), 1.0) if binomial else float(target.size)
        n_folds = int(min(_AUTO_PENALTY_CV_FOLDS, minority))
        if minority * self.val_fraction >= _AUTO_PENALTY_MIN_HELDOUT_EVENTS or n_folds < 2:
            splits = [make_validation_split(target, self.val_fraction, self.random_state)]
        else:
            folds = (StratifiedKFold if binomial else KFold)(n_folds, shuffle=True, random_state=self.random_state)
            splits = list(folds.split(np.zeros((target.size, 1)), target))
        held_out = [(validation, design[validation]) for _, validation in splits]
        held_out_target = np.concatenate([target[validation] for validation, _ in held_out])
        held_out_weight = np.concatenate([sample_weight[validation] for validation, _ in held_out])
        total_weight = held_out_weight.sum()
        low, high = _AUTO_PENALTY_BOUNDS
        start = float(np.clip(_AUTO_PENALTY_SCALE * design.shape[1] / minority, low, high))
        fits = {}

        def evaluate(strength):
            nearest = min(fits, key=lambda tried: abs(np.log(tried / strength)), default=None)
            betas, etas = [], []
            for split_index, (train, _) in enumerate(splits):
                beta, _ = self._penalized_newton(
                    design[train], target[train], sample_weight[train], penalty_for(strength),
                    initial=None if nearest is None else fits[nearest][1][split_index],
                )
                validation, validation_design = held_out[split_index]
                etas.extend(validation_design[rows].astype(np.float64) @ beta
                            for rows in _row_chunks(validation.size))
                betas.append(beta)
            eta = np.concatenate(etas)
            row_losses = held_out_weight * family.loss(eta, held_out_target)
            fits[strength] = (float(row_losses.sum() / total_weight), betas, row_losses)

        def loss(strength):
            return fits[strength][0]

        def standard_error(strength, other):
            """Standard error of the validation-loss difference, paired over the same rows."""
            difference = fits[strength][2] - fits[other][2]
            return float(np.std(difference, ddof=1) * np.sqrt(difference.size) / total_weight)

        def clearly_better(strength, other):
            return loss(other) - loss(strength) > standard_error(strength, other)

        for strength in (start, start / _AUTO_PENALTY_STEP, start * _AUTO_PENALTY_STEP):
            evaluate(strength)
        while True:
            tried = sorted(fits)
            if clearly_better(tried[0], tried[1]) and tried[0] / _AUTO_PENALTY_WALK >= low:
                evaluate(tried[0] / _AUTO_PENALTY_WALK)
            elif clearly_better(tried[-1], tried[-2]) and tried[-1] * _AUTO_PENALTY_WALK <= high:
                evaluate(tried[-1] * _AUTO_PENALTY_WALK)
            else:
                break
        # One-standard-error rule: the smoothest penalty not clearly worse than the best.
        best = min(fits, key=loss)
        chosen = max(strength for strength in fits if not clearly_better(best, strength))
        self.penalty_search_ = {strength: loss(strength) for strength in sorted(fits)}
        self.penalty_search_folds_ = len(splits)
        return chosen, fits[chosen][1][0]

    def _hessian(self, design, sample_weight, eta) -> np.ndarray:
        """``X' W X`` at ``eta``: the Hessian of the negative log-likelihood (canonical links)."""
        family = FAMILIES[self.family]
        curvature = sample_weight * family.variance(family.mean(eta))
        hessian = np.zeros((design.shape[1], design.shape[1]))
        for rows in _row_chunks(design.shape[0]):
            chunk = design[rows].astype(np.float64)
            hessian += chunk.T @ (chunk * curvature[rows, None])
        return hessian

    def _select_smoothing_reml(self, design, target, sample_weight, terms, start, fixed, initial=None,
                               start_strengths=None):
        """One smoothing parameter per penalty piece, chosen by REML with the generalised Fellner-Schall update.

        Wood & Fasiolo (2017): at the penalised fit, each parameter is updated as
        ``lambda_j <- phi * lambda_j * (tr(S^- S_j) - tr((H + S)^-1 S_j)) / (theta' S_j theta)``, where ``S`` is
        the term's total penalty, ``H`` the Hessian of the negative log-likelihood and ``phi`` the scale
        (1 for binomial and Poisson, residual variance for gaussian). Pieces that share a parameter
        (``penalty_sharing="shared"``) pool their numerators and denominators. ``fixed`` maps a term group to a
        fixed strength (a number given by the user) that is not tuned. Returns per-term strengths, the final
        coefficients and the effective degrees of freedom per term.
        """
        family = FAMILIES[self.family]
        shared = self.penalty_sharing == "shared"

        def key(term, label):
            return (term.kind, label) if shared else (term.kind, term.index, label)

        tuned = [fixed.get(t.group) is None for t in terms]
        parameter = {}
        for t, is_tuned in zip(terms, tuned):
            if is_tuned:
                for label, _ in t.pieces:
                    parameter.setdefault(key(t, label), len(parameter))
        strengths = np.full(len(parameter), float(start))
        if start_strengths is not None:  # warm start: (term kind, index) -> per-piece strengths
            for t in terms:
                for (label, _), s in zip(t.pieces, start_strengths.get((t.kind, t.index), ())):
                    if key(t, label) in parameter:
                        strengths[parameter[key(t, label)]] = s
        centred = [t.centred_pieces() for t in terms]
        offsets = np.cumsum([1] + [t.design.shape[1] for t in terms])
        low, high = _REML_BOUNDS

        def term_strengths():
            return [[fixed[t.group]] * len(t.pieces) if not is_tuned else
                    [strengths[parameter[key(t, label)]] for label, _ in t.pieces]
                    for t, is_tuned in zip(terms, tuned)]

        beta = initial
        self.reml_iterations_ = 0
        for iteration in range(1, _REML_MAX_ITERATIONS + 1):
            per_term = term_strengths()
            penalty = block_diag(np.full((1, 1), _RIDGE), *(t.penalty_with(s) for t, s in zip(terms, per_term)))
            beta, _ = self._penalized_newton(design, target, sample_weight, penalty, initial=beta)
            eta = np.concatenate([design[rows].astype(np.float64) @ beta for rows in _row_chunks(design.shape[0])])
            hessian = self._hessian(design, sample_weight, eta)
            inverse = np.linalg.inv(hessian + penalty)
            edf = (inverse * hessian.T).sum(axis=1)
            if self.family == "gaussian":
                residual = np.sum(sample_weight * (target - family.mean(eta)) ** 2)
                scale = residual / max(sample_weight.sum() - edf.sum(), 1.0)
            else:
                scale = 1.0
            numerator, denominator = np.zeros(len(parameter)), np.zeros(len(parameter))
            for t, is_tuned, pieces, s, lo, hi in zip(terms, tuned, centred, per_term, offsets[:-1], offsets[1:]):
                if not is_tuned:
                    continue
                block_penalty = sum(strength * piece for strength, piece in zip(s, pieces))
                values, vectors = np.linalg.eigh(block_penalty)
                keep = values > values.max() * 1e-10
                pseudo_inverse = (vectors[:, keep] / values[keep]) @ vectors[:, keep].T
                block_inverse, theta = inverse[lo:hi, lo:hi], beta[lo:hi]
                for (label, _), piece in zip(t.pieces, pieces):
                    j = parameter[key(t, label)]
                    numerator[j] += np.sum(pseudo_inverse * piece) - np.sum(block_inverse * piece)
                    denominator[j] += theta @ piece @ theta
            ratio = scale * numerator / np.maximum(denominator, 1e-12)
            # Damped update: at most a factor of 10 per step; a non-positive numerator means "smooth less".
            step = np.where(numerator > 0, np.log(np.maximum(ratio, 1e-300)), -np.log(10.0))
            updated = np.clip(strengths * np.exp(np.clip(step, -np.log(10.0), np.log(10.0))), low, high)
            converged = np.max(np.abs(np.log(updated) - np.log(strengths)), initial=0.0) < _REML_TOLERANCE
            strengths = updated
            self.reml_iterations_ = iteration
            if converged:
                break
        per_term = term_strengths()
        edf_per_term = [float(edf[lo:hi].sum()) for lo, hi in pairwise(offsets)]
        return per_term, beta, edf_per_term

    def _normalize_axis(self, axis, raw) -> np.ndarray:
        kind, index = axis
        return self.normalize_values(index, raw) if kind == "temporal" else self.normalize_static(index, raw)

    def _pair_basis(self, axis, raw) -> np.ndarray:
        return basis_design(self.basis, self._normalize_axis(axis, raw), self._pair_basis_a, self._pair_basis_b)

    def _pair_design(self, values, observed, static_values, pair, bases=None):
        """One pair's design ``(n, p_u * p_w)`` in centred coordinates, and its centring bases.

        mgcv ``ti()``-style: each axis basis is centred over the pair's cells
        before the tensor product, so the pair has no main effect of its own.
        """
        u, w, mask = pair_cells(values, observed, static_values, pair)
        u_basis = self._pair_basis(pair[0], u) * np.asarray(mask, dtype=np.float32)[..., None]
        w_basis = self._pair_basis(pair[1], w)
        if bases is None:
            cells = max(float(np.sum(mask)), 1.0)
            bases = (self._centering_basis(u_basis.sum(axis=(0, 1)) / cells),
                     self._centering_basis((w_basis * np.asarray(mask)[..., None]).sum(axis=(0, 1)) / cells))
        u_z, w_z = bases
        design = np.einsum("ncp,ncq->npq", u_basis @ u_z, w_basis @ w_z).reshape(u.shape[0], -1)
        return design.astype(np.float32), bases

    def _rank_pairs(self, values, observed, static_values, target, sample_weight, eta, pairs, n_keep):
        """The ``n_keep`` candidate pairs with the largest FAST gain on the main-effects residual."""
        family = FAMILIES[self.family]
        mean = family.mean(eta)
        gradient, hessian = sample_weight * (target - mean), sample_weight * family.variance(mean)

        def axis_bins(axis, raw, mask):
            finite = np.isfinite(raw) & (mask > 0)
            edges = quantile_edges(np.asarray(raw)[finite], self.n_pair_bins)
            return np.searchsorted(edges[1:-1], np.where(finite, raw, np.median(raw[finite]) if finite.any()
                                                         else 0.0), side="right"), len(edges) - 1

        gains = []
        for pair in pairs:
            u, w, mask = pair_cells(values, observed, static_values, pair)
            (u_bins, n_u), (w_bins, n_w) = axis_bins(pair[0], u, mask), axis_bins(pair[1], w, mask)
            gains.append(fast_gain(gradient, hessian, u_bins * n_w + w_bins, np.asarray(mask, float), n_u, n_w))
        return [pairs[i] for i in np.argsort(gains, kind="stable")[::-1][:n_keep]]

    def fit(self, values, observed, static_values, time_coordinates, target,
            sample_weight=None, pairs=(), n_interactions: int = 0):
        """Fit main effects; then, with ``n_interactions``, the best of the candidate ``pairs`` too.

        Pairs are picked by FAST on the main-effects residual, appended to the
        design as centred tensor-product blocks, and everything is refit
        jointly with the main-effects penalty, warm-started from the main fit.
        """
        values = np.asarray(values, dtype=np.float32)
        observed = np.asarray(observed, dtype=np.float32)
        target = np.asarray(target, dtype=np.float64).ravel()
        n_samples, n_features, _ = values.shape
        static_values = np.asarray(static_values, dtype=np.float32)
        n_static = static_values.shape[1]
        sample_weight = (np.ones(n_samples, dtype=np.float64) if sample_weight is None
                         else np.asarray(sample_weight, dtype=np.float64).ravel())

        if self.value_transform == "quantile":
            self._quantile_tables = fit_quantile_tables(values, observed)
            self._static_tables = fit_quantile_tables(static_values[:, :, None],
                                                      np.isfinite(static_values)[:, :, None])
        else:
            self._feature_center, self._feature_scale = robust_center_scale(values, observed)
            self._static_mean, self._static_scale = robust_center_scale(
                static_values[:, :, None], np.isfinite(static_values)[:, :, None])
        value_span = self.clip if self.value_transform == "quantile" else 0.9 * self.clip
        self._value_basis_a, self._value_basis_b = make_basis_params(
            self.basis, -value_span, value_span, self.n_value_bases
        )
        self._static_basis_a, self._static_basis_b = make_basis_params(
            self.basis, -value_span, value_span, self.n_static_bases
        )
        self._pair_basis_a, self._pair_basis_b = make_basis_params(
            self.basis, -value_span, value_span, self.n_pair_bases
        )
        time_coordinates = np.asarray(time_coordinates, dtype=np.float32)
        self._time_min = float(time_coordinates.min())
        self._time_range = max(float(time_coordinates.max()) - self._time_min, 1.0)
        self._time_basis_a, self._time_basis_b = make_basis_params(self.basis, 0.0, 1.0, self.n_time_bases)
        self._time_basis = basis_design(self.basis, self.normalize_time(time_coordinates),
                                        self._time_basis_a, self._time_basis_b)

        self.smoothing_params_: list[tuple[str, int, dict[str, float]]] = []
        self.edf_: list[tuple[str, int, float]] = []
        reml = "reml" in (self.temporal_penalty, self.static_penalty)
        if reml and "auto" in (self.temporal_penalty, self.static_penalty):
            raise ValueError("temporal_penalty and static_penalty cannot mix 'auto' and 'reml'.")
        temporal_design, static_design = self._build_designs(values, observed, static_values)
        # Every term is fitted in centred coordinates theta (coefficients = basis @ theta), then mapped back.
        observed_cells = np.maximum(observed.sum(axis=(0, 2)), 1.0)
        surface_pieces = self._surface_pieces(self.n_value_bases, self.n_time_bases)
        terms = []
        for k in range(n_features):
            basis = self._centering_basis(temporal_design[:, k, :].sum(axis=0) / observed_cells[k])
            terms.append(_Term("temporal", k, (temporal_design[:, k, :] @ basis).astype(np.float32), basis,
                               surface_pieces, "temporal"))
        if static_design is not None:
            static_pieces = self._static_pieces(self.n_static_bases, shrink=reml)
            for j in range(n_static):
                basis = self._centering_basis(static_design[:, j, :].mean(axis=0))
                terms.append(_Term("static", j, (static_design[:, j, :] @ basis).astype(np.float32), basis,
                                   static_pieces, "static"))

        def assemble(terms):
            """One design over all terms (intercept first), and its penalty for the two shared strengths."""
            design = np.concatenate([np.ones((n_samples, 1), dtype=np.float32), *(t.design for t in terms)], axis=1)

            def penalty_at(temporal_strength, static_strength):
                # The intercept gets the same tiny ridge as every block: without it the Hessian is
                # singular whenever the fitted probabilities saturate (e.g. a one-class bootstrap bag).
                return block_diag(np.full((1, 1), _RIDGE), *(
                    t.penalty(temporal_strength if t.group == "temporal" else static_strength) for t in terms))
            return design, penalty_at

        def strengths(shared):
            return (shared if self.temporal_penalty == "auto" else self.temporal_penalty,
                    shared if self.static_penalty == "auto" else self.static_penalty)

        fixed = {"temporal": None if self.temporal_penalty == "reml" else self.temporal_penalty,
                 "static": None if self.static_penalty == "reml" else self.static_penalty}
        start = self._start_strength(1 + sum(t.design.shape[1] for t in terms), target)

        def reml_penalty(terms, initial=None, start_strengths=None):
            """Choose the smoothing parameters by REML and return the penalty, recording them on the model."""
            design, _ = assemble(terms)
            per_term, beta, edf = self._select_smoothing_reml(design, target, sample_weight, terms, start, fixed,
                                                              initial=initial, start_strengths=start_strengths)
            self.smoothing_params_ = [(t.kind, t.index, {label: s for (label, _), s in zip(t.pieces, strengths)})
                                      for t, strengths in zip(terms, per_term)]
            self.edf_ = [(t.kind, t.index, value) for t, value in zip(terms, edf)]
            penalty = block_diag(np.full((1, 1), _RIDGE), *(t.penalty_with(s) for t, s in zip(terms, per_term)))
            return penalty, beta

        design, penalty_at = assemble(terms)
        initial = None
        self.penalty_search_ = {}
        if reml:
            main_penalty, initial = reml_penalty(terms)
            self.temporal_penalty_: float | str = "reml"
            self.static_penalty_: float | str = "reml"
        elif "auto" in (self.temporal_penalty, self.static_penalty):
            shared, initial = self._select_penalty(
                design, target, sample_weight, lambda strength: penalty_at(*strengths(strength))
            )
            self.temporal_penalty_, self.static_penalty_ = strengths(shared)
        else:
            self.temporal_penalty_, self.static_penalty_ = self.temporal_penalty, self.static_penalty

        def fit_bags(terms, initial, penalty=None):
            design, penalty_at = assemble(terms)
            if penalty is None:
                penalty = penalty_at(self.temporal_penalty_, self.static_penalty_)
            rng = np.random.RandomState(self.random_state)
            self._bagged_intercepts = []
            self._bagged_temporal_coefficients = []
            self._bagged_static_coefficients = []
            self._bagged_pair_coefficients = []
            self.n_iter_ = []
            for fit_indices in bootstrap_bags(np.arange(n_samples), self.outer_bags, rng):
                beta, n_iter = self._penalized_newton(
                    design[fit_indices], target[fit_indices], sample_weight[fit_indices], penalty,
                    initial=initial,
                )
                coefficients: dict[str, list[np.ndarray]] = {"temporal": [], "static": [], "pair": []}
                start = 1
                for term in terms:
                    size = term.design.shape[1]
                    coefficients[term.kind].append(term.uncenter(beta[start:start + size]))
                    start += size
                self._bagged_intercepts.append(float(beta[0]))
                self._bagged_temporal_coefficients.append(np.stack(coefficients["temporal"]))
                self._bagged_static_coefficients.append(
                    np.stack(coefficients["static"]) if coefficients["static"] else None)
                self._bagged_pair_coefficients.append(coefficients["pair"])
                self.n_iter_.append(n_iter)

            self._intercept = float(np.mean(self._bagged_intercepts))
            self._temporal_coefficients = np.mean(self._bagged_temporal_coefficients, axis=0)
            self._static_coefficients = (None if static_design is None
                                         else np.mean(self._bagged_static_coefficients, axis=0))
            self._pair_coefficients = [np.mean(bags, axis=0) for bags in zip(*self._bagged_pair_coefficients)]

        self._pairs, self._pair_bases = [], []
        fit_bags(terms, initial, main_penalty if reml else None)
        if n_interactions and pairs:
            pairs = list(pairs)
            if n_interactions < len(pairs):
                temporal, static, _ = self.contributions(values, observed, static_values)
                eta = self._intercept + temporal.sum(axis=(1, 2)) + static.sum(axis=1)
                pairs = self._rank_pairs(values, observed, static_values, target, sample_weight, eta,
                                         pairs, n_interactions)
            pair_blocks = [self._pair_design(values, observed, static_values, pair) for pair in pairs]
            self._pairs, self._pair_bases = pairs, [bases for _, bases in pair_blocks]
            pair_pieces = self._surface_pieces(self.n_pair_bases, self.n_pair_bases)
            pair_terms = [_Term("pair", i, block, np.kron(u_z, w_z), pair_pieces, "temporal", factors=(u_z, w_z))
                          for i, (block, (u_z, w_z)) in enumerate(pair_blocks)]
            n_pair_columns = sum(t.design.shape[1] for t in pair_terms)
            pair_initial = None if initial is None else np.concatenate([initial, np.zeros(n_pair_columns)])
            if reml:  # re-select every smoothing parameter with the pairs in, starting from the main-effects values
                previous = {(kind, index): list(values.values()) for kind, index, values in self.smoothing_params_}
                full_penalty, pair_initial = reml_penalty(terms + pair_terms, initial=pair_initial,
                                                          start_strengths=previous)
                fit_bags(terms + pair_terms, pair_initial, full_penalty)
            else:
                fit_bags(terms + pair_terms, pair_initial)
        return self

    def _start_strength(self, n_coefficients: int, target) -> float:
        """The same starting guess the ``"auto"`` search uses: ~100 x coefficients per minority-class example."""
        positives = float(np.sum(target))
        minority = max(min(positives, target.size - positives), 1.0) if self.family == "binomial" else float(target.size)
        low, high = _AUTO_PENALTY_BOUNDS
        return float(np.clip(_AUTO_PENALTY_SCALE * n_coefficients / minority, low, high))

    def contributions(self, values, observed, static_values):
        """Per-cell temporal ``(n, features, times)`` and per-column static ``(n, statics)`` terms.

        Together with ``_intercept`` they sum to the linear predictor; missing cells contribute 0.
        """
        values = np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0.0)
        observed = np.asarray(observed, dtype=np.float32)
        temporal = np.empty(values.shape, dtype=np.float64)
        for feature_index in range(values.shape[1]):
            value_basis = basis_design(self.basis, self.normalize_values(feature_index, values[:, feature_index, :]),
                                       self._value_basis_a, self._value_basis_b)
            coefficients = self._temporal_coefficients[feature_index].reshape(self.n_value_bases, self.n_time_bases)
            temporal[:, feature_index, :] = ((value_basis @ coefficients) * self._time_basis).sum(axis=2)
        temporal *= observed
        static_values = np.asarray(static_values, dtype=np.float32)
        static = np.zeros(static_values.shape, dtype=np.float64)
        for column_index in range(static_values.shape[1]):
            static[:, column_index] = self.static_effect(column_index, static_values[:, column_index])
        pairs = np.zeros((values.shape[0], len(self._pairs)))
        for pair_index, pair in enumerate(self._pairs):
            u, w, mask = pair_cells(values, observed, static_values, pair)
            surface = (self._pair_basis(pair[0], u) @ self._pair_coefficients[pair_index]) * self._pair_basis(pair[1], w)
            pairs[:, pair_index] = (surface.sum(axis=-1) * mask).sum(axis=1)
        return temporal, static, pairs

    def interaction_surface(self, pair_index: int, u_values, w_values, return_interval: bool = False):
        """Pair surface on a grid of raw values: ``(len(u_values), len(w_values))``."""
        pair = self._pairs[pair_index]
        u_basis = self._pair_basis(pair[0], np.asarray(u_values, dtype=np.float32))
        w_basis = self._pair_basis(pair[1], np.asarray(w_values, dtype=np.float32))
        surfaces = np.stack([u_basis @ bag[pair_index] @ w_basis.T for bag in self._bagged_pair_coefficients])
        return percentile_band(surfaces) if return_interval else surfaces.mean(axis=0)

    def _bagged_surfaces(self, feature_index: int, value_grid, normalized_time_grid) -> np.ndarray:
        """Stack one fitted value-time surface per bag: shape (n_bags, len(value_grid), len(time_grid))."""
        value_basis = basis_design(self.basis, np.asarray(value_grid, dtype=np.float32),
                                   self._value_basis_a, self._value_basis_b)
        time_basis = basis_design(self.basis, np.asarray(normalized_time_grid, dtype=np.float32),
                                  self._time_basis_a, self._time_basis_b)
        return np.stack([
            value_basis @ bag_coefficients[feature_index].reshape(self.n_value_bases, self.n_time_bases) @ time_basis.T
            for bag_coefficients in self._bagged_temporal_coefficients
        ], axis=0)

    def temporal_surface_normalized(self, feature_index: int, value_grid,
                                    normalized_time_grid, return_interval: bool = False):
        """Evaluate one fitted value-time surface on normalized coordinates.

        With ``return_interval=True``, returns a ``(lower, mean, upper)``
        bootstrap percentile band (2.5/97.5) computed across the
        ``outer_bags`` bagged fits instead of a single point estimate. With
        ``outer_bags=1`` the band collapses to the point estimate.
        """
        surfaces = self._bagged_surfaces(feature_index, value_grid, normalized_time_grid)
        return percentile_band(surfaces) if return_interval else surfaces.mean(axis=0)

    def mean_value_effect_normalized(self, feature_index: int, value_grid,
                                     return_interval: bool = False):
        time_grid = np.linspace(0.0, 1.0, self.n_time_bases * 3)
        per_bag_effect = self._bagged_surfaces(feature_index, value_grid, time_grid).mean(axis=2)
        return percentile_band(per_bag_effect) if return_interval else per_bag_effect.mean(axis=0)
