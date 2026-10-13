"""Smooth tensor-product GAM backend for longitudinal features."""

from __future__ import annotations

import numpy as np
from scipy.linalg import block_diag
from scipy.special import expit
from sklearn.model_selection import StratifiedKFold

from ._training import bootstrap_bags, make_validation_split
from .bases import basis_design, make_basis_params, second_diff_matrix_np
from .normalization import fit_quantile_tables, robust_center_scale, static_stats

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


def _row_chunks(n_rows: int):
    return (slice(start, min(start + _CHUNK_ROWS, n_rows)) for start in range(0, n_rows, _CHUNK_ROWS))


class SharedShapeTemporalGAMBoost:
    """Penalized GAM over smooth value-time feature surfaces, fitted by P-IRLS.

    The fit maximises ``sum_i w_i loglik_i - 1/2 beta' P beta`` jointly over
    the intercept and every surface and static coefficient block, with full
    penalized Newton steps (step-halved so each one improves the objective)
    until the objective stops changing. The model has few coefficients, so
    each step is one small dense solve; it typically converges in under 20.

    Every term is centred: each surface averages zero over the observed
    training cells and each static effect over the training rows, so the
    level lives in the intercept and an unobserved cell contributes the same
    as an average observed one.

    ``temporal_penalty``/``static_penalty`` set to ``"auto"`` share one
    penalty chosen by held-out log loss (``_select_penalty``).

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
    ) -> None:
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
    def _penalty_matrix(n_value_bases: int, n_time_bases: int,
                        strength: float) -> np.ndarray:
        value_difference = second_diff_matrix_np(n_value_bases)
        time_difference = second_diff_matrix_np(n_time_bases)
        value_penalty = value_difference.T @ value_difference
        time_penalty = time_difference.T @ time_difference
        return strength * (
            np.kron(value_penalty, np.eye(n_time_bases)) +
            np.kron(np.eye(n_value_bases), time_penalty) +
            np.eye(n_value_bases * n_time_bases)
        ) + _RIDGE * np.eye(n_value_bases * n_time_bases)

    @staticmethod
    def _static_penalty(n_bases: int, strength: float) -> np.ndarray:
        difference = second_diff_matrix_np(n_bases)
        return strength * (difference.T @ difference) + _RIDGE * np.eye(n_bases)

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

        Missing values map to the centre: the median rank or the training mean.
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

    def static_effect(self, column_index: int, values) -> np.ndarray:
        """Log-odds effect of raw values of one static covariate."""
        static_basis = basis_design(self.basis, self.normalize_static(column_index, values),
                                    self._static_basis_a, self._static_basis_b)
        return static_basis @ self._static_coefficients[column_index]

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

    @staticmethod
    def _penalized_newton(design, target, sample_weight, penalty, initial=None):
        """Maximise the penalized weighted log-likelihood of a logistic model by P-IRLS.

        ``design`` is ``(n, p)`` with an intercept column first. Starts from
        ``initial`` if given (a warm start), else from the base-rate intercept.
        Returns the coefficients and the number of Newton iterations taken.
        """
        n_rows = design.shape[0]
        if initial is not None:
            beta = np.array(initial, dtype=np.float64)
        else:
            base_rate = np.clip(np.average(target, weights=sample_weight), 1e-6, 1.0 - 1e-6)
            beta = np.zeros(design.shape[1])
            beta[0] = np.log(base_rate / (1.0 - base_rate))

        def linear_predictor(coefficients):
            return np.concatenate([design[rows].astype(np.float64) @ coefficients
                                   for rows in _row_chunks(n_rows)])

        def objective(eta, coefficients):
            negative_loglik = np.sum(sample_weight * (np.logaddexp(0.0, eta) - target * eta))
            return negative_loglik + 0.5 * coefficients @ penalty @ coefficients

        eta = linear_predictor(beta)
        current = objective(eta, beta)
        for iteration in range(1, _MAX_NEWTON_ITERATIONS + 1):
            probability = expit(eta)
            residual = sample_weight * (target - probability)
            curvature = sample_weight * probability * (1.0 - probability)
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
        """Choose the shared ``"auto"`` penalty by held-out weighted log loss.

        Scores a stratified ``val_fraction`` split when it would hold at least
        ``_AUTO_PENALTY_MIN_HELDOUT_EVENTS`` minority-class examples, else the
        out-of-fold rows of a stratified ``_AUTO_PENALTY_CV_FOLDS``-fold split.
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
        positives = target.sum()
        minority = max(min(positives, target.size - positives), 1.0)
        n_folds = int(min(_AUTO_PENALTY_CV_FOLDS, minority))
        if minority * self.val_fraction >= _AUTO_PENALTY_MIN_HELDOUT_EVENTS or n_folds < 2:
            splits = [make_validation_split(target, self.val_fraction, self.random_state)]
        else:
            folds = StratifiedKFold(n_folds, shuffle=True, random_state=self.random_state)
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
            row_losses = held_out_weight * (np.logaddexp(0.0, eta) - held_out_target * eta)
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

    def fit(self, values, observed, static_values, time_coordinates, target,
            sample_weight=None):
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
            self._static_mean, self._static_scale = static_stats(static_values)
        value_span = self.clip if self.value_transform == "quantile" else 0.9 * self.clip
        self._value_basis_a, self._value_basis_b = make_basis_params(
            self.basis, -value_span, value_span, self.n_value_bases
        )
        self._static_basis_a, self._static_basis_b = make_basis_params(
            self.basis, -value_span, value_span, self.n_static_bases
        )
        time_coordinates = np.asarray(time_coordinates, dtype=np.float32)
        self._time_min = float(time_coordinates.min())
        self._time_range = max(float(time_coordinates.max()) - self._time_min, 1.0)
        self._time_basis_a, self._time_basis_b = make_basis_params(self.basis, 0.0, 1.0, self.n_time_bases)
        self._time_basis = basis_design(self.basis, self.normalize_time(time_coordinates),
                                        self._time_basis_a, self._time_basis_b)

        temporal_design, static_design = self._build_designs(values, observed, static_values)
        # Fit in centred coordinates theta (c = Z @ theta), then map back.
        observed_cells = np.maximum(observed.sum(axis=(0, 2)), 1.0)
        temporal_bases = [self._centering_basis(temporal_design[:, k, :].sum(axis=0) / observed_cells[k])
                          for k in range(n_features)]
        temporal_design = np.stack([temporal_design[:, k, :] @ Z for k, Z in enumerate(temporal_bases)],
                                   axis=1).astype(np.float32)
        static_bases = []
        if static_design is not None:
            static_bases = [self._centering_basis(static_design[:, j, :].mean(axis=0)) for j in range(n_static)]
            static_design = np.stack([static_design[:, j, :] @ Z for j, Z in enumerate(static_bases)],
                                     axis=1).astype(np.float32)

        def uncenter(theta, bases):
            return None if theta is None else np.stack([Z @ t for Z, t in zip(bases, theta)])

        # One design over all terms: intercept, then each surface, then each static effect.
        n_temporal = temporal_design.shape[2]
        temporal_end = 1 + n_features * n_temporal
        blocks = [np.ones((n_samples, 1), dtype=np.float32), temporal_design.reshape(n_samples, -1)]
        if static_design is not None:
            blocks.append(static_design.reshape(n_samples, -1))
        design = np.concatenate(blocks, axis=1)

        def penalty_at(temporal_strength, static_strength):
            temporal = self._penalty_matrix(self.n_value_bases, self.n_time_bases, temporal_strength)
            static = self._static_penalty(self.n_static_bases, static_strength)
            # The intercept gets the same tiny ridge as every block: without it the Hessian is
            # singular whenever the fitted probabilities saturate (e.g. a one-class bootstrap bag).
            return block_diag(np.full((1, 1), _RIDGE), *(Z.T @ temporal @ Z for Z in temporal_bases),
                              *(Z.T @ static @ Z for Z in static_bases))

        def strengths(shared):
            return (shared if self.temporal_penalty == "auto" else self.temporal_penalty,
                    shared if self.static_penalty == "auto" else self.static_penalty)

        initial = None
        self.penalty_search_ = {}
        if "auto" in (self.temporal_penalty, self.static_penalty):
            shared, initial = self._select_penalty(
                design, target, sample_weight, lambda strength: penalty_at(*strengths(strength))
            )
            self.temporal_penalty_, self.static_penalty_ = strengths(shared)
        else:
            self.temporal_penalty_, self.static_penalty_ = self.temporal_penalty, self.static_penalty
        penalty = penalty_at(self.temporal_penalty_, self.static_penalty_)

        rng = np.random.RandomState(self.random_state)
        self._bagged_intercepts = []
        self._bagged_temporal_coefficients = []
        self._bagged_static_coefficients = []
        self.n_iter_ = []
        for fit_indices in bootstrap_bags(np.arange(n_samples), self.outer_bags, rng):
            beta, n_iter = self._penalized_newton(
                design[fit_indices], target[fit_indices], sample_weight[fit_indices], penalty,
                initial=initial,
            )
            self._bagged_intercepts.append(float(beta[0]))
            self._bagged_temporal_coefficients.append(
                uncenter(beta[1:temporal_end].reshape(n_features, n_temporal), temporal_bases))
            self._bagged_static_coefficients.append(
                None if static_design is None
                else uncenter(beta[temporal_end:].reshape(n_static, -1), static_bases))
            self.n_iter_.append(n_iter)

        self._intercept = float(np.mean(self._bagged_intercepts))
        self._temporal_coefficients = np.mean(self._bagged_temporal_coefficients, axis=0)
        self._static_coefficients = (None if static_design is None
                                     else np.mean(self._bagged_static_coefficients, axis=0))
        return self

    def predict_proba(self, values, observed, static_values):
        temporal_design, static_design = self._build_designs(values, observed, static_values)
        score = np.full(temporal_design.shape[0], self._intercept)
        score += np.einsum("nfp,fp->n", temporal_design, self._temporal_coefficients)
        if static_design is not None:
            score += np.einsum("nsp,sp->n", static_design, self._static_coefficients)
        return np.clip(expit(score), 1e-6, 1.0 - 1e-6)

    def predict_proba_per_bag(self, values, observed, static_values) -> np.ndarray:
        """Return each bag's own (unaveraged) predicted probability, shape ``(outer_bags, n_samples)``.

        Exposes the bagging variance-reduction property directly: the final
        ``predict_proba`` is the bag-averaged score passed through the
        logistic link, not the average of these per-bag probabilities, but
        the spread across rows here is what averaging the bags reduces.
        """
        temporal_design, static_design = self._build_designs(values, observed, static_values)
        probabilities = []
        for bag_index in range(len(self._bagged_intercepts)):
            score = np.full(temporal_design.shape[0], self._bagged_intercepts[bag_index])
            score += np.einsum("nfp,fp->n", temporal_design, self._bagged_temporal_coefficients[bag_index])
            if static_design is not None:
                score += np.einsum("nsp,sp->n", static_design, self._bagged_static_coefficients[bag_index])
            probabilities.append(expit(score))
        return np.clip(np.stack(probabilities, axis=0), 1e-6, 1.0 - 1e-6)

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
        if not return_interval:
            value_basis = basis_design(self.basis, np.asarray(value_grid, dtype=np.float32),
                                       self._value_basis_a, self._value_basis_b)
            time_basis = basis_design(self.basis, np.asarray(normalized_time_grid, dtype=np.float32),
                                      self._time_basis_a, self._time_basis_b)
            coefficients = self._temporal_coefficients[feature_index].reshape(
                self.n_value_bases, self.n_time_bases
            )
            return value_basis @ coefficients @ time_basis.T

        surfaces = self._bagged_surfaces(feature_index, value_grid, normalized_time_grid)
        lower, upper = np.percentile(surfaces, [2.5, 97.5], axis=0)
        return lower, surfaces.mean(axis=0), upper

    def mean_value_effect_normalized(self, feature_index: int, value_grid,
                                     return_interval: bool = False):
        time_grid = np.linspace(0.0, 1.0, self.n_time_bases * 3)
        if not return_interval:
            surface = self.temporal_surface_normalized(feature_index, value_grid, time_grid)
            return surface.mean(axis=1)

        per_bag_effect = self._bagged_surfaces(feature_index, value_grid, time_grid).mean(axis=2)
        lower, upper = np.percentile(per_bag_effect, [2.5, 97.5], axis=0)
        return lower, per_bag_effect.mean(axis=0), upper
