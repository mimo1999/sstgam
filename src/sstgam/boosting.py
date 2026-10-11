"""Smooth tensor-product GAM backend for longitudinal features."""

from __future__ import annotations

import numpy as np
from scipy.special import expit
from sklearn.metrics import log_loss

from ._training import make_validation_split
from .bases import basis_design, make_basis_params, second_diff_matrix_np
from .normalization import apply_value_normalization, fit_value_normalization, static_stats

_NORMALIZATION_CLIP = 4.0
_RIDGE = 1e-6


class SharedShapeTemporalGAMBoost:
    """Cyclic gradient boosting over smooth value-time feature surfaces.

    Each coordinate-block update uses a fixed quasi-Newton step: the
    preconditioner (``_preconditioner``) inverts a penalized worst-case
    Hessian bound (``p = 0.5``) computed once per bag, not the true
    per-round Hessian ``p * (1 - p)``. This is a majorization-minimization /
    IRLS-style bound, not literal Newton's method -- the step direction is
    curvature-informed but the curvature estimate is not updated as the fit
    improves.

    ``outer_bags`` bagged fits (fixed validation split, bootstrapped
    training rows) are averaged at the end, as in EBM.
    """

    def __init__(
        self,
        n_value_bases: int = 8,
        n_time_bases: int = 5,
        n_static_bases: int = 8,
        basis: str = "rbf",
        value_transform: str = "quantile",
        temporal_penalty: float = 1.0,
        static_penalty: float = 1.0,
        learning_rate: float = 0.02,
        max_rounds: int = 100,
        outer_bags: int = 4,
        early_stopping_rounds: int = 15,
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
        self.learning_rate = learning_rate
        self.max_rounds = max_rounds
        self.outer_bags = outer_bags
        self.early_stopping_rounds = early_stopping_rounds
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
    def _preconditioner(design_matrix, sample_weight, penalty):
        weighted_design = design_matrix * (0.25 * sample_weight)[:, None]
        hessian_bound = design_matrix.T @ weighted_design + penalty
        return np.linalg.inv(hessian_bound)

    def _build_designs(self, values, observed, static_values):
        values = np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0.0)
        observed = np.asarray(observed, dtype=np.float32)
        n_samples, n_features, _ = values.shape

        normalized_values = (values - self._feature_center[None, :, None]) / self._feature_scale[None, :, None]
        np.clip(normalized_values, -self.clip, self.clip, out=normalized_values)

        temporal_design = np.empty(
            (n_samples, n_features, self.n_value_bases * self.n_time_bases), dtype=np.float32
        )
        for feature_index in range(n_features):
            value_basis = basis_design(self.basis, normalized_values[:, feature_index, :],
                                       self._value_basis_a, self._value_basis_b)
            masked_basis = value_basis * observed[:, feature_index, :, None]
            temporal_design[:, feature_index, :] = np.einsum(
                "ndp,dq->npq", masked_basis, self._time_basis
            ).reshape(n_samples, -1)

        if static_values.shape[1] == 0:
            return temporal_design, None
        static_values = np.asarray(static_values, dtype=np.float32)
        filled_static = np.where(np.isfinite(static_values), static_values, self._static_mean)
        normalized_static = (filled_static - self._static_mean) / self._static_scale
        np.clip(normalized_static, -self.clip, self.clip, out=normalized_static)
        static_design = np.stack([
            basis_design(self.basis, normalized_static[:, column_index],
                        self._static_basis_a, self._static_basis_b)
            for column_index in range(static_values.shape[1])
        ], axis=1)
        return temporal_design, static_design

    @staticmethod
    def _update_block(preconditioners, train_design, validation_design,
                      coefficients, train_score, validation_score,
                      target, sample_weight, learning_rate):
        for unit_index, preconditioner in enumerate(preconditioners):
            probability = expit(train_score)
            residual = sample_weight * (target - probability)
            coefficient_update = preconditioner @ (train_design[:, unit_index, :].T @ residual)
            coefficients[unit_index] += learning_rate * coefficient_update
            train_score += learning_rate * (train_design[:, unit_index, :] @ coefficient_update)
            validation_score += learning_rate * (validation_design[:, unit_index, :] @ coefficient_update)

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

        normalized_training_values, self._feature_center, self._feature_scale, self._quantile_tables = (
            fit_value_normalization(values, observed, self.value_transform, self.clip)
        )
        self._static_mean, self._static_scale = static_stats(static_values)
        value_span = self.clip if self.value_transform == "quantile" else 0.9 * self.clip
        self._value_basis_a, self._value_basis_b = make_basis_params(
            self.basis, -value_span, value_span, self.n_value_bases
        )
        self._static_basis_a, self._static_basis_b = make_basis_params(
            self.basis, -0.9 * self.clip, 0.9 * self.clip, self.n_static_bases
        )
        time_coordinates = np.asarray(time_coordinates, dtype=np.float32)
        time_range = max(float(time_coordinates.max() - time_coordinates.min()), 1.0)
        normalized_time = (time_coordinates - time_coordinates.min()) / time_range
        self._time_basis_a, self._time_basis_b = make_basis_params(self.basis, 0.0, 1.0, self.n_time_bases)
        self._time_basis = basis_design(self.basis, normalized_time, self._time_basis_a, self._time_basis_b)

        temporal_design, static_design = self._build_designs(
            normalized_training_values, observed, static_values
        )
        temporal_penalty = self._penalty_matrix(
            self.n_value_bases, self.n_time_bases, self.temporal_penalty
        )
        static_penalty = self._static_penalty(self.n_static_bases, self.static_penalty)

        # Fixed validation split; only the bootstrapped training rows vary by bag.
        train_indices, validation_indices = make_validation_split(
            target, self.val_fraction, self.random_state
        )

        rng = np.random.RandomState(self.random_state)
        self.best_rounds_ = []
        self._bagged_intercepts = []
        self._bagged_temporal_coefficients = []
        self._bagged_static_coefficients = []
        for _ in range(self.outer_bags):
            fit_indices = (train_indices if self.outer_bags == 1
                          else rng.choice(train_indices, size=train_indices.size, replace=True))
            intercept, temporal_coefficients, static_coefficients, best_round = self._fit_one_bag(
                temporal_design, static_design, target, sample_weight,
                fit_indices, validation_indices, n_features, n_static,
                temporal_penalty, static_penalty, rng,
            )
            self.best_rounds_.append(best_round)
            self._bagged_intercepts.append(intercept)
            self._bagged_temporal_coefficients.append(temporal_coefficients)
            self._bagged_static_coefficients.append(static_coefficients)

        self._intercept = float(np.mean(self._bagged_intercepts))
        self._temporal_coefficients = np.mean(self._bagged_temporal_coefficients, axis=0)
        self._static_coefficients = (None if static_design is None
                                     else np.mean(self._bagged_static_coefficients, axis=0))
        return self

    def _fit_one_bag(self, temporal_design, static_design, target, sample_weight,
                     fit_indices, validation_indices, n_features, n_static,
                     temporal_penalty, static_penalty, rng):
        fit_temporal = temporal_design[fit_indices]
        validation_temporal = temporal_design[validation_indices]
        fit_static = None if static_design is None else static_design[fit_indices]
        validation_static = None if static_design is None else static_design[validation_indices]
        fit_target, validation_target = target[fit_indices], target[validation_indices]
        fit_weight, validation_weight = sample_weight[fit_indices], sample_weight[validation_indices]

        temporal_preconditioners = [
            self._preconditioner(fit_temporal[:, feature_index, :], fit_weight, temporal_penalty)
            for feature_index in range(n_features)
        ]
        static_preconditioners = ([] if fit_static is None else [
            self._preconditioner(fit_static[:, column_index, :], fit_weight, static_penalty)
            for column_index in range(n_static)
        ])

        base_rate = np.clip(np.average(fit_target, weights=fit_weight), 1e-6, 1.0 - 1e-6)
        intercept = float(np.log(base_rate / (1.0 - base_rate)))
        temporal_coefficients = np.zeros((n_features, temporal_design.shape[2]))
        static_coefficients = None if static_design is None else np.zeros((n_static, static_design.shape[2]))
        fit_score = np.full(fit_indices.size, intercept)
        validation_score = np.full(validation_indices.size, intercept)

        best_loss = np.inf
        best_state = None
        best_round = 0
        rounds_without_improvement = 0
        for round_index in range(1, self.max_rounds + 1):
            probability = expit(fit_score)
            residual = fit_weight * (fit_target - probability)
            intercept_update = residual.sum() / max(0.25 * fit_weight.sum(), _RIDGE)
            intercept += self.learning_rate * intercept_update
            fit_score += self.learning_rate * intercept_update
            validation_score += self.learning_rate * intercept_update

            blocks = [(temporal_preconditioners, fit_temporal, validation_temporal, temporal_coefficients)]
            if static_coefficients is not None:
                blocks.append((static_preconditioners, fit_static, validation_static, static_coefficients))
            for preconditioners, fit_design, validation_design, coefficients in blocks:
                self._update_block(
                    preconditioners, fit_design, validation_design,
                    coefficients, fit_score, validation_score,
                    fit_target, fit_weight, self.learning_rate,
                )

            validation_loss = log_loss(validation_target, expit(validation_score),
                                       sample_weight=validation_weight, labels=[0, 1])
            if validation_loss < best_loss - 1e-6:
                best_loss = validation_loss
                best_state = (intercept, temporal_coefficients.copy(),
                              None if static_coefficients is None else static_coefficients.copy())
                best_round = round_index
                rounds_without_improvement = 0
            else:
                rounds_without_improvement += 1
                if rounds_without_improvement >= self.early_stopping_rounds:
                    break

        if best_state is None:
            best_state = (intercept, temporal_coefficients, static_coefficients)
        return (*best_state, best_round)

    def predict_proba(self, values, observed, static_values):
        normalized_values = apply_value_normalization(
            np.asarray(values, dtype=np.float32), self.value_transform,
            self._quantile_tables, self.clip,
        )
        temporal_design, static_design = self._build_designs(
            normalized_values, observed, static_values
        )
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
        normalized_values = apply_value_normalization(
            np.asarray(values, dtype=np.float32), self.value_transform,
            self._quantile_tables, self.clip,
        )
        temporal_design, static_design = self._build_designs(
            normalized_values, observed, static_values
        )
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
