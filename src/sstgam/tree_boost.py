"""Histogram-based value-time GAM backend."""

import numpy as np
from scipy.special import expit
from sklearn.tree import DecisionTreeRegressor

from ._training import boost_bag, bootstrap_bags, make_validation_split

_MIN_HESSIAN = 1e-6
_N_STATIC_BINS = 16

# Per-log-candidate-count penalty on split gain, correcting CART's bias
# toward axes with more candidate thresholds. Not a formal Bonferroni or BIC
# correction -- there is no p-value or likelihood term here -- just a
# candidate-count penalty on greedy SSE gain, in the same spirit as those
# corrections (more candidates give a greedy split more chances to look good
# by chance, so the penalty scales with log of the candidate count).
_AXIS_PENALTY_LAMBDA = 2.0

# Depth of whichever axis splits first each round; the rest of max_depth
# goes to the other axis.
_FIRST_AXIS_DEPTH = 2


def _quantile_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Create deduplicated quantile bin edges with at least one bin."""
    if values.size == 0:
        return np.array([0.0, 1.0])
    edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1)))
    if edges.size == 1:
        edges = np.array([edges[0] - 0.5, edges[0] + 0.5])
    return edges.astype(np.float64)


class SharedShapeTemporalGAMTreeBoost:
    """Cyclic boosted GAM with one value-bin by time-bin table per feature.

    Tables are updated via shallow regression trees fit to aggregated Newton
    statistics, with values binned per feature from pooled training data.

    Each feature update picks which axis (value or time) splits first via a
    depth-1 trial split scored by weighted-SSE gain minus a candidate-count
    penalty (see ``_AXIS_PENALTY_LAMBDA``). Static (1-D) terms use a plain
    single-stage tree.

    ``outer_bags`` bagged fits (fixed validation split, bootstrapped training
    rows) are averaged at the end, as in EBM.
    """

    def __init__(
        self,
        n_value_bins: int = 16,
        max_depth: int = 3,
        learning_rate: float = 0.02,
        max_rounds: int = 100,
        outer_bags: int = 4,
        early_stopping_rounds: int = 15,
        val_fraction: float = 0.15,
        random_state: int = 42,
    ) -> None:
        self.n_value_bins = n_value_bins
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.max_rounds = max_rounds
        self.outer_bags = outer_bags
        self.early_stopping_rounds = early_stopping_rounds
        self.val_fraction = val_fraction
        self.random_state = random_state

    @staticmethod
    def _bin(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
        return np.searchsorted(edges[1:-1], values, side="right").astype(np.int32)

    def _fit_bins(self, values, observed, static_values) -> None:
        self._value_edges = []
        for feature_index in range(values.shape[1]):
            feature_values = values[:, feature_index, :][observed[:, feature_index, :] > 0.5]
            self._value_edges.append(_quantile_edges(feature_values, self.n_value_bins))

        self._static_edges = []
        self._static_fill = np.zeros(static_values.shape[1], dtype=np.float64)
        for column_index in range(static_values.shape[1]):
            column = static_values[:, column_index]
            finite_values = column[np.isfinite(column)]
            self._static_fill[column_index] = np.median(finite_values) if finite_values.size else 0.0
            self._static_edges.append(_quantile_edges(finite_values, _N_STATIC_BINS))

    def _feature_bins(self, values) -> list[np.ndarray]:
        return [self._bin(np.nan_to_num(values[:, feature_index, :], nan=0.0), edges)
                for feature_index, edges in enumerate(self._value_edges)]

    def _static_column_bins(self, column_index: int, values) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        filled = np.where(np.isfinite(values), values, self._static_fill[column_index])
        return self._bin(filled, self._static_edges[column_index])

    def _static_bins(self, static_values) -> list[np.ndarray]:
        return [self._static_column_bins(column_index, static_values[:, column_index])
                for column_index in range(len(self._static_edges))]

    def static_effect(self, column_index: int, values) -> np.ndarray:
        """Log-odds effect of raw values of one static covariate (missing values use the median bin)."""
        return self._static_tables[column_index][self._static_column_bins(column_index, values)]

    def _tree_update(self, coordinates, gradient_sum, hessian_sum, rng):
        """Plain single-stage tree update -- used for 1-D (static) terms."""
        active = hessian_sum > _MIN_HESSIAN
        if active.sum() < 2:
            return None
        targets = gradient_sum[active] / hessian_sum[active]
        tree = DecisionTreeRegressor(
            max_depth=self.max_depth,
            random_state=rng.randint(0, 2**31 - 1),
        )
        tree.fit(coordinates[active], targets, sample_weight=hessian_sum[active])
        return tree.predict(coordinates)

    @staticmethod
    def _weighted_gain(target, weight, prediction) -> float:
        """Reduction in weighted SSE from the weighted mean to ``prediction``."""
        if weight.sum() <= 0:
            return 0.0
        mean = float(np.average(target, weights=weight))
        loss_root = float(np.sum(weight * (target - mean) ** 2))
        loss_leaf = float(np.sum(weight * (target - prediction) ** 2))
        return max(loss_root - loss_leaf, 0.0)

    def _adaptive_tree_update(self, coordinates, gradient_sum, hessian_sum, rng):
        """Joint (value_bin, normalized_time) update with a data-driven
        choice of which axis splits first each round.

        ``coordinates`` columns are ``[value_bin, normalized_time]``.
        """
        active = hessian_sum > _MIN_HESSIAN
        if active.sum() < 2:
            return None
        targets = np.divide(gradient_sum, hessian_sum,
                            out=np.zeros_like(gradient_sum), where=hessian_sum > 0)
        value_column, time_column = coordinates[:, 0:1], coordinates[:, 1:2]

        def fit_depth1(column):
            candidate_count = max(len(np.unique(column[active])), 2)
            tree = DecisionTreeRegressor(max_depth=1, random_state=rng.randint(0, 2**31 - 1))
            tree.fit(column[active], targets[active], sample_weight=hessian_sum[active])
            prediction = tree.predict(column[active])
            gain = self._weighted_gain(targets[active], hessian_sum[active], prediction)
            return tree, gain - _AXIS_PENALTY_LAMBDA * np.log(candidate_count)

        time_tree1, time_score = fit_depth1(time_column)
        value_tree1, value_score = fit_depth1(value_column)

        first_depth = max(1, min(_FIRST_AXIS_DEPTH, self.max_depth - 1))
        second_depth = max(1, self.max_depth - first_depth)
        if time_score >= value_score:
            first_column, candidate_tree = time_column, time_tree1
            second_column = value_column
        else:
            first_column, candidate_tree = value_column, value_tree1
            second_column = time_column

        if first_depth == 1:
            # Reuse the depth-1 candidate tree instead of refitting it.
            first_tree = candidate_tree
        else:
            first_tree = DecisionTreeRegressor(max_depth=first_depth, random_state=rng.randint(0, 2**31 - 1))
            first_tree.fit(first_column[active], targets[active], sample_weight=hessian_sum[active])
        groups = first_tree.apply(first_column)
        prediction = first_tree.predict(first_column)

        update = prediction.copy()
        for group in np.unique(groups):
            mask = groups == group
            group_active = mask & active
            if group_active.sum() < 2:
                continue
            second_tree = DecisionTreeRegressor(max_depth=second_depth, random_state=rng.randint(0, 2**31 - 1))
            second_tree.fit(second_column[group_active], targets[group_active], sample_weight=hessian_sum[group_active])
            update[mask] = second_tree.predict(second_column[mask])
        return update

    def _update_temporal_term(self, gradient, hessian, cell_ids, observed,
                              coordinates, n_cells, rng):
        weighted_gradient = (gradient[:, None] * observed).ravel()
        weighted_hessian = (hessian[:, None] * observed).ravel()
        flat_cells = cell_ids.ravel()
        gradient_sum = np.bincount(flat_cells, weights=weighted_gradient, minlength=n_cells)
        hessian_sum = np.bincount(flat_cells, weights=weighted_hessian, minlength=n_cells)
        return self._adaptive_tree_update(coordinates, gradient_sum, hessian_sum, rng)

    def _update_static_term(self, gradient, hessian, bins, coordinates, rng):
        n_bins = coordinates.shape[0]
        gradient_sum = np.bincount(bins, weights=gradient, minlength=n_bins)
        hessian_sum = np.bincount(bins, weights=hessian, minlength=n_bins)
        return self._tree_update(coordinates, gradient_sum, hessian_sum, rng)

    @staticmethod
    def _grad_hess(score, target, weight):
        probability = expit(score)
        gradient = weight * (target - probability)
        hessian = weight * probability * (1.0 - probability)
        return gradient, hessian

    def fit(self, values, observed, static_values, time_coordinates, target,
            sample_weight=None):
        values = np.asarray(values, dtype=np.float32)
        observed = np.asarray(observed, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64).ravel()
        n_samples, _, n_times = values.shape
        static_values = np.asarray(static_values, dtype=np.float32)
        weights = (np.ones(n_samples, dtype=np.float64) if sample_weight is None
                   else np.asarray(sample_weight, dtype=np.float64).ravel())

        self._n_times = n_times
        self._fit_bins(values, observed, static_values)
        feature_bins = self._feature_bins(values)
        static_bins = self._static_bins(static_values)
        n_value_bins = [len(edges) - 1 for edges in self._value_edges]

        time_coordinates = np.asarray(time_coordinates, dtype=np.float64)
        time_scale = max(float(time_coordinates.max() - time_coordinates.min()), 1.0)
        normalized_time = (time_coordinates - time_coordinates.min()) / time_scale
        temporal_coordinates = [
            np.column_stack((
                np.repeat(np.arange(bin_count), n_times),
                np.tile(normalized_time, bin_count),
            ))
            for bin_count in n_value_bins
        ]
        temporal_cell_ids = [
            bins * n_times + np.arange(n_times, dtype=np.int32)[None, :]
            for bins in feature_bins
        ]
        static_coordinates = [np.arange(len(edges) - 1, dtype=np.float64)[:, None]
                              for edges in self._static_edges]

        # Fixed validation split; only the bootstrapped training rows vary by bag.
        train_indices, validation_indices = make_validation_split(
            target, self.val_fraction, self.random_state
        )

        rng = np.random.RandomState(self.random_state)
        bags = [
            self._fit_one_bag(weights, target, observed, fit_indices, validation_indices,
                              n_value_bins, n_times, static_bins, temporal_coordinates,
                              temporal_cell_ids, static_coordinates, rng)
            for fit_indices in bootstrap_bags(train_indices, self.outer_bags, rng)
        ]
        self._bagged_feature_tables = [feature_tables for _, (feature_tables, _), _ in bags]
        self.best_rounds_ = [best_round for _, _, best_round in bags]

        self._intercept = float(np.mean([intercept for intercept, _, _ in bags]))
        self._feature_tables = [np.mean(tables, axis=0) for tables in zip(*self._bagged_feature_tables)]
        self._static_tables = [np.mean(tables, axis=0)
                               for tables in zip(*(static_tables for _, (_, static_tables), _ in bags))]
        return self

    def _fit_one_bag(self, weights, target, observed, fit_indices, validation_indices,
                     n_value_bins, n_times, static_bins, temporal_coordinates,
                     temporal_cell_ids, static_coordinates, rng):
        feature_tables = [np.zeros(bin_count * n_times) for bin_count in n_value_bins]
        static_tables = [np.zeros(len(edges) - 1) for edges in self._static_edges]
        fit_target, fit_weight = target[fit_indices], weights[fit_indices]
        fit_observed, validation_observed = observed[fit_indices], observed[validation_indices]
        fit_cells = [cell_ids[fit_indices] for cell_ids in temporal_cell_ids]
        validation_cells = [cell_ids[validation_indices] for cell_ids in temporal_cell_ids]

        def update_terms(fit_score, validation_score):
            for feature_index, table in enumerate(feature_tables):
                gradient, hessian = self._grad_hess(fit_score, fit_target, fit_weight)
                update = self._update_temporal_term(
                    gradient, hessian, fit_cells[feature_index], fit_observed[:, feature_index, :],
                    temporal_coordinates[feature_index], table.size, rng,
                )
                if update is None:
                    continue
                update *= self.learning_rate
                table += update
                fit_score += (update[fit_cells[feature_index]] * fit_observed[:, feature_index, :]).sum(axis=1)
                validation_score += (update[validation_cells[feature_index]] *
                                     validation_observed[:, feature_index, :]).sum(axis=1)

            for column_index, table in enumerate(static_tables):
                gradient, hessian = self._grad_hess(fit_score, fit_target, fit_weight)
                update = self._update_static_term(
                    gradient, hessian, static_bins[column_index][fit_indices],
                    static_coordinates[column_index], rng,
                )
                if update is None:
                    continue
                update *= self.learning_rate
                table += update
                fit_score += update[static_bins[column_index][fit_indices]]
                validation_score += update[static_bins[column_index][validation_indices]]

        def snapshot():
            return [table.copy() for table in feature_tables], [table.copy() for table in static_tables]

        return boost_bag(update_terms, snapshot, fit_target, fit_weight,
                         target[validation_indices], weights[validation_indices],
                         self.learning_rate, self.max_rounds, self.early_stopping_rounds)

    def predict_proba(self, values, observed, static_values):
        values = np.asarray(values, dtype=np.float32)
        observed = np.asarray(observed, dtype=np.float64)
        score = np.full(values.shape[0], self._intercept)
        for feature_index, bins in enumerate(self._feature_bins(values)):
            cell_ids = bins * self._n_times + np.arange(self._n_times)[None, :]
            score += (self._feature_tables[feature_index][cell_ids] *
                      observed[:, feature_index, :]).sum(axis=1)
        if self._static_tables:
            for column_index, bins in enumerate(self._static_bins(static_values)):
                score += self._static_tables[column_index][bins]
        return np.clip(expit(score), 1e-6, 1.0 - 1e-6)

    def value_time_table(self, feature_index: int, return_interval: bool = False):
        """Return ``(value_bin_edges, log_odds_table)`` for one feature.

        With ``return_interval=True``, returns
        ``(value_bin_edges, lower, mean, upper)`` -- a 2.5/97.5 bootstrap
        percentile band computed across the ``outer_bags`` bagged fits
        instead of a single table. With ``outer_bags=1`` the band collapses
        to the point estimate.
        """
        if not return_interval:
            return (self._value_edges[feature_index],
                    self._feature_tables[feature_index].reshape(-1, self._n_times))

        tables = np.stack([
            bag_tables[feature_index].reshape(-1, self._n_times)
            for bag_tables in self._bagged_feature_tables
        ], axis=0)
        lower, upper = np.percentile(tables, [2.5, 97.5], axis=0)
        return self._value_edges[feature_index], lower, tables.mean(axis=0), upper
