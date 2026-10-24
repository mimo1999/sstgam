"""Histogram-based value-time GAM backend."""

import numpy as np
from sklearn.tree import DecisionTreeRegressor

from ._training import (
    FAMILIES,
    Family,
    bootstrap_bags,
    center_pair,
    distinct_edges,
    fast_gain,
    make_validation_split,
    pair_cells,
    percentile_band,
    quantile_edges,
)

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


def boost_bag(family: Family, update_terms, snapshot, fit_target, fit_weight, validation_target,
              validation_weight, learning_rate: float, max_rounds: int, patience: int, start=None):
    """Boost one bag until validation loss stops improving for ``patience`` rounds.

    Starts from the mean-outcome intercept, or from ``start = (intercept,
    fit_score, validation_score)`` of an already fitted model, which also
    counts as round 0. Each round takes a damped Newton step on the
    intercept, then calls ``update_terms(fit_score, validation_score)``,
    which updates the model terms and both score arrays in place. Returns
    ``(intercept, snapshot(), best_round)`` from the best validation round.
    """
    best_loss, best, rounds_without_improvement = np.inf, None, 0
    if start is None:
        intercept = family.link(np.average(fit_target, weights=fit_weight))
        fit_score = np.full(fit_target.size, intercept)
        validation_score = np.full(validation_target.size, intercept)
    else:
        intercept, fit_score, validation_score = start[0], start[1].copy(), start[2].copy()
        best_loss = np.average(family.loss(validation_score, validation_target), weights=validation_weight)
        best = (intercept, snapshot(), 0)

    for round_index in range(1, max_rounds + 1):
        mean = family.mean(fit_score)
        step = learning_rate * (np.sum(fit_weight * (fit_target - mean)) /
                                max(np.sum(fit_weight * family.variance(mean)), 1e-6))
        intercept += step
        fit_score += step
        validation_score += step
        update_terms(fit_score, validation_score)

        loss = np.average(family.loss(validation_score, validation_target), weights=validation_weight)
        if loss < best_loss - 1e-6:
            best_loss, best, rounds_without_improvement = loss, (intercept, snapshot(), round_index), 0
        else:
            rounds_without_improvement += 1
            if rounds_without_improvement >= patience:
                break
    return best if best is not None else (intercept, snapshot(), 0)


class SharedShapeTemporalGAMTreeBoost:
    """Cyclic boosted GAM with one value-bin by time-bin table per feature.

    Tables are updated via shallow regression trees fit to aggregated Newton
    statistics, with values binned per feature from pooled training data.

    Each feature update picks which axis (value or time) splits first via a
    depth-1 trial split scored by weighted-SSE gain minus a candidate-count
    penalty (see ``_AXIS_PENALTY_LAMBDA``). Static (1-D) terms use a plain
    single-stage tree.

    With ``min_gain > 0`` a table update is skipped (that term is left alone
    this round) unless its Newton gain reaches ``min_gain``; see ``_gated``.
    ``bin_edges="distinct"`` keeps few-level and mass-point values in their
    own bins; see ``_training.distinct_edges``.

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
        family: str = "binomial",
        bag_validation: str = "fixed",
        min_leaf_hessian: float | None = None,
        max_leaves: int | None = None,
        missing_effects: bool | str = False,
        min_gain: float = 0.0,
        bin_edges: str = "quantile",
    ) -> None:
        self.family = family
        self.min_gain = min_gain
        self.bin_edges = bin_edges
        self.missing_effects = missing_effects
        self.bag_validation = bag_validation
        self.min_leaf_hessian = min_leaf_hessian
        self.max_leaves = max_leaves
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

    def _edges(self, values, n_bins: int) -> np.ndarray:
        return (distinct_edges if self.bin_edges == "distinct" else quantile_edges)(values, n_bins)

    def _fit_bins(self, values, observed, static_values) -> None:
        self._value_edges = []
        for feature_index in range(values.shape[1]):
            feature_values = values[:, feature_index, :][observed[:, feature_index, :] > 0.5]
            self._value_edges.append(self._edges(feature_values, self.n_value_bins))

        self._static_edges = []
        self._static_fill = np.zeros(static_values.shape[1], dtype=np.float64)
        for column_index in range(static_values.shape[1]):
            column = static_values[:, column_index]
            finite_values = column[np.isfinite(column)]
            self._static_fill[column_index] = np.median(finite_values) if finite_values.size else 0.0
            self._static_edges.append(self._edges(finite_values, _N_STATIC_BINS))

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

    def static_effect(self, column_index: int, values, return_interval: bool = False):
        """Effect of raw values of one static covariate (missing values use the median bin)."""
        bins = self._static_column_bins(column_index, values)
        if not return_interval:
            return self._static_tables[column_index][bins]
        return percentile_band(np.stack([tables[column_index][bins] for tables in self._bagged_static_tables]))

    def _leaf_limits(self, hessian_total: float) -> dict:
        """Optional leaf constraints for the regression trees (none by default)."""
        limits = {}
        if self.min_leaf_hessian:
            limits["min_weight_fraction_leaf"] = min(0.5, self.min_leaf_hessian / max(hessian_total, 1e-12))
        if self.max_leaves:
            limits["max_leaf_nodes"] = self.max_leaves
        return limits

    def _gated(self, update, gradient_sum, hessian_sum):
        """``update``, or ``None`` (skip this term this round) if its Newton gain is below ``min_gain``.

        The gain ``sum_c (2 g_c u_c - h_c u_c^2)`` over cells is twice the second-order loss decrease of the
        step; for tree leaf values ``u = G/H`` it equals ``sum_leaf G^2/H``, a Wald chi-square of the step
        against zero (unit dispersion). Gaussian gains are divided by the outcome variance.
        """
        if update is None or self.min_gain <= 0:
            return update
        gain = float(np.sum(2.0 * gradient_sum * update - hessian_sum * update ** 2))
        return update if gain / self._dispersion >= self.min_gain else None

    def _tree_update(self, coordinates, gradient_sum, hessian_sum, rng):
        """Plain single-stage tree update -- used for 1-D (static) terms."""
        active = hessian_sum > _MIN_HESSIAN
        if active.sum() < 2:
            return None
        targets = gradient_sum[active] / hessian_sum[active]
        tree = DecisionTreeRegressor(
            max_depth=self.max_depth,
            random_state=rng.randint(0, 2**31 - 1),
            **self._leaf_limits(hessian_sum[active].sum()),
        )
        tree.fit(coordinates[active], targets, sample_weight=hessian_sum[active])
        return self._gated(tree.predict(coordinates), gradient_sum, hessian_sum)

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
        limits = self._leaf_limits(hessian_sum[active].sum())

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
            first_tree = DecisionTreeRegressor(max_depth=first_depth, random_state=rng.randint(0, 2**31 - 1),
                                               **limits)
            first_tree.fit(first_column[active], targets[active], sample_weight=hessian_sum[active])
        groups = first_tree.apply(first_column)
        prediction = first_tree.predict(first_column)

        update = prediction.copy()
        for group in np.unique(groups):
            mask = groups == group
            group_active = mask & active
            if group_active.sum() < 2:
                continue
            second_tree = DecisionTreeRegressor(max_depth=second_depth, random_state=rng.randint(0, 2**31 - 1),
                                                **limits)
            second_tree.fit(second_column[group_active], targets[group_active], sample_weight=hessian_sum[group_active])
            update[mask] = second_tree.predict(second_column[mask])
        return self._gated(update, gradient_sum, hessian_sum)

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

    def _grad_hess(self, score, target, weight):
        family = FAMILIES[self.family]
        mean = family.mean(score)
        return weight * (target - mean), weight * family.variance(mean)

    def _axis_bins(self, axis, raw):
        """Bin raw values of one pair axis with that term's own main-effect bins: ``(bins, n_bins)``."""
        kind, index = axis
        if kind == "temporal":
            edges = self._value_edges[index]
            return self._bin(np.nan_to_num(raw, nan=0.0), edges), len(edges) - 1
        filled = np.where(np.isfinite(raw), raw, self._static_fill[index])
        return self._bin(filled, self._static_edges[index]), len(self._static_edges[index]) - 1

    def _pair_bins(self, values, observed, static_values, pair):
        """``(cell ids, mask, n_u, n_w)`` of one pair; cell id is ``u_bin * n_w + w_bin``."""
        u, w, mask = pair_cells(values, observed, static_values, pair)
        (u_bins, n_u), (w_bins, n_w) = self._axis_bins(pair[0], u), self._axis_bins(pair[1], w)
        return u_bins * n_w + w_bins, np.asarray(mask, dtype=np.float64), n_u, n_w

    def _score(self, values, observed, static_values):
        return self._intercept + sum(term.sum(axis=tuple(range(1, term.ndim)))
                                     for term in self.contributions(values, observed, static_values))

    def fit(self, values, observed, static_values, time_coordinates, target,
            sample_weight=None, pairs=(), n_interactions: int = 0):
        """Fit main effects; then, with ``n_interactions``, the best of the candidate ``pairs`` too.

        Pairs are picked by FAST on the main-effects residual and fitted in a
        second boosting stage that starts from the averaged main effects and
        keeps updating them, with each pair update purified (``center_pair``)
        so a pair only carries what its main effects cannot.
        """
        values = np.asarray(values, dtype=np.float32)
        observed = np.asarray(observed, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64).ravel()
        n_samples, _, n_times = values.shape
        static_values = np.asarray(static_values, dtype=np.float32)
        weights = (np.ones(n_samples, dtype=np.float64) if sample_weight is None
                   else np.asarray(sample_weight, dtype=np.float64).ravel())

        self._n_times = n_times
        # binomial and poisson have unit dispersion; a gaussian gain is in outcome units squared
        self._dispersion = (max(float(np.cov(target, aweights=weights)), 1e-12) if self.family == "gaussian" else 1.0)
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
        # "not measured" tables are functions of time only, fitted like a static term over the time axis
        missing_coordinates = normalized_time[:, None] if self.missing_effects else None

        # "fixed": one validation split, only the bootstrapped training rows vary by bag.
        # "per_bag": every bag draws its own validation split, as EBM does.
        train_indices, validation_indices = make_validation_split(
            target, self.val_fraction, self.random_state
        )

        rng = np.random.RandomState(self.random_state)
        self._pairs, self._pair_shapes = [], []

        def bag_rows():
            if self.bag_validation == "fixed":
                for fit_indices in bootstrap_bags(train_indices, self.outer_bags, rng):
                    yield fit_indices, validation_indices
                return
            for bag in range(self.outer_bags):
                bag_train, bag_validation = make_validation_split(target, self.val_fraction, self.random_state + bag)
                yield (bag_train if self.outer_bags == 1
                       else rng.choice(bag_train, size=bag_train.size, replace=True)), bag_validation

        def fit_bags(start=None, pair_data=()):
            bags = [
                self._fit_one_bag(weights, target, observed, fit_indices, bag_validation_indices,
                                  n_value_bins, n_times, static_bins, temporal_coordinates,
                                  temporal_cell_ids, static_coordinates, rng, start, pair_data,
                                  missing_coordinates)
                for fit_indices, bag_validation_indices in bag_rows()
            ]
            self._bagged_feature_tables = [feature_tables for _, (feature_tables, _, _, _), _ in bags]
            self._bagged_static_tables = [static_tables for _, (_, static_tables, _, _), _ in bags]
            self._bagged_pair_tables = [pair_tables for _, (_, _, pair_tables, _), _ in bags]
            self._bagged_missing_tables = [missing_tables for _, (_, _, _, missing_tables), _ in bags]
            self.best_rounds_ = [best_round for _, _, best_round in bags]
            self._intercept = float(np.mean([intercept for intercept, _, _ in bags]))
            self._feature_tables = [np.mean(tables, axis=0) for tables in zip(*self._bagged_feature_tables)]
            self._static_tables = [np.mean(tables, axis=0) for tables in zip(*self._bagged_static_tables)]
            self._pair_tables = [np.mean(tables, axis=0) for tables in zip(*self._bagged_pair_tables)]
            self._missing_tables = [np.mean(tables, axis=0) for tables in zip(*self._bagged_missing_tables)]

        fit_bags()
        if n_interactions and pairs:
            score = self._score(values, observed, static_values)
            pair_data = [self._pair_bins(values, observed, static_values, pair) for pair in pairs]
            if n_interactions < len(pairs):
                # rank on the rows the bags fit from, so the validation rows stay unseen by the pair choice
                gradient, hessian = self._grad_hess(score[train_indices], target[train_indices], weights[train_indices])
                gains = [fast_gain(gradient, hessian, cells[train_indices], mask[train_indices], n_u, n_w)
                         for cells, mask, n_u, n_w in pair_data]
                keep = np.argsort(gains, kind="stable")[::-1][:n_interactions]
                pairs, pair_data = [pairs[i] for i in keep], [pair_data[i] for i in keep]
            self._pairs, self._pair_shapes = list(pairs), [(n_u, n_w) for _, _, n_u, n_w in pair_data]
            fit_bags(start=(self._intercept, [table.copy() for table in self._feature_tables],
                            [table.copy() for table in self._static_tables], score,
                            [table.copy() for table in self._missing_tables]), pair_data=pair_data)
        return self

    def _fit_one_bag(self, weights, target, observed, fit_indices, validation_indices,
                     n_value_bins, n_times, static_bins, temporal_coordinates,
                     temporal_cell_ids, static_coordinates, rng, start=None, pair_data=(),
                     missing_coordinates=None):
        if start is None:
            feature_tables = [np.zeros(bin_count * n_times) for bin_count in n_value_bins]
            static_tables = [np.zeros(len(edges) - 1) for edges in self._static_edges]
            missing_tables = ([np.zeros(n_times) for _ in n_value_bins] if missing_coordinates is not None else [])
        else:
            feature_tables = [table.copy() for table in start[1]]
            static_tables = [table.copy() for table in start[2]]
            missing_tables = [table.copy() for table in start[4]]
        pair_tables = [np.zeros(n_u * n_w) for _, _, n_u, n_w in pair_data]
        pair_coordinates = [np.column_stack((np.repeat(np.arange(n_u), n_w), np.tile(np.arange(n_w), n_u)))
                            .astype(np.float64) for _, _, n_u, n_w in pair_data]
        pair_weights = [np.bincount(cells[fit_indices].ravel(), mask[fit_indices].ravel(), n_u * n_w)
                        .reshape(n_u, n_w) for cells, mask, n_u, n_w in pair_data]
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

            # each group of features shares one "not measured" table: every feature alone, or all ("pooled")
            groups = ([list(range(len(missing_tables)))] if self.missing_effects == "pooled" and missing_tables
                      else [[k] for k in range(len(missing_tables))])
            for group in groups:
                gradient, hessian = self._grad_hess(fit_score, fit_target, fit_weight)
                fit_missing = (1.0 - fit_observed[:, group, :]).sum(axis=1)
                update = self._tree_update(missing_coordinates, gradient @ fit_missing, hessian @ fit_missing, rng)
                if update is None:
                    continue
                update *= self.learning_rate
                for feature_index in group:
                    missing_tables[feature_index] += update
                fit_score += fit_missing @ update
                validation_score += (1.0 - validation_observed[:, group, :]).sum(axis=1) @ update

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

            for pair_index, table in enumerate(pair_tables):
                cells, mask, n_u, n_w = pair_data[pair_index]
                gradient, hessian = self._grad_hess(fit_score, fit_target, fit_weight)
                fit_cells_pair, fit_mask = cells[fit_indices].ravel(), mask[fit_indices]
                update = self._tree_update(
                    pair_coordinates[pair_index],
                    np.bincount(fit_cells_pair, (gradient[:, None] * fit_mask).ravel(), table.size),
                    np.bincount(fit_cells_pair, (hessian[:, None] * fit_mask).ravel(), table.size), rng,
                )
                if update is None:
                    continue
                update = center_pair(self.learning_rate * update.reshape(n_u, n_w), pair_weights[pair_index]).ravel()
                table += update
                fit_score += (update[cells[fit_indices]] * fit_mask).sum(axis=1)
                validation_score += (update[cells[validation_indices]] * mask[validation_indices]).sum(axis=1)

        def snapshot():
            return ([table.copy() for table in feature_tables], [table.copy() for table in static_tables],
                    [table.copy() for table in pair_tables], [table.copy() for table in missing_tables])

        return boost_bag(FAMILIES[self.family], update_terms, snapshot, fit_target, fit_weight,
                         target[validation_indices], weights[validation_indices],
                         self.learning_rate, self.max_rounds, self.early_stopping_rounds,
                         start=None if start is None else (start[0], start[3][fit_indices],
                                                           start[3][validation_indices]))

    def contributions(self, values, observed, static_values):
        """Per-cell temporal, per-column static, per-pair and per-cell "not measured" terms.

        Shapes ``(n, features, times)``, ``(n, statics)``, ``(n, pairs)`` and ``(n, features, times)``; with
        ``_intercept`` they sum to the linear predictor. A missing cell adds nothing to ``temporal``; with
        ``missing_effects`` it adds its "not measured" effect to the last array instead.
        """
        values = np.asarray(values, dtype=np.float32)
        observed = np.asarray(observed, dtype=np.float64)
        temporal = np.zeros(values.shape, dtype=np.float64)
        for feature_index, bins in enumerate(self._feature_bins(values)):
            cell_ids = bins * self._n_times + np.arange(self._n_times)[None, :]
            temporal[:, feature_index, :] = self._feature_tables[feature_index][cell_ids] * observed[:, feature_index, :]
        static_values = np.asarray(static_values)
        static = np.zeros(static_values.shape, dtype=np.float64)
        for column_index, bins in enumerate(self._static_bins(static_values)):
            static[:, column_index] = self._static_tables[column_index][bins]
        pairs = np.zeros((values.shape[0], len(self._pairs)))
        for pair_index, pair in enumerate(self._pairs):
            cells, mask, _, _ = self._pair_bins(values, observed, static_values, pair)
            pairs[:, pair_index] = (self._pair_tables[pair_index][cells] * mask).sum(axis=1)
        missing = np.zeros(values.shape, dtype=np.float64)
        for feature_index, table in enumerate(self._missing_tables):
            missing[:, feature_index, :] = (1.0 - observed[:, feature_index, :]) * table[None, :]
        return temporal, static, pairs, missing

    def missing_surface(self, feature_index: int, time_positions, return_interval: bool = False):
        """The fitted "not measured" effect of one feature at the given time positions (indices of the time ids)."""
        positions = np.asarray(time_positions, dtype=int)
        tables = np.stack([bag[feature_index][positions] for bag in self._bagged_missing_tables])
        return percentile_band(tables) if return_interval else self._missing_tables[feature_index][positions]

    def interaction_surface(self, pair_index: int, u_values, w_values, return_interval: bool = False):
        """Pair table looked up on a grid of raw values: ``(len(u_values), len(w_values))``."""
        pair, (n_u, n_w) = self._pairs[pair_index], self._pair_shapes[pair_index]
        u_bins, _ = self._axis_bins(pair[0], np.asarray(u_values, dtype=np.float64))
        w_bins, _ = self._axis_bins(pair[1], np.asarray(w_values, dtype=np.float64))
        tables = np.stack([bag[pair_index].reshape(n_u, n_w)[np.ix_(u_bins, w_bins)]
                           for bag in self._bagged_pair_tables])
        return percentile_band(tables) if return_interval else self._pair_tables[pair_index].reshape(
            n_u, n_w)[np.ix_(u_bins, w_bins)]

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
        return (self._value_edges[feature_index], *percentile_band(tables))
