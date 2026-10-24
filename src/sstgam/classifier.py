"""Public SSTGAM estimators: a binary classifier and a regressor for counts or continuous outcomes."""

import itertools
import re
import warnings
from collections.abc import Iterable
from typing import NamedTuple

import numpy as np
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted

from ._training import FAMILIES
from .boosting import SharedShapeTemporalGAMBoost
from .tree_boost import SharedShapeTemporalGAMTreeBoost

_TEMPORAL_COLUMN_PATTERN = re.compile(r"^(.+)_(\d+(?:\.\d+)?)$")

# A point estimate, or a ``(lower, mean, upper)`` band when ``return_interval=True``.
_Effect = np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]


class Explanation(NamedTuple):
    """Per-row additive breakdown of the linear predictor (log-odds for ``SSTGAM``).

    ``intercept + temporal.sum(axis=(1, 2)) + static.sum(axis=1) +
    pairs.sum(axis=1) + missing.sum(axis=(1, 2))`` equals ``decision_function(X)``.
    ``temporal[i, k, t]`` is what row ``i``'s value of ``temporal_features_[k]``
    at ``time_ids_[t]`` adds (0 when missing); ``static[i, s]`` is what
    ``static_features_[s]`` adds; ``pairs[i, j]`` is what interaction
    ``interaction_terms_[j]`` adds; ``missing[i, k, t]`` is what *not* measuring
    feature ``k`` at time ``t`` adds (all zero unless ``missing_effects=True``).
    """

    intercept: float
    temporal: np.ndarray
    static: np.ndarray
    pairs: np.ndarray
    missing: np.ndarray


class _SSTGAMBase(BaseEstimator):
    """Column parsing, fitting and inspection shared by ``SSTGAM`` and ``SSTGAMRegressor``."""

    _family = "binomial"

    def __init__(
        self,
        method: str = "spline",
        n_value_bases: int = 8,
        n_time_bases: int = 5,
        n_static_bases: int = 8,
        n_value_bins: int = 16,
        tree_depth: int = 3,
        basis: str = "bspline",
        value_transform: str = "quantile",
        temporal_penalty: float | str = "auto",
        static_penalty: float | str = "auto",
        learning_rate: float = 0.02,
        max_rounds: int = 100,
        outer_bags: int = 4,
        early_stopping_rounds: int = 15,
        include_missingness_indicators: bool = False,
        val_fraction: float = 0.15,
        static_columns: Iterable[str] | None = None,
        interactions: int | Iterable[tuple[str, str]] = 0,
        n_pair_bases: int = 6,
        penalty_sharing: str = "per_term",
        missing_effects: bool | str = False,
        bag_validation: str = "fixed",
        min_leaf_hessian: float | None = None,
        max_leaves: int | None = None,
        min_gain: float = 0.0,
        bin_edges: str = "quantile",
        context_channels: Iterable[str] = (),
        context_order: str = "ascending",
        random_state: int = 42,
    ) -> None:
        self.method = method
        self.n_value_bases = n_value_bases
        self.n_time_bases = n_time_bases
        self.n_static_bases = n_static_bases
        self.n_value_bins = n_value_bins
        self.tree_depth = tree_depth
        self.basis = basis
        self.value_transform = value_transform
        self.temporal_penalty = temporal_penalty
        self.static_penalty = static_penalty
        self.learning_rate = learning_rate
        self.max_rounds = max_rounds
        self.outer_bags = outer_bags
        self.early_stopping_rounds = early_stopping_rounds
        self.include_missingness_indicators = include_missingness_indicators
        self.val_fraction = val_fraction
        self.static_columns = static_columns
        self.interactions = interactions
        self.n_pair_bases = n_pair_bases
        self.penalty_sharing = penalty_sharing
        self.missing_effects = missing_effects
        self.bag_validation = bag_validation
        self.min_leaf_hessian = min_leaf_hessian
        self.max_leaves = max_leaves
        self.min_gain = min_gain
        self.bin_edges = bin_edges
        self.context_channels = context_channels
        self.context_order = context_order
        self.random_state = random_state

    def _parse_columns(self, column_names: list[str]):
        forced_static = set(self.static_columns or ())
        unknown = sorted(forced_static - set(column_names))
        if unknown:
            raise ValueError(f"static_columns not found in X: {unknown}")
        temporal_columns: dict[str, dict[float, int]] = {}
        static_columns: list[int] = []
        for column_index, column_name in enumerate(column_names):
            match = None if column_name in forced_static else _TEMPORAL_COLUMN_PATTERN.match(column_name)
            if match is None:
                static_columns.append(column_index)
                continue
            suffix = match.group(2)
            feature_name, time_id = match.group(1), float(suffix) if "." in suffix else int(suffix)
            feature_columns = temporal_columns.setdefault(feature_name, {})
            if time_id in feature_columns:
                raise ValueError(f"Duplicate temporal column: {column_name!r}")
            feature_columns[time_id] = column_index
        # A lone column at a time no other feature uses is usually a static name like "icd_10".
        for feature_name, columns in temporal_columns.items():
            if len(columns) != 1:
                continue
            (time_id, column_index), = columns.items()
            if sum(time_id in other for other in temporal_columns.values()) == 1:
                warnings.warn(
                    f"Column {column_names[column_index]!r} is parsed as temporal feature {feature_name!r} at "
                    f"time {time_id}, the only column at that time. If it is static, list it in static_columns.",
                    stacklevel=4,
                )
        return temporal_columns, static_columns

    @staticmethod
    def _as_float_matrix(X) -> np.ndarray:
        matrix = np.asarray(X.values if hasattr(X, "values") else X, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError("X must be a two-dimensional matrix.")
        if np.isinf(matrix).any():
            raise ValueError("X contains infinite values; use NaN for missing values.")
        return matrix

    def _prepare_fit_input(self, X, feature_names: Iterable[str] | None):
        matrix = self._as_float_matrix(X)
        if feature_names is not None:
            names = list(feature_names)
        elif hasattr(X, "columns"):
            names = list(X.columns)
        else:
            # Made-up names would match the temporal pattern and silently misparse every column.
            raise ValueError(
                "X has no column names. Pass a DataFrame, or the names through feature_names=. "
                "In a Pipeline, call set_output(transform='pandas') on the preceding steps."
            )
        names = [str(name) for name in names]
        if len(names) != matrix.shape[1]:
            raise ValueError("feature_names must contain one name per column in X.")
        if len(set(names)) != len(names):
            raise ValueError("Feature names must be unique.")
        return matrix, names

    def _prepare_predict_input(self, X) -> np.ndarray:
        if hasattr(X, "columns"):
            received = list(X.columns)
            missing = [name for name in self.feature_names_in_ if name not in received]
            extra = [name for name in received if name not in self.feature_names_in_]
            if missing or extra:
                raise ValueError(
                    f"Prediction columns differ from fitted columns; missing={missing}, extra={extra}."
                )
            X = X.loc[:, self.feature_names_in_]
        matrix = self._as_float_matrix(X)
        if matrix.shape[1] != len(self.feature_names_in_):
            raise ValueError("X has a different number of columns than the fitted data.")
        return matrix

    def _build_tensors(self, matrix: np.ndarray):
        n_samples = matrix.shape[0]
        n_features = len(self._base_features_)
        n_times = len(self.time_ids_)
        values = np.zeros((n_samples, n_features, n_times), dtype=np.float32)
        observed = np.zeros_like(values)
        time_positions = {time_id: index for index, time_id in enumerate(self.time_ids_)}

        for feature_index, feature_name in enumerate(self._base_features_):
            for time_id, column_index in self._temporal_columns[feature_name].items():
                column = matrix[:, column_index]
                is_observed = np.isfinite(column)
                time_index = time_positions[time_id]
                values[:, feature_index, time_index] = np.where(is_observed, column, 0.0)
                observed[:, feature_index, time_index] = is_observed

        if self._static_column_indices:
            static_values = matrix[:, self._static_column_indices]
        else:
            static_values = np.empty((n_samples, 0), dtype=np.float32)
        static_values = static_values.astype(np.float32, copy=False)

        if self.include_missingness_indicators:
            missingness_fraction = observed.mean(axis=2).astype(np.float32)
            static_values = np.concatenate([static_values, missingness_fraction], axis=1)

        if self.context_channels:
            derived_values, derived_observed = self._context_tensors(values, observed)
            values = np.concatenate([values, derived_values], axis=1)
            observed = np.concatenate([observed, derived_observed], axis=1)
        return values, observed, static_values

    def _context_tensors(self, values: np.ndarray, observed: np.ndarray):
        """History-relative channels, each a pseudo-feature with its own value-by-time surface.

        Cells are visited in chronological order (``context_order``) and every channel at a cell uses only that
        cell and strictly earlier ones, so nothing looks ahead. ``"change"``: at an observed cell with an earlier
        observation, the value minus the previous observed value. ``"staleness"``: at every cell from the first
        observation on, the time since the last observation (0 at an observed cell).
        """
        n_samples, n_features, n_times = values.shape
        ids = np.asarray(self.time_ids_, dtype=np.float64)
        order = np.arange(n_times) if self.context_order == "ascending" else np.arange(n_times)[::-1]
        last_value = np.zeros((n_samples, n_features), dtype=np.float32)
        last_time = np.zeros((n_samples, n_features))
        has_previous = np.zeros((n_samples, n_features), dtype=bool)
        change = np.zeros_like(values)
        change_seen = np.zeros_like(values)
        stale = np.zeros_like(values)
        stale_seen = np.zeros_like(values)
        for t in order:
            seen = observed[:, :, t] > 0
            usable = seen & has_previous
            change[:, :, t] = np.where(usable, values[:, :, t] - last_value, 0.0)
            change_seen[:, :, t] = usable
            known = seen | has_previous
            stale[:, :, t] = np.where(known & ~seen, np.abs(ids[t] - last_time), 0.0)
            stale_seen[:, :, t] = known
            last_value = np.where(seen, values[:, :, t], last_value)
            last_time = np.where(seen, ids[t], last_time)
            has_previous |= seen
        channels = {"change": (change, change_seen), "staleness": (stale, stale_seen)}
        chosen = [name for name in ("change", "staleness") if name in self.context_channels]
        return (np.concatenate([channels[name][0] for name in chosen], axis=1),
                np.concatenate([channels[name][1] for name in chosen], axis=1))

    def _fit(self, matrix: np.ndarray, names: list[str], target: np.ndarray, sample_weight) -> None:
        self._validate_hyperparameters()
        if target.shape[0] != matrix.shape[0]:
            raise ValueError("X and y must contain the same number of rows.")
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=np.float64).ravel()
            if (sample_weight.shape[0] != matrix.shape[0] or np.any(sample_weight < 0)
                    or sample_weight.sum() <= 0):
                raise ValueError(
                    "sample_weight must be non-negative, have positive total weight, and match X rows."
                )

        temporal_columns, static_columns = self._parse_columns(names)
        if not temporal_columns:
            raise ValueError("At least one '<feature_name>_<time_id>' column is required.")
        empty = [name for name, column in zip(names, matrix.T) if not np.isfinite(column).any()]
        if empty:
            warnings.warn(f"Columns with no observed values: {empty}", stacklevel=3)

        self.feature_names_in_ = np.asarray(names, dtype=object)
        self.n_features_in_ = len(names)
        self._temporal_columns = temporal_columns
        self._static_column_indices = static_columns
        self.temporal_features_ = sorted(temporal_columns)
        self.static_features_ = [names[index] for index in static_columns]
        if self.include_missingness_indicators:
            self.static_features_ += [f"{name}_observed_fraction" for name in self.temporal_features_]
        self.time_ids_ = sorted({
            time_id for feature_name in self.temporal_features_
            for time_id in temporal_columns[feature_name]
        })
        self._base_features_ = list(self.temporal_features_)
        self.temporal_features_ = self._base_features_ + [
            f"{name} [{channel}]" for channel in ("change", "staleness") if channel in self.context_channels
            for name in self._base_features_]

        values, observed, static_values = self._build_tensors(matrix)
        if self.missing_effects and self.include_missingness_indicators:
            warnings.warn("missing_effects and include_missingness_indicators both encode how often features are "
                          "measured; missing_effects alone is usually enough.", stacklevel=3)
        pairs, n_interactions = self._candidate_pairs()
        self.model_ = self._make_trainer().fit(
            values, observed, static_values,
            np.asarray(self.time_ids_, dtype=np.float32), target,
            sample_weight=sample_weight, pairs=pairs, n_interactions=n_interactions,
        )
        self.interaction_terms_ = [(self._axis_name(u), self._axis_name(w)) for u, w in self.model_._pairs]
        self.smoothing_params_ = {self._term_name(kind, index): values
                                  for kind, index, values in getattr(self.model_, "smoothing_params_", [])}
        self.edf_ = {self._term_name(kind, index): value for kind, index, value in getattr(self.model_, "edf_", [])}
        explanation = self._explain_matrix(matrix)
        importances = {name: float(np.abs(explanation.temporal[:, index].sum(axis=1)).mean())
                       for index, name in enumerate(self.temporal_features_)}
        importances.update({name: float(np.abs(explanation.static[:, index]).mean())
                            for index, name in enumerate(self.static_features_)})
        importances.update({f"{u} x {w}": float(np.abs(explanation.pairs[:, index]).mean())
                            for index, (u, w) in enumerate(self.interaction_terms_)})
        if self.missing_effects:
            importances.update({f"{name} [not measured]": float(np.abs(explanation.missing[:, index].sum(axis=1)).mean())
                                for index, name in enumerate(self.temporal_features_)})
        self.term_importances_ = dict(sorted(importances.items(), key=lambda item: -item[1]))

    def _axis(self, name: str):
        if name in self.temporal_features_:
            return ("temporal", self.temporal_features_.index(name))
        if name in self.static_features_:
            return ("static", self.static_features_.index(name))
        raise ValueError(f"Unknown feature {name!r} in interactions; fitted features: "
                         f"{self.temporal_features_ + self.static_features_}")

    def _term_name(self, kind: str, index: int) -> str:
        if kind == "pair":
            return " x ".join(self._axis_name(axis) for axis in self.model_._pairs[index])
        if kind == "missing":
            return ("[not measured]" if self.missing_effects == "pooled"
                    else f"{self.temporal_features_[index]} [not measured]")
        return (self.temporal_features_ if kind == "temporal" else self.static_features_)[index]

    def _axis_name(self, axis) -> str:
        kind, index = axis
        return (self.temporal_features_ if kind == "temporal" else self.static_features_)[index]

    def _candidate_pairs(self):
        """``(candidate pairs, how many to keep)``; each pair is two axes, a temporal one first."""
        if isinstance(self.interactions, int | np.integer):
            axes = ([("temporal", index) for index in range(len(self.temporal_features_))]
                    + [("static", index) for index in range(len(self.static_features_))])
            return list(itertools.combinations(axes, 2)), int(self.interactions)
        pairs = []
        for first, second in self.interactions:
            pair = sorted((self._axis(first), self._axis(second)), key=lambda axis: axis[0] != "temporal")
            if pair[0] == pair[1]:
                raise ValueError(f"An interaction needs two different features, got {(first, second)}")
            pairs.append(tuple(pair))
        return pairs, len(pairs)

    def _explain_matrix(self, matrix: np.ndarray) -> Explanation:
        temporal, static, pairs, missing = self.model_.contributions(*self._build_tensors(matrix))
        return Explanation(self.model_._intercept, temporal, static, pairs, missing)

    def explain(self, X) -> Explanation:
        """Break each row's prediction into the intercept plus one term per observed cell and static column.

        See ``Explanation``. Units are the link scale: log-odds for
        ``SSTGAM``, the outcome itself for a gaussian ``SSTGAMRegressor``,
        log-rate for a poisson one.
        """
        check_is_fitted(self, "model_")
        return self._explain_matrix(self._prepare_predict_input(X))

    def decision_function(self, X) -> np.ndarray:
        """Linear predictor per row (log-odds for ``SSTGAM``): the sum of ``explain(X)``."""
        explanation = self.explain(X)
        return (explanation.intercept + explanation.temporal.sum(axis=(1, 2)) + explanation.static.sum(axis=1)
                + explanation.pairs.sum(axis=1) + explanation.missing.sum(axis=(1, 2)))

    def _validate_hyperparameters(self) -> None:
        if self.method not in {"spline", "tree"}:
            raise ValueError(f"method must be 'spline' or 'tree', got {self.method!r}")
        if self.basis not in {"rbf", "bspline"}:
            raise ValueError(f"basis must be 'rbf' or 'bspline', got {self.basis!r}")
        if self.value_transform not in {"quantile", "zscore"}:
            raise ValueError(f"value_transform must be 'quantile' or 'zscore', got {self.value_transform!r}")
        if min(self.n_value_bases, self.n_time_bases, self.n_static_bases,
               self.n_value_bins, self.tree_depth) < 1:
            raise ValueError("Basis counts, bin count, and tree_depth must be positive.")
        for name in ("temporal_penalty", "static_penalty"):
            value = getattr(self, name)
            if value not in ("auto", "reml") and (isinstance(value, str) or value <= 0):
                raise ValueError(f"{name} must be 'auto', 'reml' or a positive number, got {value!r}")
        if {self.temporal_penalty, self.static_penalty} == {"auto", "reml"}:
            raise ValueError("temporal_penalty and static_penalty cannot mix 'auto' and 'reml'.")
        if self.missing_effects not in (False, True, "pooled"):
            raise ValueError(f"missing_effects must be True, False or 'pooled', got {self.missing_effects!r}")
        if self.penalty_sharing not in {"per_term", "shared"}:
            raise ValueError(f"penalty_sharing must be 'per_term' or 'shared', got {self.penalty_sharing!r}")
        if self.bag_validation not in {"fixed", "per_bag"}:
            raise ValueError(f"bag_validation must be 'fixed' or 'per_bag', got {self.bag_validation!r}")
        if self.min_leaf_hessian is not None and self.min_leaf_hessian < 0:
            raise ValueError("min_leaf_hessian must be non-negative.")
        if self.min_gain < 0:
            raise ValueError("min_gain must be non-negative.")
        unknown = set(self.context_channels) - {"change", "staleness"}
        if unknown:
            raise ValueError(f"context_channels must be drawn from 'change' and 'staleness', got {sorted(unknown)}")
        if self.context_order not in {"ascending", "descending"}:
            raise ValueError(f"context_order must be 'ascending' or 'descending', got {self.context_order!r}")
        if self.bin_edges not in {"quantile", "distinct"}:
            raise ValueError(f"bin_edges must be 'quantile' or 'distinct', got {self.bin_edges!r}")
        if self.max_leaves is not None and self.max_leaves < 2:
            raise ValueError("max_leaves must be at least 2.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if self.max_rounds < 1 or self.early_stopping_rounds < 1 or self.outer_bags < 1:
            raise ValueError("max_rounds, early_stopping_rounds, and outer_bags must be positive.")
        if isinstance(self.interactions, int | np.integer) and self.interactions < 0:
            raise ValueError("interactions must be a non-negative number of pairs or a list of feature pairs.")
        if self.n_pair_bases < 2:
            raise ValueError("n_pair_bases must be at least 2.")

    def _make_trainer(self):
        if self.method == "tree":
            return SharedShapeTemporalGAMTreeBoost(
                n_value_bins=self.n_value_bins,
                max_depth=self.tree_depth,
                learning_rate=self.learning_rate,
                max_rounds=self.max_rounds,
                outer_bags=self.outer_bags,
                early_stopping_rounds=self.early_stopping_rounds,
                val_fraction=self.val_fraction,
                random_state=self.random_state,
                family=self._family,
                bag_validation=self.bag_validation,
                min_leaf_hessian=self.min_leaf_hessian,
                max_leaves=self.max_leaves,
                min_gain=self.min_gain,
                bin_edges=self.bin_edges,
                missing_effects=self.missing_effects,
            )
        return SharedShapeTemporalGAMBoost(
            n_value_bases=self.n_value_bases,
            n_time_bases=self.n_time_bases,
            n_static_bases=self.n_static_bases,
            basis=self.basis,
            value_transform=self.value_transform,
            temporal_penalty=self.temporal_penalty,
            static_penalty=self.static_penalty,
            outer_bags=self.outer_bags,
            val_fraction=self.val_fraction,
            random_state=self.random_state,
            family=self._family,
            n_pair_bases=self.n_pair_bases,
            n_pair_bins=self.n_value_bins,
            penalty_sharing=self.penalty_sharing,
            missing_effects=self.missing_effects,
        )

    def _temporal_index(self, feature: str) -> int:
        check_is_fitted(self, "model_")
        if feature not in self.temporal_features_:
            raise ValueError(f"Unknown temporal feature {feature!r}; fitted temporal features: "
                             f"{self.temporal_features_}")
        return self.temporal_features_.index(feature)

    def value_effect(self, feature: str, values, return_interval: bool = False) -> _Effect:
        """Return the fitted spline effect of values averaged across time.

        ``values`` may be a scalar or 1-D array; the result has one entry per
        value. With ``return_interval=True``, returns a ``(lower, mean, upper)``
        pointwise 95% Bayesian band (see ``temporal_surface``).
        """
        if self.method != "spline":
            raise ValueError("value_effect is only available when method='spline'.")
        feature_index = self._temporal_index(feature)
        normalized_values = self.model_.normalize_values(feature_index, np.atleast_1d(values))
        return self.model_.mean_value_effect_normalized(
            feature_index, normalized_values, return_interval=return_interval
        )

    def temporal_surface(self, feature: str, values, time_coordinates,
                         return_interval: bool = False, center_time: bool = False) -> _Effect:
        """Evaluate a spline feature's value-time surface on the link scale.

        The per-time level of the surface is not identified by the data (adding a function of time alone
        that sums to zero over time leaves every prediction unchanged), so ``center_time=True`` removes the
        mean over the training cells at each time: use it to read or plot a surface.
        With ``return_interval=True``, returns ``(lower, mean, upper)``, a pointwise 95% Bayesian band from
        the penalised posterior covariance. It is conditional on the smoothing parameters and ignores
        penalisation bias, so expect under-coverage at extreme values and where the surface is steep;
        it is not a simultaneous band.
        """
        if self.method != "spline":
            raise ValueError("temporal_surface is only available when method='spline'.")
        feature_index = self._temporal_index(feature)
        normalized_values = self.model_.normalize_values(feature_index, values)
        normalized_time = self.model_.normalize_time(time_coordinates)
        return self.model_.temporal_surface_normalized(
            feature_index, normalized_values, normalized_time, return_interval=return_interval,
            center_time=center_time,
        )

    def static_effect(self, feature: str, values, return_interval: bool = False) -> _Effect:
        """Return the fitted effect of a static covariate at ``values`` on the link scale.

        ``feature`` is a static column name, or ``<temporal_feature>_observed_fraction``
        for a missingness indicator. Missing values get the effect the model
        uses for them at prediction time. Spline effects average zero over the
        training rows. With ``return_interval=True``, returns a
        ``(lower, mean, upper)`` bootstrap percentile band across the bagged fits.
        """
        check_is_fitted(self, "model_")
        if feature not in self.static_features_:
            raise ValueError(f"Unknown static feature {feature!r}; fitted static features: {self.static_features_}")
        return self.model_.static_effect(self.static_features_.index(feature), np.atleast_1d(values),
                                         return_interval=return_interval)

    def interaction_surface(self, first: str, second: str, first_values, second_values,
                            return_interval: bool = False) -> _Effect:
        """Evaluate a fitted interaction on a grid: shape ``(len(first_values), len(second_values))``.

        The pair must be in ``interaction_terms_`` (in either order). Units
        are the link scale; the surface has no main effect of its own, so it
        reads as "how much more (or less) the two together add than their
        separate effects". Signal x signal pairs act on each time both were
        measured, signal x static pairs on each time the signal was measured.
        With ``return_interval=True``, returns a ``(lower, mean, upper)``
        bootstrap percentile band across the bagged fits.
        """
        check_is_fitted(self, "model_")
        if (first, second) in self.interaction_terms_:
            index, swap = self.interaction_terms_.index((first, second)), False
        elif (second, first) in self.interaction_terms_:
            index, swap = self.interaction_terms_.index((second, first)), True
        else:
            raise ValueError(f"No fitted interaction between {first!r} and {second!r}; fitted: "
                             f"{self.interaction_terms_}")
        u_values, w_values = (second_values, first_values) if swap else (first_values, second_values)
        result = self.model_.interaction_surface(index, np.atleast_1d(u_values), np.atleast_1d(w_values),
                                                 return_interval=return_interval)
        if not swap:
            return result
        return tuple(part.T for part in result) if return_interval else result.T

    def missing_surface(self, feature: str, time_coordinates, return_interval: bool = False) -> _Effect:
        """The fitted effect of *not* measuring ``feature`` at each of ``time_coordinates`` (link scale).

        Needs ``missing_effects=True``. The spline method accepts any time
        coordinate; the tree method only the fitted ``time_ids_``. Effects are
        centred over the training rows.
        """
        if not self.missing_effects:
            raise ValueError("missing_surface needs missing_effects=True.")
        index = self._temporal_index(feature)
        times = np.atleast_1d(time_coordinates)
        if self.method == "spline":
            return self.model_.missing_surface(index, self.model_.normalize_time(times), return_interval=return_interval)
        unknown = [t for t in times.tolist() if t not in self.time_ids_]
        if unknown:
            raise ValueError(f"The tree method has missing effects only at the fitted time ids; got {unknown}")
        positions = [self.time_ids_.index(t) for t in times.tolist()]
        return self.model_.missing_surface(index, positions, return_interval=return_interval)

    def value_time_table(self, feature: str, return_interval: bool = False) -> tuple[np.ndarray, ...]:
        """Return the tree backend's fitted value-bin by time-bin table.

        With ``return_interval=True``, returns
        ``(value_bin_edges, lower, mean, upper)`` -- a bootstrap percentile
        band computed across the ``outer_bags`` bagged fits instead of a
        single table.
        """
        if self.method != "tree":
            raise ValueError("value_time_table is only available when method='tree'.")
        return self.model_.value_time_table(self._temporal_index(feature), return_interval=return_interval)


class SSTGAM(ClassifierMixin, _SSTGAMBase):
    """Explainable binary classifier for wide longitudinal data.

    Temporal columns use the name ``<feature_name>_<time_id>``. For example,
    ``signal_a_0`` and ``signal_a_6`` are two observations of the same
    feature at time coordinates 0 and 6; time ids may also be decimals
    (``signal_a_0.5``). Columns without a numeric suffix, and columns listed
    in ``static_columns``, are static covariates.

    Each temporal feature has one shared value-time effect. The model sums the
    effects of observed temporal cells, adds optional static feature effects,
    and maps the result to a binary probability through a logistic link.
    Missing temporal values are represented by an observation mask; they do
    not contribute to the temporal surface.

    Parameters
    ----------
    method : {"spline", "tree"}, default="spline"
        ``"spline"`` fits smooth tensor-product value-time surfaces.
        ``"tree"`` fits discrete value-bin by time-bin tables.
    n_value_bases, n_time_bases, n_static_bases : int
        Number of spline bases used by the spline method.
    n_value_bins, tree_depth : int
        Value bin count and shallow-tree depth used by the tree method.
    basis : {"bspline", "rbf"}, default="bspline"
        Basis family used by the spline method's value-time surfaces.
        ``"bspline"`` fits a clamped, evenly knotted cubic B-spline basis.
        ``"rbf"`` selects fixed-bandwidth Gaussian bumps on evenly spaced
        centers for backwards-compatible experiments. Both use the same
        second-difference roughness penalty. Ignored when ``method="tree"``.
    value_transform : {"quantile", "zscore"}, default="quantile"
        Per-feature value normalization for the spline method, applied to
        temporal values and static covariates alike.
    temporal_penalty, static_penalty : float, "auto" or "reml", default="auto"
        Smoothness penalties for spline temporal and static effects. The
        right value depends on how much information each coefficient gets
        and differs by orders of magnitude between datasets. With
        ``"auto"``, terms share one penalty chosen by held-out loss (a
        short search from a starting guess of ~100 x coefficients per
        minority-class example, picking the smoothest penalty within one
        standard error of the best), then the model is refit on all rows.
        The held-out loss comes from one ``val_fraction`` split when that
        split holds at least 500 minority-class examples, otherwise from
        5-fold cross-validation, which is slower but stable on small data. The chosen values are on ``model_`` as
        ``temporal_penalty_`` / ``static_penalty_``, and every value tried
        with its validation loss as ``penalty_search_``. With ``"reml"``,
        every penalty piece of every term gets its own smoothing parameter
        (value-axis roughness, time-axis roughness and overall size for a
        surface; roughness and size for a static effect), chosen by REML with
        the generalised Fellner-Schall update, without cross-validation. The
        chosen values are in ``smoothing_params_`` and the effective degrees
        of freedom per term in ``edf_``. A number for one of the two fixes that
        group's pieces while the other is tuned; ``"auto"`` and ``"reml"``
        cannot be mixed.
    penalty_sharing : {"per_term", "shared"}, default="per_term"
        Under ``"reml"``: one smoothing parameter per piece of every term, or
        one per piece type shared by all terms of a kind (fewer parameters, for
        very small data).
    val_fraction : float, default=0.15
        Held-out fraction: the tree method's early-stopping split, and the
        spline method's ``"auto"`` penalty search.
    learning_rate, max_rounds, early_stopping_rounds
        Boosting controls for the tree method. The spline method has no
        boosting rounds: it solves its penalized fit directly by P-IRLS, so
        it ignores these.
    outer_bags : int, default=4
        Number of bootstrap-resampled fits (the validation split is fixed
        across bags) whose coefficients/tables are averaged at the end --
        the same bagging EBM uses for variance reduction. ``outer_bags=1``
        disables bagging (a single fit on the full training split). The
        ``return_interval`` bands of the tree method and of ``static_effect``,
        ``interaction_surface`` and ``missing_surface`` are percentiles across
        these fits (a stability check); the spline surfaces use a Bayesian band.
    include_missingness_indicators : bool, default=False
        When ``True``, adds one extra static covariate per temporal feature:
        the fraction of its time points that were observed for that sample.
        This lets the model learn from informative missingness (e.g. a
        feature is measured more or less often depending on the outcome)
        through the existing static-effect pathway, without asserting
        anything about which times or features that applies to. Off by
        default; see ``ARCHITECTURE.md`` for the missingness assumption this
        opts out of.
    static_columns : list of str, optional
        Column names to treat as static even though they end in ``_<number>``
        (e.g. ``icd_10``).
    interactions : int or list of (str, str), default=0
        Pairwise interaction terms, as in EBM/GA2M. An integer K fits the K
        pairs with the largest FAST gain on the main-effects residual, out of
        every signal x signal, signal x static and static x static pair; a
        list such as ``[("creatinine", "urea"), ("lactate", "age")]`` fits
        exactly those. Each pair is a 2-D surface shared across time with no
        main effect of its own, so the main effects keep their meaning.
    n_pair_bases : int, default=6
        Bases per axis of each spline interaction surface. Tree interactions
        reuse their two terms' bins, and pair ranking uses ``n_value_bins``.
    missing_effects : bool or "pooled", default=False
        Adds one term per temporal feature for the measurement process: a
        smooth (spline) or binned (tree) effect over time of the feature *not*
        being measured. Without it a missing cell adds nothing, which assumes
        missingness carries no information; with it the model learns, for
        example, that not checking a lab on a given day signals lower risk.
        Inspect it with ``missing_surface``; ``explain(X).missing`` holds the
        per-cell contributions. ``"pooled"`` fits one effect over time shared
        by every feature (an unmeasured cell of any feature counts the same):
        far fewer coefficients, so it costs less when missingness carries no
        information, and suits data where sicker patients are measured more
        overall rather than one lab at a time.
    bag_validation : {"fixed", "per_bag"}, default="fixed"
        Tree method: one validation split shared by every bag, or a fresh
        split per bag (as EBM does), so the early-stopping points differ.
    min_leaf_hessian, max_leaves : optional
        Tree method: the smallest total curvature a leaf may hold, and the
        largest number of leaves per tree update.
    context_channels : iterable of {"change", "staleness"}, default=()
        History-relative channels added as extra temporal terms, each with its own value-by-time surface and
        named ``"<feature> [change]"`` / ``"<feature> [staleness]"``. ``"change"`` is the value minus the
        previous observed value; ``"staleness"`` is the time since the last observation (0 when observed). A cell
        uses only itself and strictly earlier cells.
    context_order : {"ascending", "descending"}, default="ascending"
        Whether larger time ids are later (``"ascending"``) or earlier (``"descending"``, e.g. lag-indexed
        columns where 0 is the most recent), which fixes what "previous" means for the channels.
    min_gain : float, default=0.0
        Tree method: skip a table update (leave that term alone this round)
        unless its Newton gain, twice the second-order loss decrease, reaches
        this value; ``0`` never skips. About 8 (a chi-square(1) 99.5% point)
        stops tables from being fitted to noise every round. Applies to
        temporal, static, "not measured" and pair updates.
    bin_edges : {"quantile", "distinct"}, default="quantile"
        Tree method: ``"distinct"`` gives each value its own bin when a
        covariate has at most ``n_value_bins`` (static: 16) distinct values
        and keeps a mass point at the maximum in its own bin.
    random_state
        Random seed (validation split, bootstrap bags, tree fitting).

    Attributes
    ----------
    classes_ : ndarray of shape (2,)
        The two labels; probabilities and effects refer to ``classes_[1]``.
    temporal_features_, static_features_, time_ids_
        Parsed feature names and the sorted time ids.
    interaction_terms_ : list of (str, str)
        The fitted interaction pairs.
    term_importances_ : dict
        Mean absolute log-odds contribution of each term over the training
        rows, largest first (the temporal terms summed over time per row;
        interactions named ``"a x b"``).
    smoothing_params_, edf_ : dict
        With ``"reml"`` penalties: the chosen smoothing parameter of each
        penalty piece, and the effective degrees of freedom, per term name.
        Empty otherwise.
    """

    def fit(self, X, y, feature_names: Iterable[str] | None = None, sample_weight=None) -> "SSTGAM":
        matrix, names = self._prepare_fit_input(X, feature_names)
        labels = np.asarray(y).ravel()
        check_classification_targets(labels)
        classes = np.unique(labels)
        if len(classes) != 2:
            raise ValueError(f"SSTGAM needs a binary target with exactly two classes; y has {len(classes)}: "
                             f"{classes[:5].tolist()}")
        self.classes_ = classes
        self._fit(matrix, names, (labels == classes[1]).astype(np.float64), sample_weight)
        return self

    def predict_proba(self, X) -> np.ndarray:
        probability = np.clip(expit(self.decision_function(X)), 1e-6, 1.0 - 1e-6)
        return np.column_stack((1.0 - probability, probability))

    def predict(self, X) -> np.ndarray:
        return self.classes_[(self.decision_function(X) >= 0.0).astype(int)]

    def predict_log_proba(self, X) -> np.ndarray:
        return np.log(np.clip(self.predict_proba(X), 1e-10, 1.0))


class SSTGAMRegressor(RegressorMixin, _SSTGAMBase):
    """SSTGAM for a continuous (``family="gaussian"``) or count (``family="poisson"``) outcome.

    Same model, data format and parameters as ``SSTGAM``; only the link
    changes. Gaussian uses the identity link, so effects are in the outcome's
    units; poisson uses the log link, so effects are log-rate and combine
    multiplicatively on the rate. ``predict`` returns the expected outcome.
    """

    def __init__(
        self,
        family: str = "gaussian",
        method: str = "spline",
        n_value_bases: int = 8,
        n_time_bases: int = 5,
        n_static_bases: int = 8,
        n_value_bins: int = 16,
        tree_depth: int = 3,
        basis: str = "bspline",
        value_transform: str = "quantile",
        temporal_penalty: float | str = "auto",
        static_penalty: float | str = "auto",
        learning_rate: float = 0.02,
        max_rounds: int = 100,
        outer_bags: int = 4,
        early_stopping_rounds: int = 15,
        include_missingness_indicators: bool = False,
        val_fraction: float = 0.15,
        static_columns: Iterable[str] | None = None,
        interactions: int | Iterable[tuple[str, str]] = 0,
        n_pair_bases: int = 6,
        penalty_sharing: str = "per_term",
        missing_effects: bool | str = False,
        bag_validation: str = "fixed",
        min_leaf_hessian: float | None = None,
        max_leaves: int | None = None,
        min_gain: float = 0.0,
        bin_edges: str = "quantile",
        context_channels: Iterable[str] = (),
        context_order: str = "ascending",
        random_state: int = 42,
    ) -> None:
        super().__init__(
            method=method, n_value_bases=n_value_bases, n_time_bases=n_time_bases,
            n_static_bases=n_static_bases, n_value_bins=n_value_bins, tree_depth=tree_depth, basis=basis,
            value_transform=value_transform, temporal_penalty=temporal_penalty, static_penalty=static_penalty,
            learning_rate=learning_rate, max_rounds=max_rounds, outer_bags=outer_bags,
            early_stopping_rounds=early_stopping_rounds,
            include_missingness_indicators=include_missingness_indicators, val_fraction=val_fraction,
            static_columns=static_columns, interactions=interactions, n_pair_bases=n_pair_bases,
            penalty_sharing=penalty_sharing, missing_effects=missing_effects, bag_validation=bag_validation,
            min_leaf_hessian=min_leaf_hessian,
            max_leaves=max_leaves, min_gain=min_gain, bin_edges=bin_edges, context_channels=context_channels,
            context_order=context_order, random_state=random_state,
        )
        self.family = family

    @property
    def _family(self) -> str:  # type: ignore[override]
        return self.family

    def fit(self, X, y, feature_names: Iterable[str] | None = None, sample_weight=None) -> "SSTGAMRegressor":
        if self.family not in {"gaussian", "poisson"}:
            raise ValueError(f"family must be 'gaussian' or 'poisson', got {self.family!r}")
        matrix, names = self._prepare_fit_input(X, feature_names)
        target = np.asarray(y, dtype=np.float64).ravel()
        if not np.isfinite(target).all():
            raise ValueError("y must be finite.")
        if self.family == "poisson" and (target < 0).any():
            raise ValueError("family='poisson' needs a non-negative target.")
        self._fit(matrix, names, target, sample_weight)
        return self

    def predict(self, X) -> np.ndarray:
        return FAMILIES[self.family].mean(self.decision_function(X))
