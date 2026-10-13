"""Public SSTGAM classifier."""

import re
from collections.abc import Iterable

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from .boosting import SharedShapeTemporalGAMBoost
from .tree_boost import SharedShapeTemporalGAMTreeBoost

_TEMPORAL_COLUMN_PATTERN = re.compile(r"^(.+)_(\d+)$")

# A point estimate, or a ``(lower, mean, upper)`` band when ``return_interval=True``.
_Effect = np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]


class SSTGAM(ClassifierMixin, BaseEstimator):
    """Explainable binary classifier for wide longitudinal data.

    Temporal columns use the name ``<feature_name>_<time_id>``. For example,
    ``signal_a_0`` and ``signal_a_6`` are two observations of the same
    feature at time coordinates 0 and 6. Columns without an integer suffix are
    static covariates.

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
        Number of RBF bases used by the spline method.
    n_value_bins, tree_depth : int
        Value bin count and shallow-tree depth used by the tree method.
    basis : {"rbf", "bspline"}, default="rbf"
        Basis family used by the spline method's value-time surfaces.
        ``"rbf"`` places fixed-bandwidth Gaussian bumps on evenly spaced
        centers. ``"bspline"`` fits a clamped, evenly knotted B-spline basis
        of the same size. Both use the same second-difference roughness
        penalty, so smoothness is controlled the same way regardless of
        basis. Ignored when ``method="tree"``.
    value_transform : {"quantile", "zscore"}, default="quantile"
        Per-feature value normalization for the spline method, applied to
        temporal values and static covariates alike.
    temporal_penalty, static_penalty : float or "auto", default="auto"
        Smoothness penalties for spline temporal and static effects. The
        right value depends on how much information each coefficient gets
        and differs by orders of magnitude between datasets. With
        ``"auto"``, terms share one penalty chosen by held-out log loss (a
        short search from a starting guess of ~100 x coefficients per
        minority-class example, picking the smoothest penalty within one
        standard error of the best), then the model is refit on all rows.
        The held-out loss comes from one ``val_fraction`` split when that
        split holds at least 500 minority-class examples, otherwise from
        5-fold cross-validation, which is slower but stable on small data. The chosen values are on ``model_`` as
        ``temporal_penalty_`` / ``static_penalty_``, and every value tried
        with its validation loss as ``penalty_search_``.
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
        disables bagging (a single fit on the full training split).
    include_missingness_indicators : bool, default=False
        When ``True``, adds one extra static covariate per temporal feature:
        the fraction of its time points that were observed for that sample.
        This lets the model learn from informative missingness (e.g. a
        feature is measured more or less often depending on the outcome)
        through the existing static-effect pathway, without asserting
        anything about which times or features that applies to. Off by
        default; see ``ARCHITECTURE.md`` for the missingness assumption this
        opts out of.
    random_state
        Random seed (validation split, bootstrap bags, tree fitting).
    """

    def __init__(
        self,
        method: str = "spline",
        n_value_bases: int = 8,
        n_time_bases: int = 5,
        n_static_bases: int = 8,
        n_value_bins: int = 16,
        tree_depth: int = 3,
        basis: str = "rbf",
        value_transform: str = "quantile",
        temporal_penalty: float | str = "auto",
        static_penalty: float | str = "auto",
        learning_rate: float = 0.02,
        max_rounds: int = 100,
        outer_bags: int = 4,
        early_stopping_rounds: int = 15,
        include_missingness_indicators: bool = False,
        val_fraction: float = 0.15,
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
        self.random_state = random_state

    @staticmethod
    def _parse_columns(column_names: list[str]):
        temporal_columns: dict[str, dict[int, int]] = {}
        static_columns: list[int] = []
        for column_index, column_name in enumerate(column_names):
            match = _TEMPORAL_COLUMN_PATTERN.match(column_name)
            if match is None:
                static_columns.append(column_index)
                continue
            feature_name, time_id = match.group(1), int(match.group(2))
            feature_columns = temporal_columns.setdefault(feature_name, {})
            if time_id in feature_columns:
                raise ValueError(f"Duplicate temporal column: {column_name!r}")
            feature_columns[time_id] = column_index
        return temporal_columns, static_columns

    @staticmethod
    def _as_float_matrix(X) -> np.ndarray:
        matrix = np.asarray(X.values if hasattr(X, "values") else X, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError("X must be a two-dimensional matrix.")
        return matrix

    def _prepare_fit_input(self, X, feature_names: Iterable[str] | None):
        matrix = self._as_float_matrix(X)
        if feature_names is None:
            names = list(X.columns) if hasattr(X, "columns") else [
                f"feature_{index}" for index in range(matrix.shape[1])
            ]
        else:
            names = list(feature_names)
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
        n_features = len(self.temporal_features_)
        n_times = len(self.time_ids_)
        values = np.zeros((n_samples, n_features, n_times), dtype=np.float32)
        observed = np.zeros_like(values)
        time_positions = {time_id: index for index, time_id in enumerate(self.time_ids_)}

        for feature_index, feature_name in enumerate(self.temporal_features_):
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

        return values, observed, static_values

    def fit(self, X, y, feature_names: Iterable[str] | None = None, sample_weight=None) -> "SSTGAM":
        matrix, names = self._prepare_fit_input(X, feature_names)
        target = np.asarray(y, dtype=np.float64).ravel()
        if target.shape[0] != matrix.shape[0]:
            raise ValueError("X and y must contain the same number of rows.")
        if not np.isin(target, (0.0, 1.0)).all():
            raise ValueError("SSTGAM supports binary targets encoded as 0 and 1.")
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

        values, observed, static_values = self._build_tensors(matrix)
        self._validate_hyperparameters()
        self.model_ = self._make_trainer().fit(
            values, observed, static_values,
            np.asarray(self.time_ids_, dtype=np.float32), target,
            sample_weight=sample_weight,
        )
        self.classes_ = np.array([0, 1])
        return self

    def _positive_probability(self, X) -> np.ndarray:
        check_is_fitted(self, "model_")
        values, observed, static_values = self._build_tensors(self._prepare_predict_input(X))
        return self.model_.predict_proba(values, observed, static_values)

    def predict_proba(self, X) -> np.ndarray:
        probability = self._positive_probability(X)
        return np.column_stack((1.0 - probability, probability))

    def predict(self, X) -> np.ndarray:
        return (self._positive_probability(X) >= 0.5).astype(int)

    def predict_log_proba(self, X) -> np.ndarray:
        return np.log(np.clip(self.predict_proba(X), 1e-10, 1.0))

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
            if value != "auto" and (isinstance(value, str) or value <= 0):
                raise ValueError(f"{name} must be 'auto' or a positive number, got {value!r}")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if self.max_rounds < 1 or self.early_stopping_rounds < 1 or self.outer_bags < 1:
            raise ValueError("max_rounds, early_stopping_rounds, and outer_bags must be positive.")

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
        )

    def value_effect(self, feature: str, values, return_interval: bool = False) -> _Effect:
        """Return the fitted spline effect of values averaged across time.

        ``values`` may be a scalar or 1-D array; the result has one entry per
        value. With ``return_interval=True``, returns a ``(lower, mean, upper)``
        bootstrap percentile band computed across the ``outer_bags`` bagged
        fits (see ``ARCHITECTURE.md``) instead of a single point estimate.
        """
        if self.method != "spline":
            raise ValueError("value_effect is only available when method='spline'.")
        check_is_fitted(self, "model_")
        feature_index = self.temporal_features_.index(feature)
        normalized_values = self.model_.normalize_values(feature_index, np.atleast_1d(values))
        return self.model_.mean_value_effect_normalized(
            feature_index, normalized_values, return_interval=return_interval
        )

    def temporal_surface(self, feature: str, values, time_coordinates,
                         return_interval: bool = False) -> _Effect:
        """Evaluate a spline feature's value-time log-odds surface.

        With ``return_interval=True``, returns a ``(lower, mean, upper)``
        bootstrap percentile band computed across the ``outer_bags`` bagged
        fits instead of a single point estimate.
        """
        if self.method != "spline":
            raise ValueError("temporal_surface is only available when method='spline'.")
        check_is_fitted(self, "model_")
        feature_index = self.temporal_features_.index(feature)
        normalized_values = self.model_.normalize_values(feature_index, values)
        normalized_time = self.model_.normalize_time(time_coordinates)
        return self.model_.temporal_surface_normalized(
            feature_index, normalized_values, normalized_time, return_interval=return_interval
        )

    def static_effect(self, feature: str, values) -> np.ndarray:
        """Return the fitted log-odds effect of a static covariate at ``values``.

        ``feature`` is a static column name, or ``<temporal_feature>_observed_fraction``
        for a missingness indicator. Missing values get the effect the model
        uses for them at prediction time. Spline effects average zero over the
        training rows.
        """
        check_is_fitted(self, "model_")
        if feature not in self.static_features_:
            raise ValueError(f"Unknown static feature {feature!r}; fitted static features: {self.static_features_}")
        return self.model_.static_effect(self.static_features_.index(feature), np.atleast_1d(values))

    def value_time_table(self, feature: str, return_interval: bool = False) -> tuple[np.ndarray, ...]:
        """Return the tree backend's fitted value-bin by time-bin table.

        With ``return_interval=True``, returns
        ``(value_bin_edges, lower, mean, upper)`` -- a bootstrap percentile
        band computed across the ``outer_bags`` bagged fits instead of a
        single table.
        """
        if self.method != "tree":
            raise ValueError("value_time_table is only available when method='tree'.")
        check_is_fitted(self, "model_")
        return self.model_.value_time_table(
            self.temporal_features_.index(feature), return_interval=return_interval
        )
