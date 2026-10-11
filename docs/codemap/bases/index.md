# `bases.py`

Pure numerics: RBF basis-function and roughness-penalty primitives used only
by the spline backend (`boosting.py`). Unit-testable in isolation — no
dependency on the rest of the package.

| Function                                            | Purpose                                                |
| --------------------------------------------------- | ------------------------------------------------------ |
| [`make_rbf_centers`](make_rbf_centers.md)           | Place `n` evenly-spaced RBF centers + pick a bandwidth |
| [`rbf_np`](rbf_np.md)                               | Vectorized Gaussian RBF design matrix                  |
| [`second_diff_matrix_np`](second_diff_matrix_np.md) | Discrete second-difference (curvature) operator        |

**Used by:** [../boosting/index.md](../boosting/index.md) only.

**Back to:** [../INDEX.md](../INDEX.md)
