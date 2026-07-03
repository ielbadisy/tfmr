# Controlling TabFM execution

Controlling TabFM execution

## Usage

``` r
control_tab_fm(
  backend = c("auto", "jax", "pytorch"),
  checkpoint_path = NULL,
  step = NULL,
  device = NULL,
  use_cache = TRUE,
  col_attention_impl = "flash",
  row_attention_impl = "jax",
  icl_attention_impl = "flash",
  n_estimators = 32L,
  norm_methods = NULL,
  feat_shuffle_method = "random",
  class_shift = TRUE,
  permute_categorical = FALSE,
  outlier_threshold = 4,
  max_num_features = 500L,
  max_num_rows = NULL,
  softmax_temperature = 0.9,
  average_logits = TRUE,
  use_amp = TRUE,
  batch_size = 1L,
  random_state = 42L,
  verbose = FALSE,
  cat_encoder_mode = "appearance",
  binary_calibration_method = NULL,
  multiclass_calibration_method = NULL,
  num_folds_for_cv = 5L,
  n_feature_crosses = 0,
  n_svd_features = 0,
  total_svd_pool = NULL,
  enable_nnls = FALSE,
  nnls_beta = 0.75,
  calibration_lambda = 0.01,
  min_rows_for_single_val_split = 2000L,
  ...
)
```

## Arguments

- backend:

  Backend selector. Use `"auto"` to prefer JAX when available and
  otherwise fall back to PyTorch.

- checkpoint_path:

  Optional local checkpoint directory.

- step:

  Optional checkpoint step for the JAX loader.

- device:

  Optional PyTorch device such as `"cpu"` or `"cuda"`.

- use_cache:

  Reuse a process-wide cached pretrained model when possible.

- col_attention_impl:

  JAX column-attention implementation.

- row_attention_impl:

  JAX row-attention implementation.

- icl_attention_impl:

  JAX ICL attention implementation.

- n_estimators:

  Number of ensemble members.

- norm_methods:

  Normalization methods passed to the Python constructor.

- feat_shuffle_method:

  Feature permutation strategy.

- class_shift:

  Whether to apply class-label shifts during classification.

- permute_categorical:

  Whether to randomly permute categorical values.

- outlier_threshold:

  Z-score threshold for outlier clipping.

- max_num_features:

  Maximum number of features per ensemble member.

- max_num_rows:

  Maximum number of rows per ensemble member.

- softmax_temperature:

  Temperature for classification probabilities.

- average_logits:

  Whether to average logits before probabilities.

- use_amp:

  Automatic mixed precision setting.

- batch_size:

  Number of ensemble members to process together.

- random_state:

  Random seed.

- verbose:

  Whether to print Python-side training details.

- cat_encoder_mode:

  Categorical encoding mode.

- binary_calibration_method:

  Binary calibration method.

- multiclass_calibration_method:

  Multiclass calibration method.

- num_folds_for_cv:

  Number of folds for out-of-fold calibration.

- n_feature_crosses:

  Number of feature crosses or `"sqrt"`.

- n_svd_features:

  Number of SVD features or `"sqrt"`.

- total_svd_pool:

  Pool size for SVD features.

- enable_nnls:

  Enable NNLS ensemble blending.

- nnls_beta:

  Blend weight for NNLS.

- calibration_lambda:

  L2 regularization strength for calibration.

- min_rows_for_single_val_split:

  Minimum rows for a single validation split.

- ...:

  Additional named arguments passed directly to the Python constructor.

## Value

A list with class `"control_tab_fm"`.
