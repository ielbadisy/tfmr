#' Controlling TabFM execution
#'
#' @param backend Backend selector. Use `"auto"` to prefer JAX when available
#'   and otherwise fall back to PyTorch.
#' @param checkpoint_path Optional local checkpoint directory.
#' @param step Optional checkpoint step for the JAX loader.
#' @param device Optional PyTorch device such as `"cpu"` or `"cuda"`.
#' @param use_cache Reuse a process-wide cached pretrained model when possible.
#' @param col_attention_impl JAX column-attention implementation.
#' @param row_attention_impl JAX row-attention implementation.
#' @param icl_attention_impl JAX ICL attention implementation.
#' @param n_estimators Number of ensemble members.
#' @param norm_methods Normalization methods passed to the Python constructor.
#' @param feat_shuffle_method Feature permutation strategy.
#' @param class_shift Whether to apply class-label shifts during classification.
#' @param permute_categorical Whether to randomly permute categorical values.
#' @param outlier_threshold Z-score threshold for outlier clipping.
#' @param max_num_features Maximum number of features per ensemble member.
#' @param max_num_rows Maximum number of rows per ensemble member.
#' @param softmax_temperature Temperature for classification probabilities.
#' @param average_logits Whether to average logits before probabilities.
#' @param use_amp Automatic mixed precision setting.
#' @param batch_size Number of ensemble members to process together.
#' @param random_state Random seed.
#' @param verbose Whether to print Python-side training details.
#' @param cat_encoder_mode Categorical encoding mode.
#' @param binary_calibration_method Binary calibration method.
#' @param multiclass_calibration_method Multiclass calibration method.
#' @param num_folds_for_cv Number of folds for out-of-fold calibration.
#' @param n_feature_crosses Number of feature crosses or `"sqrt"`.
#' @param n_svd_features Number of SVD features or `"sqrt"`.
#' @param total_svd_pool Pool size for SVD features.
#' @param enable_nnls Enable NNLS ensemble blending.
#' @param nnls_beta Blend weight for NNLS.
#' @param calibration_lambda L2 regularization strength for calibration.
#' @param min_rows_for_single_val_split Minimum rows for a single validation split.
#' @param ... Additional named arguments passed directly to the Python constructor.
#' @return A list with class `"control_tab_fm"`.
#' @export
control_tab_fm <- function(
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
  outlier_threshold = 4.0,
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
  calibration_lambda = 1e-2,
  min_rows_for_single_val_split = 2000L,
  ...
) {
  backend <- rlang::arg_match(backend)

  check_string(feat_shuffle_method)
  check_string(col_attention_impl)
  check_string(row_attention_impl)
  check_string(icl_attention_impl)
  check_number_whole(n_estimators, min = 1)
  check_number_decimal(outlier_threshold, min = 0)
  check_number_whole(max_num_features, min = 1)
  check_number_decimal(softmax_temperature, min = .Machine$double.eps)
  check_logical(class_shift)
  check_logical(permute_categorical)
  check_logical(average_logits)
  check_logical(use_amp)
  check_logical(enable_nnls)
  check_number_whole(batch_size, min = 1)
  check_number_whole(random_state)
  check_logical(verbose)
  check_string(cat_encoder_mode)
  check_number_whole(num_folds_for_cv, min = 2)
  check_number_whole(min_rows_for_single_val_split, min = 0)
  if (!is.null(checkpoint_path)) check_string(checkpoint_path)
  if (!is.null(step)) check_number_whole(step, min = 0)
  if (!is.null(device)) check_string(device)
  if (!is.null(max_num_rows)) check_number_whole(max_num_rows, min = 1)
  if (!is.null(total_svd_pool)) check_number_whole(total_svd_pool, min = 1)
  if (!is.null(binary_calibration_method)) check_string(binary_calibration_method)
  if (!is.null(multiclass_calibration_method)) check_string(multiclass_calibration_method)

  reserved <- c(
    "backend",
    "checkpoint_path",
    "step",
    "device",
    "use_cache",
    "col_attention_impl",
    "row_attention_impl",
    "icl_attention_impl",
    "n_estimators",
    "norm_methods",
    "feat_shuffle_method",
    "class_shift",
    "permute_categorical",
    "outlier_threshold",
    "max_num_features",
    "max_num_rows",
    "softmax_temperature",
    "average_logits",
    "use_amp",
    "batch_size",
    "random_state",
    "verbose",
    "cat_encoder_mode",
    "binary_calibration_method",
    "multiclass_calibration_method",
    "num_folds_for_cv",
    "n_feature_crosses",
    "n_svd_features",
    "total_svd_pool",
    "enable_nnls",
    "nnls_beta",
    "calibration_lambda",
    "min_rows_for_single_val_split"
  )

  dot_args <- rlang::list2(...)
  conflicts <- intersect(names(dot_args), reserved)
  if (length(conflicts) > 0) {
    cli::cli_abort(
      "Argument{?s} {.arg {conflicts}} must be passed as named argument{?s}, not via {.code ...}."
    )
  }

  args <- c(
    list(
      backend = backend,
      checkpoint_path = checkpoint_path,
      step = step,
      device = device,
      use_cache = use_cache,
      col_attention_impl = col_attention_impl,
      row_attention_impl = row_attention_impl,
      icl_attention_impl = icl_attention_impl,
      n_estimators = as.integer(n_estimators),
      norm_methods = norm_methods,
      feat_shuffle_method = feat_shuffle_method,
      class_shift = class_shift,
      permute_categorical = permute_categorical,
      outlier_threshold = outlier_threshold,
      max_num_features = as.integer(max_num_features),
      max_num_rows = max_num_rows,
      softmax_temperature = softmax_temperature,
      average_logits = average_logits,
      use_amp = use_amp,
      batch_size = as.integer(batch_size),
      random_state = as.integer(random_state),
      verbose = verbose,
      cat_encoder_mode = cat_encoder_mode,
      binary_calibration_method = binary_calibration_method,
      multiclass_calibration_method = multiclass_calibration_method,
      num_folds_for_cv = as.integer(num_folds_for_cv),
      n_feature_crosses = n_feature_crosses,
      n_svd_features = n_svd_features,
      total_svd_pool = total_svd_pool,
      enable_nnls = enable_nnls,
      nnls_beta = nnls_beta,
      calibration_lambda = calibration_lambda,
      min_rows_for_single_val_split = as.integer(min_rows_for_single_val_split)
    ),
    dot_args
  )

  args <- args[!vapply(args, is.null, logical(1))]
  class(args) <- "control_tab_fm"
  args
}

#' @export
print.control_tab_fm <- function(x, ...) {
  cli::cli_inform("control object for {.fn tab_fm}")
  cat("\n")
  lst <- purrr::map2(
    names(x),
    x,
    ~ cli::format_inline("{.arg {.x}}: {.val {.y}}")
  )
  names(lst) <- rep("*", length(lst))
  cli::cli_bullets(lst)
  invisible(x)
}
