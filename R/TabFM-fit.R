#' Fit a TabFM model
#'
#' `tab_fm()` fits the Python `tabfm` classifier or regressor through
#' `reticulate` and returns an R object with `predict()` and `augment()`
#' methods.
#'
#' @param x A data frame, matrix, recipe, or formula.
#' @param y Outcome vector for the data-frame and matrix interfaces.
#' @param data Data frame for formula and recipe interfaces.
#' @param formula Formula for the formula interface.
#' @param training_set_limit Maximum number of training rows retained before
#'   fitting. Use `Inf` to disable downsampling.
#' @param control A list from [control_tab_fm()].
#' @param ... Not currently used.
#' @return A `tab_fm` model object.
#' @export
tab_fm <- function(x, ...) {
  UseMethod("tab_fm")
}

#' @export
#' @rdname tab_fm
tab_fm.default <- function(x, ...) {
  cli::cli_abort("{.fn tab_fm} is not defined for {obj_type_friendly(x)}.")
}

#' @export
#' @rdname tab_fm
tab_fm.data.frame <- function(
  x,
  y,
  training_set_limit = Inf,
  control = control_tab_fm(),
  ...
) {
  processed <- hardhat::mold(x, y)
  processed <- limit_training_set(processed, training_set_limit)
  tab_fm_bridge(processed, control, ...)
}

#' @export
#' @rdname tab_fm
tab_fm.matrix <- function(
  x,
  y,
  training_set_limit = Inf,
  control = control_tab_fm(),
  ...
) {
  processed <- hardhat::mold(x, y)
  processed <- limit_training_set(processed, training_set_limit)
  tab_fm_bridge(processed, control, ...)
}

#' @export
#' @rdname tab_fm
tab_fm.formula <- function(
  formula,
  data,
  training_set_limit = Inf,
  control = control_tab_fm(),
  ...
) {
  bp <- hardhat::default_formula_blueprint(
    intercept = FALSE,
    allow_novel_levels = FALSE,
    indicators = "none",
    composition = "tibble"
  )
  processed <- hardhat::mold(formula, data, blueprint = bp)
  processed <- limit_training_set(processed, training_set_limit)
  tab_fm_bridge(processed, control, ...)
}

#' @export
#' @rdname tab_fm
tab_fm.recipe <- function(
  x,
  data,
  training_set_limit = Inf,
  control = control_tab_fm(),
  ...
) {
  processed <- hardhat::mold(x, data)
  processed <- limit_training_set(processed, training_set_limit)
  tab_fm_bridge(processed, control, ...)
}

tab_fm_bridge <- function(processed, options, ...) {
  rlang::check_dots_empty()

  predictors <- processed$predictors
  outcome <- processed$outcomes[[1]]
  res <- tab_fm_impl(predictors, outcome, options)

  new_tab_fm(
    fit = res$fit,
    levels = res$lvls,
    training = res$train,
    logging = res$logging,
    blueprint = processed$blueprint
  )
}

tabfm_require_backend <- function(backend) {
  req <- switch(
    backend,
    auto = "tabfm[pytorch]",
    jax = "tabfm[jax]",
    pytorch = "tabfm[pytorch]"
  )
  try(reticulate::py_require(req), silent = TRUE)
  invisible(NULL)
}

tabfm_loader_module <- function(backend) {
  tabfm_require_backend(backend)

  pkg <- import_tabfm()
  if (is.null(pkg)) {
    pkg <- reticulate::import(
      "tabfm",
      delay_load = list(
        on_error = function(e) {
          cli::cli_abort("The {.pkg tabfm} Python package is not installed.")
        },
        before_load = check_libomp
      )
    )
  }

  if (backend == "auto") {
    if (reticulate::py_has_attr(pkg, "tabfm_v1_0_0")) {
      return(list(module = pkg$tabfm_v1_0_0, backend = "jax"))
    }
    if (reticulate::py_has_attr(pkg, "tabfm_v1_0_0_pytorch")) {
      return(list(module = pkg$tabfm_v1_0_0_pytorch, backend = "pytorch"))
    }
    if (reticulate::py_has_attr(pkg, "tabfm_v1_0_0_jax")) {
      return(list(module = pkg$tabfm_v1_0_0_jax, backend = "jax"))
    }
    cli::cli_abort("Could not find a TabFM loader in the installed Python package.")
  }

  module_name <- switch(
    backend,
    jax = "tabfm.src.jax.tabfm_v1_0_0",
    pytorch = "tabfm.src.pytorch.tabfm_v1_0_0"
  )
  list(module = reticulate::import(module_name), backend = backend)
}

tabfm_model_type <- function(y) {
  if (is.factor(y)) {
    "classification"
  } else if (is.numeric(y)) {
    "regression"
  } else {
    cli::cli_abort("`y` must be numeric for regression or a factor for classification.")
  }
}

tabfm_estimator_args <- function(kind, options) {
  args <- list(
    n_estimators = options$n_estimators,
    norm_methods = options$norm_methods,
    feat_shuffle_method = options$feat_shuffle_method,
    permute_categorical = options$permute_categorical,
    outlier_threshold = options$outlier_threshold,
    max_num_features = options$max_num_features,
    max_num_rows = options$max_num_rows,
    use_amp = options$use_amp,
    batch_size = options$batch_size,
    random_state = options$random_state,
    verbose = options$verbose,
    cat_encoder_mode = options$cat_encoder_mode,
    num_folds_for_cv = options$num_folds_for_cv,
    n_feature_crosses = options$n_feature_crosses,
    n_svd_features = options$n_svd_features,
    total_svd_pool = options$total_svd_pool,
    enable_nnls = options$enable_nnls,
    nnls_beta = options$nnls_beta,
    min_rows_for_single_val_split = options$min_rows_for_single_val_split
  )

  if (kind == "classification") {
    args$class_shift <- options$class_shift
    args$softmax_temperature <- options$softmax_temperature
    args$average_logits <- options$average_logits
    args$binary_calibration_method <- options$binary_calibration_method
    args$multiclass_calibration_method <- options$multiclass_calibration_method
    args$calibration_lambda <- options$calibration_lambda
  }

  args
}

tabfm_load_model <- function(loader, kind, options, backend) {
  if (backend == "pytorch") {
    return(loader$load(
      model_type = kind,
      checkpoint_path = options$checkpoint_path,
      device = options$device,
      use_cache = options$use_cache
    ))
  }

  if (backend == "jax") {
    return(loader$load(
      model_type = kind,
      checkpoint_path = options$checkpoint_path,
      step = options$step,
      col_attention_impl = options$col_attention_impl,
      row_attention_impl = options$row_attention_impl,
      icl_attention_impl = options$icl_attention_impl,
      use_cache = options$use_cache
    ))
  }

  cli::cli_abort("Could not load a TabFM model for backend {.val {options$backend}}.")
}

tab_fm_impl <- function(x, y, opts) {
  kind <- tabfm_model_type(y)

  if (kind == "classification" && is.factor(y) && length(levels(y)) > 10) {
    cli::cli_abort(
      c(
        i = "There are {length(levels(y))} classes in the outcome.",
        x = "TabFM supports at most 10 classes."
      )
    )
  }

  backend_request <- opts$backend
  if (backend_request == "auto") {
    if (!is.null(opts$device)) {
      backend_request <- "pytorch"
    } else if (
      !is.null(opts$step) ||
      !identical(opts$col_attention_impl, "flash") ||
      !identical(opts$row_attention_impl, "jax") ||
      !identical(opts$icl_attention_impl, "flash")
    ) {
      backend_request <- "jax"
    }
  }

  loader_info <- tabfm_loader_module(backend_request)
  loader <- loader_info$module
  backend <- loader_info$backend
  tabfm_pkg <- import_tabfm()
  if (is.null(tabfm_pkg)) {
    tabfm_pkg <- reticulate::import("tabfm")
  }

  model <- tabfm_load_model(loader, kind, opts, backend)

  class_ctor <- if (kind == "classification") {
    tabfm_pkg$TabFMClassifier
  } else {
    tabfm_pkg$TabFMRegressor
  }

  est_args <- tabfm_estimator_args(kind, opts)
  est_call <- rlang::call2(class_ctor, model = model, !!!est_args)
  est <- rlang::eval_bare(est_call)

  if (kind == "classification") {
    y_fit <- as.character(y)
    lvls <- levels(y)
  } else {
    y_fit <- y
    lvls <- NULL
  }

  captured <- capture_python_or_r(model_fit <- try(est$fit(x, y_fit), silent = TRUE))
  py_msg <- captured$output

  if (inherits(model_fit, "try-error")) {
    msgs <- as.character(model_fit)
    cli::cli_abort("Model failed: {msgs}")
  } else {
    msgs <- character(0)
  }

  res <- list(
    fit = model_fit,
    lvls = lvls,
    train = dim(x),
    logging = c(r = msgs, py = py_msg)
  )
  class(res) <- c("tab_fm")
  res
}

#' @export
print.tab_fm <- function(x, ...) {
  type <- ifelse(is.null(x$levels), "Regression", "Classification")
  cli::cli_inform("TabFM {type} Model")
  cat("\n")
  cli::cli_inform("Training set\n\n")
  cli::cli_inform(c(i = "{x$training[1]} data point{?s}"))
  cli::cli_inform(c(i = "{x$training[2]} predictor{?s}"))

  if (!is.null(x$levels)) {
    cli::cli_inform(c(i = "class levels: {.val {x$levels}}"))
  }

  invisible(x)
}
