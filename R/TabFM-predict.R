#' Predict using `TabFM`
#'
#' @param object,x A `tab_fm` object.
#'
#' @param new_data A data frame or matrix of new predictors.
#'
#' @param type The type of prediction. For classification, can be `"class"` or
#' `"prob"`. Defaults to `NULL` which gives all prediction types possible.
#'
#' @param ... Not used, but required for extensibility.
#'
#' @return
#'
#' [predict()] returns a tibble of predictions and [augment()] appends the
#' columns in `new_data`. In either case, the number of rows in the tibble is
#' guaranteed to be the same as the number of rows in `new_data`.
#'
#' For regression data, the prediction is in the column `.pred`. For
#' classification, the class predictions are in `.pred_class` and the
#' probability estimates are in columns with the pattern `.pred_{level}` where
#' `level` is the levels of the outcome factor vector.
#'
#' @export
predict.tab_fm <- function(object, new_data, type = NULL, ...) {
  rlang::check_dots_empty()
  forged <- hardhat::forge(new_data, object$blueprint)$predictors

  if (is.null(object$levels)) {
    out <- try(object$fit$predict(forged), silent = TRUE)
    if (inherits(out, "try-error")) {
      cli::cli_abort("Prediction failed: {as.character(out)}")
    }
    return(tibble::tibble(.pred = as.vector(out)))
  }

  out <- try(object$fit$predict_proba(forged), silent = TRUE)
  if (inherits(out, "try-error")) {
    cli::cli_abort("Prediction failed: {as.character(out)}")
  }

  out <- as.matrix(out)
  cls <- as.character(object$fit$classes_)
  colnames(out) <- paste0(".pred_", cls)
  cls_ind <- apply(out, 1, which.max)
  res <- tibble::as_tibble(out)
  res$.pred_class <- factor(cls[cls_ind], levels = object$levels)

  if (!is.null(type)) {
    type <- rlang::arg_match(type, c("class", "prob"))
    if (type == "class") {
      res <- res[, ".pred_class"]
    } else if (type == "prob") {
      res <- res[, names(res) != ".pred_class"]
    }
  }

  res
}

#' @rdname predict.tab_fm
#' @export
augment.tab_fm <- function(x, new_data, type = NULL, ...) {
  new_data <- tibble::new_tibble(new_data)
  res <- predict(x, new_data, type = type)
  res <- cbind(res, new_data)
  tibble::new_tibble(res)
}
