#' Fit a tabular foundation model
#'
#' `fit_tfmr()` is a small convenience wrapper over [tab_pfn()],
#' [tab_icl()], and [tab_fm()]. Use the explicit backend functions when you
#' need backend-specific arguments.
#'
#' @param x A data frame, matrix, recipe, or formula.
#' @param ... Arguments passed to the selected backend.
#' @param engine One of `"tabpfn"`, `"tabicl"`, or `"tabfm"`.
#' @return A fitted foundation-model object.
#' @export
fit_tfmr <- function(x, ..., engine = c("tabpfn", "tabicl", "tabfm")) {
  engine <- match.arg(engine)
  switch(
    engine,
    tabpfn = tab_pfn(x, ...),
    tabicl = tab_icl(x, ...),
    tabfm = tab_fm(x, ...)
  )
}

#' @rdname fit_tfmr
#' @export
fit_tabfm <- fit_tfmr
