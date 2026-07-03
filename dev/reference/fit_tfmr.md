# Fit a tabular foundation model

`fit_tfmr()` is a small convenience wrapper over
[`tab_pfn()`](https://tabpfn.tidymodels.org/dev/reference/tab_pfn.md),
[`tab_icl()`](https://tabpfn.tidymodels.org/dev/reference/tab_icl.md),
and [`tab_fm()`](https://tabpfn.tidymodels.org/dev/reference/tab_fm.md).
Use the explicit backend functions when you need backend-specific
arguments.

## Usage

``` r
fit_tfmr(x, ..., engine = c("tabpfn", "tabicl", "tabfm"))

fit_tabfm(x, ..., engine = c("tabpfn", "tabicl", "tabfm"))
```

## Arguments

- x:

  A data frame, matrix, recipe, or formula.

- ...:

  Arguments passed to the selected backend.

- engine:

  One of `"tabpfn"`, `"tabicl"`, or `"tabfm"`.

## Value

A fitted foundation-model object.
