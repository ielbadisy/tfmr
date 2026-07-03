# Fit a TabFM model

`tab_fm()` fits the Python `tabfm` classifier or regressor through
`reticulate` and returns an R object with
[`predict()`](https://rdrr.io/r/stats/predict.html) and
[`augment()`](https://generics.r-lib.org/reference/augment.html)
methods.

## Usage

``` r
tab_fm(x, ...)

# Default S3 method
tab_fm(x, ...)

# S3 method for class 'data.frame'
tab_fm(x, y, training_set_limit = Inf, control = control_tab_fm(), ...)

# S3 method for class 'matrix'
tab_fm(x, y, training_set_limit = Inf, control = control_tab_fm(), ...)

# S3 method for class 'formula'
tab_fm(
  formula,
  data,
  training_set_limit = Inf,
  control = control_tab_fm(),
  ...
)

# S3 method for class 'recipe'
tab_fm(x, data, training_set_limit = Inf, control = control_tab_fm(), ...)
```

## Arguments

- x:

  A data frame, matrix, recipe, or formula.

- ...:

  Not currently used.

- y:

  Outcome vector for the data-frame and matrix interfaces.

- training_set_limit:

  Maximum number of training rows retained before fitting. Use `Inf` to
  disable downsampling.

- control:

  A list from
  [`control_tab_fm()`](https://tabpfn.tidymodels.org/dev/reference/control_tab_fm.md).

- formula:

  Formula for the formula interface.

- data:

  Data frame for formula and recipe interfaces.

## Value

A `tab_fm` model object.
