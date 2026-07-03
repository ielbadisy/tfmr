# Fit a TabICLv2 model

`tab_icl()` fits the Python `tabicl` classifier or regressor through
`reticulate` and returns an R object with
[`predict()`](https://rdrr.io/r/stats/predict.html) and
[`augment()`](https://generics.r-lib.org/reference/augment.html)
methods.

## Usage

``` r
tab_icl(x, ...)

# Default S3 method
tab_icl(x, ...)

# S3 method for class 'data.frame'
tab_icl(x, y, training_set_limit = Inf, control = control_tab_icl(), ...)

# S3 method for class 'matrix'
tab_icl(x, y, training_set_limit = Inf, control = control_tab_icl(), ...)

# S3 method for class 'formula'
tab_icl(
  formula,
  data,
  training_set_limit = Inf,
  control = control_tab_icl(),
  ...
)

# S3 method for class 'recipe'
tab_icl(x, data, training_set_limit = Inf, control = control_tab_icl(), ...)
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
  [`control_tab_icl()`](https://ielbadisy.github.io/tfmr/dev/reference/control_tab_icl.md).

- formula:

  Formula for the formula interface.

- data:

  Data frame for formula and recipe interfaces.

## Value

A `tab_icl` model object.
