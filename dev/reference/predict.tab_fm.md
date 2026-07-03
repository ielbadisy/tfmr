# Predict using `TabFM`

Predict using `TabFM`

## Usage

``` r
# S3 method for class 'tab_fm'
predict(object, new_data, type = NULL, ...)

# S3 method for class 'tab_fm'
augment(x, new_data, type = NULL, ...)
```

## Arguments

- object, x:

  A `tab_fm` object.

- new_data:

  A data frame or matrix of new predictors.

- type:

  The type of prediction. For classification, can be `"class"` or
  `"prob"`. Defaults to `NULL` which gives all prediction types
  possible.

- ...:

  Not used, but required for extensibility.

## Value

[`predict()`](https://rdrr.io/r/stats/predict.html) returns a tibble of
predictions and
[`augment()`](https://generics.r-lib.org/reference/augment.html) appends
the columns in `new_data`. In either case, the number of rows in the
tibble is guaranteed to be the same as the number of rows in `new_data`.

For regression data, the prediction is in the column `.pred`. For
classification, the class predictions are in `.pred_class` and the
probability estimates are in columns with the pattern `.pred_{level}`
where `level` is the levels of the outcome factor vector.
