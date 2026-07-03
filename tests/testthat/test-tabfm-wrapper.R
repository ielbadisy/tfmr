test_that("tab_fm regression wrapper fits and predicts through sklearn-style API", {
  fake_fit <- structure(
    list(
      predict = function(newdata) rep(1.5, nrow(newdata))
    ),
    class = "tab_fm_mock"
  )
  fake_regressor <- function(...) {
    structure(
      list(fit = function(x, y) fake_fit),
      class = "tab_fm_mock"
    )
  }
  fake_loader <- list(load = function(...) "fake_model")
  fake_module <- list(TabFMRegressor = fake_regressor)

  local_mocked_bindings(
    import_tabfm = function() fake_module,
    tabfm_loader_module = function(...) list(module = fake_loader, backend = "pytorch")
  )

  mod <- tab_fm(mpg ~ wt + hp, data = mtcars[1:10, ])
  expect_s3_class(mod, "tab_fm")
  pred <- predict(mod, mtcars[11:12, ])
  expect_equal(pred, tibble::tibble(.pred = c(1.5, 1.5)))
  expect_s3_class(fit_tfmr(mpg ~ wt + hp, data = mtcars[1:10, ], engine = "tabfm"), "tab_fm")
})

test_that("tab_fm classification wrapper returns class and probability predictions", {
  fake_fit <- structure(
    list(
      classes_ = c("no", "yes"),
      predict_proba = function(newdata) matrix(
        c(0.8, 0.2, 0.3, 0.7),
        nrow = nrow(newdata),
        byrow = TRUE
      )
    ),
    class = "tab_fm_mock"
  )
  fake_classifier <- function(...) {
    structure(
      list(fit = function(x, y) fake_fit),
      class = "tab_fm_mock"
    )
  }
  fake_loader <- list(load = function(...) "fake_model")
  fake_module <- list(TabFMClassifier = fake_classifier)

  local_mocked_bindings(
    import_tabfm = function() fake_module,
    tabfm_loader_module = function(...) list(module = fake_loader, backend = "pytorch")
  )

  dat <- data.frame(
    y = factor(c("no", "yes", "no", "yes")),
    x1 = 1:4,
    x2 = c(1, 1, 0, 0)
  )
  mod <- tab_fm(y ~ ., data = dat)
  pred <- predict(mod, dat[1:2, ])
  expect_equal(names(pred), c(".pred_no", ".pred_yes", ".pred_class"))
  expect_equal(as.character(pred$.pred_class), c("no", "yes"))
  expect_equal(
    predict(mod, dat[1:2, ], type = "prob")[0, ],
    tibble::tibble(.pred_no = numeric(), .pred_yes = numeric())
  )
})
