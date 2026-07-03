skip_if_no_tabpfn <- function() {
  skip_if(
    !is_tab_pfn_installed(),
    message = "TabPFN Python library is not installed"
  )
  skip_on_cran()
}

exp_cls <- c("tab_pfn", "hardhat_model", "hardhat_scalar")
exp_cls_fm <- c("tab_fm", "hardhat_model", "hardhat_scalar")

skip_if_no_tabfm <- function() {
  skip_if(
    !is_tab_fm_installed(),
    message = "TabFM Python library is not installed"
  )
  skip_on_cran()
}

predictors <- mtcars[, -1]
outcome <- mtcars[, 1]
