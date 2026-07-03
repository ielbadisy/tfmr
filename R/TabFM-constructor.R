new_tab_fm <- function(
  fit,
  levels,
  training,
  logging,
  blueprint,
  call = NULL
) {
  check_character(levels, allow_null = TRUE)

  hardhat::new_model(
    fit = fit,
    levels = levels,
    training = training,
    logging = logging,
    blueprint = blueprint,
    class = "tab_fm"
  )
}
