if (!requireNamespace("testthat", quietly = TRUE)) {
  stop("Package 'testthat' is required to run tests.", call. = FALSE)
}

if (!dir.exists("R")) {
  message("Skipping legacy R tests: the R implementation is not present in this revision.")
  quit(save = "no", status = 0)
}

testthat::test_dir(file.path("tests", "testthat"))
