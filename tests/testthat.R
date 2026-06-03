if (!requireNamespace("testthat", quietly = TRUE)) {
  stop("Package 'testthat' is required to run tests.", call. = FALSE)
}

testthat::test_dir(file.path("tests", "testthat"))
