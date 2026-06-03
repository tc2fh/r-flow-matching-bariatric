project_root <- if (file.exists(file.path("R", "config.R"))) "." else file.path("..", "..")

source(file.path(project_root, "R", "config.R"))
source(file.path(project_root, "R", "data.R"))

testthat::test_that("loader creates 18 target dimensions and matching masks", {
  dataset <- load_flow_dataset(file.path(project_root, "fake_data", "fake_mbs_cohort.csv"))

  testthat::expect_equal(ncol(dataset$x), 18)
  testthat::expect_equal(dim(dataset$x), dim(dataset$mask))
  testthat::expect_equal(nrow(dataset$x), length(dataset$subject_ids))
  testthat::expect_true(all(dataset$mask %in% c(0, 1)))
  testthat::expect_equal(colnames(dataset$surgery_one_hot), c("surgery_sleeve", "surgery_rnygb"))
})

testthat::test_that("loader accepts loaded data frames with DB-style column casing", {
  df <- read.csv(
    file.path(project_root, "fake_data", "fake_mbs_cohort.csv"),
    stringsAsFactors = FALSE,
    na.strings = c("", "NA", "NaN"),
    colClasses = "character"
  )
  names(df)[names(df) == "PatKey"] <- "Patkey"
  names(df)[names(df) == "Sex"] <- "sex"
  names(df)[names(df) == "InsulinStatus"] <- "Insulinstatus"
  names(df)[names(df) == "MACEinterval"] <- "MACEInterval"

  dataset <- load_flow_dataset_from_data_frame(df, source_label = "test data frame")

  testthat::expect_equal(ncol(dataset$x), 18)
  testthat::expect_equal(nrow(dataset$x), nrow(df))
  testthat::expect_true(all(dataset$surgery_type %in% c("sleeve", "rnygb")))
})

testthat::test_that("write_flow_input_csv canonicalizes loaded data frame columns", {
  df <- read.csv(
    file.path(project_root, "fake_data", "fake_mbs_cohort.csv"),
    stringsAsFactors = FALSE,
    na.strings = c("", "NA", "NaN"),
    colClasses = "character"
  )
  names(df)[names(df) == "PatKey"] <- "Patkey"
  output_path <- tempfile(fileext = ".csv")

  write_flow_input_csv(df, output_path)
  exported <- read.csv(output_path, stringsAsFactors = FALSE, colClasses = "character")

  testthat::expect_true("PatKey" %in% names(exported))
  testthat::expect_false("Patkey" %in% names(exported))
})

testthat::test_that("MACE censoring rules produce expected cumulative targets", {
  mace <- c(1, 1, 0, 0, NA)
  mace_interval <- c(100, 500, NA, NA, NA)
  active_end <- c(1000, 1000, 400, 100, 1000)

  value <- derive_mace_by_horizon(mace, mace_interval, active_end, horizon_days = 365)

  testthat::expect_equal(value[1], 1)
  testthat::expect_equal(value[2], 0)
  testthat::expect_equal(value[3], 0)
  testthat::expect_true(is.na(value[4]))
  testthat::expect_equal(value[5], 0)
})

testthat::test_that("surgery and sex encodings are deterministic", {
  testthat::expect_equal(map_surgery_type(c("43775", "43644", "43846")), c("sleeve", "rnygb", "rnygb"))
  testthat::expect_equal(encode_sex_male(c("Female", "Male", "F", "M")), c(0, 1, 0, 1))
})
