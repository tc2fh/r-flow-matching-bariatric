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
  testthat::expect_equal(
    dataset$target_metadata$name[16:18],
    c("retinopathy_ever", "nephropathy_ever", "mace_ever")
  )
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
  names(df)[names(df) == "Retinopathy"] <- "retinopathy"
  names(df)[names(df) == "Nephropathy"] <- "nephropathy"

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

testthat::test_that("binary event targets treat NA and 0 as 0 and 1 as 1", {
  value <- binary_event_from_column(c(1, 0, NA, "1", "0", ""))

  testthat::expect_equal(value, c(1, 0, 0, 1, 0, 0))
})

testthat::test_that("event targets are fully observed and do not require interval columns", {
  df <- read.csv(
    file.path(project_root, "fake_data", "fake_mbs_cohort.csv"),
    stringsAsFactors = FALSE,
    na.strings = c("", "NA", "NaN"),
    colClasses = "character"
  )
  df$MACEinterval <- NULL
  df$ActiveEndInterval <- NULL

  dataset <- load_flow_dataset_from_data_frame(df, source_label = "no interval columns")
  event_dims <- dataset$target_metadata$dim[dataset$target_metadata$group %in% c("retinopathy", "nephropathy", "mace")]

  testthat::expect_equal(event_dims, 16:18)
  testthat::expect_true(all(dataset$mask[, event_dims] == 1))
  testthat::expect_true(all(dataset$x[, event_dims] %in% c(0, 1)))
})

testthat::test_that("surgery and sex encodings are deterministic", {
  testthat::expect_equal(
    map_surgery_type(c("43775", "43644", "43846", "43645")),
    c("sleeve", "rnygb", "rnygb", "rnygb")
  )
  testthat::expect_equal(encode_sex_male(c("Female", "Male", "F", "M")), c(0, 1, 0, 1))
})
