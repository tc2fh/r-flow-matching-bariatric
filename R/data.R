# Data loading, target construction, preprocessing, and patient splits.

normalize_cpt_code <- function(x) {
  if (is.numeric(x)) {
    return(ifelse(is.na(x), NA_character_, sprintf("%.0f", x)))
  }
  code <- trimws(as.character(x))
  code <- sub("\\.0$", "", code)
  code[code == "" | is.na(x)] <- NA_character_
  code
}

map_surgery_type <- function(cpt_code) {
  code <- normalize_cpt_code(cpt_code)
  out <- unname(SURGERY_CPT_MAP[code])
  out[is.na(code)] <- NA_character_
  out
}

encode_sex_male <- function(sex) {
  value <- tolower(trimws(as.character(sex)))
  out <- rep(NA_real_, length(value))
  out[value %in% c("male", "m")] <- 1
  out[value %in% c("female", "f")] <- 0
  out
}

normalize_column_name <- function(name) {
  tolower(gsub("[^A-Za-z0-9]", "", name))
}

find_compatible_column <- function(existing_names, canonical_name) {
  if (canonical_name %in% existing_names) {
    return(canonical_name)
  }

  normalized_existing <- normalize_column_name(existing_names)
  normalized_canonical <- normalize_column_name(canonical_name)

  exact_case_insensitive <- which(normalized_existing == normalized_canonical)
  if (length(exact_case_insensitive) > 0) {
    return(existing_names[exact_case_insensitive[1]])
  }

  suffixes <- c(
    ".y",
    "_y",
    ".mbs",
    "_mbs",
    ".mbscohort",
    "_mbscohort",
    ".x",
    "_x",
    ".glp1",
    "_glp1",
    ".glp1cohort",
    "_glp1cohort"
  )
  for (suffix in suffixes) {
    normalized_candidate <- normalize_column_name(paste0(canonical_name, suffix))
    suffix_match <- which(normalized_existing == normalized_candidate)
    if (length(suffix_match) > 0) {
      return(existing_names[suffix_match[1]])
    }
  }

  NA_character_
}

required_wide_columns <- function() {
  unique(c(
    "PatKey",
    "CptCode",
    "AgeAtEvent",
    "Sex",
    "CreatinineAtEvent",
    "HbA1cAtEvent",
    "BMIatEvent",
    "InsulinStatus",
    make_target_metadata()$source_column
  ))
}

canonicalize_flow_columns <- function(df) {
  required_cols <- required_wide_columns()
  original_names <- names(df)

  for (canonical_name in required_cols) {
    matched_name <- find_compatible_column(original_names, canonical_name)
    if (!is.na(matched_name) && matched_name != canonical_name) {
      names(df)[names(df) == matched_name] <- canonical_name
      original_names[original_names == matched_name] <- canonical_name
    }
  }

  df
}

assert_required_columns <- function(df, required_cols, source_path) {
  missing_cols <- setdiff(required_cols, names(df))
  if (length(missing_cols) > 0) {
    stop(
      "Required columns missing from ",
      source_path,
      ": ",
      paste(missing_cols, collapse = ", "),
      call. = FALSE
    )
  }
}

is_binary_event_target <- function(group) {
  group %in% c("retinopathy", "nephropathy", "mace")
}

binary_event_from_column <- function(x) {
  value <- suppressWarnings(as.numeric(x))
  as.numeric(!is.na(value) & value == 1)
}

build_target_matrix <- function(df, metadata = make_target_metadata()) {
  x <- matrix(0, nrow = nrow(df), ncol = nrow(metadata))
  mask <- matrix(0, nrow = nrow(df), ncol = nrow(metadata))
  colnames(x) <- metadata$name
  colnames(mask) <- paste0(metadata$name, "_observed")

  for (i in seq_len(nrow(metadata))) {
    if (is_binary_event_target(metadata$group[i])) {
      value <- binary_event_from_column(df[[metadata$source_column[i]]])
      observed <- rep(TRUE, length(value))
    } else {
      value <- suppressWarnings(as.numeric(df[[metadata$source_column[i]]]))
      observed <- !is.na(value)
    }
    x[observed, i] <- value[observed]
    mask[observed, i] <- 1
  }

  list(x = x, mask = mask)
}

make_patient_feature_frame <- function(df) {
  data.frame(
    age_at_surgery = suppressWarnings(as.numeric(df$AgeAtEvent)),
    sex_male = encode_sex_male(df$Sex),
    creatinine_at_surgery = suppressWarnings(as.numeric(df$CreatinineAtEvent)),
    hba1c_at_surgery = suppressWarnings(as.numeric(df$HbA1cAtEvent)),
    bmi_at_surgery = suppressWarnings(as.numeric(df$BMIatEvent)),
    insulin_status = suppressWarnings(as.numeric(df$InsulinStatus)),
    stringsAsFactors = FALSE
  )
}

make_surgery_one_hot <- function(surgery_type) {
  one_hot <- matrix(0, nrow = length(surgery_type), ncol = length(SURGERY_LEVELS))
  colnames(one_hot) <- paste0("surgery_", SURGERY_LEVELS)
  for (i in seq_along(SURGERY_LEVELS)) {
    one_hot[, i] <- as.numeric(surgery_type == SURGERY_LEVELS[i])
  }
  one_hot
}

prepare_flow_dataset <- function(df, source_path = "loaded data frame") {
  metadata <- make_target_metadata()
  df <- canonicalize_flow_columns(df)
  assert_required_columns(df, required_wide_columns(), source_path)

  df$cpt_code_normalized <- normalize_cpt_code(df$CptCode)
  df$surgery_type <- map_surgery_type(df$cpt_code_normalized)

  unknown_codes <- sort(unique(df$cpt_code_normalized[is.na(df$surgery_type)]))
  unknown_codes <- unknown_codes[!is.na(unknown_codes)]
  if (length(unknown_codes) > 0) {
    warning(
      "Excluding rows with unrecognized CptCode values: ",
      paste(unknown_codes, collapse = ", "),
      call. = FALSE
    )
  }
  df <- df[!is.na(df$surgery_type), , drop = FALSE]

  if (any(duplicated(df$PatKey))) {
    duplicated_ids <- unique(df$PatKey[duplicated(df$PatKey)])
    stop(
      "Duplicate PatKey rows found in wide patient input: ",
      paste(head(duplicated_ids, 10), collapse = ", "),
      call. = FALSE
    )
  }

  patient_features_df <- make_patient_feature_frame(df)
  complete_conditioning <- complete.cases(patient_features_df)
  dropped_conditioning <- sum(!complete_conditioning)
  if (dropped_conditioning > 0) {
    warning(
      "Dropping ",
      dropped_conditioning,
      " rows with missing required conditioning fields.",
      call. = FALSE
    )
  }

  df <- df[complete_conditioning, , drop = FALSE]
  patient_features_df <- patient_features_df[complete_conditioning, , drop = FALSE]

  targets <- build_target_matrix(df, metadata)
  surgery_one_hot <- make_surgery_one_hot(df$surgery_type)
  surgery_idx <- match(df$surgery_type, SURGERY_LEVELS)

  list(
    source_path = source_path,
    data = df,
    subject_ids = as.character(df$PatKey),
    surgery_type = df$surgery_type,
    surgery_idx = surgery_idx,
    surgery_one_hot = surgery_one_hot,
    patient_features_raw = as.matrix(patient_features_df[, PATIENT_FEATURES, drop = FALSE]),
    patient_feature_names = PATIENT_FEATURES,
    x = targets$x,
    mask = targets$mask,
    target_metadata = metadata
  )
}

load_flow_dataset_from_data_frame <- function(df, source_label = "loaded data frame") {
  prepare_flow_dataset(df, source_path = source_label)
}

load_flow_dataset <- function(csv_path = default_flow_config()$csv_path) {
  if (!file.exists(csv_path)) {
    stop("CSV not found: ", csv_path, call. = FALSE)
  }
  df <- read.csv(
    csv_path,
    stringsAsFactors = FALSE,
    na.strings = c("", "NA", "NaN"),
    colClasses = "character"
  )
  prepare_flow_dataset(df, source_path = csv_path)
}

write_flow_input_csv <- function(df, output_path) {
  df <- canonicalize_flow_columns(df)
  assert_required_columns(df, required_wide_columns(), "loaded data frame")
  output_dir <- dirname(output_path)
  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  }
  write.csv(df, output_path, row.names = FALSE, na = "")
  output_path
}

split_counts <- function(n, train_frac, val_frac) {
  n_train <- floor(n * train_frac)
  n_val <- floor(n * val_frac)
  n_test <- n - n_train - n_val
  c(train = n_train, val = n_val, test = n_test)
}

make_patient_splits <- function(dataset, config = default_flow_config()) {
  validate_flow_config(config)
  set.seed(config$split_seed)

  train_idx <- integer(0)
  val_idx <- integer(0)
  test_idx <- integer(0)

  for (surgery in sort(unique(dataset$surgery_type))) {
    idx <- which(dataset$surgery_type == surgery)
    idx <- sample(idx, length(idx), replace = FALSE)
    counts <- split_counts(length(idx), config$train_frac, config$val_frac)

    train_idx <- c(train_idx, idx[seq_len(counts["train"])])

    val_start <- counts["train"] + 1
    val_end <- counts["train"] + counts["val"]
    if (counts["val"] > 0) {
      val_idx <- c(val_idx, idx[val_start:val_end])
    }

    test_start <- val_end + 1
    if (counts["test"] > 0) {
      test_idx <- c(test_idx, idx[test_start:length(idx)])
    }
  }

  list(
    train_idx = sample(train_idx, length(train_idx), replace = FALSE),
    val_idx = sample(val_idx, length(val_idx), replace = FALSE),
    test_idx = sample(test_idx, length(test_idx), replace = FALSE)
  )
}

assert_no_patient_leakage <- function(dataset, splits) {
  split_subjects <- list(
    train = dataset$subject_ids[splits$train_idx],
    val = dataset$subject_ids[splits$val_idx],
    test = dataset$subject_ids[splits$test_idx]
  )
  overlaps <- list(
    train_val = intersect(split_subjects$train, split_subjects$val),
    train_test = intersect(split_subjects$train, split_subjects$test),
    val_test = intersect(split_subjects$val, split_subjects$test)
  )
  bad <- overlaps[vapply(overlaps, length, integer(1)) > 0]
  if (length(bad) > 0) {
    stop("Patient leakage across splits detected.", call. = FALSE)
  }
  invisible(TRUE)
}

fit_preprocessing <- function(dataset, train_idx) {
  x_train <- dataset$x[train_idx, , drop = FALSE]
  m_train <- dataset$mask[train_idx, , drop = FALSE]
  observed_count <- colSums(m_train)
  if (any(observed_count == 0)) {
    empty_dims <- which(observed_count == 0)
    empty_names <- dataset$target_metadata$name[empty_dims]
    stop(
      "Target dimensions have no train observations: ",
      paste(empty_names, collapse = ", "),
      call. = FALSE
    )
  }

  target_mean <- colSums(x_train * m_train) / observed_count
  centered <- sweep(x_train, 2, target_mean, "-")
  target_var <- colSums((centered^2) * m_train) / observed_count
  target_std <- sqrt(target_var)
  target_std[target_std < 1e-8 | is.na(target_std)] <- 1

  raw <- dataset$patient_features_raw[train_idx, , drop = FALSE]
  static_mean <- rep(0, ncol(raw))
  static_std <- rep(1, ncol(raw))
  names(static_mean) <- dataset$patient_feature_names
  names(static_std) <- dataset$patient_feature_names

  continuous_idx <- match(CONTINUOUS_PATIENT_FEATURES, dataset$patient_feature_names)
  static_mean[continuous_idx] <- colMeans(raw[, continuous_idx, drop = FALSE])
  static_std[continuous_idx] <- apply(raw[, continuous_idx, drop = FALSE], 2, stats::sd)
  bad_static_std <- static_std[continuous_idx] < 1e-8 | is.na(static_std[continuous_idx])
  static_std[continuous_idx[bad_static_std]] <- 1

  list(
    target_mean = target_mean,
    target_std = target_std,
    static_mean = static_mean,
    static_std = static_std,
    static_continuous_idx = continuous_idx,
    patient_feature_names = dataset$patient_feature_names,
    target_metadata = dataset$target_metadata
  )
}

transform_targets <- function(x, mask, preprocessing) {
  z <- sweep(x, 2, preprocessing$target_mean, "-")
  z <- sweep(z, 2, preprocessing$target_std, "/")
  z * mask
}

transform_patient_features <- function(patient_features_raw, preprocessing) {
  out <- patient_features_raw
  idx <- preprocessing$static_continuous_idx
  out[, idx] <- sweep(out[, idx, drop = FALSE], 2, preprocessing$static_mean[idx], "-")
  out[, idx] <- sweep(out[, idx, drop = FALSE], 2, preprocessing$static_std[idx], "/")
  out
}

split_preprocessed_arrays <- function(dataset, splits, preprocessing) {
  x_std <- transform_targets(dataset$x, dataset$mask, preprocessing)
  patient_std <- transform_patient_features(dataset$patient_features_raw, preprocessing)

  make_split <- function(idx) {
    list(
      x = x_std[idx, , drop = FALSE],
      mask = dataset$mask[idx, , drop = FALSE],
      surgery_one_hot = dataset$surgery_one_hot[idx, , drop = FALSE],
      patient_features = patient_std[idx, , drop = FALSE],
      subject_ids = dataset$subject_ids[idx],
      original_x = dataset$x[idx, , drop = FALSE],
      original_mask = dataset$mask[idx, , drop = FALSE]
    )
  }

  list(
    train = make_split(splits$train_idx),
    val = make_split(splits$val_idx),
    test = make_split(splits$test_idx)
  )
}

write_dataset_summary <- function(dataset, splits, path) {
  summary <- data.frame(
    item = c(
      "source_path",
      "patients",
      "x_dim",
      "mean_observed_targets",
      "train_patients",
      "val_patients",
      "test_patients",
      "sleeve_patients",
      "rnygb_patients"
    ),
    value = c(
      dataset$source_path,
      length(dataset$subject_ids),
      ncol(dataset$x),
      sprintf("%.3f", mean(rowSums(dataset$mask))),
      length(splits$train_idx),
      length(splits$val_idx),
      length(splits$test_idx),
      sum(dataset$surgery_type == "sleeve"),
      sum(dataset$surgery_type == "rnygb")
    ),
    stringsAsFactors = FALSE
  )
  write.csv(summary, path, row.names = FALSE)
}
