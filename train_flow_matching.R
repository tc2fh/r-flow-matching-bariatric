# Monolithic R torch concat-conditioned flow-matching trainer.
#
# Direct command-line use:
#   Rscript train_flow_matching.R --csv fake_data/fake_mbs_cohort.csv
#
# Collaborator database-script use:
#   mbs <- DBI::dbGetQuery(...)
#   source("train_flow_matching.R")
#
# If sourced, this script automatically trains from a loaded data frame named
# `mbs` when present. Optional sourced-run overrides can be supplied before
# sourcing, for example:
#   flow_config_overrides <- list(num_steps = 1000, batch_size = 32)
#   source("train_flow_matching.R")

detect_project_root <- function() {
  script_args <- commandArgs(trailingOnly = FALSE)
  file_arg <- "--file="
  script_path <- sub(file_arg, "", script_args[startsWith(script_args, file_arg)])
  if (length(script_path) > 0) {
    return(dirname(normalizePath(script_path[1])))
  }

  frames <- sys.frames()
  ofiles <- vapply(frames, function(frame) {
    if (!is.null(frame$ofile)) frame$ofile else ""
  }, character(1))
  ofiles <- ofiles[nzchar(ofiles)]
  if (length(ofiles) > 0) {
    return(dirname(normalizePath(tail(ofiles, 1))))
  }

  getwd()
}

PROJECT_ROOT <- detect_project_root()

require_package <- function(package_name, purpose = "this script") {
  if (!requireNamespace(package_name, quietly = TRUE)) {
    stop("Package '", package_name, "' is required for ", purpose, ".", call. = FALSE)
  }
}

require_package("torch", "training")

# ---- Config and schema ----

BMI_TARGETS <- data.frame(
  group = "bmi",
  horizon_label = c("3m", "6m", "9m", "12m", "2y", "3y", "4y", "5y", "6y"),
  horizon_months = c(3, 6, 9, 12, 24, 36, 48, 60, 72),
  horizon_days = NA_integer_,
  source_column = c(
    "BMI3mPostEvent",
    "BMI6mPostEvent",
    "BMI9mPostEvent",
    "BMI12mPostEvent",
    "BMI2yPostEvent",
    "BMI3yPostEvent",
    "BMI4yPostEvent",
    "BMI5yPostEvent",
    "BMI6yPostEvent"
  ),
  stringsAsFactors = FALSE
)

HBA1C_TARGETS <- data.frame(
  group = "hba1c",
  horizon_label = c("12m", "2y", "3y", "4y", "5y", "6y"),
  horizon_months = c(12, 24, 36, 48, 60, 72),
  horizon_days = NA_integer_,
  source_column = c(
    "HbA1c12mPostEvent",
    "HbA1c2yPostEvent",
    "HbA1c3yPostEvent",
    "HbA1c4yPostEvent",
    "HbA1c5yPostEvent",
    "HbA1c6yPostEvent"
  ),
  stringsAsFactors = FALSE
)

EVENT_TARGETS <- data.frame(
  group = c("retinopathy", "nephropathy", "mace"),
  horizon_label = "ever",
  horizon_months = NA_integer_,
  horizon_days = NA_integer_,
  source_column = c("Retinopathy", "Nephropathy", "MACE"),
  stringsAsFactors = FALSE
)

PATIENT_FEATURES <- c(
  "age_at_surgery",
  "sex_male",
  "creatinine_at_surgery",
  "hba1c_at_surgery",
  "bmi_at_surgery",
  "insulin_status"
)

CONTINUOUS_PATIENT_FEATURES <- c(
  "age_at_surgery",
  "creatinine_at_surgery",
  "hba1c_at_surgery",
  "bmi_at_surgery"
)

SURGERY_LEVELS <- c("sleeve", "rnygb")
SURGERY_CPT_MAP <- c("43775" = "sleeve", "43644" = "rnygb", "43846" = "rnygb")

make_target_metadata <- function() {
  metadata <- rbind(BMI_TARGETS, HBA1C_TARGETS, EVENT_TARGETS)
  metadata$dim <- seq_len(nrow(metadata))
  metadata$name <- paste(metadata$group, metadata$horizon_label, sep = "_")
  metadata <- metadata[
    ,
    c(
      "dim",
      "name",
      "group",
      "horizon_label",
      "horizon_months",
      "horizon_days",
      "source_column"
    )
  ]
  rownames(metadata) <- NULL
  metadata
}

default_flow_config <- function(
  csv_path = file.path(PROJECT_ROOT, "fake_data", "fake_mbs_cohort.csv"),
  output_dir = file.path(PROJECT_ROOT, "runs", "r_flow_matching"),
  device = "cpu",
  seed = 0,
  split_seed = 0,
  train_frac = 0.70,
  val_frac = 0.15,
  test_frac = 0.15,
  time_emb_dim = 64,
  time_scale = 10.0,
  surgery_emb_dim = 8,
  hidden_dim = 64,
  num_hidden_layers = 2,
  learning_rate = 3e-4,
  weight_decay = 1e-2,
  num_steps = 6000,
  batch_size = 64,
  early_stop_patience = 5,
  early_stop_min_delta = 0.005,
  log_every = 100,
  val_every = 250,
  val_repeats = 8,
  sample_steps = 50,
  n_samples_per_patient = 50
) {
  list(
    csv_path = csv_path,
    output_dir = output_dir,
    device = device,
    seed = seed,
    split_seed = split_seed,
    train_frac = train_frac,
    val_frac = val_frac,
    test_frac = test_frac,
    x_dim = nrow(make_target_metadata()),
    patient_feature_dim = length(PATIENT_FEATURES),
    patient_features = PATIENT_FEATURES,
    continuous_patient_features = CONTINUOUS_PATIENT_FEATURES,
    surgery_levels = SURGERY_LEVELS,
    conditioning = "concat",
    time_emb_dim = time_emb_dim,
    time_scale = time_scale,
    surgery_emb_dim = surgery_emb_dim,
    hidden_dim = hidden_dim,
    num_hidden_layers = num_hidden_layers,
    learning_rate = learning_rate,
    weight_decay = weight_decay,
    num_steps = num_steps,
    batch_size = batch_size,
    early_stop_patience = early_stop_patience,
    early_stop_min_delta = early_stop_min_delta,
    log_every = log_every,
    val_every = val_every,
    val_repeats = val_repeats,
    sample_steps = sample_steps,
    n_samples_per_patient = n_samples_per_patient
  )
}

validate_flow_config <- function(config) {
  if (!identical(config$conditioning, "concat")) {
    stop("Only concat conditioning is implemented.", call. = FALSE)
  }
  if (config$time_emb_dim %% 2 != 0) {
    stop("time_emb_dim must be even.", call. = FALSE)
  }
  split_sum <- config$train_frac + config$val_frac + config$test_frac
  if (!isTRUE(all.equal(split_sum, 1.0))) {
    stop("train_frac + val_frac + test_frac must equal 1.0.", call. = FALSE)
  }
  if (config$num_hidden_layers < 1) {
    stop("num_hidden_layers must be at least 1.", call. = FALSE)
  }
  invisible(config)
}

save_config_json <- function(config, path) {
  if (!requireNamespace("jsonlite", quietly = TRUE)) {
    warning("Package 'jsonlite' not installed; skipping config JSON.", call. = FALSE)
    return(invisible(FALSE))
  }
  jsonlite::write_json(config, path, pretty = TRUE, auto_unbox = TRUE)
  invisible(TRUE)
}

# ---- Data loading and preprocessing ----

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

# ---- Torch model and flow matching ----

silu <- function(x) {
  x * torch::torch_sigmoid(x)
}

sinusoidal_time_embedding <- function(t, dim, time_scale = 10.0) {
  if (dim %% 2 != 0) {
    stop("dim must be even for sinusoidal_time_embedding.", call. = FALSE)
  }
  half <- dim / 2
  freqs <- exp(-log(10000.0) * (seq_len(half) - 1) / half)
  freqs <- torch::torch_tensor(freqs, dtype = torch::torch_float(), device = t$device)
  args <- t$view(c(-1, 1)) * time_scale * freqs$view(c(1, -1))
  torch::torch_cat(list(torch::torch_sin(args), torch::torch_cos(args)), dim = 2)
}

VectorFieldNet <- torch::nn_module(
  "VectorFieldNet",
  initialize = function(
    x_dim = 18,
    patient_feature_dim = 6,
    num_surgery_types = 2,
    time_emb_dim = 64,
    time_scale = 10.0,
    surgery_emb_dim = 8,
    hidden_dim = 64,
    num_hidden_layers = 2
  ) {
    if (time_emb_dim %% 2 != 0) {
      stop("time_emb_dim must be even.", call. = FALSE)
    }
    self$x_dim <- x_dim
    self$patient_feature_dim <- patient_feature_dim
    self$num_surgery_types <- num_surgery_types
    self$time_emb_dim <- time_emb_dim
    self$time_scale <- time_scale
    self$surgery_emb_dim <- surgery_emb_dim
    self$hidden_dim <- hidden_dim
    self$num_hidden_layers <- num_hidden_layers

    self$surgery_embed <- torch::nn_linear(num_surgery_types, surgery_emb_dim, bias = FALSE)

    cond_dim <- time_emb_dim + surgery_emb_dim + patient_feature_dim
    layer_input_dim <- x_dim + cond_dim
    self$hidden_layers <- torch::nn_module_list()

    for (i in seq_len(num_hidden_layers)) {
      self$hidden_layers$append(torch::nn_linear(layer_input_dim, hidden_dim))
      layer_input_dim <- hidden_dim + cond_dim
    }
    self$out <- torch::nn_linear(layer_input_dim, x_dim)
  },
  forward = function(x_t, t, surgery_one_hot, patient_features) {
    t_emb <- sinusoidal_time_embedding(t, self$time_emb_dim, self$time_scale)
    surgery_emb <- self$surgery_embed(surgery_one_hot)
    cond <- torch::torch_cat(list(t_emb, surgery_emb, patient_features), dim = 2)

    h <- torch::torch_cat(list(x_t, cond), dim = 2)
    for (i in seq_len(length(self$hidden_layers))) {
      h <- self$hidden_layers[[i]](h)
      h <- silu(h)
      h <- torch::torch_cat(list(h, cond), dim = 2)
    }
    self$out(h)
  }
)

to_torch_float <- function(x, device) {
  torch::torch_tensor(as.matrix(x), dtype = torch::torch_float(), device = device)
}

sample_conditional_path <- function(x1) {
  batch_size <- x1$size()[1]
  t <- torch::torch_rand(c(batch_size), dtype = torch::torch_float(), device = x1$device)
  x0 <- torch::torch_randn_like(x1)
  t_expand <- t$view(c(batch_size, 1))
  x_t <- (1 - t_expand) * x0 + t_expand * x1
  u_t <- x1 - x0
  list(x_t = x_t, t = t, u_t = u_t)
}

flow_matching_loss <- function(model, x_t, t, surgery_one_hot, patient_features, u_t, mask) {
  v_pred <- model(x_t, t, surgery_one_hot, patient_features)
  sq_err <- (v_pred - u_t)$pow(2)
  (mask * sq_err)$sum() / (mask$sum() + 1e-8)
}

sample_training_batch <- function(split_arrays, batch_size) {
  n <- nrow(split_arrays$x)
  if (n == 0) {
    stop("Cannot sample a training batch from an empty split.", call. = FALSE)
  }
  idx <- sample(seq_len(n), size = batch_size, replace = batch_size > n)
  list(
    x = split_arrays$x[idx, , drop = FALSE],
    mask = split_arrays$mask[idx, , drop = FALSE],
    surgery_one_hot = split_arrays$surgery_one_hot[idx, , drop = FALSE],
    patient_features = split_arrays$patient_features[idx, , drop = FALSE]
  )
}

evaluate_flow_loss <- function(model, split_arrays, device, n_repeats = 8) {
  if (nrow(split_arrays$x) == 0) {
    return(NA_real_)
  }

  x1 <- to_torch_float(split_arrays$x, device)
  mask <- to_torch_float(split_arrays$mask, device)
  surgery_one_hot <- to_torch_float(split_arrays$surgery_one_hot, device)
  patient_features <- to_torch_float(split_arrays$patient_features, device)
  losses <- numeric(n_repeats)

  model$eval()
  torch::with_no_grad({
    for (i in seq_len(n_repeats)) {
      path <- sample_conditional_path(x1)
      loss <- flow_matching_loss(
        model,
        path$x_t,
        path$t,
        surgery_one_hot,
        patient_features,
        path$u_t,
        mask
      )
      losses[i] <- loss$item()
    }
  })
  model$train()
  mean(losses)
}

compute_observed_metrics <- function(pred_mean, observed, mask, metadata) {
  rows <- list()
  add_row <- function(group_name, dim_idx) {
    obs <- mask[, dim_idx, drop = FALSE] == 1
    n_obs <- sum(obs)
    if (n_obs == 0) {
      mae <- NA_real_
      rmse <- NA_real_
    } else {
      diff <- pred_mean[, dim_idx, drop = FALSE][obs] - observed[, dim_idx, drop = FALSE][obs]
      mae <- mean(abs(diff))
      rmse <- sqrt(mean(diff^2))
    }
    data.frame(
      group = group_name,
      n_observed = n_obs,
      mae = mae,
      rmse = rmse,
      stringsAsFactors = FALSE
    )
  }

  rows[[length(rows) + 1]] <- add_row("overall", seq_len(ncol(observed)))
  for (group_name in unique(metadata$group)) {
    dim_idx <- metadata$dim[metadata$group == group_name]
    rows[[length(rows) + 1]] <- add_row(group_name, dim_idx)
  }

  do.call(rbind, rows)
}

# ---- Sampling and prediction decoding ----

sample_trajectories <- function(
  model,
  surgery_one_hot,
  patient_features,
  x_dim,
  n_samples_per_patient = 50,
  n_steps = 50,
  device = torch::torch_device("cpu"),
  initial_noise = NULL
) {
  n_patients <- nrow(patient_features)
  total <- n_patients * n_samples_per_patient
  tiled_idx <- rep(seq_len(n_patients), each = n_samples_per_patient)

  surgery_tiled <- to_torch_float(surgery_one_hot[tiled_idx, , drop = FALSE], device)
  patient_tiled <- to_torch_float(patient_features[tiled_idx, , drop = FALSE], device)

  if (is.null(initial_noise)) {
    x <- torch::torch_randn(c(total, x_dim), dtype = torch::torch_float(), device = device)
  } else {
    x <- torch::torch_tensor(as.matrix(initial_noise), dtype = torch::torch_float(), device = device)
    expected <- c(total, x_dim)
    if (!identical(as.integer(x$size()), as.integer(expected))) {
      stop("initial_noise must have shape N * samples by x_dim.", call. = FALSE)
    }
  }

  dt <- 1 / n_steps
  model$eval()
  torch::with_no_grad({
    for (i in seq_len(n_steps)) {
      t_value <- (i - 1) * dt
      t <- torch::torch_full(c(total), t_value, dtype = torch::torch_float(), device = device)
      v <- model(x, t, surgery_tiled, patient_tiled)
      x <- x + dt * v
    }
  })

  arr <- as.array(x$cpu())
  samples_first <- array(arr, dim = c(n_samples_per_patient, n_patients, x_dim))
  aperm(samples_first, c(2, 1, 3))
}

unstandardize_samples <- function(samples, preprocessing) {
  out <- samples
  for (dim_idx in seq_along(preprocessing$target_mean)) {
    out[, , dim_idx] <- out[, , dim_idx] * preprocessing$target_std[dim_idx] +
      preprocessing$target_mean[dim_idx]
  }
  out
}

clip01 <- function(x) {
  pmin(pmax(x, 0), 1)
}

summarize_samples <- function(samples_original, metadata) {
  mean_flat <- apply(samples_original, c(1, 3), mean)
  p10_flat <- apply(samples_original, c(1, 3), stats::quantile, probs = 0.10)
  p90_flat <- apply(samples_original, c(1, 3), stats::quantile, probs = 0.90)

  event_dims <- metadata$dim[metadata$group %in% c("retinopathy", "nephropathy", "mace")]
  mace_dims <- metadata$dim[metadata$group == "mace"]
  if (length(event_dims) > 0) {
    mean_flat[, event_dims] <- clip01(mean_flat[, event_dims])
    p10_flat[, event_dims] <- clip01(p10_flat[, event_dims])
    p90_flat[, event_dims] <- clip01(p90_flat[, event_dims])
  }

  list(
    mean_flat = mean_flat,
    p10_flat = p10_flat,
    p90_flat = p90_flat,
    bmi = mean_flat[, metadata$dim[metadata$group == "bmi"], drop = FALSE],
    hba1c = mean_flat[, metadata$dim[metadata$group == "hba1c"], drop = FALSE],
    event_probability = mean_flat[, event_dims, drop = FALSE],
    event_class = 1 * (mean_flat[, event_dims, drop = FALSE] >= 0.5),
    mace_probability = mean_flat[, mace_dims, drop = FALSE],
    mace_class = 1 * (mean_flat[, mace_dims, drop = FALSE] >= 0.5)
  )
}

build_prediction_frame <- function(subject_ids, split_name, summary, observed, mask, metadata) {
  out <- data.frame(
    subject_id = subject_ids,
    split = split_name,
    stringsAsFactors = FALSE
  )

  for (i in seq_len(nrow(metadata))) {
    name <- metadata$name[i]
    out[[paste0("pred_mean_", name)]] <- summary$mean_flat[, i]
    out[[paste0("pred_p10_", name)]] <- summary$p10_flat[, i]
    out[[paste0("pred_p90_", name)]] <- summary$p90_flat[, i]

    observed_value <- observed[, i]
    observed_value[mask[, i] == 0] <- NA
    out[[paste0("observed_", name)]] <- observed_value
    out[[paste0("observed_mask_", name)]] <- mask[, i]
  }

  out
}

# ---- Training ----

parse_cli_args <- function(config) {
  args <- commandArgs(trailingOnly = TRUE)
  if ("--help" %in% args || "-h" %in% args) {
    cat(
      "Usage: Rscript train_flow_matching.R [options]\n",
      "  --csv PATH\n",
      "  --output-dir PATH\n",
      "  --device cpu\n",
      "  --num-steps N\n",
      "  --batch-size N\n",
      "  --seed N\n",
      "  --split-seed N\n",
      "  --n-samples N\n",
      "  --sample-steps N\n",
      sep = ""
    )
    quit(save = "no", status = 0)
  }

  i <- 1
  while (i <= length(args)) {
    key <- args[i]
    if (i == length(args)) {
      stop("Missing value for argument ", key, call. = FALSE)
    }
    value <- args[i + 1]
    if (key == "--csv") config$csv_path <- value
    else if (key == "--output-dir") config$output_dir <- value
    else if (key == "--device") config$device <- value
    else if (key == "--num-steps") config$num_steps <- as.integer(value)
    else if (key == "--batch-size") config$batch_size <- as.integer(value)
    else if (key == "--seed") config$seed <- as.integer(value)
    else if (key == "--split-seed") config$split_seed <- as.integer(value)
    else if (key == "--n-samples") config$n_samples_per_patient <- as.integer(value)
    else if (key == "--sample-steps") config$sample_steps <- as.integer(value)
    else stop("Unknown argument: ", key, call. = FALSE)
    i <- i + 2
  }

  config
}

make_run_dir <- function(output_dir) {
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  stamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
  run_dir <- file.path(output_dir, paste0("run_", stamp))
  dir.create(run_dir, recursive = TRUE, showWarnings = FALSE)
  run_dir
}

select_prediction_split <- function(arrays) {
  if (nrow(arrays$test$x) > 0) return("test")
  if (nrow(arrays$val$x) > 0) return("val")
  "train"
}

train_flow_model <- function(config = default_flow_config(), data_frame = NULL, source_label = NULL) {
  validate_flow_config(config)
  set.seed(config$seed)
  torch::torch_manual_seed(config$seed)

  device <- torch::torch_device(config$device)
  run_dir <- make_run_dir(config$output_dir)
  checkpoint_path <- file.path(run_dir, "model_state.pt")

  if (is.null(data_frame)) {
    message("Loading data from ", config$csv_path)
    dataset <- load_flow_dataset(config$csv_path)
  } else {
    source_label <- if (is.null(source_label)) "loaded data frame" else source_label
    message("Loading data from ", source_label)
    dataset <- load_flow_dataset_from_data_frame(data_frame, source_label = source_label)
  }

  splits <- make_patient_splits(dataset, config)
  assert_no_patient_leakage(dataset, splits)
  preprocessing <- fit_preprocessing(dataset, splits$train_idx)
  arrays <- split_preprocessed_arrays(dataset, splits, preprocessing)

  config$x_dim <- ncol(dataset$x)
  config$patient_feature_dim <- ncol(dataset$patient_features_raw)
  config$target_names <- dataset$target_metadata$name
  config$run_dir <- run_dir

  save_config_json(config, file.path(run_dir, "config.json"))
  saveRDS(preprocessing, file.path(run_dir, "preprocessing.rds"))
  write_dataset_summary(dataset, splits, file.path(run_dir, "dataset_summary.csv"))

  message(
    "Patients: ",
    length(dataset$subject_ids),
    " (train=",
    length(splits$train_idx),
    ", val=",
    length(splits$val_idx),
    ", test=",
    length(splits$test_idx),
    ", x_dim=",
    config$x_dim,
    ")"
  )

  model <- VectorFieldNet(
    x_dim = config$x_dim,
    patient_feature_dim = config$patient_feature_dim,
    num_surgery_types = length(SURGERY_LEVELS),
    time_emb_dim = config$time_emb_dim,
    time_scale = config$time_scale,
    surgery_emb_dim = config$surgery_emb_dim,
    hidden_dim = config$hidden_dim,
    num_hidden_layers = config$num_hidden_layers
  )
  model$to(device = device)

  optimizer <- torch::optim_adamw(
    model$parameters,
    lr = config$learning_rate,
    weight_decay = config$weight_decay
  )

  train_log <- data.frame(
    step = integer(0),
    train_loss = numeric(0),
    val_loss = numeric(0),
    stringsAsFactors = FALSE
  )

  best_val <- Inf
  best_step <- NA_integer_
  evals_since_improve <- 0
  early_stopped <- FALSE
  effective_batch_size <- min(config$batch_size, max(1, nrow(arrays$train$x)))

  message(
    "Training for up to ",
    config$num_steps,
    " steps with batch_size=",
    effective_batch_size
  )

  for (step in seq_len(config$num_steps)) {
    model$train()
    batch <- sample_training_batch(arrays$train, effective_batch_size)
    x1 <- to_torch_float(batch$x, device)
    mask <- to_torch_float(batch$mask, device)
    surgery_one_hot <- to_torch_float(batch$surgery_one_hot, device)
    patient_features <- to_torch_float(batch$patient_features, device)

    path <- sample_conditional_path(x1)
    optimizer$zero_grad()
    loss <- flow_matching_loss(
      model,
      path$x_t,
      path$t,
      surgery_one_hot,
      patient_features,
      path$u_t,
      mask
    )
    loss$backward()
    optimizer$step()

    train_loss <- loss$item()
    should_eval <- step == 1 || step %% config$val_every == 0 || step == config$num_steps

    if (should_eval) {
      val_loss <- evaluate_flow_loss(model, arrays$val, device, config$val_repeats)
      score <- if (is.na(val_loss)) train_loss else val_loss
      improved <- score < best_val - config$early_stop_min_delta

      if (improved) {
        best_val <- score
        best_step <- step
        evals_since_improve <- 0
        torch::torch_save(model$state_dict(), checkpoint_path)
      } else {
        evals_since_improve <- evals_since_improve + 1
      }

      train_log <- rbind(
        train_log,
        data.frame(
          step = step,
          train_loss = train_loss,
          val_loss = val_loss,
          stringsAsFactors = FALSE
        )
      )
      write.csv(train_log, file.path(run_dir, "training_log.csv"), row.names = FALSE)

      message(
        "Step ",
        step,
        "/",
        config$num_steps,
        " train=",
        sprintf("%.4f", train_loss),
        " val=",
        ifelse(is.na(val_loss), "NA", sprintf("%.4f", val_loss)),
        ifelse(improved, " *", ""),
        " best=",
        sprintf("%.4f", best_val),
        "@",
        best_step
      )

      if (!is.na(val_loss) && evals_since_improve >= config$early_stop_patience) {
        early_stopped <- TRUE
        message("Early stopping at step ", step)
        break
      }
    } else if (step %% config$log_every == 0) {
      message("Step ", step, "/", config$num_steps, " train=", sprintf("%.4f", train_loss))
    }
  }

  if (!file.exists(checkpoint_path)) {
    torch::torch_save(model$state_dict(), checkpoint_path)
    best_step <- config$num_steps
    best_val <- NA_real_
  }

  model$load_state_dict(torch::torch_load(checkpoint_path))

  prediction_split <- select_prediction_split(arrays)
  prediction_arrays <- arrays[[prediction_split]]
  message("Sampling predictions for ", prediction_split, " split")

  samples_std <- sample_trajectories(
    model = model,
    surgery_one_hot = prediction_arrays$surgery_one_hot,
    patient_features = prediction_arrays$patient_features,
    x_dim = config$x_dim,
    n_samples_per_patient = config$n_samples_per_patient,
    n_steps = config$sample_steps,
    device = device
  )
  samples_original <- unstandardize_samples(samples_std, preprocessing)
  prediction_summary <- summarize_samples(samples_original, dataset$target_metadata)

  predictions <- build_prediction_frame(
    subject_ids = prediction_arrays$subject_ids,
    split_name = prediction_split,
    summary = prediction_summary,
    observed = prediction_arrays$original_x,
    mask = prediction_arrays$original_mask,
    metadata = dataset$target_metadata
  )
  write.csv(predictions, file.path(run_dir, paste0("predictions_", prediction_split, ".csv")), row.names = FALSE)

  metrics <- compute_observed_metrics(
    prediction_summary$mean_flat,
    prediction_arrays$original_x,
    prediction_arrays$original_mask,
    dataset$target_metadata
  )
  metrics$split <- prediction_split
  metrics$best_step <- best_step
  metrics$early_stopped <- early_stopped
  write.csv(metrics, file.path(run_dir, "metrics.csv"), row.names = FALSE)
  if (requireNamespace("jsonlite", quietly = TRUE)) {
    jsonlite::write_json(metrics, file.path(run_dir, "metrics.json"), pretty = TRUE)
  }

  message("Saved run artifacts to ", run_dir)
  invisible(list(
    run_dir = run_dir,
    model = model,
    dataset = dataset,
    splits = splits,
    preprocessing = preprocessing,
    metrics = metrics
  ))
}

# ---- Auto-run entrypoint ----

find_loaded_flow_data_frame <- function(env = .GlobalEnv) {
  candidate_names <- c("mbs", "mbscoh", "mbs_cohort", "cosmos_mbs", "merged")
  for (candidate_name in candidate_names) {
    if (exists(candidate_name, envir = env, inherits = TRUE)) {
      candidate <- get(candidate_name, envir = env, inherits = TRUE)
      if (is.data.frame(candidate)) {
        return(list(name = candidate_name, data = candidate))
      }
    }
  }
  NULL
}

apply_config_overrides <- function(config, env = .GlobalEnv) {
  if (exists("flow_config", envir = env, inherits = TRUE)) {
    supplied <- get("flow_config", envir = env, inherits = TRUE)
    if (is.list(supplied)) {
      config <- utils::modifyList(config, supplied)
    }
  }
  if (exists("flow_config_overrides", envir = env, inherits = TRUE)) {
    overrides <- get("flow_config_overrides", envir = env, inherits = TRUE)
    if (is.list(overrides)) {
      config <- utils::modifyList(config, overrides)
    }
  }
  config
}

is_direct_script <- any(grepl("train_flow_matching[.]R$", commandArgs(trailingOnly = FALSE)))

if (is_direct_script) {
  flow_config <- parse_cli_args(default_flow_config())
  flow_training_result <- train_flow_model(flow_config)
} else {
  auto_run_enabled <- !(
    exists("flow_auto_run", envir = .GlobalEnv, inherits = TRUE) &&
      identical(get("flow_auto_run", envir = .GlobalEnv, inherits = TRUE), FALSE)
  )
  loaded_data <- find_loaded_flow_data_frame(.GlobalEnv)
  if (!auto_run_enabled) {
    message("train_flow_matching.R loaded. flow_auto_run is FALSE, so training did not auto-run.")
  } else if (!is.null(loaded_data)) {
    flow_config <- apply_config_overrides(default_flow_config(), .GlobalEnv)
    flow_training_result <- train_flow_model(
      flow_config,
      data_frame = loaded_data$data,
      source_label = paste0("loaded data frame `", loaded_data$name, "`")
    )
  } else {
    message(
      "train_flow_matching.R loaded. No data frame named mbs, mbscoh, ",
      "mbs_cohort, cosmos_mbs, or merged was found, so training did not auto-run."
    )
  }
}
