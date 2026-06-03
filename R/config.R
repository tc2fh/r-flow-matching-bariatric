# Configuration and schema constants for the R flow-matching model.

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
  csv_path = file.path("fake_data", "fake_mbs_cohort.csv"),
  output_dir = file.path("runs", "r_flow_matching"),
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
    stop("Package 'jsonlite' is required to save config JSON.", call. = FALSE)
  }
  jsonlite::write_json(config, path, pretty = TRUE, auto_unbox = TRUE)
}
