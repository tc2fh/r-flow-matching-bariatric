# Train the R torch concat-conditioned flow-matching model.
#
# Example:
#   Rscript train_flow_matching.R --num-steps 25 --batch-size 16

script_args <- commandArgs(trailingOnly = FALSE)
file_arg <- "--file="
script_path <- sub(file_arg, "", script_args[startsWith(script_args, file_arg)])
PROJECT_ROOT <- if (length(script_path) > 0) dirname(normalizePath(script_path[1])) else getwd()

source(file.path(PROJECT_ROOT, "R", "config.R"))
source(file.path(PROJECT_ROOT, "R", "data.R"))

if (!requireNamespace("torch", quietly = TRUE)) {
  stop(
    "Package 'torch' is required for training. Install it with ",
    "install.packages('torch') and then run torch::install_torch() if needed.",
    call. = FALSE
  )
}

source(file.path(PROJECT_ROOT, "R", "model.R"))
source(file.path(PROJECT_ROOT, "R", "flow_matching.R"))
source(file.path(PROJECT_ROOT, "R", "sample.R"))

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

train_flow_model <- function(config = default_flow_config()) {
  validate_flow_config(config)
  set.seed(config$seed)
  torch::torch_manual_seed(config$seed)

  device <- torch::torch_device(config$device)
  run_dir <- make_run_dir(config$output_dir)
  checkpoint_path <- file.path(run_dir, "model_state.pt")

  message("Loading data from ", config$csv_path)
  dataset <- load_flow_dataset(config$csv_path)
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

config <- parse_cli_args(default_flow_config())
train_flow_model(config)
