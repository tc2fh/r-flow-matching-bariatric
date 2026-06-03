# Sampling and prediction decoding for trained flow-matching models.

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

  mace_dims <- metadata$dim[metadata$group == "mace"]
  if (length(mace_dims) > 0) {
    mean_flat[, mace_dims] <- clip01(mean_flat[, mace_dims])
    p10_flat[, mace_dims] <- clip01(p10_flat[, mace_dims])
    p90_flat[, mace_dims] <- clip01(p90_flat[, mace_dims])
  }

  list(
    mean_flat = mean_flat,
    p10_flat = p10_flat,
    p90_flat = p90_flat,
    bmi = mean_flat[, metadata$dim[metadata$group == "bmi"], drop = FALSE],
    hba1c = mean_flat[, metadata$dim[metadata$group == "hba1c"], drop = FALSE],
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
