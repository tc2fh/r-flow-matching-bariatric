# Flow-matching path sampling, loss, and evaluation helpers.

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
