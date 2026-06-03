# Torch vector-field network for concat-conditioned flow matching.

require_torch <- function() {
  if (!requireNamespace("torch", quietly = TRUE)) {
    stop(
      "Package 'torch' is required. Install it with install.packages('torch') ",
      "and then run torch::install_torch() if your R torch setup requires it.",
      call. = FALSE
    )
  }
  invisible(TRUE)
}

require_torch()

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

    self$surgery_embed <- torch::nn_linear(
      num_surgery_types,
      surgery_emb_dim,
      bias = FALSE
    )

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
