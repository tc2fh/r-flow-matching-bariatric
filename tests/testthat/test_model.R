testthat::skip_if_not_installed("torch")

project_root <- if (file.exists(file.path("R", "config.R"))) "." else file.path("..", "..")

source(file.path(project_root, "R", "config.R"))
source(file.path(project_root, "R", "model.R"))
source(file.path(project_root, "R", "flow_matching.R"))
source(file.path(project_root, "R", "sample.R"))

testthat::test_that("VectorFieldNet forward pass returns batch by target dimension", {
  device <- torch::torch_device("cpu")
  model <- VectorFieldNet(
    x_dim = 18,
    patient_feature_dim = 6,
    num_surgery_types = 2,
    time_emb_dim = 16,
    surgery_emb_dim = 4,
    hidden_dim = 8,
    num_hidden_layers = 2
  )
  model$to(device = device)

  x_t <- torch::torch_randn(c(5, 18), device = device)
  t <- torch::torch_rand(c(5), device = device)
  surgery_one_hot <- torch::torch_tensor(
    matrix(c(1, 0, 0, 1, 1, 0, 0, 1, 1, 0), nrow = 5, byrow = TRUE),
    dtype = torch::torch_float(),
    device = device
  )
  patient_features <- torch::torch_randn(c(5, 6), device = device)

  out <- model(x_t, t, surgery_one_hot, patient_features)
  testthat::expect_equal(as.integer(out$size()), c(5, 18))
})

testthat::test_that("one training step produces finite loss", {
  device <- torch::torch_device("cpu")
  model <- VectorFieldNet(
    x_dim = 18,
    patient_feature_dim = 6,
    num_surgery_types = 2,
    time_emb_dim = 16,
    surgery_emb_dim = 4,
    hidden_dim = 8,
    num_hidden_layers = 2
  )
  optimizer <- torch::optim_adamw(model$parameters, lr = 1e-3)

  x1 <- torch::torch_randn(c(4, 18), device = device)
  mask <- torch::torch_ones(c(4, 18), device = device)
  surgery_one_hot <- torch::torch_tensor(
    matrix(c(1, 0, 0, 1, 1, 0, 0, 1), nrow = 4, byrow = TRUE),
    dtype = torch::torch_float(),
    device = device
  )
  patient_features <- torch::torch_randn(c(4, 6), device = device)
  path <- sample_conditional_path(x1)
  first_param_before <- model$parameters[[1]]$detach()$clone()

  optimizer$zero_grad()
  loss <- flow_matching_loss(model, path$x_t, path$t, surgery_one_hot, patient_features, path$u_t, mask)
  loss$backward()
  optimizer$step()
  first_param_after <- model$parameters[[1]]$detach()$clone()

  testthat::expect_true(is.finite(loss$item()))
  testthat::expect_false(torch::torch_equal(first_param_before, first_param_after))
})

testthat::test_that("sampler returns patient by sample by target arrays", {
  device <- torch::torch_device("cpu")
  model <- VectorFieldNet(
    x_dim = 18,
    patient_feature_dim = 6,
    num_surgery_types = 2,
    time_emb_dim = 16,
    surgery_emb_dim = 4,
    hidden_dim = 8,
    num_hidden_layers = 2
  )

  surgery_one_hot <- matrix(c(1, 0, 0, 1), nrow = 2, byrow = TRUE)
  patient_features <- matrix(rnorm(12), nrow = 2)

  samples <- sample_trajectories(
    model,
    surgery_one_hot,
    patient_features,
    x_dim = 18,
    n_samples_per_patient = 3,
    n_steps = 2,
    device = device
  )

  testthat::expect_equal(dim(samples), c(2, 3, 18))
})
