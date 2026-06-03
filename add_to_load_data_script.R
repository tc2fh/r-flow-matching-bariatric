# add this to the end of the load data script to train a model from the loaded dataframe (change 'mbs' variable)

source("train_flow_matching.R")

cfg <- default_flow_config(num_steps = 1000, batch_size = 32)
train_flow_model(cfg, data_frame = mbs, source_label = "Cosmos MBSCohort")