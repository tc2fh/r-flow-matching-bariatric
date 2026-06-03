# R Flow-Matching Model For Bariatric Outcome Vectors

This folder contains a pure-R implementation of a concat-conditioned flow-matching
model for wide MBSCohort-style patient rows. It defaults to the synthetic file
`fake_data/fake_mbs_cohort.csv`.

The model predicts an 18-dimensional flattened target vector:

- BMI at 3m, 6m, 9m, 12m, 2y, 3y, 4y, 5y, 6y
- HbA1c at 12m, 2y, 3y, 4y, 5y, 6y
- cumulative MACE any-event indicators by 1y, 3y, 5y

## Dependencies

Required for training:

```r
install.packages("torch")
torch::install_torch()
```

Useful supporting packages:

```r
install.packages(c("jsonlite", "testthat"))
```

The scripts do not install packages automatically.

## Train

Smoke run:

```sh
Rscript train_flow_matching.R --num-steps 25 --batch-size 16 --n-samples 10
```

Full default run:

```sh
Rscript train_flow_matching.R
```

Useful options:

```sh
Rscript train_flow_matching.R \
  --csv fake_data/fake_mbs_cohort.csv \
  --output-dir runs/r_flow_matching \
  --num-steps 1000 \
  --batch-size 32 \
  --seed 0 \
  --split-seed 0
```

Each run writes a timestamped folder under `runs/r_flow_matching/` containing:

- `config.json`
- `preprocessing.rds`
- `model_state.pt`
- `dataset_summary.csv`
- `training_log.csv`
- `metrics.csv` and `metrics.json`
- `predictions_<split>.csv`

## Data Contract

Input is one row per surgery patient with MBSCohort-style columns. Recognized
surgery mappings are:

- `43775`: sleeve gastrectomy
- `43644`, `43846`: Roux-en-y gastric bypass

Conditioning fields are:

- `AgeAtEvent`
- `Sex`, encoded as `sex_male`
- surgery type, embedded separately
- `CreatinineAtEvent`
- `HbA1cAtEvent`
- `BMIatEvent`
- `InsulinStatus`

MACE targets are derived from `MACE`, `MACEinterval`, and `ActiveEndInterval`.
If follow-up has not reached a horizon and no event was observed before that
horizon, that target is masked during training.

## Cosmos Data

If you already loaded the collaborator's database query into an R data frame
named `mbs`, you can train directly from that object in an interactive R session:

```r
source("train_flow_matching.R")

cfg <- default_flow_config(num_steps = 1000, batch_size = 32)
train_flow_model(cfg, data_frame = mbs, source_label = "Cosmos MBSCohort")
```

To export a CSV from the loaded `mbs` object and then train from the command
line:

```r
source("R/config.R")
source("R/data.R")
write_flow_input_csv(mbs, "data/cosmos_mbs_flow_input.csv")
```

```sh
Rscript train_flow_matching.R --csv data/cosmos_mbs_flow_input.csv
```

The helper script `load_cosmos_flow_data.R` also contains a corrected SQL Server
loader for MBSCohort:

```sh
Rscript load_cosmos_flow_data.R --output data/cosmos_mbs_flow_input.csv
```

The `data/` directory is git-ignored because real Cosmos exports may contain
sensitive records. Do not commit database exports.

## Tests

After installing `testthat`, run:

```sh
Rscript tests/testthat.R
```
