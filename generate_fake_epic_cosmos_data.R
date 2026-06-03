# Synthetic EPIC Cosmos-like cohort data generator.
#
# This creates fake data for ML prototyping only. It does not use, copy, or
# de-identify real patient data, and it should not be used for clinical inference.

set.seed(20260603)

n_entries <- 100
output_dir <- "fake_data"

dir.create(output_dir, showWarnings = FALSE)

clip <- function(x, lower, upper) {
  pmin(pmax(x, lower), upper)
}

sample_values <- function(values, n, probs = NULL) {
  sample(values, size = n, replace = TRUE, prob = probs)
}

random_dates <- function(n, start_date, end_date) {
  start_date <- as.Date(start_date)
  end_date <- as.Date(end_date)
  start_date + sample.int(as.integer(end_date - start_date) + 1, n, replace = TRUE) - 1
}

add_missing <- function(x, probability) {
  x[runif(length(x)) < probability] <- NA
  x
}

binary_from_probability <- function(probability) {
  as.integer(runif(length(probability)) < probability)
}

make_event_interval <- function(event_flag, active_end_interval) {
  interval <- rep(NA_integer_, length(event_flag))
  event_rows <- which(event_flag == 1L)

  for (row in event_rows) {
    max_interval <- max(30L, active_end_interval[row])
    interval[row] <- sample(30:max_interval, 1)
  }

  interval
}

make_mace_date <- function(anchor_date, mace_interval) {
  mace_date <- as.Date(rep(NA_character_, length(anchor_date)))
  event_rows <- which(!is.na(mace_interval))
  mace_date[event_rows] <- anchor_date[event_rows] + mace_interval[event_rows]
  mace_date
}

make_follow_up_value <- function(baseline, months_after_event, annual_change,
                                 noise_sd, missing_probability,
                                 lower = -Inf, upper = Inf) {
  years_after_event <- months_after_event / 12
  value <- baseline + annual_change * years_after_event +
    rnorm(length(baseline), mean = 0, sd = noise_sd)
  add_missing(round(clip(value, lower, upper), 4), missing_probability)
}

make_glp1_fields <- function(has_glp1) {
  n <- length(has_glp1)
  glp1_names <- c(
    "semaglutide",
    "tirzepatide",
    "dulaglutide",
    "liraglutide",
    "exenatide microspheres"
  )
  dose_options <- list(
    "semaglutide" = c(0.25, 0.5, 1.0, 2.0),
    "tirzepatide" = c(2.5, 5.0, 7.5, 10.0, 12.5, 15.0),
    "dulaglutide" = c(0.75, 1.5, 3.0, 4.5),
    "liraglutide" = c(0.6, 1.2, 1.8, 3.0),
    "exenatide microspheres" = c(2.0)
  )

  glp1_name <- rep(NA_character_, n)
  glp1_route <- rep(NA_character_, n)
  max_glp1_dose <- rep(NA_real_, n)
  most_recent_dose <- rep(NA_real_, n)
  most_recent_dose_unit <- rep(NA_character_, n)
  glp1_start_date <- as.Date(rep(NA_character_, n))
  glp1_end_date <- as.Date(rep(NA_character_, n))
  glp1_duration <- rep(NA_integer_, n)

  glp1_rows <- which(has_glp1)

  for (row in glp1_rows) {
    drug <- sample_values(glp1_names, 1, probs = c(0.35, 0.25, 0.2, 0.15, 0.05))
    possible_doses <- dose_options[[drug]]
    max_dose <- sample_values(possible_doses, 1)
    most_recent <- sample_values(possible_doses[possible_doses <= max_dose], 1)

    glp1_name[row] <- drug
    glp1_route[row] <- "subcutaneous"
    max_glp1_dose[row] <- max_dose
    most_recent_dose[row] <- most_recent
    most_recent_dose_unit[row] <- "mg"
    glp1_start_date[row] <- random_dates(1, "2018-01-01", "2023-04-30")
    glp1_duration[row] <- sample(180:1800, 1)
    glp1_end_date[row] <- glp1_start_date[row] + glp1_duration[row]
  }

  data.frame(
    GLP1StartDate = glp1_start_date,
    GLP1EndDate = glp1_end_date,
    GLP1Duration = glp1_duration,
    GLP1Name = glp1_name,
    GLP1Route = glp1_route,
    MaxGLP1Dose = round(max_glp1_dose, 4),
    MostRecentDose = round(most_recent_dose, 4),
    MostRecentDoseUnit = most_recent_dose_unit,
    stringsAsFactors = FALSE
  )
}

make_fake_cosmos_cohort <- function(n = 100) {
  cohort <- sample_values(c("GLP1", "MBS"), n, probs = c(0.5, 0.5))
  is_glp1 <- cohort == "GLP1"
  is_mbs <- cohort == "MBS"

  pat_key <- 900000000L + seq_len(n)
  age <- sample(35:82, n, replace = TRUE)
  sex <- sample_values(c("Female", "Male"), n, probs = c(0.58, 0.42))
  coverage_class <- sample_values(
    c("Commercial", "Medicare", "Medicaid", "Self-pay", "Other"),
    n,
    probs = c(0.45, 0.28, 0.18, 0.04, 0.05)
  )

  first_race <- sample_values(
    c(
      "White",
      "Black or African American",
      "Asian",
      "American Indian or Alaska Native",
      "Native Hawaiian or Other Pacific Islander",
      "Other"
    ),
    n,
    probs = c(0.55, 0.22, 0.09, 0.03, 0.02, 0.09)
  )
  second_race <- ifelse(
    runif(n) < 0.1,
    sample_values(c("White", "Black or African American", "Asian", "Other"), n),
    NA_character_
  )
  multi_racial <- as.integer(!is.na(second_race))

  preferred_language <- sample_values(
    c("English", "Spanish", "Chinese", "Arabic", "Other"),
    n,
    probs = c(0.82, 0.11, 0.03, 0.02, 0.02)
  )
  state_or_province <- sample_values(
    c("PA", "NJ", "DE", "NY", "MD", "VA", "OH"),
    n,
    probs = c(0.42, 0.22, 0.08, 0.1, 0.08, 0.06, 0.04)
  )
  ruca <- sample_values(
    c("1 metropolitan", "4 micropolitan", "7 small town", "10 rural"),
    n,
    probs = c(0.72, 0.14, 0.09, 0.05)
  )

  svi_overall <- round(runif(n, 0.02, 0.98), 4)
  svi_household <- round(clip(svi_overall + rnorm(n, 0, 0.15), 0, 1), 4)
  svi_transportation <- round(clip(svi_overall + rnorm(n, 0, 0.18), 0, 1), 4)
  svi_minority <- round(clip(svi_overall + rnorm(n, 0, 0.2), 0, 1), 4)
  svi_ses <- round(clip(svi_overall + rnorm(n, 0, 0.15), 0, 1), 4)

  pmh_dm2 <- rep(1L, n)
  pmh_hypertension <- binary_from_probability(plogis(-1.1 + 0.04 * (age - 50)))
  pmh_osa <- binary_from_probability(plogis(-1.3 + 0.025 * (age - 50)))
  pmh_dyslipidemia <- binary_from_probability(plogis(-0.9 + 0.035 * (age - 50)))
  pmh_retinopathy <- rep(0L, n)
  pmh_vte <- binary_from_probability(rep(0.07, n))
  pmh_afib <- binary_from_probability(plogis(-2.8 + 0.05 * (age - 55)))
  pmh_mi <- binary_from_probability(plogis(-2.7 + 0.035 * (age - 55)))
  pmh_stroke <- binary_from_probability(plogis(-3.1 + 0.035 * (age - 55)))

  insulin_status <- binary_from_probability(rep(0.22, n))
  biguanide_status <- binary_from_probability(rep(0.62, n))
  sglt2_status <- binary_from_probability(rep(0.18, n))
  pmh_dialysis_transplant <- rep(0L, n)
  pmh_prior_mbs <- rep(0L, n)
  mbs_during_glp1 <- ifelse(is_glp1, 0L, NA_integer_)

  bmi_at_event <- round(clip(rnorm(n, mean = 44, sd = 6.5), 35, 74.5), 4)
  height_inches <- clip(rnorm(n, mean = 66.5, sd = 4.2), 57, 78)
  weight_at_event <- round(bmi_at_event * height_inches^2 / 703, 4)

  weight_loss_rate <- ifelse(is_mbs, runif(n, 22, 42), runif(n, 5, 18))
  bmi_loss_rate <- weight_loss_rate / (height_inches^2 / 703)

  bmi_3m <- make_follow_up_value(bmi_at_event, 3, -bmi_loss_rate, 1.1, 0.08, 22, 75)
  bmi_6m <- make_follow_up_value(bmi_at_event, 6, -bmi_loss_rate, 1.3, 0.10, 22, 75)
  bmi_9m <- make_follow_up_value(bmi_at_event, 9, -bmi_loss_rate, 1.5, 0.14, 22, 75)
  bmi_12m <- make_follow_up_value(bmi_at_event, 12, -bmi_loss_rate, 1.8, 0.18, 22, 75)
  bmi_2y <- make_follow_up_value(bmi_at_event, 24, -0.7 * bmi_loss_rate, 2.2, 0.28, 22, 75)
  bmi_3y <- make_follow_up_value(bmi_at_event, 36, -0.45 * bmi_loss_rate, 2.7, 0.38, 22, 75)
  bmi_4y <- make_follow_up_value(bmi_at_event, 48, -0.32 * bmi_loss_rate, 3.0, 0.48, 22, 75)
  bmi_5y <- make_follow_up_value(bmi_at_event, 60, -0.24 * bmi_loss_rate, 3.4, 0.58, 22, 75)
  bmi_6y <- make_follow_up_value(bmi_at_event, 72, -0.18 * bmi_loss_rate, 3.8, 0.68, 22, 75)

  weight_3m <- round(bmi_3m * height_inches^2 / 703, 4)
  weight_6m <- round(bmi_6m * height_inches^2 / 703, 4)
  weight_9m <- round(bmi_9m * height_inches^2 / 703, 4)
  weight_12m <- round(bmi_12m * height_inches^2 / 703, 4)
  weight_2y <- round(bmi_2y * height_inches^2 / 703, 4)
  weight_3y <- round(bmi_3y * height_inches^2 / 703, 4)
  weight_4y <- round(bmi_4y * height_inches^2 / 703, 4)
  weight_5y <- round(bmi_5y * height_inches^2 / 703, 4)
  weight_6y <- round(bmi_6y * height_inches^2 / 703, 4)

  hba1c_at_event <- round(clip(rnorm(n, mean = 8.1, sd = 1.2), 5.8, 13.5), 4)
  hba1c_change <- ifelse(is_mbs, runif(n, 0.55, 1.4), runif(n, 0.35, 1.2))
  hba1c_12m <- make_follow_up_value(hba1c_at_event, 12, -hba1c_change, 0.45, 0.16, 4.5, 14)
  hba1c_2y <- make_follow_up_value(hba1c_at_event, 24, -0.6 * hba1c_change, 0.6, 0.3, 4.5, 14)
  hba1c_3y <- make_follow_up_value(hba1c_at_event, 36, -0.42 * hba1c_change, 0.75, 0.42, 4.5, 14)
  hba1c_4y <- make_follow_up_value(hba1c_at_event, 48, -0.33 * hba1c_change, 0.85, 0.52, 4.5, 14)
  hba1c_5y <- make_follow_up_value(hba1c_at_event, 60, -0.25 * hba1c_change, 0.95, 0.62, 4.5, 14)
  hba1c_6y <- make_follow_up_value(hba1c_at_event, 72, -0.18 * hba1c_change, 1.05, 0.72, 4.5, 14)

  creatinine_at_event <- round(clip(rlnorm(n, meanlog = log(0.95), sdlog = 0.25), 0.45, 2.6), 4)
  egfr_at_event <- round(clip(115 - 0.75 * age - 13 * (creatinine_at_event - 1) +
    rnorm(n, 0, 10), 20, 125), 4)

  active_end_interval <- sample(700:2400, n, replace = TRUE)
  outcome_risk <- -2.15 + 0.025 * (age - 55) + 0.45 * pmh_hypertension +
    0.35 * insulin_status + 0.12 * (hba1c_at_event - 7) -
    0.012 * (egfr_at_event - 75)
  nephropathy <- binary_from_probability(plogis(outcome_risk))
  nephropathy_interval <- make_event_interval(nephropathy, active_end_interval)

  kidney_failure <- binary_from_probability(plogis(outcome_risk - 2.0))
  kidney_failure_interval <- make_event_interval(kidney_failure, active_end_interval)

  dialysis <- binary_from_probability(ifelse(kidney_failure == 1L, 0.25, 0.015))
  dialysis_interval <- make_event_interval(dialysis, active_end_interval)

  transplant <- binary_from_probability(ifelse(dialysis == 1L, 0.08, 0.005))
  transplant_interval <- make_event_interval(transplant, active_end_interval)

  retinopathy <- binary_from_probability(plogis(-2.4 + 0.16 * (hba1c_at_event - 7) +
    0.3 * insulin_status))
  retinopathy_interval <- make_event_interval(retinopathy, active_end_interval)

  mace <- binary_from_probability(plogis(-2.8 + 0.035 * (age - 55) +
    0.65 * pmh_mi + 0.5 * pmh_stroke + 0.45 * pmh_afib +
    0.3 * pmh_hypertension))
  mace_interval <- make_event_interval(mace, active_end_interval)
  mace_type <- rep(NA_character_, n)
  mace_type[mace == 1L] <- sample_values(
    c("myocardial infarction", "stroke", "heart failure hospitalization"),
    sum(mace == 1L),
    probs = c(0.38, 0.28, 0.34)
  )

  deceased <- binary_from_probability(plogis(-4.0 + 0.055 * (age - 65) + 0.75 * mace))
  death_interval <- make_event_interval(deceased, active_end_interval)
  cvdeath <- as.integer(deceased == 1L & mace == 1L & runif(n) < 0.35)

  procedure_date <- as.Date(rep(NA_character_, n))
  procedure_date[is_mbs] <- random_dates(sum(is_mbs), "2018-01-01", "2023-04-30")
  procedure_date_bigint <- rep(NA_integer_, n)
  procedure_date_bigint[is_mbs] <- as.integer(format(procedure_date[is_mbs], "%Y%m%d"))
  cpt_code <- rep(NA_character_, n)
  cpt_code[is_mbs] <- sample_values(
    c("43775", "43644", "43846", "43645"),
    sum(is_mbs),
    probs = c(0.55, 0.34, 0.07, 0.04)
  )

  postop_glp1 <- rep(NA_integer_, n)
  postop_glp1[is_mbs] <- binary_from_probability(rep(0.28, sum(is_mbs)))
  glp1_interval <- rep(NA_integer_, n)
  postop_glp1_rows <- which(postop_glp1 == 1L)
  glp1_interval[postop_glp1_rows] <- sample(30:720, length(postop_glp1_rows), replace = TRUE)

  prior_glp1 <- ifelse(is_glp1, binary_from_probability(rep(0.12, n)), 0L)
  has_glp1 <- is_glp1 | postop_glp1 == 1L
  glp1_fields <- make_glp1_fields(has_glp1)
  glp1_fields$GLP1StartDate[postop_glp1_rows] <-
    procedure_date[postop_glp1_rows] + glp1_interval[postop_glp1_rows]
  glp1_fields$GLP1EndDate[postop_glp1_rows] <-
    glp1_fields$GLP1StartDate[postop_glp1_rows] + glp1_fields$GLP1Duration[postop_glp1_rows]

  anchor_date <- ifelse(is_mbs, procedure_date, glp1_fields$GLP1StartDate)
  anchor_date <- as.Date(anchor_date, origin = "1970-01-01")
  mace_date <- make_mace_date(anchor_date, mace_interval)

  data.frame(
    Cohort = cohort,
    PatKey = pat_key,
    CptCode = cpt_code,
    ProcedureDate = procedure_date_bigint,
    ProcDateValue = procedure_date,
    AgeAtEvent = age,
    Sex = sex,
    CoverageClass = coverage_class,
    PriorGLP1 = prior_glp1,
    PostOpGLP1 = postop_glp1,
    GLP1Interval = glp1_interval,
    GLP1StartDate = glp1_fields$GLP1StartDate,
    GLP1EndDate = glp1_fields$GLP1EndDate,
    GLP1Duration = glp1_fields$GLP1Duration,
    GLP1Name = glp1_fields$GLP1Name,
    GLP1Route = glp1_fields$GLP1Route,
    MaxGLP1Dose = glp1_fields$MaxGLP1Dose,
    MostRecentDose = glp1_fields$MostRecentDose,
    MostRecentDoseUnit = glp1_fields$MostRecentDoseUnit,
    FirstRace = first_race,
    SecondRace = second_race,
    MultiRacial = multi_racial,
    PreferredLanguage = preferred_language,
    StateOrProvince = state_or_province,
    RUCA = ruca,
    SviOverall = svi_overall,
    SviHousehold = svi_household,
    SviTransportation = svi_transportation,
    SviMinority = svi_minority,
    SviSES = svi_ses,
    PMH_DM2 = pmh_dm2,
    PMH_hypertension = pmh_hypertension,
    PMH_OSA = pmh_osa,
    PMH_dyslipidemia = pmh_dyslipidemia,
    PMH_retinopathy = pmh_retinopathy,
    PMH_VTE = pmh_vte,
    PMH_AFib = pmh_afib,
    PMH_MI = pmh_mi,
    PMH_stroke = pmh_stroke,
    InsulinStatus = insulin_status,
    BiguanideStatus = biguanide_status,
    SGLT2Status = sglt2_status,
    PMH_dialysis_transplant = pmh_dialysis_transplant,
    PMH_PriorMBS = pmh_prior_mbs,
    MBSduringGLP1 = mbs_during_glp1,
    BMIatEvent = bmi_at_event,
    BMI3mPostEvent = bmi_3m,
    BMI6mPostEvent = bmi_6m,
    BMI9mPostEvent = bmi_9m,
    BMI12mPostEvent = bmi_12m,
    BMI2yPostEvent = bmi_2y,
    BMI3yPostEvent = bmi_3y,
    BMI4yPostEvent = bmi_4y,
    BMI5yPostEvent = bmi_5y,
    BMI6yPostEvent = bmi_6y,
    WeightAtEvent = weight_at_event,
    Weight3mPostEvent = weight_3m,
    Weight6mPostEvent = weight_6m,
    Weight9mPostEvent = weight_9m,
    Weight12mPostEvent = weight_12m,
    Weight2yPostEvent = weight_2y,
    Weight3yPostEvent = weight_3y,
    Weight4yPostEvent = weight_4y,
    Weight5yPostEvent = weight_5y,
    Weight6yPostEvent = weight_6y,
    HbA1cAtEvent = hba1c_at_event,
    HbA1c12mPostEvent = hba1c_12m,
    HbA1c2yPostEvent = hba1c_2y,
    HbA1c3yPostEvent = hba1c_3y,
    HbA1c4yPostEvent = hba1c_4y,
    HbA1c5yPostEvent = hba1c_5y,
    HbA1c6yPostEvent = hba1c_6y,
    CreatinineAtEvent = creatinine_at_event,
    eGFRatEvent = egfr_at_event,
    Nephropathy = nephropathy,
    NephropathyInterval = nephropathy_interval,
    KidneyFailureInterval = kidney_failure_interval,
    DialysisInterval = dialysis_interval,
    TransplantInterval = transplant_interval,
    Retinopathy = retinopathy,
    RetinopathyInterval = retinopathy_interval,
    Deceased = deceased,
    DeathInterval = death_interval,
    CVdeath = cvdeath,
    MACE = mace,
    MACEtype = mace_type,
    MACEdate = mace_date,
    MACEinterval = mace_interval,
    ActiveEndInterval = active_end_interval,
    stringsAsFactors = FALSE
  )
}

fake_modeling_cohort <- make_fake_cosmos_cohort(n_entries)

glp1_columns <- c(
  "PatKey", "AgeAtEvent", "Sex", "CoverageClass", "PriorGLP1",
  "GLP1StartDate", "GLP1EndDate", "GLP1Duration", "GLP1Name", "GLP1Route",
  "MaxGLP1Dose", "MostRecentDose", "MostRecentDoseUnit", "FirstRace",
  "SecondRace", "MultiRacial", "PreferredLanguage", "StateOrProvince", "RUCA",
  "SviOverall", "SviHousehold", "SviTransportation", "SviMinority", "SviSES",
  "PMH_DM2", "PMH_hypertension", "PMH_OSA", "PMH_dyslipidemia",
  "PMH_retinopathy", "PMH_VTE", "PMH_AFib", "PMH_MI", "PMH_stroke",
  "InsulinStatus", "BiguanideStatus", "SGLT2Status",
  "PMH_dialysis_transplant", "PMH_PriorMBS", "MBSduringGLP1", "BMIatEvent",
  "BMI3mPostEvent", "BMI6mPostEvent", "BMI9mPostEvent", "BMI12mPostEvent",
  "BMI2yPostEvent", "BMI3yPostEvent", "BMI4yPostEvent", "BMI5yPostEvent",
  "BMI6yPostEvent", "WeightAtEvent", "Weight3mPostEvent", "Weight6mPostEvent",
  "Weight9mPostEvent", "Weight12mPostEvent", "Weight2yPostEvent",
  "Weight3yPostEvent", "Weight4yPostEvent", "Weight5yPostEvent",
  "Weight6yPostEvent", "HbA1cAtEvent", "HbA1c12mPostEvent",
  "HbA1c2yPostEvent", "HbA1c3yPostEvent", "HbA1c4yPostEvent",
  "HbA1c5yPostEvent", "HbA1c6yPostEvent", "CreatinineAtEvent", "eGFRatEvent",
  "Nephropathy", "NephropathyInterval", "KidneyFailureInterval",
  "DialysisInterval", "TransplantInterval", "Retinopathy",
  "RetinopathyInterval", "Deceased", "DeathInterval", "CVdeath", "MACE",
  "MACEtype", "MACEdate", "MACEinterval", "ActiveEndInterval"
)

mbs_columns <- c(
  "PatKey", "CptCode", "ProcedureDate", "ProcDateValue", "AgeAtEvent",
  "CoverageClass", "PriorGLP1", "PostOpGLP1", "GLP1Interval",
  "GLP1StartDate", "GLP1EndDate", "GLP1Duration", "GLP1Name", "GLP1Route",
  "MaxGLP1Dose", "MostRecentDose", "MostRecentDoseUnit", "Sex", "FirstRace",
  "SecondRace", "MultiRacial", "PreferredLanguage", "StateOrProvince", "RUCA",
  "SviOverall", "SviHousehold", "SviTransportation", "SviMinority", "SviSES",
  "PMH_DM2", "PMH_hypertension", "PMH_OSA", "PMH_dyslipidemia",
  "PMH_retinopathy", "PMH_VTE", "PMH_AFib", "PMH_MI", "PMH_stroke",
  "InsulinStatus", "BiguanideStatus", "SGLT2Status",
  "PMH_dialysis_transplant", "PMH_PriorMBS", "BMIatEvent", "BMI3mPostEvent",
  "BMI6mPostEvent", "BMI9mPostEvent", "BMI12mPostEvent", "BMI2yPostEvent",
  "BMI3yPostEvent", "BMI4yPostEvent", "BMI5yPostEvent", "BMI6yPostEvent",
  "WeightAtEvent", "Weight3mPostEvent", "Weight6mPostEvent",
  "Weight9mPostEvent", "Weight12mPostEvent", "Weight2yPostEvent",
  "Weight3yPostEvent", "Weight4yPostEvent", "Weight5yPostEvent",
  "Weight6yPostEvent", "HbA1cAtEvent", "HbA1c12mPostEvent",
  "HbA1c2yPostEvent", "HbA1c3yPostEvent", "HbA1c4yPostEvent",
  "HbA1c5yPostEvent", "HbA1c6yPostEvent", "CreatinineAtEvent", "eGFRatEvent",
  "Nephropathy", "NephropathyInterval", "KidneyFailureInterval",
  "DialysisInterval", "TransplantInterval", "Retinopathy",
  "RetinopathyInterval", "Deceased", "DeathInterval", "CVdeath", "MACE",
  "MACEtype", "MACEdate", "MACEinterval", "ActiveEndInterval"
)

fake_glp1_cohort <- fake_modeling_cohort[fake_modeling_cohort$Cohort == "GLP1", glp1_columns]
fake_mbs_cohort <- fake_modeling_cohort[fake_modeling_cohort$Cohort == "MBS", mbs_columns]

write.csv(
  fake_modeling_cohort,
  file.path(output_dir, "fake_modeling_cohort.csv"),
  row.names = FALSE,
  na = ""
)
write.csv(
  fake_glp1_cohort,
  file.path(output_dir, "fake_glp1_cohort.csv"),
  row.names = FALSE,
  na = ""
)
write.csv(
  fake_mbs_cohort,
  file.path(output_dir, "fake_mbs_cohort.csv"),
  row.names = FALSE,
  na = ""
)

message("Wrote ", nrow(fake_modeling_cohort), " synthetic modeling rows to ",
        file.path(output_dir, "fake_modeling_cohort.csv"))
message("Wrote ", nrow(fake_glp1_cohort), " synthetic GLP1 rows to ",
        file.path(output_dir, "fake_glp1_cohort.csv"))
message("Wrote ", nrow(fake_mbs_cohort), " synthetic MBS rows to ",
        file.path(output_dir, "fake_mbs_cohort.csv"))
