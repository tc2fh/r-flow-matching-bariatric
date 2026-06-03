# Load MBSCohort from EPIC Cosmos/SQL Server and export a flow-compatible CSV.
#
# This script intentionally writes to data/ by default. The data/ directory is
# ignored by git because real database exports may contain sensitive records.
#
# Example:
#   Rscript load_cosmos_flow_data.R --output data/cosmos_mbs_flow_input.csv

script_args <- commandArgs(trailingOnly = FALSE)
file_arg <- "--file="
script_path <- sub(file_arg, "", script_args[startsWith(script_args, file_arg)])
PROJECT_ROOT <- if (length(script_path) > 0) dirname(normalizePath(script_path[1])) else getwd()

source(file.path(PROJECT_ROOT, "R", "config.R"))
source(file.path(PROJECT_ROOT, "R", "data.R"))

require_package <- function(package_name) {
  if (!requireNamespace(package_name, quietly = TRUE)) {
    stop("Package '", package_name, "' is required for Cosmos loading.", call. = FALSE)
  }
}

default_connection_string <- paste(
  "Driver={ODBC Driver 17 for SQL Server};",
  "Server=tcp:PROJECTS;",
  "Database=ProjectD332AFD;",
  "Trusted_Connection=yes;",
  sep = "\n"
)

parse_cosmos_args <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  out <- list(
    output = file.path("data", "cosmos_mbs_flow_input.csv"),
    connection_string = default_connection_string
  )

  if ("--help" %in% args || "-h" %in% args) {
    cat(
      "Usage: Rscript load_cosmos_flow_data.R [options]\n",
      "  --output PATH\n",
      "  --connection-string TEXT\n",
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
    if (key == "--output") out$output <- value
    else if (key == "--connection-string") out$connection_string <- value
    else stop("Unknown argument: ", key, call. = FALSE)
    i <- i + 2
  }

  out
}

load_cosmos_mbs_cohort <- function(connection_string = default_connection_string, timeout = 1000) {
  require_package("DBI")
  require_package("odbc")

  projects <- DBI::dbConnect(
    odbc::odbc(),
    .connection_string = connection_string,
    timeout = timeout
  )
  on.exit(DBI::dbDisconnect(projects), add = TRUE)

  sql <- "
SELECT *
FROM MBSCohort
WHERE Sex NOT IN (N'#Masked', N'*Unspecified', N'Unknown', N'other')
  AND CoverageClass NOT IN (N'*Not Applicable', N'*Unspecified')
  AND BMIatEvent < WeightAtEvent
  AND BMIatEvent BETWEEN 35 AND 75
  AND (eGFRatEvent IS NULL OR eGFRatEvent >= 20)
  AND PMH_dialysis_transplant = 0
  AND (NephropathyInterval IS NULL OR NephropathyInterval >= 0)
  AND PMH_PriorMBS = 0
  AND PriorGLP1 = 0
  AND PMH_DM2 = 1
  AND PMH_retinopathy = 0
  AND ActiveEndInterval >= 700
  AND ProcDateValue <= '2023-05-01'
"

  unique(DBI::dbGetQuery(projects, sql))
}

export_cosmos_mbs_for_flow <- function(
  output_path = file.path("data", "cosmos_mbs_flow_input.csv"),
  connection_string = default_connection_string
) {
  mbs <- load_cosmos_mbs_cohort(connection_string = connection_string)
  write_flow_input_csv(mbs, output_path)

  dataset <- load_flow_dataset(output_path)
  message("Wrote ", nrow(dataset$data), " flow-compatible MBS rows to ", output_path)
  message("Recognized surgery counts:")
  print(table(dataset$surgery_type))
  invisible(output_path)
}

is_direct_script <- any(grepl("load_cosmos_flow_data[.]R$", commandArgs(trailingOnly = FALSE)))
if (is_direct_script) {
  args <- parse_cosmos_args()
  export_cosmos_mbs_for_flow(
    output_path = args$output,
    connection_string = args$connection_string
  )
}
