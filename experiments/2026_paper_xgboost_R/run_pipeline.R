#!/usr/bin/env Rscript
#' Run the full R pipeline for the 2026 XGBoost paper (steps 00-10, in order).
#' Scripts resolve data paths automatically; set TRAFFIC_BASE to the directory
#' holding the input data (see README.md) before running.
#' Usage: Rscript experiments/2026_paper_xgboost_R/run_pipeline.R

scripts_dir <- file.path("experiments", "2026_paper_xgboost_R", "scripts")
steps <- sort(list.files(scripts_dir, pattern = "^[0-9]+.*\\.R$"))

cat(rep("=", 60), "\n", sep = "")
cat("  traffic_interpolation -- 2026 XGBoost paper R pipeline\n")
cat(sprintf("  %d steps from %s\n", length(steps), scripts_dir))
cat(rep("=", 60), "\n\n", sep = "")

for (i in seq_along(steps)) {
  step <- file.path(scripts_dir, steps[i])
  cat(sprintf("\n[%d/%d] %s\n", i, length(steps), steps[i]))
  status <- system2("Rscript", shQuote(step))
  if (status != 0) stop(sprintf("Step failed (exit %d): %s", status, steps[i]))
}

cat("\nDone.\n")
