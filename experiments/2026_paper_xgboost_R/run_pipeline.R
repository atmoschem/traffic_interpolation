#' Run the full XGBoost pipeline
#' Usage: Rscript experiments/2026_paper_xgboost_R/run_pipeline.R

library(data.table); library(yaml)
cfg_path <- "experiments/2026_paper_xgboost_R/config.yaml"
if (!file.exists(cfg_path)) cfg_path <- "config.default.yaml"
cfg <- yaml::read_yaml(cfg_path)

cat(rep("=",60),"\n",sep=""); cat("  traffic_interpolation -- 2026 XGBoost Paper\n"); cat(rep("=",60),"\n\n",sep="")

source("src/R/train_xgboost_flow.R")
source("src/R/validate_flow.R")
source("src/R/validate_cetrio.R")
source("src/R/generate_figures.R")

cat("\n[1/4] Training flow models...\n")
if (file.exists(cfg$training_data)) {
  train_flow_models(cfg$training_data, cfg$model_dir, cfg$output_dir)
} else cat("  Training data not found -- using pre-trained models.\n")

cat("\n[2/4] Validating on SP test set (Thu-Sun)...\n")
if (file.exists(cfg$training_data)) validate_flow_models(cfg$training_data, cfg$model_dir, cfg$output_dir)

cat("\n[3/4] Cross-validating on CET-Rio...\n")
if (file.exists(cfg$cetrio_data)) validate_cetrio(cfg$cetrio_data, cfg$model_dir, cfg$output_dir)

cat("\n[4/4] Generating figures...\n")
cat("  See experiments/2026_paper_xgboost_R/scripts/ for individual figure scripts.\n")
cat("\nDone.\n")
