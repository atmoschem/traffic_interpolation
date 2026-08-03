#!/usr/bin/env Rscript
# Generate SHAP figures for both flow and speed models
# with clear vehicle type / speed labels + inferno_r grid maps

library(data.table)
library(xgboost)
library(ggplot2)
library(hexbin)

# ── Dynamic Base Path Resolution ──
BASE <- Sys.getenv("TRAFFIC_BASE", unset = "/home/sibarra/.hermes/profiles/aire/home/hermes_vein/utfpr2025")
if (!dir.exists(BASE) || !dir.exists(file.path(BASE, "data"))) {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("--file=", args, value = TRUE)
  if (length(file_arg) > 0) {
    script_path <- sub("--file=", "", file_arg[1])
    potential_base <- normalizePath(file.path(dirname(script_path), "..", "..", ".."))
    if (dir.exists(file.path(potential_base, "data"))) {
      BASE <- potential_base
    }
  } else {
    if (dir.exists("../../../data")) {
      BASE <- normalizePath("../../../data/..")
    } else {
      BASE <- getwd()
    }
  }
}

cat("Using BASE path:", BASE, "\n")

SP_DATA <- file.path(BASE, "xgboost_june_2026/training_data_fixed_rd.csv")
MODEL_DIR <- file.path(BASE, "xgboost_june_2026/models")
FIGS_DIR  <- file.path(BASE, "xgboost_june_2026/final/rfigs")
dir.create(FIGS_DIR, showWarnings = FALSE, recursive = TRUE)

STATIC <- c("lanes","maxspeed","nightlight","pop_density",
            "road_density_500m","road_density_1000m","road_density_5000m",
            "highway_code","surface_code","oneway_code",
            "lu_tree","lu_shrubland","lu_grassland","lu_cropland",
            "lu_builtup","lu_bare","lu_water","lu_wetland",
            "lu_mangroves","lu_moss")
TEMPORAL <- c("hour","day_of_week","is_weekend")
FEATURES <- c(STATIC, TEMPORAL)

FILL_VALS <- list(lanes = 2, maxspeed = 60, nightlight = 0, pop_density = 0,
                  road_density_500m = 0, road_density_1000m = 0, road_density_5000m = 0,
                  highway_code = 4, surface_code = 1, oneway_code = 1)
for (c in STATIC[grep("^lu_", STATIC)]) FILL_VALS[[c]] <- 0

FLOW_CLASSES <- list(
  flow_pc    = list(label = "PC",    color = "#2166AC", short = "pc",    shap_title = "Passenger Car Flow"),
  flow_lcv   = list(label = "LCV",   color = "#D6604D", short = "lcv",   shap_title = "LCV Flow"),
  flow_mc    = list(label = "MC",    color = "#4DAF4A", short = "mc",    shap_title = "Motorcycle Flow"),
  flow_truck = list(label = "Truck", color = "#984EA3", short = "truck", shap_title = "Truck Flow"),
  flow_bus   = list(label = "Bus",   color = "#FF7F00", short = "bus",   shap_title = "Bus Flow")
)
FLOW_TARGETS <- names(FLOW_CLASSES)

if (!requireNamespace("shapviz", quietly = TRUE)) {
  install.packages("shapviz", repos = "https://cloud.r-project.org")
}
library(shapviz)

# ── Flow SHAP figures ──
cat("=== Generating Flow Model SHAP Figures ===\n")

sp <- fread(SP_DATA)
for (c in FEATURES) {
  if (c %in% names(sp)) {
    val <- FILL_VALS[[c]]; if (is.null(val)) val <- 0
    set(sp, which(is.na(sp[[c]])), c, val)
  }
}

for (target in FLOW_TARGETS) {
  cls <- FLOW_CLASSES[[target]]
  model_path <- file.path(MODEL_DIR, sprintf("xgb_flow_%s_sprj_r.json", cls$short))
  if (!file.exists(model_path)) {
    model_path <- file.path(MODEL_DIR, sprintf("xgb_flow_%s_sp_r.json", cls$short))
  }
  
  if (!file.exists(model_path)) {
    cat(sprintf("  SKIP %s: model not found\n", cls$label))
    next
  }
  
  model <- xgb.load(model_path)
  
  set.seed(42)
  n_sample <- min(2000, nrow(sp))
  idx_sample <- sample(nrow(sp), n_sample)
  X_sample <- as.matrix(sp[idx_sample, ..FEATURES])
  
  shp <- shapviz(model, X_pred = X_sample, X = X_sample)
  p_shap <- sv_importance(shp, kind = "both", max_display = 15, fill = cls$color) +
    ggtitle(sprintf("%s — SHAP Summary", cls$shap_title)) +
    theme_minimal(base_size = 14) +
    theme(plot.title = element_text(face = "bold", size = 16))
  
  fig_path <- file.path(FIGS_DIR, sprintf("fig05_shap_%s_r.png", cls$short))
  ggsave(fig_path, p_shap, width = 11, height = 9, dpi = 300)
  cat(sprintf("  Saved: %s\n", fig_path))
}

# ── Speed SHAP figure ──
cat("\n=== Generating Speed Model SHAP Figures ===\n")

speed_model_path <- file.path(MODEL_DIR, "xgb_speed_hourly_r.json")
if (file.exists(speed_model_path)) {
  speed_model <- xgb.load(speed_model_path)
  
  SPEED_FEATURES <- FEATURES
  
  set.seed(42)
  n_sample <- min(2000, nrow(sp))
  idx_sample <- sample(nrow(sp), n_sample)
  X_sample <- as.matrix(sp[idx_sample, ..SPEED_FEATURES])
  
  shp_speed <- shapviz(speed_model, X_pred = X_sample, X = X_sample)
  p_speed_shap <- sv_importance(shp_speed, kind = "both", max_display = 15, fill = "#E41A1C") +
    ggtitle("Hourly Speed Model — SHAP Summary") +
    theme_minimal(base_size = 14) +
    theme(plot.title = element_text(face = "bold", size = 16))
  
  fig_path <- file.path(FIGS_DIR, "fig05_shap_speed_r.png")
  ggsave(fig_path, p_speed_shap, width = 9, height = 7, dpi = 300)
  cat(sprintf("  Saved: %s\n", fig_path))
} else {
  cat(sprintf("  Speed model not found at %s\n", speed_model_path))
}

cat("\nDone. All SHAP figures generated.\n")
