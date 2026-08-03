#!/usr/bin/env Rscript
# Speed validation hexbin scatter + SHAP, matching flow figure style.
# Uses the SP+RJ flow data (which includes speed) and the hourly speed model.

library(data.table)
library(xgboost)
library(ggplot2)
library(hexbin)
library(patchwork)

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
FIGS_DIR <- file.path(BASE, "xgboost_june_2026/final/rfigs")
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

r2_score <- function(y_true, y_pred) {
  1 - sum((y_true - y_pred)^2) / sum((y_true - mean(y_true))^2)
}

cat("Loading SP training data (has speed)...\n")
full <- fread(SP_DATA)
full <- full[source == "artesp" & !is.na(speed)]
cat(sprintf("ARTESP hourly: %d rows\n", nrow(full)))

# Fill NAs in features
for (c in FEATURES) {
  v <- FILL_VALS[[c]]; if (is.null(v)) v <- 0
  set(full, which(is.na(full[[c]])), c, v)
}

# ── Load speed model ──
speed_model_path <- file.path(MODEL_DIR, "xgb_speed_hourly_r.json")
if (!file.exists(speed_model_path)) {
  stop("Speed model not found at ", speed_model_path)
}
speed_model <- xgb.load(speed_model_path)
cat(sprintf("Loaded: %s\n", speed_model_path))

# ── Predict on all data with speed ──
valid <- full[!is.na(speed)]
X <- as.matrix(valid[, ..FEATURES])
y_obs <- valid$speed
y_pred <- predict(speed_model, X)

r2_v <- r2_score(y_obs, y_pred)
mae_v <- mean(abs(y_obs - y_pred))
bias <- mean(y_pred) / mean(y_obs)

cat(sprintf("Speed model: R²=%.3f, MAE=%.1f, bias=%.2fx, n=%d\n",
            r2_v, mae_v, bias, length(y_obs)))

# ── Hexbin scatter plot (matching flow style) ──
df_scat <- data.table(observed = y_obs, predicted = y_pred)
lims <- c(0, max(y_obs, y_pred))

p <- ggplot(df_scat, aes(x = observed, y = predicted)) +
  geom_hex(bins = 40, alpha = 0.85) +
  scale_fill_gradient(low = "lightblue", high = "darkblue", trans = "log") +
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed", alpha = 0.4, linewidth = 1) +
  coord_fixed(xlim = lims, ylim = lims) +
  labs(x = "Observed speed (km/h)", y = "Predicted speed (km/h)",
       title = sprintf("Speed: R²=%.3f, bias=%.2fx", r2_v, bias), fill = "Count") +
  theme_minimal(base_size = 14) +
  theme(plot.title = element_text(face = "bold", size = 16), panel.grid.minor = element_blank())

ggsave(file.path(FIGS_DIR, "fig_speed_validation_hex_r.png"), p, width = 9, height = 8, dpi = 300)
cat(sprintf("Saved: %s/fig_speed_validation_hex_r.png\n", FIGS_DIR))

# ── Speed SHAP ──
cat("\nGenerating speed SHAP...\n")
if (requireNamespace("shapviz", quietly = TRUE)) {
  library(shapviz)
  set.seed(42)
  n_sample <- min(2000, nrow(valid))
  idx_sample <- sample(nrow(valid), n_sample)
  X_sample <- as.matrix(valid[idx_sample, ..FEATURES])
  
  shp <- shapviz(speed_model, X_pred = X_sample, X = X_sample)
  p_shap <- sv_importance(shp, kind = "both", max_display = 15, fill = "#2166AC") +
    ggtitle("Hourly Speed Model — SHAP Summary") +
    theme_minimal(base_size = 14) +
    theme(plot.title = element_text(face = "bold", size = 16))
  
  ggsave(file.path(FIGS_DIR, "fig05_shap_speed_r.png"), p_shap, width = 11, height = 9, dpi = 300)
  cat(sprintf("Saved: %s/fig05_shap_speed_r.png\n", FIGS_DIR))
} else {
  cat("shapviz not installed, skipping SHAP\n")
}

cat("\nDone.\n")
