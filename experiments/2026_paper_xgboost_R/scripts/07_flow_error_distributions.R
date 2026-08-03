#!/usr/bin/env Rscript
# Step 7: Error histograms + Shapiro-Wilk normality test for each flow model
# Manuscript Figure 2: Prediction Error Distributions — Flow Models (Thu-Sun test set)
# Uses pre-trained SP-only models on the fixed road-density SP data.

library(data.table)
library(xgboost)
library(ggplot2)
library(patchwork)
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

CLASSES <- list(
  flow_pc    = list(label = "PC",    color = "#2166AC", short = "pc"),
  flow_lcv   = list(label = "LCV",   color = "#D6604D", short = "lcv"),
  flow_mc    = list(label = "MC",    color = "#4DAF4A", short = "mc"),
  flow_truck = list(label = "Truck", color = "#984EA3", short = "truck")
)
TARGETS <- names(CLASSES)

r2_score <- function(y_true, y_pred) {
  1 - sum((y_true - y_pred)^2) / sum((y_true - mean(y_true))^2)
}

cat("Loading data...\n")
sp <- fread(SP_DATA)
test_mask <- sp$day_of_week %in% c(3, 4, 5, 6)
df_test <- sp[test_mask]
for (c in FEATURES) {
  if (c %in% names(df_test)) {
    val <- FILL_VALS[[c]]; if (is.null(val)) val <- 0
    set(df_test, which(is.na(df_test[[c]])), c, val)
  }
}
cat(sprintf("Test set: %d rows\n", nrow(df_test)))

results <- list()
p_list <- list()
shapiro_results <- data.table()

for (target in TARGETS) {
  cls <- CLASSES[[target]]
  model_path <- file.path(MODEL_DIR, sprintf("xgb_flow_%s_sp_r.json", cls$short))
  if (!file.exists(model_path)) next

  model <- xgb.load(model_path)
  y_obs <- df_test[[target]]
  y_pred <- predict(model, as.matrix(df_test[, ..FEATURES]))

  valid <- !is.na(y_obs) & is.finite(y_pred)
  y_obs <- y_obs[valid]
  y_pred <- y_pred[valid]

  residuals <- y_obs - y_pred
  r2 <- r2_score(y_obs, y_pred)
  mae <- mean(abs(residuals))
  rmse <- sqrt(mean(residuals^2))

  # Shapiro-Wilk test (max 5000 samples per R limitation)
  n_shap <- min(5000, length(residuals))
  set.seed(42)
  idx_shap <- sample(length(residuals), n_shap)
  sw <- shapiro.test(residuals[idx_shap])

  cat(sprintf("\n%s:\n", cls$label))
  cat(sprintf("  R²=%.3f, MAE=%.0f, RMSE=%.0f\n", r2, mae, rmse))
  cat(sprintf("  Residuals: mean=%.1f, sd=%.0f, n=%d\n", mean(residuals), sd(residuals), length(residuals)))
  cat(sprintf("  Shapiro-Wilk: W=%.4f, p=%s\n", sw$statistic, format(sw$p.value, scientific = TRUE, digits = 3)))

  shapiro_results <- rbind(shapiro_results, data.table(
    class = cls$label,
    n = length(residuals),
    mean_resid = round(mean(residuals), 1),
    sd_resid = round(sd(residuals), 0),
    skewness = round(mean((residuals - mean(residuals))^3) / (sd(residuals)^3), 3),
    shapiro_W = round(sw$statistic, 4),
    shapiro_p = sw$p.value
  ))

  # Histogram with normal curve overlay
  df_err <- data.table(residual = residuals)

  sw_label <- sprintf("Shapiro-Wilk\nW = %.4f\np = %s", sw$statistic, format(sw$p.value, scientific = TRUE, digits = 3))

  p <- ggplot(df_err, aes(x = residual)) +
    geom_histogram(aes(y = after_stat(density)), bins = 80,
                   fill = cls$color, alpha = 0.7, color = "white", linewidth = 0.2) +
    stat_function(fun = dnorm, args = list(mean = mean(residuals), sd = sd(residuals)),
                  color = "black", linewidth = 1, linetype = "dashed") +
    annotate("label", x = Inf, y = Inf, label = sw_label,
             hjust = 1.1, vjust = 1.5, size = 5, fontface = "bold",
             fill = "white", alpha = 0.85, label.size = 0.3) +
    labs(x = "Residual (observed - predicted, veh/h)",
         y = "Density",
         title = sprintf("%s — Residuals (R²=%.3f, MAE=%.0f)", cls$label, r2, mae)) +
    theme_minimal(base_size = 14) +
    theme(plot.title = element_text(face = "bold"),
          panel.grid.minor = element_blank())
  p_list[[target]] <- p
}

# Print summary table
cat("\n\n=== Normality Test Summary ===\n")
print(shapiro_results)

# Save combined figure
p_all <- wrap_plots(p_list, ncol = 2) +
  plot_annotation(title = "Prediction Error Distributions — Flow Models (Thu-Sun test set)",
                  theme = theme(plot.title = element_text(face = "bold", size = 18)))

fig_path <- file.path(FIGS_DIR, "fig07_flow_error_distributions_r.png")
ggsave(fig_path, p_all, width = 14, height = 12, dpi = 300)
cat(sprintf("\nSaved: %s\n", fig_path))

# Save Shapiro results
fwrite(shapiro_results, file.path(FIGS_DIR, "fig07_shapiro_results_r.csv"))
cat("Saved: fig07_shapiro_results_r.csv\n")
cat("Done.\n")
