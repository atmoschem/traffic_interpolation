#!/usr/bin/env Rscript
# Step 6: Regenerate Figure 1 (flow validation scatter) with p-value annotations
# Loads pre-trained SP-only models, so no retraining needed.

library(data.table)
library(xgboost)
library(ggplot2)
library(patchwork)
library(hexbin)
library(ggpmisc)

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

CLASSES <- list(
  flow_pc    = list(label = "PC",    color = "#2166AC", short = "pc"),
  flow_lcv   = list(label = "LCV",   color = "#D6604D", short = "lcv"),
  flow_mc    = list(label = "MC",    color = "#4DAF4A", short = "mc"),
  flow_truck = list(label = "Truck", color = "#984EA3", short = "truck"),
  flow_bus   = list(label = "Bus",   color = "#FF7F00", short = "bus")
)
TARGETS <- names(CLASSES)

FILL_VALS <- list(lanes = 2, maxspeed = 60, nightlight = 0, pop_density = 0,
                  road_density_500m = 0, road_density_1000m = 0, road_density_5000m = 0,
                  highway_code = 4, surface_code = 1, oneway_code = 1)
for (c in STATIC[grep("^lu_", STATIC)]) FILL_VALS[[c]] <- 0

r2_score <- function(y_true, y_pred) {
  1 - sum((y_true - y_pred)^2) / sum((y_true - mean(y_true))^2)
}

cat("Loading data...\n")
sp <- fread(SP_DATA)
full <- copy(sp)

# Fill NAs
for (c in FEATURES) {
  if (c %in% names(full)) {
    val <- FILL_VALS[[c]]; if (is.null(val)) val <- 0
    set(full, which(is.na(full[[c]])), c, val)
  }
}

# Test set: Thu-Sun
test_mask <- full$day_of_week %in% c(3, 4, 5, 6)
df_test <- full[test_mask]
cat(sprintf("Test set: %d rows\n", nrow(df_test)))

cat("Loading pre-trained models and generating Figure 1...\n")
p_list <- list()

for (target in TARGETS) {
  cls <- CLASSES[[target]]
  model_path <- file.path(MODEL_DIR, sprintf("xgb_flow_%s_sp_r.json", cls$short))
  
  if (!file.exists(model_path)) {
    cat(sprintf("  Model not found: %s — skipping\n", model_path))
    next
  }
  
  model <- xgb.load(model_path)
  
  y_obs <- df_test[[target]]
  y_pred <- predict(model, as.matrix(df_test[, ..FEATURES]))
  
  # Remove NAs
  valid <- !is.na(y_obs) & is.finite(y_pred)
  y_obs <- y_obs[valid]
  y_pred <- y_pred[valid]
  
  df_scat <- data.table(observed = y_obs, predicted = y_pred)
  lims <- c(min(c(y_obs, y_pred)), max(c(y_obs, y_pred)))
  
  r2_v <- r2_score(y_obs, y_pred)
  mae_v <- mean(abs(y_obs - y_pred))
  
  # ---- P-VALUE COMPUTATION ----
  cor_test <- cor.test(y_obs, y_pred)
  pval <- cor_test$p.value
  pval_str <- ifelse(pval < 0.001, "p<0.001", sprintf("p=%.3f", pval))
  cat(sprintf("  %s: R²=%.3f, MAE=%.0f, %s (r=%.3f, n=%d)\n",
              cls$label, r2_v, mae_v, pval_str, cor_test$estimate, length(y_obs)))
  
  use_hex <- diff(range(y_obs)) > 1 && diff(range(y_pred)) > 1
  
  if (use_hex) {
    p <- ggplot(df_scat, aes(x = observed, y = predicted)) +
      geom_hex(bins = 45, alpha = 0.8) +
      scale_fill_gradient(low = "lightblue", high = "darkblue", trans = "log") +
      geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed",
                  alpha = 0.4, linewidth = 1)
  } else {
    p <- ggplot(df_scat, aes(x = observed, y = predicted)) +
      geom_point(alpha = 0.3, size = 0.8, color = cls$color) +
      geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed",
                  alpha = 0.4, linewidth = 1)
  }
  
  p <- p +
    coord_fixed(xlim = lims, ylim = lims) +
    labs(x = sprintf("Observed %s flow (veh/h)", cls$label),
         y = sprintf("Predicted %s flow (veh/h)", cls$label),
         title = sprintf("%s: R²=%.3f, MAE=%.0f, %s", cls$label, r2_v, mae_v, pval_str)) +
    theme_minimal(base_size = 14) +
    theme(plot.title = element_text(face = "bold"),
          panel.grid.minor = element_blank())
  
  p_list[[target]] <- p
}

# Arrange panels (4 main classes, exclude bus)
p_fig1 <- wrap_plots(p_list[1:4], ncol = 2) +
  plot_annotation(title = "Flow Model Validation — SP-only (Test set: Thu–Sun)",
                  theme = theme(plot.title = element_text(face = "bold", size = 18)))

fig_path <- file.path(FIGS_DIR, "fig01_flow_validation_scatter_pval.png")
ggsave(fig_path, p_fig1, width = 14, height = 12, dpi = 300)
cat(sprintf("\nSaved: %s\n", fig_path))
cat("Done.\n")
