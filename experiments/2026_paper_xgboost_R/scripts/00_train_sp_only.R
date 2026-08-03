#!/usr/bin/env Rscript
# Step: Train flow models on SP-only, validate per-class on SP + total on RJ
# R port: data.table + xgboost + ggplot2 + patchwork
#
# Strategy:
#   Train:  SP-only per-class flow (Mon-Wed)
#   Validate: SP test set (Thu-Sun) per-class
#   Cross-validate: Predict on RJ, sum classes → compare vs observed RJ total

library(data.table)
library(xgboost)
library(ggplot2)
library(patchwork)
library(jsonlite)
library(hexbin)
library(ggpmisc)

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
SP_DATA <- file.path(BASE, "xgboost_june_2026/training_data_fixed_rd.csv")
RJ_DATA <- file.path(BASE, "xgboost_june_2026/cetrio_training_data_r.csv")
MODEL_DIR <- file.path(BASE, "xgboost_june_2026/models")
FIGS_DIR  <- file.path(BASE, "xgboost_june_2026/final/rfigs")
dir.create(MODEL_DIR, showWarnings = FALSE, recursive = TRUE)
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
  flow_pc    = list(label = "PC",    color = "#2166AC", short = "pc",    rj = TRUE,  shap_title = "Passenger Car Flow"),
  flow_lcv   = list(label = "LCV",   color = "#D6604D", short = "lcv",   rj = TRUE,  shap_title = "LCV Flow"),
  flow_mc    = list(label = "MC",    color = "#4DAF4A", short = "mc",    rj = TRUE,  shap_title = "Motorcycle Flow"),
  flow_truck = list(label = "Truck", color = "#984EA3", short = "truck", rj = TRUE,  shap_title = "Truck Flow"),
  flow_bus   = list(label = "Bus",   color = "#FF7F00", short = "bus",   rj = FALSE, shap_title = "Bus Flow")
)
TARGETS <- names(CLASSES)

TRAIN_PARAMS <- list(objective = "reg:squarederror", max_depth = 8,
                     learning_rate = 0.08, subsample = 0.8, colsample_bytree = 0.8,
                     min_child_weight = 5, gamma = 1, seed = 42, nthread = -1)

FILL_VALS <- list(lanes = 2, maxspeed = 60, nightlight = 0, pop_density = 0,
                  road_density_500m = 0, road_density_1000m = 0, road_density_5000m = 0,
                  highway_code = 4, surface_code = 1, oneway_code = 1)
for (c in STATIC[grep("^lu_", STATIC)]) FILL_VALS[[c]] <- 0

r2_score <- function(y_true, y_pred) {
  1 - sum((y_true - y_pred)^2) / sum((y_true - mean(y_true))^2)
}

save_fig <- function(name, plot_obj, w = 20, h = 4.5) {
  path <- file.path(FIGS_DIR, name)
  ggsave(path, plot_obj, dpi = 300, width = w, height = h, units = "in")
  cat(sprintf("  Saved: %s\n", path))
}

cat(rep("=", 60), "\n", sep = "")
cat("Flow Models: SP-only train, per-class + RJ total validation\n")
cat(rep("=", 60), "\n", sep = "")

# ── 1. Load data ──
cat("\n[1/7] Loading data...\n")
t0 <- Sys.time()
sp <- fread(SP_DATA)
rj <- fread(RJ_DATA)
cat(sprintf("  SP: %d rows | RJ: %d rows\n", nrow(sp), nrow(rj)))
cat(sprintf("  Time: %.0fs\n", difftime(Sys.time(), t0, units = "secs")))

# ── 2. Train SP-only flow models ──
cat("\n[2/7] Training SP-only flow models (Mon-Wed train, Thu-Sun test)...\n")
models <- list()
train_results <- list()

for (target in TARGETS) {
  if (!target %in% names(sp)) {
    cat(sprintf("  %s: not in SP data, skipping\n", target))
    next
  }
  cls <- CLASSES[[target]]
  df <- sp[!is.na(get(target))]

  # Fill NaN features
  for (c in FEATURES) {
    val <- FILL_VALS[[c]]
    if (is.null(val)) val <- 0
    set(df, which(is.na(df[[c]])), c, val)
  }

  X <- as.matrix(df[, ..FEATURES])
  y <- df[[target]]

  # Mon-Wed (0,1,2) train, Thu-Sun (3,4,5,6) test
  train_mask <- df$day_of_week %in% c(0, 1, 2)
  test_mask  <- df$day_of_week %in% c(3, 4, 5, 6)

  X_train <- X[train_mask, ]
  X_test  <- X[test_mask, ]
  y_train <- y[train_mask]
  y_test  <- y[test_mask]

  dtrain <- xgb.DMatrix(X_train, label = y_train)
  dtest  <- xgb.DMatrix(X_test, label = y_test)

  # Capture verbose output for learning curve
  log_file <- tempfile(fileext = ".log")
  sink(log_file)
  model <- xgb.train(
    params = TRAIN_PARAMS, data = dtrain, nrounds = 500,
    evals = list(train = dtrain, test = dtest),
    early_stopping_rounds = 30, verbose = 1
  )
  sink()

  log_lines <- readLines(log_file)
  unlink(log_file)
  rmse_pattern <- "\\[([0-9]+)\\]\\s+train-rmse:([0-9.]+)\\s+test-rmse:([0-9.]+)"
  parsed <- data.table(round = integer(), train_rmse = numeric(), test_rmse = numeric())
  for (line in log_lines) {
    m <- regmatches(line, regexec(rmse_pattern, line))[[1]]
    if (length(m) == 4) {
      parsed <- rbind(parsed, data.table(
        round = as.integer(m[2]),
        train_rmse = as.numeric(m[3]),
        test_rmse = as.numeric(m[4])
      ))
    }
  }

  best_round <- parsed$round[which.min(parsed$test_rmse)]
  y_pred    <- predict(model, dtest)
  y_pred_train <- predict(model, dtrain)

  r2     <- r2_score(y_test, y_pred)
  r2_tr  <- r2_score(y_train, y_pred_train)
  mae    <- mean(abs(y_test - y_pred))
  rmse_v <- sqrt(mean((y_test - y_pred)^2))

  models[[target]] <- model

  train_results[[target]] <- list(
    r2 = round(r2, 4), mae = round(mae, 2), rmse = round(rmse_v, 2),
    r2_train = round(r2_tr, 4), rounds = best_round,
    n_train = length(y_train), n_test = length(y_test),
    mean_obs = round(mean(y_test), 1),
    mean_pred = round(mean(y_pred), 1),
    evals = list(train_rmse = parsed$train_rmse, test_rmse = parsed$test_rmse)
  )

  model_path <- file.path(MODEL_DIR, sprintf("xgb_flow_%s_sp_r.json", cls$short))
  xgb.save(model, model_path)

  cat(sprintf("  %s (SP-only): R²=%.4f (train: %.4f)  MAE=%.1f  RMSE=%.1f  rounds=%d\n",
              cls$label, r2, r2_tr, mae, rmse_v, best_round))
}

write_json(train_results, file.path(MODEL_DIR, "results_sp_flow_r.json"),
           pretty = TRUE, auto_unbox = TRUE, null = "null")

# ── 3. Figure 1: Learning Curves (SP test set) ──
cat("\n[3/7] Figure 1: Learning curves (SP test set)...\n")
p_list <- list()
for (target in TARGETS) {
  if (!target %in% names(models)) next
  cls <- CLASSES[[target]]
  evals <- train_results[[target]]$evals
  if (is.null(evals$train_rmse) || length(evals$train_rmse) == 0) next

  best <- train_results[[target]]$rounds
  df_curve <- data.table(
    round = seq_along(evals$train_rmse),
    rmse_train = evals$train_rmse,
    rmse_test  = evals$test_rmse
  )

  p <- ggplot(df_curve) +
    geom_line(aes(x = round, y = .data[["rmse_train"]]), color = "#333333", alpha = 0.5, linewidth = 0.8) +
    geom_line(aes(x = round, y = .data[["rmse_test"]]), color = cls$color, linewidth = 1.2) +
    geom_vline(xintercept = best, color = "red", linetype = "dashed", alpha = 0.4, linewidth = 0.8) +
    annotate("text", x = best, y = max(df_curve$rmse_test) * 0.9,
             label = sprintf("n=%d", best), size = 2.5, color = "red", hjust = 0, vjust = 1) +
    labs(x = "Boosting round", y = "RMSE (veh/h)", title = cls$label) +
    xlim(0, length(evals$train_rmse)) +
    theme_minimal(base_size = 10) +
    theme(plot.title = element_text(face = "bold", size = 12),
          panel.grid.minor = element_blank())
  p_list[[target]] <- p
}

if (length(p_list) > 0) {
  p_lc <- wrap_plots(p_list, ncol = 5) +
    plot_annotation(title = "SP-only Flow Models — Learning Curves (Mon-Wed train, Thu-Sun test)",
                    theme = theme(plot.title = element_text(face = "bold", size = 14)))
  save_fig("fig01_flow_learning_curves_r.png", p_lc, w = 22, h = 5)
}

# ── 4. Figure 2: Per-class Validation Scatter (SP test set only) ──
cat("\n[4/7] Figure 2: Per-class validation (SP test set hexbin)...\n")
p_list2 <- list()
for (target in TARGETS) {
  if (!target %in% names(models)) next
  cls <- CLASSES[[target]]
  model <- models[[target]]

  df <- sp[!is.na(get(target))]
  for (c in FEATURES) {
    val <- FILL_VALS[[c]]
    if (is.null(val)) val <- 0
    set(df, which(is.na(df[[c]])), c, val)
  }

  # Test set only (Thu-Sun)
  test_mask <- df$day_of_week %in% c(3, 4, 5, 6)
  df_test <- df[test_mask]

  y_obs <- df_test[[target]]
  y_pred <- predict(model, as.matrix(df_test[, ..FEATURES]))

  df_scat <- data.table(observed = y_obs, predicted = y_pred)
  lims <- c(min(c(y_obs, y_pred)), max(c(y_obs, y_pred)))

  r2_v <- r2_score(y_obs, y_pred)
  mae_v <- mean(abs(y_obs - y_pred))
  # Compute correlation p-value
  cor_test <- cor.test(y_obs, y_pred)
  pval <- cor_test$p.value
  pval_str <- ifelse(pval < 0.001, "p<0.001", sprintf("p=%.3f", pval))

  # Use points for low-variance targets (bus: all predictions ~constant)
  use_hex <- diff(range(y_obs)) > 1 && diff(range(y_pred)) > 1

  if (use_hex) {
    p <- ggplot(df_scat, aes(x = observed, y = predicted)) +
      geom_hex(bins = 45, alpha = 0.8) +
      scale_fill_gradient(low = "lightblue", high = "darkblue", trans = "log") +
      geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed",
                  alpha = 0.4, linewidth = 1) +
      stat_poly_eq(aes(label = paste(after_stat(rr.label), after_stat(p.value.label), sep = "~~~")),
                   formula = y ~ x, parse = TRUE, size = 3,
                   label.x = "left", label.y = "top")
  } else {
    cat(sprintf("  Low variance — using points for %s\n", cls$label))
    p <- ggplot(df_scat, aes(x = observed, y = predicted)) +
      geom_point(alpha = 0.3, size = 0.8, color = cls$color) +
      geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed",
                  alpha = 0.4, linewidth = 1)
  }

  p <- p +
    coord_fixed(xlim = lims, ylim = lims) +
    labs(x = sprintf("Observed %s flow (veh/h)", cls$label),
         y = sprintf("Predicted %s flow (veh/h)", cls$label),
         title = sprintf("%s (SP test set): R²=%.3f, MAE=%.0f, %s", cls$label, r2_v, mae_v, pval_str)) +
    theme_minimal(base_size = 10) +
    theme(plot.title = element_text(face = "bold"),
          panel.grid.minor = element_blank())
  p_list2[[target]] <- p
  cat(sprintf("  %s: R²=%.3f MAE=%.0f (n=%d)\n", cls$label, r2_v, mae_v, length(y_obs)))
}

p_scatter <- wrap_plots(p_list2, ncol = 3) +
  plot_annotation(title = "SP-only Flow Models — Test Set Validation (Thu-Sun)",
                  theme = theme(plot.title = element_text(face = "bold", size = 14)))
ggsave(file.path(FIGS_DIR, "fig02_flow_validation_scatter_r.png"), p_scatter,
       width = 15, height = 10, dpi = 300)
cat(sprintf("  Saved: %s/fig02_flow_validation_scatter_r.png\n", FIGS_DIR))

# ── 5. Figure 3: Feature Importance (SP models) ──
cat("\n[5/7] Figure 3: Feature importance (SP models)...\n")
imp_list <- list()
for (target in TARGETS) {
  if (!target %in% names(models)) next
  cls <- CLASSES[[target]]
  model <- models[[target]]

  imp <- xgb.importance(model = model)
  n_top <- min(12, nrow(imp))
  imp_top <- imp[order(Gain, decreasing = TRUE)][1:n_top]
  imp_top[, Feature := factor(Feature, levels = rev(Feature))]

  p <- ggplot(imp_top, aes(x = Gain, y = Feature)) +
    geom_col(fill = cls$color, alpha = 0.8, width = 0.7) +
    labs(x = "Gain", y = NULL, title = cls$label) +
    theme_minimal(base_size = 9) +
    theme(plot.title = element_text(face = "bold"),
          panel.grid.minor = element_blank())
  imp_list[[target]] <- p
}

p_imp <- wrap_plots(imp_list, ncol = 5) +
  plot_annotation(title = "Feature Importance (Gain) — SP-only Flow Models",
                  theme = theme(plot.title = element_text(face = "bold", size = 14)))
ggsave(file.path(FIGS_DIR, "fig03_flow_importance_r.png"), p_imp,
       width = 22, height = 4.5, dpi = 300)
cat(sprintf("  Saved: %s/fig03_flow_importance_r.png\n", FIGS_DIR))

# ── 6. Figure 4: RJ Total Flow Validation ──
cat("\n[6/7] Figure 4: RJ total flow validation...\n")
# Compute RJ observed total (sum of 4 estimated classes = original total from CET-Rio)
rj[, rj_total_obs := flow_pc + flow_lcv + flow_mc + flow_truck]

# Predict each class on RJ data
for (c in FEATURES) {
  val <- FILL_VALS[[c]]
  if (is.null(val)) val <- 0
  set(rj, which(is.na(rj[[c]])), c, val)
}
X_rj <- as.matrix(rj[, ..FEATURES])

# Predict per-class
for (target in TARGETS) {
  if (!target %in% names(models)) next
  cls <- CLASSES[[target]]
  pred_col <- paste0("pred_", cls$short)
  rj[, (pred_col) := predict(models[[target]], X_rj)]
}

# Total predicted = sum of all class predictions
rj[, rj_total_pred := pred_pc + pred_lcv + pred_mc + pred_truck]
# Note: bus predicted is 0 for RJ (no bus model with RJ data)

y_rj_obs <- rj[, rj_total_obs]
y_rj_pred <- rj[, rj_total_pred]

r2_rj <- r2_score(y_rj_obs, y_rj_pred)
mae_rj <- mean(abs(y_rj_obs - y_rj_pred))
rmse_rj <- sqrt(mean((y_rj_obs - y_rj_pred)^2))
bias_rj <- mean(y_rj_pred) / mean(y_rj_obs)

cat(sprintf("  RJ total flow: R²=%.4f  MAE=%.1f  RMSE=%.1f  bias=%.3f\n",
            r2_rj, mae_rj, rmse_rj, bias_rj))

df_rj <- data.table(observed = y_rj_obs, predicted = y_rj_pred)
lims_rj <- c(min(c(y_rj_obs, y_rj_pred)), max(c(y_rj_obs, y_rj_pred)))

p_rj <- ggplot(df_rj, aes(x = observed, y = predicted)) +
  geom_hex(bins = 50, alpha = 0.8) +
  scale_fill_gradient(low = "lightblue", high = "darkblue", trans = "log") +
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed",
              alpha = 0.4, linewidth = 1) +
  coord_fixed(xlim = lims_rj, ylim = lims_rj) +
  labs(x = "Observed total flow (veh/h) — CET-Rio",
       y = "Predicted total flow (veh/h) — sum of 4 classes",
       title = sprintf("RJ Total Flow Validation: R²=%.3f, MAE=%.0f, bias=%.2f",
                       r2_rj, mae_rj, bias_rj)) +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold"),
        panel.grid.minor = element_blank())

ggsave(file.path(FIGS_DIR, "fig04_rj_total_validation_r.png"), p_rj,
       width = 8, height = 8, dpi = 300)
cat(sprintf("  Saved: %s/fig04_rj_total_validation_r.png\n", FIGS_DIR))

# Also save RJ validation to results
rj_results <- list(
  r2 = round(r2_rj, 4), mae = round(mae_rj, 2), rmse = round(rmse_rj, 2),
  bias = round(bias_rj, 4), n = length(y_rj_obs)
)
write_json(rj_results, file.path(MODEL_DIR, "results_rj_validation_r.json"),
           pretty = TRUE, auto_unbox = TRUE)

# ── 7. Figure 5: R² by Day of Week (SP test set) ──
cat("\n[7/7] Figure 5: R² by day of week (SP test set)...\n")
dow_names <- c("Mon","Tue","Wed","Thu","Fri","Sat","Sun")
r2_data <- data.table()

for (target in TARGETS) {
  if (!target %in% names(models)) next
  cls <- CLASSES[[target]]
  model <- models[[target]]

  df <- sp[!is.na(get(target))]
  for (c in FEATURES) {
    val <- FILL_VALS[[c]]
    if (is.null(val)) val <- 0
    set(df, which(is.na(df[[c]])), c, val)
  }

  # Only test set days (Thu-Sun)
  for (d in 3:6) {
    mask <- df$day_of_week == d
    if (sum(mask) > 10) {
      y_o <- df[mask][[target]]
      y_p <- predict(model, as.matrix(df[mask, ..FEATURES]))
      r2_d <- r2_score(y_o, y_p)
      r2_data <- rbind(r2_data, data.table(
        target = cls$label, day = d, r2 = r2_d,
        color = cls$color
      ))
    }
  }
}
r2_data[, day_name := factor(dow_names[day + 1], levels = dow_names)]

p_dow <- ggplot(r2_data, aes(x = day_name, y = r2, fill = target)) +
  geom_col(position = position_dodge(width = 0.85), alpha = 0.85, width = 0.75) +
  scale_fill_manual(values = setNames(
    sapply(TARGETS, function(t) CLASSES[[t]]$color),
    sapply(TARGETS, function(t) CLASSES[[t]]$label))) +
  geom_hline(yintercept = 0, color = "gray", linewidth = 0.3) +
  ylim(-0.5, 1.0) +
  labs(x = "Day of week (test set)", y = expression(R^2),
       title = "SP-only Flow Models — Validation R² by Day of Week", fill = "Class") +
  theme_minimal(base_size = 11) +
  theme(plot.title = element_text(face = "bold"),
        panel.grid.minor = element_blank(),
        legend.position = "bottom")

ggsave(file.path(FIGS_DIR, "fig05_flow_dow_validation_r.png"), p_dow,
       width = 8, height = 5, dpi = 300)
cat(sprintf("  Saved: %s/fig05_flow_dow_validation_r.png\n", FIGS_DIR))

# ── Final summary ──
cat("\n" , rep("=", 60), "\n", sep = "")
cat("SP-only Flow Model Summary\n")
cat(rep("=", 60), "\n", sep = "")
for (target in TARGETS) {
  if (!target %in% names(train_results)) next
  r <- train_results[[target]]
  cat(sprintf("  %6s: R²=%.3f  MAE=%.1f  RMSE=%.1f  rounds=%d\n",
              CLASSES[[target]]$label, r$r2, r$mae, r$rmse, r$rounds))
}
cat(sprintf("\n  RJ total validation: R²=%.3f  MAE=%.1f  bias=%.2f\n", r2_rj, mae_rj, bias_rj))

cat("\nModels saved to:", MODEL_DIR, "/\n")
cat("Figures saved to:", FIGS_DIR, "/\n")
cat("Done.\n")
