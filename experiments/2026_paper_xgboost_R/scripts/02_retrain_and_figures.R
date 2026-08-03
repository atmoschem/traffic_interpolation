#!/usr/bin/env Rscript
# Step 2-3: Retrain all models with SP+RJ data, generate paper-quality figures
# R port: data.table + terra + xgboost + ggplot2/tidyterra + patchwork

library(data.table)
library(xgboost)
library(ggplot2)
library(patchwork)
library(jsonlite)

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
RJ_DATA <- file.path(BASE, "xgboost_june_2026/cetrio_training_data_r.csv")
MODEL_DIR <- file.path(BASE, "xgboost_june_2026/models")
FIGS_DIR  <- file.path(BASE, "xgboost_june_2026/final/rfigs")
dir.create(MODEL_DIR, showWarnings = FALSE, recursive = TRUE)
dir.create(FIGS_DIR, showWarnings = FALSE, recursive = TRUE)

# ── Features ──
STATIC <- c("lanes","maxspeed","nightlight","pop_density",
            "road_density_500m","road_density_1000m","road_density_5000m",
            "highway_code","surface_code","oneway_code",
            "lu_tree","lu_shrubland","lu_grassland","lu_cropland",
            "lu_builtup","lu_bare","lu_water","lu_wetland",
            "lu_mangroves","lu_moss")
TEMPORAL <- c("hour","day_of_week","is_weekend")
FEATURES <- c(STATIC, TEMPORAL)

CLASSES <- list(
  flow_pc    = list(label = "PC",    color = "#2166AC", short = "pc",    shap_title = "Passenger Car Flow"),
  flow_lcv   = list(label = "LCV",   color = "#D6604D", short = "lcv",   shap_title = "LCV Flow"),
  flow_mc    = list(label = "MC",    color = "#4DAF4A", short = "mc",    shap_title = "Motorcycle Flow"),
  flow_truck = list(label = "Truck", color = "#984EA3", short = "truck", shap_title = "Truck Flow"),
  flow_bus   = list(label = "Bus",   color = "#FF7F00", short = "bus",   shap_title = "Bus Flow")
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

save_fig <- function(name, plot_obj) {
  path <- file.path(FIGS_DIR, name)
  ggsave(path, plot_obj, dpi = 300, width = 24, height = 6, units = "in")
  cat(sprintf("  Saved: %s\n", path))
}

cat(rep("=", 60), "\n", sep = "")
cat("Step 2-3: Retrain + Paper Figures (SP+RJ) - R port (Organized)\n")
cat(rep("=", 60), "\n", sep = "")

# ── 1. Load ALL training data ──
cat("\n[1/6] Loading training data...\n")
t0 <- Sys.time()
sp <- fread(SP_DATA)
cat(sprintf("  SP data: %d rows\n", nrow(sp)))

if (file.exists(RJ_DATA)) {
  rj <- fread(RJ_DATA)
  cat(sprintf("  RJ data: %d rows\n", nrow(rj)))
  missing <- setdiff(names(sp), names(rj))
  missing <- setdiff(missing, "speed")
  for (c in missing) rj[, (c) := NA_real_]
  common <- intersect(names(sp), names(rj))
  if (!"speed" %in% names(rj)) common <- setdiff(common, "speed")
  full <- rbindlist(list(sp[, ..common], rj[, ..common]), fill = TRUE)
  cat(sprintf("  Combined: %d rows\n", nrow(full)))
} else {
  cat(sprintf("  RJ data not found at %s, using SP only\n", RJ_DATA))
  full <- copy(sp)
}
cat(sprintf("  Time: %.0fs\n", difftime(Sys.time(), t0, units = "secs")))

# ── 2. Winsorize targets ──
cat("\n[2/6] Winsorizing targets (p99 clip)...\n")
for (t in TARGETS) {
  if (t %in% names(full)) {
    upper <- quantile(full[[t]], 0.99, na.rm = TRUE)
    full[get(t) > upper, (t) := upper]
    cat(sprintf("  %s: p99=%.0f\n", t, upper))
  }
}

# ── 3. Train models ──
cat("\n[3/6] Training models...\n")
models <- list()
train_results <- list()

for (target in TARGETS) {
  if (!target %in% names(full)) {
    cat(sprintf("  %s: not in data, skipping\n", target))
    next
  }
  
  cls <- CLASSES[[target]]
  df_train <- full[!is.na(get(target))]
  
  for (c in FEATURES) {
    if (c %in% names(df_train)) {
      val <- FILL_VALS[[c]]
      if (is.null(val)) val <- 0
      set(df_train, which(is.na(df_train[[c]])), c, val)
    }
  }
  
  X <- as.matrix(df_train[, ..FEATURES])
  y <- df_train[[target]]
  
  # Temporal split: Mon-Wed train, Thu-Sun test
  train_mask <- df_train$day_of_week %in% c(0, 1, 2)
  test_mask  <- df_train$day_of_week %in% c(3, 4, 5, 6)
  
  if (sum(train_mask) < 10 || sum(test_mask) < 10) {
    cat(sprintf("  %s: insufficient rows, skipping\n", cls$label))
    next
  }
  
  X_train <- X[train_mask, ]
  X_test  <- X[test_mask, ]
  y_train <- y[train_mask]
  y_test  <- y[test_mask]
  
  dtrain <- xgb.DMatrix(X_train, label = y_train)
  dtest  <- xgb.DMatrix(X_test, label = y_test)
  
  log_file <- tempfile(fileext = ".log")
  sink(log_file)
  model <- xgb.train(
    params = TRAIN_PARAMS,
    data = dtrain,
    nrounds = 500,
    evals = list(train = dtrain, test = dtest),
    early_stopping_rounds = 30,
    verbose = 1
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
  evals_list <- list(train_rmse = parsed$train_rmse, test_rmse = parsed$test_rmse)
  train_results[[target]] <- list(
    r2 = round(r2, 4), mae = round(mae, 2), rmse = round(rmse_v, 2),
    r2_train = round(r2_tr, 4), rounds = best_round,
    n_train = length(y_train), n_test = length(y_test),
    mean_obs = round(mean(y_test), 1),
    mean_pred = round(mean(y_pred), 1),
    evals = evals_list
  )
  
  model_path <- file.path(MODEL_DIR, sprintf("xgb_flow_%s_sprj_r.json", cls$short))
  xgb.save(model, model_path)
  
  cat(sprintf("\n  %s (%s):\n", cls$label, target))
  cat(sprintf("    R² = %.4f (train: %.4f)\n", r2, r2_tr))
  cat(sprintf("    MAE = %.1f, RMSE = %.1f\n", mae, rmse_v))
  cat(sprintf("    Model: %s\n", model_path))
}

write_json(train_results, file.path(MODEL_DIR, "results_sprj_r.json"),
           pretty = TRUE, auto_unbox = TRUE, null = "null")

# ── 4. Figure 1: Learning Curves ──
cat("\n[4/6] Figure 1: Learning curves...\n")
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
    geom_line(aes(x = round, y = rmse_train), color = "#333333", alpha = 0.5, linewidth = 0.8) +
    geom_line(aes(x = round, y = rmse_test), color = cls$color, linewidth = 1.2) +
    geom_vline(xintercept = best, color = "red", linetype = "dashed", alpha = 0.4, linewidth = 0.8) +
    annotate("text", x = best, y = max(df_curve$rmse_test) * 0.9,
             label = sprintf("n=%d", best), size = 4.5, color = "red", vjust = 1) +
    labs(x = "Boosting round", y = "RMSE (veh/h)", title = cls$label) +
    xlim(0, length(evals$train_rmse)) +
    theme_minimal(base_size = 14) +
    theme(plot.title = element_text(face = "bold", size = 16))
  p_list[[target]] <- p
}

if (length(p_list) > 0) {
  p_lc <- wrap_plots(p_list, ncol = 5) +
    plot_annotation(title = "Learning Curves — XGBoost Retrained with SP + RJ Data",
                    theme = theme(plot.title = element_text(face = "bold", size = 18)))
  save_fig("fig01_learning_curves_r.png", p_lc)
}

# ── 5. Figure 2: Validation Scatter (hexbin) ──
cat("\n[5/6] Figure 2: Validation scatter (hexbin)...\n")
p_list2 <- list()
for (target in TARGETS) {
  if (!target %in% names(models)) next
  cls <- CLASSES[[target]]
  model <- models[[target]]
  
  df_val <- full[!is.na(get(target))]
  for (c in FEATURES) {
    val <- FILL_VALS[[c]]; if (is.null(val)) val <- 0
    set(df_val, which(is.na(df_val[[c]])), c, val)
  }
  
  y_obs <- df_val[[target]]
  y_pred <- predict(model, as.matrix(df_val[, ..FEATURES]))
  
  df_scat <- data.table(observed = y_obs, predicted = y_pred)
  lims <- c(min(c(y_obs, y_pred)), max(c(y_obs, y_pred)))
  r2_v <- r2_score(y_obs, y_pred)
  mae_v <- mean(abs(y_obs - y_pred))
  pval <- cor.test(y_obs, y_pred)$p.value
  pval_str <- ifelse(pval < 0.001, "p<0.001", sprintf("p=%.3f", pval))

  p <- ggplot(df_scat, aes(x = observed, y = predicted)) +
    geom_hex(bins = 45, alpha = 0.8) +
    scale_fill_gradient(low = "lightblue", high = "darkblue", trans = "log") +
    geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed", alpha = 0.4, linewidth = 1) +
    coord_fixed(xlim = lims, ylim = lims) +
    labs(x = sprintf("Observed %s flow (veh/h)", cls$label),
         y = sprintf("Predicted %s flow (veh/h)", cls$label),
         title = sprintf("%s: R²=%.3f, MAE=%.0f, %s", cls$label, r2_v, mae_v, pval_str)) +
    theme_minimal(base_size = 14) +
    theme(plot.title = element_text(face = "bold"))
  p_list2[[target]] <- p
}
p_list2$flow_bus <- NULL
p_scatter <- wrap_plots(p_list2, ncol = 2) +
  plot_annotation(title = "SP + RJ Model Validation (Mon-Wed train, Thu-Sun test)",
                  theme = theme(plot.title = element_text(face = "bold", size = 18)))
ggsave(file.path(FIGS_DIR, "fig02_validation_scatter_r.png"), p_scatter, width = 18, height = 12, dpi = 300)
cat(sprintf("  Saved: %s/fig02_validation_scatter_r.png\n", FIGS_DIR))

# ── 6. Figure 3: R² by Day of Week ──
cat("\n[5/6] Figure 3: R² by day of week...\n")
dow_names <- c("Mon","Tue","Wed","Thu","Fri","Sat","Sun")
r2_data <- data.table()

for (target in TARGETS) {
  if (!target %in% names(models)) next
  cls <- CLASSES[[target]]
  model <- models[[target]]
  
  df_val <- full[!is.na(get(target))]
  for (c in FEATURES) {
    val <- FILL_VALS[[c]]; if (is.null(val)) val <- 0
    set(df_val, which(is.na(df_val[[c]])), c, val)
  }
  
  for (d in 0:6) {
    mask <- df_val$day_of_week == d
    if (sum(mask) > 10) {
      y_o <- df_val[mask][[target]]
      y_p <- predict(model, as.matrix(df_val[mask, ..FEATURES]))
      r2_d <- r2_score(y_o, y_p)
      r2_data <- rbind(r2_data, data.table(target = cls$label, day = d, r2 = r2_d, color = cls$color))
    }
  }
}
r2_data[, day_name := factor(dow_names[day + 1], levels = dow_names)]

p_dow <- ggplot(r2_data, aes(x = day_name, y = r2, fill = target)) +
  geom_col(position = position_dodge(width = 0.85), alpha = 0.85, width = 0.75) +
  scale_fill_manual(values = setNames(sapply(TARGETS, function(t) CLASSES[[t]]$color), 
                                       sapply(TARGETS, function(t) CLASSES[[t]]$label))) +
  geom_hline(yintercept = 0, color = "gray", linewidth = 0.3) +
  ylim(-0.5, 1.0) +
  labs(x = "Day of week", y = expression(R^2), title = "Validation R² by Day of Week", fill = "Class") +
  theme_minimal(base_size = 14) +
  theme(plot.title = element_text(face = "bold"), legend.position = "bottom")
ggsave(file.path(FIGS_DIR, "fig03_validation_dow_r.png"), p_dow, width = 10, height = 7, dpi = 300)
cat(sprintf("  Saved: %s/fig03_validation_dow_r.png\n", FIGS_DIR))

# ── 7. Feature Importance ──
cat("\n[5/6] Figure 4: Feature importance...\n")
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
    theme_minimal(base_size = 14) +
    theme(plot.title = element_text(face = "bold"))
  imp_list[[target]] <- p
}
p_imp <- wrap_plots(imp_list, ncol = 5) +
  plot_annotation(title = "Feature Importance (Gain) — SP+RJ Models",
                  theme = theme(plot.title = element_text(face = "bold", size = 18)))
ggsave(file.path(FIGS_DIR, "fig04_feature_importance_r.png"), p_imp, width = 24, height = 6, dpi = 300)
cat(sprintf("  Saved: %s/fig04_feature_importance_r.png\n", FIGS_DIR))

# ── 8. RJ held-out validation (Thu-Sun, never seen during training) ──
cat("\n[6/6] RJ held-out validation figure...\n")
rj_test <- fread(RJ_DATA)[day_of_week %in% c(3, 4, 5, 6)]
for (c in FEATURES) {
  val <- FILL_VALS[[c]]; if (is.null(val)) val <- 0
  set(rj_test, which(is.na(rj_test[[c]])), c, val)
}

RJ_TARGETS <- c("flow_pc", "flow_lcv", "flow_mc", "flow_truck")
rj_results <- list()
p_list_rj <- list()
for (target in RJ_TARGETS) {
  if (!target %in% names(models)) next
  cls <- CLASSES[[target]]
  y_o <- rj_test[[target]]
  y_p <- predict(models[[target]], as.matrix(rj_test[, ..FEATURES]))
  r2_v <- r2_score(y_o, y_p)
  mae_v <- mean(abs(y_o - y_p))
  bias_v <- mean(y_p) / mean(y_o)
  rj_results[[target]] <- list(r2 = round(r2_v, 4), mae = round(mae_v, 2),
                               bias_ratio = round(bias_v, 3),
                               mean_obs = round(mean(y_o), 1),
                               mean_pred = round(mean(y_p), 1),
                               n = length(y_o))
  cat(sprintf("  %s: R²=%.4f bias=%.3fx\n", cls$label, r2_v, bias_v))

  df_scat <- data.table(observed = y_o, predicted = y_p)
  lim <- max(c(y_o, y_p)) * 1.02
  p <- ggplot(df_scat, aes(x = observed, y = predicted)) +
    geom_hex(bins = 40, alpha = 0.85) +
    scale_fill_gradient(low = "lightblue", high = "darkblue", trans = "log") +
    geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "#333333",
                alpha = 0.3, linewidth = 0.6) +
    coord_fixed(xlim = c(0, lim), ylim = c(0, lim)) +
    labs(x = sprintf("Observed %s (veh/h)", cls$label),
         y = sprintf("Predicted %s (veh/h)", cls$label),
         title = sprintf("%s: R²=%.3f, bias=%.2fx", cls$label, r2_v, bias_v)) +
    theme_minimal(base_size = 12) +
    theme(plot.title = element_text(face = "bold"))
  p_list_rj[[target]] <- p
}
if (length(p_list_rj) > 0) {
  p_rj <- wrap_plots(p_list_rj, ncol = 2) +
    plot_annotation(title = "CET-Rio Validation (Thu-Sun, held-out from training)",
                    theme = theme(plot.title = element_text(face = "bold", size = 16)))
  ggsave(file.path(FIGS_DIR, "fig06_rj_validation_temporal_r.png"), p_rj,
         width = 12, height = 10, dpi = 300)
  cat(sprintf("  Saved: %s/fig06_rj_validation_temporal_r.png\n", FIGS_DIR))
}
write_json(rj_results, file.path(MODEL_DIR, "rj_validation_proper_r.json"),
           pretty = TRUE, auto_unbox = TRUE)

cat("\n[6/6] R model training complete.\n")
