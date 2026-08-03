#!/usr/bin/env Rscript
# Step 3: Build speed model using hourly data from ARTESP + CET-Rio + static TomTom
# R port: terra + data.table + xgboost + ggplot2/tidyterra

library(data.table)
library(terra)
library(xgboost)
library(FNN)
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

CETRIO_CSV  <- file.path(BASE, "data/cetrio/volumes e velocidades 2025-2024-2023-001.csv")
TRAINING_CSV <- file.path(BASE, "xgboost_june_2026/training_data_fixed_rd.csv")
OSM_GPKG     <- file.path(BASE, "data/phase1/osm_se/SE_predicted.gpkg")
TOMATO_GPKG  <- file.path(BASE, "data/flow_model/cetsptrans_with_tomtom.gpkg")
OUT_DIR  <- file.path(BASE, "xgboost_june_2026")
MODEL_DIR <- file.path(OUT_DIR, "models")
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
FEATURES_STATIC   <- STATIC
FEATURES_TEMPORAL <- c(STATIC, TEMPORAL)
ALL_COLS <- c("speed","source", FEATURES_TEMPORAL)

FILL_VALS <- list(lanes = 2, maxspeed = 60, nightlight = 0, pop_density = 0,
                  road_density_500m = 0, road_density_1000m = 0, road_density_5000m = 0,
                  highway_code = 4, surface_code = 1, oneway_code = 1)

PARAMS <- list(objective = "reg:squarederror", max_depth = 8, 
               learning_rate = 0.08, subsample = 0.8, colsample_bytree = 0.8,
               min_child_weight = 5, gamma = 1, seed = 42, nthread = -1)

cat(rep("=", 60), "\n", sep = "")
cat("Hourly Speed Model: ARTESP + CET-Rio hourly, TomTom static (R port) (Organized)\n")
cat(rep("=", 60), "\n", sep = "")

# ── 1. Load OSM+ feature grid ──
cat("\n[1] Loading OSM+ feature grid...\n")
t0 <- Sys.time()
osm <- vect(OSM_GPKG, layer = "SE_predicted")
osm <- project(osm, "EPSG:3857")
osm_dt <- as.data.table(osm)[, .SD, .SDcols = intersect(STATIC, names(osm))]
for (c in STATIC) {
  if (!c %in% names(osm_dt)) osm_dt[, (c) := NA_real_]
}
osm_cents <- centroids(osm)
osm_xy <- crds(osm_cents)
cat(sprintf("  %d segments | %.1fs\n", nrow(osm_dt), difftime(Sys.time(), t0, units = "secs")))

# ── 2. ARTESP hourly ──
cat("\n[2] Loading ARTESP hourly speed...\n")
t0 <- Sys.time()
artesp_raw <- fread(TRAINING_CSV)
artesp <- artesp_raw[source == "artesp" & !is.na(speed)]
artesp[, source := "ARTESP"]
cat(sprintf("  %d hourly obs | %d stations | %.0fs\n",
            nrow(artesp), uniqueN(artesp$snap_dist_m), difftime(Sys.time(), t0, units = "secs")))

# ── 3. CET-Rio hourly ──
cat(sprintf("\n[3] Loading CET-Rio hourly speed...\n"))
t0 <- Sys.time()
rj_raw <- fread(CETRIO_CSV, sep = ";", dec = ",", encoding = "UTF-8",
                header = FALSE, col.names = c("loc","ts","vol","spd","lat","lon"))
rj_raw[, ts_str := as.character(ts)]
rj_raw <- rj_raw[between(substr(ts_str, 1, 8), "20240304", "20240310")]
rj_raw[, `:=`(hour = as.integer(substr(ts_str, 9, 10)),
              date_str = substr(ts_str, 1, 8))]
rj_raw[, date := as.Date(date_str, format = "%Y%m%d")]
rj_raw[, day_of_week := as.integer(format(date, "%u")) - 1L]
rj_raw[, is_weekend := fifelse(day_of_week >= 5, 1L, 0L)]
rj_raw[, spd := as.numeric(gsub(",", ".", spd))]
rj <- rj_raw[!is.na(spd) & !is.na(lat) & !is.na(lon) & !is.na(hour) & !is.na(day_of_week)]
cat(sprintf("  %d hourly obs | %d stations | %.0fs\n",
            nrow(rj), uniqueN(rj$loc), difftime(Sys.time(), t0, units = "secs")))

# ── 4. Snap CET-Rio to OSM+ ──
cat("\n[4] Snapping CET-Rio to OSM+...\n")
t0 <- Sys.time()
# One coordinate pair per station (first occurrence), matching the Python pipeline
stations <- unique(rj[, .(loc, lat, lon)], by = "loc")
st_pts <- vect(cbind(stations$lon, stations$lat), crs = "EPSG:4326")
st_pts <- project(st_pts, "EPSG:3857")
st_xy <- crds(st_pts)
nn <- get.knnx(osm_xy, st_xy, k = 1)
valid <- nn$nn.dist[, 1] <= 200
cat(sprintf("  Snapped: %d/%d stations (within 200m)\n", sum(valid), nrow(stations)))

st_dt <- data.table(loc = stations$loc)
for (c in STATIC) st_dt[, (c) := NA_real_]
for (i in which(valid)) {
  idx <- nn$nn.index[i, 1]
  for (c in STATIC) {
    set(st_dt, i, c, osm_dt[[c]][idx])
  }
}
st_dt_valid <- st_dt[valid]
rj <- rj[loc %in% st_dt_valid$loc]
rj <- merge(rj, st_dt_valid[, c("loc", STATIC), with = FALSE], by = "loc", all.x = TRUE)
setnames(rj, "spd", "speed")
rj[, source := "CET-Rio"]
for (c in names(FILL_VALS)) {
  set(rj, which(is.na(rj[[c]])), c, FILL_VALS[[c]])
}
cat(sprintf("  %d hourly obs after snapping | %.0fs\n",
            nrow(rj), difftime(Sys.time(), t0, units = "secs")))

# ── 5. TomTom static speed ──
cat("\n[5] Loading TomTom static speed...\n")
t0 <- Sys.time()
tt_raw <- vect(TOMATO_GPKG)
tt <- project(tt_raw, "EPSG:3857")
tt_df <- as.data.frame(tt)[, c("avg_speed"), drop = FALSE]
tt_df <- tt_df[!is.na(tt_df$avg_speed), , drop = FALSE]

speed_idx <- which(!is.na(as.data.frame(tt)$avg_speed))
tt_pts <- centroids(tt[speed_idx])
tt_xy <- crds(tt_pts)
nn_tt <- get.knnx(osm_xy, tt_xy, k = 1)
valid_tt <- nn_tt$nn.dist[, 1] <= 200

tt_list <- vector("list", sum(valid_tt))
idx_tt <- which(valid_tt)
for (i in seq_along(idx_tt)) {
  j <- idx_tt[i]
  entry <- list(speed = tt_df$avg_speed[j], source = "TomTom")
  for (c in STATIC) entry[[c]] <- osm_dt[[c]][nn_tt$nn.index[j, 1]]
  for (t in TEMPORAL) entry[[t]] <- NA_real_
  tt_list[[i]] <- entry
}
tt_df <- rbindlist(tt_list, fill = TRUE)
cat(sprintf("  %d segments snapped to OSM | %.0fs\n", nrow(tt_df), difftime(Sys.time(), t0, units = "secs")))

# ── 6. Combine datasets ──
cat("\n[6] Building combined dataset...\n")
for (c in FEATURES_TEMPORAL) {
  if (c %in% names(artesp)) set(artesp, j = c, value = as.numeric(artesp[[c]]))
  if (c %in% names(rj))      set(rj, j = c, value = as.numeric(rj[[c]]))
  if (c %in% names(tt_df))   set(tt_df, j = c, value = as.numeric(tt_df[[c]]))
}
tt_df[, speed := as.numeric(speed)]

combined <- rbindlist(list(artesp[, .SD, .SDcols = intersect(ALL_COLS, names(artesp))],
                           rj[, .SD, .SDcols = intersect(ALL_COLS, names(rj))],
                           tt_df[, .SD, .SDcols = intersect(ALL_COLS, names(tt_df))]),
                      fill = TRUE)
for (c in STATIC) {
  if (c %in% names(combined)) {
    na_idx <- which(is.na(combined[[c]]))
    if (length(na_idx) > 0) {
      set(combined, na_idx, c, FILL_VALS[[c]])
    }
  }
}

cat(sprintf("\n  Combined dataset: %d rows\n", nrow(combined)))
for (s in unique(combined$source)) {
  sub <- combined[source == s]
  cat(sprintf("    %-8s: %7d rows, speed mean=%.1f\n", s, nrow(sub), mean(sub$speed, na.rm = TRUE)))
}

# ── 7. Train: STATIC ONLY vs TEMPORAL ──
cat("\n[7] Training XGBoost models...\n")

train_and_eval <- function(features, variant_name) {
  X <- as.matrix(combined[, ..features])
  y <- combined$speed
  
  src_labels <- combined[, as.integer(factor(source, levels = c("ARTESP","CET-Rio","TomTom")))]
  set.seed(42)
  test_ratio <- 0.2
  n <- length(y)
  test_idx <- unlist(tapply(seq_len(n), src_labels, function(idx) {
    sample(idx, size = round(length(idx) * test_ratio))
  }))
  train_idx <- setdiff(seq_len(n), test_idx)
  
  X_train <- X[train_idx, ]
  X_test  <- X[test_idx, ]
  y_train <- y[train_idx]
  y_test  <- y[test_idx]
  test_sources <- combined$source[test_idx]
  
  dtrain <- xgb.DMatrix(X_train, label = y_train)
  dtest  <- xgb.DMatrix(X_test, label = y_test)
  
  log_file <- tempfile(fileext = ".log")
  sink(log_file)
  model <- xgb.train(
    params = PARAMS,
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
  y_pred <- predict(model, dtest)
  
  r2  <- 1 - sum((y_test - y_pred)^2) / sum((y_test - mean(y_test))^2)
  mae <- mean(abs(y_test - y_pred))
  rmse_val <- sqrt(mean((y_test - y_pred)^2))
  
  cat(sprintf("\n  --- %s (%d features) ---\n", variant_name, length(features)))
  cat(sprintf("    R² = %.4f\n", r2))
  cat(sprintf("    MAE = %.1f km/h\n", mae))
  cat(sprintf("    RMSE = %.1f km/h\n", rmse_val))
  
  cat("    Per-source test metrics:\n")
  for (src in unique(combined$source)) {
    mask <- test_sources == src
    if (sum(mask) > 10) {
      y_s <- y_test[mask]
      p_s <- predict(model, X_test[mask, ])
      r2_s <- 1 - sum((y_s - p_s)^2) / sum((y_s - mean(y_s))^2)
      bias_s <- mean(p_s) / mean(y_s)
      cat(sprintf("      %-8s: R²=%.3f bias=%.2fx n=%d obs=%.1f\n",
                  src, r2_s, bias_s, sum(mask), mean(y_s)))
    }
  }
  
  list(model = model, r2 = r2, mae = mae, rmse = rmse_val,
       best_round = best_round, n_train = length(y_train),
       n_test = length(y_test), evals = list(train_rmse = parsed$train_rmse, test_rmse = parsed$test_rmse))
}

results <- list()
for (pair in list(c("static_only", FEATURES_STATIC), c("temporal", FEATURES_TEMPORAL))) {
  variant <- pair[1]
  features <- pair[-1]
  res <- train_and_eval(features, variant)
  results[[variant]] <- res
}

# ── 8. Save models ──
cat("\n[8] Saving models...\n")
model_path <- file.path(MODEL_DIR, "xgb_speed_hourly_r.json")
xgb.save(results[["temporal"]]$model, model_path)
cat(sprintf("  Hourly model: %s\n", model_path))

model_path_old <- file.path(MODEL_DIR, "xgb_speed_hourly_static_r.json")
xgb.save(results[["static_only"]]$model, model_path_old)
cat(sprintf("  Static-only (baseline): %s\n", model_path_old))

out_json <- list(
  static_only = list(r2 = round(results[["static_only"]]$r2, 4),
                     mae = round(results[["static_only"]]$mae, 1),
                     rmse = round(results[["static_only"]]$rmse, 1),
                     rounds = results[["static_only"]]$best_round,
                     n_train = results[["static_only"]]$n_train,
                     n_test = results[["static_only"]]$n_test),
  temporal = list(r2 = round(results[["temporal"]]$r2, 4),
                  mae = round(results[["temporal"]]$mae, 1),
                  rmse = round(results[["temporal"]]$rmse, 1),
                  rounds = results[["temporal"]]$best_round,
                  n_train = results[["temporal"]]$n_train,
                  n_test = results[["temporal"]]$n_test)
)
write_json(out_json, file.path(MODEL_DIR, "speed_hourly_results_r.json"), pretty = TRUE, auto_unbox = TRUE)

# ── 9. Figures ──
cat("\n[9] Generating figures...\n")

# Fig 1: Learning curves
p_list <- list()
for (variant in c("static_only", "temporal")) {
  evals <- results[[variant]]$evals
  if (is.null(evals$train_rmse) || length(evals$train_rmse) == 0) next
  df_curve <- data.table(round = seq_along(evals$train_rmse), rmse_train = evals$train_rmse, rmse_test = evals$test_rmse)
  best_r <- results[[variant]]$best_round
  r2_val <- results[[variant]]$r2
  label <- ifelse(variant == "static_only", "Static-only (20 features)", "Temporal (23 features)")
  
  p <- ggplot(df_curve) +
    geom_line(aes(x = round, y = rmse_train), color = "#999999", alpha = 0.4, linewidth = 0.7) +
    geom_line(aes(x = round, y = rmse_test), color = "#2166AC", linewidth = 1.2) +
    geom_vline(xintercept = best_r, color = "red", linetype = "dashed", alpha = 0.4, linewidth = 0.7) +
    labs(x = "Boosting round", y = "RMSE (km/h)", title = sprintf("%s\nR²=%.3f", label, r2_val)) +
    theme_minimal(base_size = 14) +
    theme(plot.title = element_text(face = "bold"))
  p_list[[variant]] <- p
}
if (length(p_list) > 0) {
  p_combined <- wrap_plots(p_list, ncol = 2) +
    plot_annotation(title = "Speed Model: Static vs Temporal Features",
                    theme = theme(plot.title = element_text(face = "bold", size = 13)))
  ggsave(file.path(FIGS_DIR, "fig_speed_hourly_learning_curves_r.png"), p_combined, width = 16, height = 7, dpi = 300)
  cat(sprintf("  Saved: %s/fig_speed_hourly_learning_curves_r.png\n", FIGS_DIR))
}

# Fig 2: Validation scatter — held-out test set (same seeded split as
# train_and_eval, so the plotted cloud matches the R² in the title)
model_tmp <- results[["temporal"]]$model
src_labels <- combined[, as.integer(factor(source, levels = c("ARTESP","CET-Rio","TomTom")))]
set.seed(42)
test_idx <- unlist(tapply(seq_len(nrow(combined)), src_labels, function(idx) {
  sample(idx, size = round(length(idx) * 0.2))
}))
test <- combined[test_idx]
X_test <- as.matrix(test[, ..FEATURES_TEMPORAL])
y_pred_test <- predict(model_tmp, X_test)
y_test <- test$speed
src_colors <- c("ARTESP" = "#2166AC", "CET-Rio" = "#D6604D", "TomTom" = "#4DAF4A")
df_scatter <- data.table(observed = y_test, predicted = y_pred_test, source = test$source)
lims <- c(0, max(c(y_test, y_pred_test), na.rm = TRUE) * 1.02)
p2 <- ggplot(df_scatter, aes(x = observed, y = predicted, color = source)) +
  geom_point(alpha = 0.2, size = 0.5, shape = 16) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "#333333", alpha = 0.3, linewidth = 0.6) +
  scale_color_manual(values = src_colors) +
  coord_fixed(xlim = lims, ylim = lims) +
  labs(x = "Observed speed (km/h)", y = "Predicted speed (km/h)",
       title = sprintf("Hourly Speed Model: R²=%.3f", results[["temporal"]]$r2), color = "Source") +
  theme_minimal(base_size = 14) +
  theme(plot.title = element_text(face = "bold"), legend.position = "bottom")
ggsave(file.path(FIGS_DIR, "fig_speed_hourly_validation_r.png"), p2, width = 9, height = 8, dpi = 300)
cat(sprintf("  Saved: %s/fig_speed_hourly_validation_r.png\n", FIGS_DIR))

cat("\nSpeed modeling in R completed.\n")
