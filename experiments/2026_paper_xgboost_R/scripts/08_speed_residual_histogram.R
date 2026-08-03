#!/usr/bin/env Rscript
# Step 8: Speed residuals histogram — using R-trained model
# Manuscript Figure 6: Speed Model Residuals — Hourly
# Same style as flow error histograms (Figure 2).
# Rebuilds the combined hourly speed dataset (ARTESP + CET-Rio + TomTom)
# exactly as the R training script, then predicts with the R-trained model on
# the held-out 20% test set (same seeded split as R/03), so the metrics match
# the speed validation figures.

library(data.table)
library(terra)
library(xgboost)
library(FNN)
library(ggplot2)

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

# ── Paths ──
CETRIO_CSV  <- file.path(BASE, "data/cetrio/volumes e velocidades 2025-2024-2023-001.csv")
TRAINING_CSV <- file.path(BASE, "xgboost_june_2026/training_data_fixed_rd.csv")
OSM_GPKG     <- file.path(BASE, "data/phase1/osm_se/SE_predicted.gpkg")
TOMATO_GPKG  <- file.path(BASE, "data/flow_model/cetsptrans_with_tomtom.gpkg")
MODEL_DIR <- file.path(BASE, "xgboost_june_2026/models")
FIGS_DIR  <- file.path(BASE, "xgboost_june_2026/final/rfigs")
dir.create(FIGS_DIR, showWarnings = FALSE, recursive = TRUE)

# ── Features (same as R training script) ──
STATIC <- c("lanes","maxspeed","nightlight","pop_density",
            "road_density_500m","road_density_1000m","road_density_5000m",
            "highway_code","surface_code","oneway_code",
            "lu_tree","lu_shrubland","lu_grassland","lu_cropland",
            "lu_builtup","lu_bare","lu_water","lu_wetland",
            "lu_mangroves","lu_moss")
TEMPORAL <- c("hour","day_of_week","is_weekend")
FEATURES_TEMPORAL <- c(STATIC, TEMPORAL)
ALL_COLS <- c("speed","source", FEATURES_TEMPORAL)

FILL_VALS <- list(lanes = 2, maxspeed = 60, nightlight = 0, pop_density = 0,
                  road_density_500m = 0, road_density_1000m = 0, road_density_5000m = 0,
                  highway_code = 4, surface_code = 1, oneway_code = 1)

cat(rep("=", 60), "\n", sep = "")
cat("Speed Residuals Histogram (R model)\n")
cat(rep("=", 60), "\n", sep = "")

# ── 1. Load OSM+ ──
cat("\n[1] Loading OSM+ grid...\n")
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

# ── 2. ARTESP ──
cat("\n[2] Loading ARTESP hourly speed...\n")
t0 <- Sys.time()
artesp_raw <- fread(TRAINING_CSV)
artesp <- artesp_raw[source == "artesp" & !is.na(speed)]
artesp[, source := "ARTESP"]
cat(sprintf("  %d hourly obs | speed mean=%.1f\n", nrow(artesp), mean(artesp$speed, na.rm = TRUE)))

# ── 3. CET-Rio ──
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
cat(sprintf("  %d hourly obs | %d stations\n", nrow(rj), uniqueN(rj$loc)))

# Snap to OSM
cat("\n[4] Snapping CET-Rio to OSM+...\n")
t0 <- Sys.time()
stations <- unique(rj[, .(loc, lat, lon)], by = "loc")
st_pts <- vect(cbind(stations$lon, stations$lat), crs = "EPSG:4326")
st_pts <- project(st_pts, "EPSG:3857")
st_xy <- crds(st_pts)
nn <- get.knnx(osm_xy, st_xy, k = 1)
valid <- nn$nn.dist[, 1] <= 200
cat(sprintf("  Snapped: %d/%d stations\n", sum(valid), nrow(stations)))

st_dt <- data.table(loc = stations$loc)
for (c in STATIC) st_dt[, (c) := NA_real_]
for (i in which(valid)) {
  idx <- nn$nn.index[i, 1]
  for (c in STATIC) set(st_dt, i, c, osm_dt[[c]][idx])
}
st_dt_valid <- st_dt[valid]
rj <- rj[loc %in% st_dt_valid$loc]
rj <- merge(rj, st_dt_valid[, c("loc", STATIC), with = FALSE], by = "loc", all.x = TRUE)
setnames(rj, "spd", "speed")
rj[, source := "CET-Rio"]
for (c in names(FILL_VALS)) {
  set(rj, which(is.na(rj[[c]])), c, FILL_VALS[[c]])
}
cat(sprintf("  %d obs after snapping | %.0fs\n", nrow(rj), difftime(Sys.time(), t0, units = "secs")))

# ── 5. TomTom ──
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
cat(sprintf("  %d segments\n", nrow(tt_df)))

# ── 6. Combine ──
cat("\n[6] Building combined dataset...\n")
for (c in FEATURES_TEMPORAL) {
  if (c %in% names(artesp)) set(artesp, j = c, value = as.numeric(artesp[[c]]))
  if (c %in% names(rj))      set(rj, j = c, value = as.numeric(rj[[c]]))
  if (c %in% names(tt_df))   set(tt_df, j = c, value = as.numeric(tt_df[[c]]))
}
tt_df[, speed := as.numeric(speed)]

combined <- rbindlist(list(
  artesp[, .SD, .SDcols = intersect(ALL_COLS, names(artesp))],
  rj[, .SD, .SDcols = intersect(ALL_COLS, names(rj))],
  tt_df[, .SD, .SDcols = intersect(ALL_COLS, names(tt_df))]
), fill = TRUE)
for (c in STATIC) {
  if (c %in% names(combined)) {
    na_idx <- which(is.na(combined[[c]]))
    if (length(na_idx) > 0) set(combined, na_idx, c, FILL_VALS[[c]])
  }
}

cat(sprintf("  Total: %d rows\n", nrow(combined)))
for (s in unique(combined$source)) {
  sub <- combined[source == s]
  cat(sprintf("    %-8s: %7d rows, speed mean=%.1f\n", s, nrow(sub), mean(sub$speed, na.rm = TRUE)))
}

# ── 7. Load R model + predict on the held-out test set ──
# Reproduces the exact 20% stratified split from R/03_build_speed_hourly.R
# (same combined row order + set.seed(42)) so these metrics match the
# speed validation figures.
cat("\n[7] Loading R-trained model and predicting (held-out test set)...\n")
model <- xgb.load(file.path(MODEL_DIR, "xgb_speed_hourly_r.json"))
src_labels <- combined[, as.integer(factor(source, levels = c("ARTESP","CET-Rio","TomTom")))]
set.seed(42)
test_idx <- unlist(tapply(seq_len(nrow(combined)), src_labels, function(idx) {
  sample(idx, size = round(length(idx) * 0.2))
}))
test <- combined[test_idx]
X_all <- as.matrix(test[, ..FEATURES_TEMPORAL])
y_obs <- test$speed
y_pred <- predict(model, X_all)

residuals <- y_pred - y_obs

# Compute metrics
r2 <- 1 - sum((y_obs - y_pred)^2) / sum((y_obs - mean(y_obs))^2)
mae <- mean(abs(residuals))
rmse_val <- sqrt(mean(residuals^2))

# Shapiro-Wilk (max 5000)
set.seed(42)
n_shap <- min(5000, length(residuals))
idx_shap <- sample(length(residuals), n_shap)
sw <- shapiro.test(residuals[idx_shap])

cat(sprintf("  R² = %.4f\n", r2))
cat(sprintf("  MAE = %.1f km/h\n", mae))
cat(sprintf("  RMSE = %.1f km/h\n", rmse_val))
cat(sprintf("  Residual mean = %.1f, sd = %.1f\n", mean(residuals), sd(residuals)))
cat(sprintf("  Shapiro-Wilk: W=%.4f, p=%s\n", sw$statistic, format(sw$p.value, scientific = TRUE, digits = 3)))

# Per-source
cat("\n  Per-source:\n")
for (s in unique(test$source)) {
  mask <- test$source == s
  y_s <- y_obs[mask]; p_s <- y_pred[mask]
  r2_s <- 1 - sum((y_s - p_s)^2) / sum((y_s - mean(y_s))^2)
  mae_s <- mean(abs(p_s - y_s))
  bias_s <- mean(p_s) / mean(y_s)
  cat(sprintf("    %-8s: R²=%.3f MAE=%.1f bias=%.2fx n=%d\n", s, r2_s, mae_s, bias_s, sum(mask)))
}

# ── 8. Generate histogram (same style as flow error fig) ──
cat("\n[8] Generating histogram...\n")
df_err <- data.table(residual = residuals)

sw_label <- sprintf("Shapiro-Wilk\nW = %.4f\np = %s", sw$statistic, format(sw$p.value, scientific = TRUE, digits = 3))
stats_label <- sprintf("R² = %.3f\nMAE = %.1f km/h\nRMSE = %.1f km/h\nMean = %.1f\nSD = %.1f\nn = %d",
                       r2, mae, rmse_val, mean(residuals), sd(residuals), length(residuals))

p <- ggplot(df_err, aes(x = residual)) +
  geom_histogram(aes(y = after_stat(density)), bins = 80,
                 fill = "#2166AC", alpha = 0.7, color = "white", linewidth = 0.2) +
  stat_function(fun = dnorm, args = list(mean = mean(residuals), sd = sd(residuals)),
                color = "black", linewidth = 1, linetype = "dashed") +
  annotate("label", x = Inf, y = Inf, label = sw_label,
           hjust = 1.1, vjust = 1.5, size = 5, fontface = "bold",
           fill = "white", alpha = 0.85) +
  annotate("label", x = -Inf, y = Inf, label = stats_label,
           hjust = -0.1, vjust = 1.5, size = 4.5, fill = "white", alpha = 0.85) +
  labs(x = "Speed residual (predicted − observed, km/h)",
       y = "Density",
       title = sprintf("Speed Model Residuals — Hourly (R²=%.3f, MAE=%.1f km/h)", r2, mae),
       subtitle = "Held-out test set (20%), all sources") +
  xlim(-60, 60) +
  theme_minimal(base_size = 14) +
  theme(plot.title = element_text(face = "bold"),
        panel.grid.minor = element_blank())

fig_path <- file.path(FIGS_DIR, "fig08_speed_residual_histogram_r.png")
ggsave(fig_path, p, width = 10, height = 8, dpi = 300)
cat(sprintf("  Saved: %s\n", fig_path))
cat(sprintf("  Size: %.0f KB\n", file.info(fig_path)$size / 1024))
cat("Done.\n")
