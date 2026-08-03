#!/usr/bin/env Rscript
# Step 1: Prepare CET-Rio training data
# Reads CET-Rio CSV, snaps to OSM+, splits by ARTESP proportions

library(data.table)
library(terra)
library(FNN)

# ── Dynamic Base Path Resolution ──
BASE <- Sys.getenv("TRAFFIC_BASE", unset = "/home/sibarra/.hermes/profiles/aire/home/hermes_vein/utfpr2025")
if (!dir.exists(BASE) || !dir.exists(file.path(BASE, "data"))) {
  # Try resolving relative to script location: final/R/01_prepare_cetrio.R -> final/R/ -> final/ -> xgboost_june_2026/ -> utfpr2025/
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
OSM_GPKG    <- file.path(BASE, "data/phase1/osm_se/SE_predicted.gpkg")
OUT_DIR     <- file.path(BASE, "xgboost_june_2026")
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

STATIC_FEATURES <- c("lanes","maxspeed","nightlight","pop_density",
                     "road_density_500m","road_density_1000m","road_density_5000m",
                     "highway_code","surface_code","oneway_code",
                     "lu_tree","lu_shrubland","lu_grassland","lu_cropland",
                     "lu_builtup","lu_bare","lu_water","lu_wetland",
                     "lu_mangroves","lu_moss")
FEATURES <- c(STATIC_FEATURES, "hour","day_of_week","is_weekend")

# ARTESP hourly class proportions (24h)
ARTESP_PCT <- list(
  flow_pc    = c(64.2,56.3,50.4,47.1,51.9,64.5,73.1,75.9,78.3,74.7,73.7,74.2,
                 75.1,75.0,74.4,74.7,76.0,77.7,78.3,78.4,76.7,75.3,73.9,70.1),
  flow_mc    = c(7.4,8.0,8.3,7.5,6.9,7.3,7.3,6.8,9.5,5.2,5.1,5.3,
                 5.4,5.4,5.4,5.3,5.5,6.2,6.0,5.3,5.4,5.8,6.4,6.5),
  flow_lcv   = c(19.9,25.0,28.9,31.8,28.9,19.7,13.8,12.1,10.5,14.1,14.8,14.4,
                 13.7,13.7,14.2,14.0,12.9,11.3,11.0,11.4,12.6,13.2,13.8,16.4),
  flow_truck = c(8.5,10.7,12.4,13.6,12.3,8.4,5.9,5.2,1.7,6.0,6.3,6.1,
                 5.8,5.8,6.0,6.0,5.5,4.8,4.7,4.9,5.4,5.7,5.9,7.0)
)
TARGETS <- names(ARTESP_PCT)

FILL_VALS <- list(lanes = 2, maxspeed = 60, nightlight = 0, pop_density = 0,
                  road_density_500m = 0, road_density_1000m = 0, road_density_5000m = 0,
                  highway_code = 4, surface_code = 1, oneway_code = 1)

cat(rep("=", 60), "\n", sep = "")
cat("Prepare CET-Rio Training Data (R port) (Organized)\n")
cat(rep("=", 60), "\n", sep = "")

# ── 1. Read CET-Rio ──
cat("\n[1/6] Reading CET-Rio CSV...\n")
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
df <- rj_raw[!is.na(vol) & !is.na(lat) & !is.na(lon)]
cat(sprintf("  %d obs | %d stations | %.0fs\n",
            nrow(df), uniqueN(df$loc), difftime(Sys.time(), t0, units = "secs")))

# ── 2. Station coords ──
cat("\n[2/6] Extracting station coordinates...\n")
# One coordinate pair per station (first occurrence), matching the Python pipeline.
# Using unique on (loc, lat, lon) would keep multiple variants per station and
# duplicate observations in the merge below.
stations <- unique(df[, .(loc, lat, lon)], by = "loc")
cat(sprintf("  %d stations\n", nrow(stations)))

# ── 3. Load OSM+ ──
cat("\n[3/6] Loading OSM+ features...\n")
t0 <- Sys.time()
osm <- vect(OSM_GPKG, layer = "SE_predicted")
osm <- project(osm, "EPSG:3857")
osm_dt <- as.data.table(osm)[, .SD, .SDcols = intersect(STATIC_FEATURES, names(osm))]
for (c in STATIC_FEATURES) {
  if (!c %in% names(osm_dt)) osm_dt[, (c) := NA_real_]
}
osm_cents <- centroids(osm)
osm_xy <- crds(osm_cents)
cat(sprintf("  %d OSM segments | %.0fs\n", nrow(osm_dt), difftime(Sys.time(), t0, units = "secs")))

# ── 4. Snap ──
cat("\n[4/6] Snapping stations to OSM+...\n")
st_pts <- vect(cbind(stations$lon, stations$lat), crs = "EPSG:4326")
st_pts <- project(st_pts, "EPSG:3857")
st_xy <- crds(st_pts)
nn <- get.knnx(osm_xy, st_xy, k = 1)
valid <- nn$nn.dist[, 1] <= 200
cat(sprintf("  Matched: %d/%d\n", sum(valid), nrow(stations)))

st_dt <- data.table(loc = stations$loc)
for (c in STATIC_FEATURES) st_dt[, (c) := NA_real_]
for (i in which(valid)) {
  idx <- nn$nn.index[i, 1]
  for (c in STATIC_FEATURES) {
    set(st_dt, i, c, osm_dt[[c]][idx])
  }
}
st_dt <- st_dt[valid]

# Fill missing values
for (c in names(FILL_VALS)) {
  set(st_dt, which(is.na(st_dt[[c]])), c, FILL_VALS[[c]])
}
cat(sprintf("  %d stations with features\n", nrow(st_dt)))

# ── 5. Merge features + split flow ──
cat("\n[5/6] Building training rows...\n")
t0 <- Sys.time()
feat_map <- st_dt
setkey(feat_map, loc)
obs_all <- merge(df[loc %in% feat_map$loc], feat_map, by = "loc", all.x = TRUE)
for (c in names(FILL_VALS)) {
  if (c %in% names(obs_all)) {
    set(obs_all, which(is.na(obs_all[[c]])), c, FILL_VALS[[c]])
  }
}

# Build training rows
rows_list <- vector("list", nrow(obs_all))
for (i in seq_len(nrow(obs_all))) {
  row <- obs_all[i]
  h <- as.integer(row$hour) %% 24
  total <- row$vol
  entry <- list()
  for (f in STATIC_FEATURES) entry[[f]] <- as.numeric(row[[f]])
  entry$hour <- as.numeric(row$hour)
  entry$day_of_week <- as.numeric(row$day_of_week)
  entry$is_weekend <- as.numeric(row$is_weekend)
  for (t in TARGETS) {
    entry[[t]] <- total * ARTESP_PCT[[t]][h + 1] / 100
  }
  rows_list[[i]] <- entry
}
rj_train <- rbindlist(rows_list)
cat(sprintf("  %d training rows from CET-Rio | %.0fs\n",
            nrow(rj_train), difftime(Sys.time(), t0, units = "secs")))

# ── 6. Save ──
cat("\n[6/6] Saving...\n")
fwrite(rj_train, file.path(OUT_DIR, "cetrio_training_data_r.csv"))
cat(sprintf("  Saved: %s/cetrio_training_data_r.csv\n", OUT_DIR))
cat(sprintf("  Columns: %s\n", paste(names(rj_train), collapse = ", ")))
for (t in TARGETS) {
  cat(sprintf("  %s: mean=%.1f max=%.0f\n", t, mean(rj_train[[t]]), max(rj_train[[t]])))
}
cat(sprintf("\nDone in %.0fs\n", difftime(Sys.time(), t0, units = "secs")))
