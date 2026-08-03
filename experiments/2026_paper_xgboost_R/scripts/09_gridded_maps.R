#!/usr/bin/env Rscript
# Step 9 (R): 2D histogram gridded maps for PC, LCV, MC, Truck, Bus + Speed
# Manuscript Figure 8 — "Predicted Traffic Flow & Speed — Weekday 8am"
# Using SP-only flow models + the hourly speed model at weekday peak (Tue 08:00).
# Inferno (flows) / viridis (speed): low=yellow, high=dark. State boundaries from GeoJSON.
# Port of scripts/plot_gridded_maps.R; mirrors final/python/10_gridded_maps.py.
library(data.table)
library(terra)
library(xgboost)
library(ggplot2)

# Helper: convert SpatVector to ggplot-compatible data.frame (no tidyterra needed)
spatvector_to_df <- function(sv) {
  g <- geom(sv)
  as.data.frame(g)
}

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

OSM_GPKG <- file.path(BASE, "data/phase1/osm_se/SE_predicted.gpkg")
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
FILL_VALS <- list(lanes=2, maxspeed=60, nightlight=0, pop_density=0,
                  road_density_500m=0, road_density_1000m=0, road_density_5000m=0,
                  highway_code=4, surface_code=1, oneway_code=1)
for (c in STATIC[grep("^lu_", STATIC)]) FILL_VALS[[c]] <- 0

# 'files' is ordered: primary choice first, fallbacks after (xgboost JSON is
# cross-language). Speed gets a fixed vmax and the viridis palette, matching
# the Python figure.
CLASSES <- list(
  PC    = list(files=c("xgb_flow_pc_sp_r.json",    "xgb_flow_pc_sp.json"),    label="Passenger Cars",  abbr="PC"),
  LCV   = list(files=c("xgb_flow_lcv_sp_r.json",   "xgb_flow_lcv_sp.json"),   label="Light Commercial", abbr="LCV"),
  MC    = list(files=c("xgb_flow_mc_sp_r.json",    "xgb_flow_mc_sp.json"),    label="Motorcycles",     abbr="MC"),
  Truck = list(files=c("xgb_flow_truck_sp_r.json", "xgb_flow_truck_sp.json"), label="Trucks",          abbr="Truck"),
  Bus   = list(files=c("xgb_flow_bus_sp_r.json",   "xgb_flow_bus_sp.json"),   label="Buses",           abbr="Bus"),
  Speed = list(files=c("xgb_speed_hourly_r.json",  "xgb_speed_hourly.json"),  label="Speed",           abbr="Speed",
               vmax=110, cmap="viridis", legend="Speed (km/h)")
)

cat(paste(rep("=",60),collapse=""),"\n")
cat("2D Histogram Gridded Maps — Flow Classes\n")
cat(paste(rep("=",60),collapse=""),"\n")

# ── 1. Load OSM grid ──
cat("\n[1] Loading OSM+ grid...\n")
t0 <- Sys.time()
osm <- vect(OSM_GPKG, layer="SE_predicted")
osm <- project(osm, "EPSG:3857")
osm_dt <- as.data.table(osm)
for (c in STATIC) {
  if (!c %in% names(osm_dt)) osm_dt[, (c) := NA_real_]
}
# Fill NAs
for (c in names(FILL_VALS)) {
  na_idx <- which(is.na(osm_dt[[c]]))
  if (length(na_idx) > 0) set(osm_dt, na_idx, c, FILL_VALS[[c]])
}
# Add temporal features for Tue 08:00
osm_dt[, hour := 8]
osm_dt[, day_of_week := 1]
osm_dt[, is_weekend := 0]

# Centroid coordinates for gridding (in EPSG:4326 for geographic grid)
cents <- centroids(osm)
cents_4326 <- project(cents, "EPSG:4326")
cents_xy <- crds(cents_4326)
cat(sprintf("  %d segments | %.1fs\n", nrow(osm_dt), difftime(Sys.time(), t0, units="secs")))

# ── 2. Load state boundaries from GeoJSON ──
cat("\n[2] Loading Brazil state boundaries...\n")
t0 <- Sys.time()
se_states <- c("São Paulo","Rio de Janeiro","Minas Gerais","Espírito Santo","Paraná","Santa Catarina")
states_v <- vect("/tmp/brazil_states.geojson")
se <- states_v[states_v$name %in% se_states, ]
se_df <- spatvector_to_df(se)
cat(sprintf("  %d states | %.1fs\n", nrow(se), difftime(Sys.time(), t0, units="secs")))

# ── 3. Predict all classes ──
cat("\n[3] Predicting flow for all classes...\n")
X <- as.matrix(osm_dt[, ..FEATURES])
results_list <- list()
for (cls_name in names(CLASSES)) {
  cls <- CLASSES[[cls_name]]
  model_path <- NULL
  for (f in cls$files) {
    cand <- file.path(MODEL_DIR, f)
    if (file.exists(cand)) { model_path <- cand; break }
  }
  if (is.null(model_path)) {
    cat(sprintf("  WARNING: no model file found for %s (%s), skipping\n",
                cls_name, paste(cls$files, collapse=", ")))
    next
  }
  model <- xgb.load(model_path)
  pred <- predict(model, X)
  pred[pred < 0] <- 0  # clip negatives
  if (!is.null(cls$vmax)) pred[pred > cls$vmax] <- cls$vmax  # Speed: cap like the Python figure
  results_list[[cls_name]] <- pred
  cat(sprintf("  %-10s: mean=%.0f max=%.0f %s [%s]\n", cls$abbr, mean(pred), max(pred),
              ifelse(cls_name == "Speed", "km/h", "veh/h"), basename(model_path)))
}

# ── 4. Create 2D histogram grid ──
cat("\n[4] Creating 2D histogram grids...\n")
RES <- 0.03  # ~3 km grid
lon <- cents_xy[, 1]
lat <- cents_xy[, 2]
xmin <- min(lon); xmax <- max(lon)
ymin <- min(lat); ymax <- max(lat)
nx <- floor((xmax - xmin) / RES) + 1
ny <- floor((ymax - ymin) / RES) + 1
cat(sprintf("  Grid: %d x %d cells (%.0f x %.0f km)\n", nx, ny, 
            (xmax-xmin)*111, (ymax-ymin)*111))

make_grid <- function(vals) {
  lon_bins <- seq(xmin, xmax, length.out=nx+1)
  lat_bins <- seq(ymin, ymax, length.out=ny+1)
  lon_idx <- findInterval(lon, lon_bins, rightmost.closed=TRUE)
  lat_idx <- findInterval(lat, lat_bins, rightmost.closed=TRUE)
  lon_idx[lon_idx < 1] <- 1; lon_idx[lon_idx > nx] <- nx
  lat_idx[lat_idx < 1] <- 1; lat_idx[lat_idx > ny] <- ny
  cell_idx <- (lat_idx - 1) * nx + lon_idx
  
  sum_vals <- tapply(vals, cell_idx, sum, na.rm=TRUE)
  count <- tapply(rep(1, length(vals)), cell_idx, sum, na.rm=TRUE)
  
  grid <- matrix(NA, nrow=nx, ncol=ny)
  cell_nums <- as.numeric(names(sum_vals))
  grid[cell_nums] <- sum_vals / count[names(sum_vals)]
  list(grid=grid, count=count)
}

# ── 5. Plot each class ──
cat("\n[5] Generating gridded maps...\n")
extent_4326 <- c(xmin, xmax, ymin, ymax)

# Common theme
map_theme <- theme_minimal(base_size=14) +
  theme(plot.title=element_text(face="bold", size=16),
        plot.subtitle=element_text(color="grey40", size=11),
        panel.grid=element_line(color="grey85", linewidth=0.15),
        legend.position="right",
        legend.key.height=unit(1.5, "cm"),
        legend.key.width=unit(0.5, "cm"),
        axis.title=element_blank(),
        plot.margin=margin(5,5,5,5))

for (cls_name in names(CLASSES)) {
  if (!cls_name %in% names(results_list)) next
  cls <- CLASSES[[cls_name]]
  vals <- results_list[[cls_name]]
  
  gridded <- make_grid(vals)
  grid <- gridded$grid
  count <- gridded$count
  
  # Fixed vmax for Speed (as the Python figure); 99th percentile otherwise
  vmax <- if (!is.null(cls$vmax)) cls$vmax else quantile(vals, 0.99, na.rm=TRUE)
  grid[grid > vmax] <- vmax
  cmap <- if (!is.null(cls$cmap)) cls$cmap else "inferno"
  leg  <- if (!is.null(cls$legend)) cls$legend else "Flow (veh/h)"
  
  # Convert to SpatRaster
  r <- rast(nrows=ny, ncols=nx, 
            xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
            crs="EPSG:4326")
  values(r) <- as.vector(grid[, ncol(grid):1])  # terra cell order: lat rows ymax→ymin, lon cols xmin→xmax
  
  # Convert to data.frame for ggplot
  r_df <- as.data.frame(r, xy=TRUE, na.rm=TRUE)
  names(r_df)[3] <- "flow"
  
  # Cap display
  r_df$flow[r_df$flow > vmax] <- vmax
  r_df <- r_df[r_df$flow > 0, ]
  
  p <- ggplot() +
    geom_raster(data=r_df, aes(x=x, y=y, fill=flow)) +
    geom_path(data=se_df, aes(x=x, y=y, group=geom), color="#333333", linewidth=0.3) +
    scale_fill_viridis_c(option=cmap, direction=-1,
                          name=leg,
                          limits=c(0, vmax),
                          oob=scales::squish,
                          na.value=NA) +
    coord_fixed(xlim=c(-54, -39), ylim=c(-27, -14),
             expand=FALSE) +
    labs(title=paste(cls$label, "— Weekday Peak (Tue 08:00)"),
         subtitle=sprintf("XGBoost SP Model — 2D histogram (%.0f km grid)", RES*111)) +
    map_theme
  
  fig_name <- sprintf("fig_gridded_%s.png", tolower(cls$abbr))
  fig_path <- file.path(FIGS_DIR, fig_name)
  ggsave(fig_path, p, width=9, height=8, dpi=300)
  cat(sprintf("  %-10s: %s (%.0f KB)\n", cls$abbr, fig_name, file.info(fig_path)$size/1024))
}

# ── 6. Combined 6-panel figure ──
cat("\n[6] Generating combined 6-panel figure...\n")
panel_list <- list()
for (cls_name in names(CLASSES)) {
  if (!cls_name %in% names(results_list)) next
  cls <- CLASSES[[cls_name]]
  vals <- results_list[[cls_name]]
  
  gridded <- make_grid(vals)
  grid <- gridded$grid
  vmax <- if (!is.null(cls$vmax)) cls$vmax else quantile(vals, 0.99, na.rm=TRUE)
  grid[grid > vmax] <- vmax
  cmap <- if (!is.null(cls$cmap)) cls$cmap else "inferno"
  
  r <- rast(nrows=ny, ncols=nx, 
            xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
            crs="EPSG:4326")
  values(r) <- as.vector(grid[, ncol(grid):1])  # terra cell order: lat rows ymax→ymin, lon cols xmin→xmax
  
  r_df <- as.data.frame(r, xy=TRUE, na.rm=TRUE)
  names(r_df)[3] <- "flow"
  r_df$flow[r_df$flow > vmax] <- vmax
  r_df <- r_df[r_df$flow > 0, ]
  
  p_panel <- ggplot() +
    geom_raster(data=r_df, aes(x=x, y=y, fill=flow)) +
    geom_path(data=se_df, aes(x=x, y=y, group=geom), color="#333333", linewidth=0.25) +
    scale_fill_viridis_c(option=cmap, direction=-1,
                          name=NULL,
                          limits=c(0, vmax),
                          oob=scales::squish,
                          na.value=NA) +
    coord_fixed(xlim=c(-54, -39), ylim=c(-27, -14),
             expand=FALSE) +
    labs(title=cls$abbr,
         subtitle=sprintf("Mean %.0f | Max %.0f", mean(vals), vmax)) +
    theme_minimal(base_size=12) +
    theme(plot.title=element_text(face="bold", size=14),
          plot.subtitle=element_text(color="grey40", size=10),
          panel.grid=element_line(color="grey85", linewidth=0.1),
          legend.position="right",
          legend.key.height=unit(0.8, "cm"),
          legend.key.width=unit(0.3, "cm"),
          axis.title=element_blank(),
          axis.text=element_blank(),
          plot.margin=margin(2,2,2,2))
  
  panel_list[[cls_name]] <- p_panel
}

# Arrange with patchwork
library(patchwork)
p_combined <- wrap_plots(panel_list, ncol=2) +
  plot_annotation(
    title="Predicted Traffic Flow & Speed — Weekday Peak (Tue 08:00)",
    subtitle="XGBoost SP Models — 2D Histogram Grid (~3 km resolution)",
    theme=theme(plot.title=element_text(face="bold", size=18),
                plot.subtitle=element_text(color="grey40", size=13)))

fig_path <- file.path(FIGS_DIR, "fig09_gridded_maps_r.png")
ggsave(fig_path, p_combined, width=14, height=18, dpi=300)
cat(sprintf("  Combined: %s (%.0f KB)\n", fig_path, file.info(fig_path)$size/1024))

cat("\nDone.\n")
