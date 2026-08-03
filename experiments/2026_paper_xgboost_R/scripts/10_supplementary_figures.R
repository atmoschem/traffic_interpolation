#!/usr/bin/env Rscript
# Step 10: Supplementary Figures S1–S5 (R versions)
#
#   S1: ARTESP stations by mean traffic volume      (map, median flow)
#   S2: ARTESP 168-Hour Weekly Profiles             (5 panels, median + IQR ribbons)
#   S3: CET-Rio stations by mean traffic volume     (map, mean flow)
#   S4: CET-Rio 168-Hour Weekly Profiles            (2 panels, median + IQR ribbons)
#   S5: TomTom probe speed coverage                 (segment-centroid map, harmonic speed)
#
# Faithfully ported from the original exploratory scripts in scripts/:
#   S1 <- scripts/plot_artesp_station_map_v2.R   (identical to plot_artesp_station_map.R)
#   S2 <- scripts/plot_artesp_hourly_patterns_v2.R  (IQR-ribbon variant; v1 uses mean±SD)
#   S3 <- scripts/plot_cetrio_station_map.R
#   S4 <- scripts/plot_cetrio_hourly_patterns.R
#   S5 <- scripts/plot_tomtom_segments.R
#
# Data logic, titles, subtitles, color scales, and caption text are identical
# to the originals. Only changes: dynamic BASE resolution, output directory
# (final/rfigs/), output file names, the CET-Rio CSV path (the originals read
# from a temporary /tmp extraction; the file now lives in data/cetrio/), and
# per-section guards so one failure does not kill the remaining figures.

library(data.table)
library(ggplot2)

# PROJ database fix (needed for Figure S5): when this script is run with the
# environment's Rscript but without conda/micromamba activation, PROJ_DATA is
# unset and sf/GDAL cannot open proj.db, so st_transform/st_crs fail with
# "crs not found". Point PROJ at the environment's proj.db *before* loading
# sf, which locks the PROJ search path at load time. No-op when PROJ_DATA /
# PROJ_LIB are already set (activated env) or no proj.db is found.
r_home_proj <- normalizePath(file.path(R.home(), "..", "..", "share", "proj"),
                             mustWork = FALSE)
if (nchar(Sys.getenv("PROJ_DATA")) == 0 && nchar(Sys.getenv("PROJ_LIB")) == 0 &&
    file.exists(file.path(r_home_proj, "proj.db"))) {
  Sys.setenv(PROJ_DATA = r_home_proj)
}
library(sf)

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

FIGS_DIR <- file.path(BASE, "xgboost_june_2026/final/rfigs")
dir.create(FIGS_DIR, showWarnings = FALSE, recursive = TRUE)

# Input files
ARTESP_FILES <- file.path(BASE, "data/artesp",
                          c("CONTAGEM_SAT_2024.csv", "CONTAGEM_SAT_2025.csv"))
CETRIO <- file.path(BASE, "data/cetrio/volumes e velocidades 2025-2024-2023-001.csv")
GPKG   <- file.path(BASE, "data/flow_model/cetsptrans_with_tomtom.gpkg")

# ══════════════════════════════════════════════════════════════════════════
# Figure S1: ARTESP stations by mean traffic volume
# Ported from scripts/plot_artesp_station_map_v2.R
# ══════════════════════════════════════════════════════════════════════════
cat("\n══ Figure S1: ARTESP station map ══\n")
if (!all(file.exists(ARTESP_FILES))) {
  warning("S1: ARTESP input file(s) missing — skipping figure S1")
} else tryCatch({
  world_df <- map_data("world")
  map_data <- subset(world_df, region == "Brazil")

  stations <- rbindlist(lapply(ARTESP_FILES, function(f) {
    d <- fread(f, select = c("ID", "LATITUDE", "LONGITUDE", "QTD_MOTO", "QTD_PASSEIO", "QTD_COMERCIAL", "CONCESSIONARIA"))
    d[, total_flow := QTD_MOTO + QTD_PASSEIO + QTD_COMERCIAL]
    d
  }))
  stations <- stations[, .(lat = median(LATITUDE, na.rm = TRUE),
                           lon = mean(LONGITUDE, na.rm = TRUE),
                           mean_flow = quantile(total_flow, probs = 0.5, na.rm = TRUE),
                           concessionaria = names(sort(table(CONCESSIONARIA), decreasing = TRUE))[1]),
                        by = ID]
  stations <- stations[!is.na(lat) & !is.na(lon)]
  cat(sprintf("Stations: %d\n", nrow(stations)))

  p <- ggplot() +
    geom_polygon(data = map_data, aes(x = long, y = lat, group = group),
                 fill = "transparent", color = "black", linewidth = 0.3) +
    geom_point(data = stations, aes(x = lon, y = lat, color = mean_flow),
               alpha = 0.4) +
    scale_color_gradient(low = "#4575B4", high = "#D73027", name = "Median flow\n(veh/h)") +
    scale_size_continuous(range = c(1.5, 6), guide = "none") +
    coord_quickmap(xlim = c(-54, -44), ylim = c(-26, -20)) +
    labs(x = "Longitude", y = "Latitude",
         title = "ARTESP stations by mean traffic volume",
         subtitle = "446 stations | 2024-2025") +
    theme_minimal(base_size = 14) +
    theme(plot.title = element_text(face = "bold"),
          legend.position = "right")

  out <- file.path(FIGS_DIR, "fig_s1_artesp_station_map_r.png")
  ggsave(out, p, width = 10, height = 8, dpi = 300)
  cat(sprintf("Saved: %s\n", out))
  rm(stations, p); gc(verbose = FALSE)
}, error = function(e) warning(sprintf("S1 failed: %s — continuing", conditionMessage(e))))

# ══════════════════════════════════════════════════════════════════════════
# Figure S2: ARTESP 168-Hour Weekly Profiles (March 4-10, 2024)
# Ported from scripts/plot_artesp_hourly_patterns_v2.R (IQR-ribbon variant)
# ══════════════════════════════════════════════════════════════════════════
cat("\n══ Figure S2: ARTESP 168-hour weekly profiles ══\n")
if (!file.exists(ARTESP_FILES[1])) {
  warning("S2: ARTESP 2024 input file missing — skipping figure S2")
} else tryCatch({
  # Read ARTESP data
  message("Reading data...")
  artesp <- fread(ARTESP_FILES[1],
                  select = c("ID", "HORA", "QTD_MOTO", "QTD_PASSEIO", "QTD_COMERCIAL", "VEL_MEDIA", "DATA"))

  artesp[, hour := HORA]
  artesp[, time := as.POSIXct(paste0(DATA, " ", sprintf("%02d:00:00", hour)), tz = "UTC")]
  artesp[, date := as.Date(time)]

  # Filter for the specific week: March 4 to 10, 2024 (Monday to Sunday)
  dt <- artesp[date >= "2024-03-04" & date <= "2024-03-10"]

  # Apply the vehicle splits from the paper
  dt[, MC := QTD_MOTO]
  dt[, PC := QTD_PASSEIO]
  dt[, LCV := QTD_COMERCIAL * 0.70]
  dt[, Truck := QTD_COMERCIAL * 0.30]
  dt[, Speed := VEL_MEDIA]

  # Melt the dataset to facilitate ggplot faceting
  df <- melt(dt,
             id.vars = c("ID", "time"),
             measure.vars = c("PC", "MC", "LCV", "Truck", "Speed"))

  # Compute hourly statistics across all 446 stations
  message("Computing statistics...")
  stats <- df[, .(
    median_val = median(value, na.rm = TRUE),
    q25 = quantile(value, 0.25, na.rm = TRUE),
    q75 = quantile(value, 0.75, na.rm = TRUE)
  ), by = .(time, variable)]

  # Define custom labels and colors
  var_labels <- c(
    "PC" = "Passenger Cars (PC)",
    "MC" = "Motorcycles (MC)",
    "LCV" = "Light Commercial Vehicles (LCV)",
    "Truck" = "Trucks",
    "Speed" = "Average Speed (km/h)"
  )

  var_colors <- c(
    "PC" = "#2166AC",
    "MC" = "#4DAF4A",
    "LCV" = "#E69F00",
    "Truck" = "#984EA3",
    "Speed" = "#D6604D"
  )

  # Plotting the 168-hour profiles with IQR ribbons
  message("Generating plot...")
  p <- ggplot(stats, aes(x = time, y = median_val, color = variable, fill = variable)) +
    # IQR Ribbon
    geom_ribbon(aes(ymin = q25, ymax = q75), alpha = 0.2, color = NA) +
    # Median Line
    geom_line(linewidth = 0.8) +
    # Facet by variable with independent y-scales
    facet_wrap(~variable, scales = "free_y", ncol = 1,
               labeller = labeller(variable = var_labels)) +
    # Day boundaries (vertical lines at midnight of each day)
    geom_vline(xintercept = as.numeric(seq(as.POSIXct("2024-03-04 00:00:00", tz="UTC"),
                                           as.POSIXct("2024-03-11 00:00:00", tz="UTC"),
                                           by="1 day")),
               linetype = "dashed", color = "gray60", linewidth = 0.4) +
    scale_x_datetime(date_breaks = "1 day", date_labels = "%a\n%b %d", expand = c(0, 0)) +
    scale_color_manual(values = var_colors, guide = "none") +
    scale_fill_manual(values = var_colors, guide = "none") +
    labs(
      x = "Time (Monday March 4 to Sunday March 10, 2024)",
      y = "Value (hourly flow [veh/h] or speed [km/h])",
      title = "ARTESP 168-Hour Weekly Profiles (March 4-10, 2024)",
      subtitle = "Solid line shows the median across 446 stations; shaded area shows the interquartile range (25th to 75th percentile)."
    ) +
    theme_minimal(base_size = 11) +
    theme(
      plot.title = element_text(face = "bold", size = 13, margin = margin(b = 5)),
      plot.subtitle = element_text(size = 10, color = "gray30", margin = margin(b = 15)),
      strip.text = element_text(face = "bold", size = 10, hjust = 0),
      panel.grid.minor = element_blank(),
      panel.grid.major.x = element_blank(), # Remove default vertical grid lines to use our vlines instead
      panel.spacing = unit(1, "lines"),
      axis.title.x = element_text(margin = margin(t = 10)),
      axis.title.y = element_text(margin = margin(r = 10))
    )

  out <- file.path(FIGS_DIR, "fig_s2_artesp_weekly_profiles_r.png")
  ggsave(out, p, width = 10, height = 12, dpi = 300)
  cat(sprintf("Saved: %s\n", out))
  rm(artesp, dt, df, stats, p); gc(verbose = FALSE)
}, error = function(e) warning(sprintf("S2 failed: %s — continuing", conditionMessage(e))))

# ══════════════════════════════════════════════════════════════════════════
# Figure S3: CET-Rio stations by mean traffic volume
# Ported from scripts/plot_cetrio_station_map.R
# ══════════════════════════════════════════════════════════════════════════
cat("\n══ Figure S3: CET-Rio station map ══\n")
if (!file.exists(CETRIO)) {
  warning("S3: CET-Rio input CSV missing — skipping figure S3")
} else tryCatch({
  world_df <- map_data("world")
  map_data <- subset(world_df, region == "Brazil")

  cat("Reading CET-Rio CSV...\n")
  cet <- fread(CETRIO, sep = ";", encoding = "UTF-8", header = FALSE,
               col.names = c("ID", "timestamp", "volume", "speed", "lat", "lon"),
               select = 1:6,
               na.strings = "")

  # Filter to March 4-10, 2024
  cet[, ts_str := as.character(timestamp)]
  cet[, date_str := substr(ts_str, 1, 8)]
  cet <- cet[date_str >= "20240304" & date_str <= "20240310"]

  # Parse numeric — comma-to-dot
  cet[, volume := as.numeric(volume)]
  cet[, lat    := as.numeric(gsub(",", ".", lat))]
  cet[, lon    := as.numeric(gsub(",", ".", lon))]

  cat(sprintf("Observations: %d\n", nrow(cet)))
  cat(sprintf("Unique stations: %d\n", uniqueN(cet$ID)))

  # Aggregate per station — same structure as ARTESP script
  stations <- cet[!is.na(lat) & !is.na(lon),
                  .(lat = mean(lat, na.rm = TRUE),
                    lon = mean(lon, na.rm = TRUE),
                    mean_flow = mean(volume, na.rm = TRUE)),
                  by = ID]

  stations <- stations[!is.na(lat) & !is.na(lon)]
  cat(sprintf("Stations: %d\n", nrow(stations)))

  # ── Plot: SAME ggplot call as ARTESP ──
  p <- ggplot() +
    geom_polygon(data = map_data, aes(x = long, y = lat, group = group),
                 fill = "transparent", color = "black", linewidth = 0.3) +
    geom_point(data = stations, aes(x = lon, y = lat, color = mean_flow),
               alpha = 0.4) +
    scale_color_gradient(low = "#4575B4", high = "#D73027", name = "Mean flow\n(veh/h)") +
    scale_size_continuous(range = c(1.5, 6), guide = "none") +
    coord_quickmap(xlim = c(-44.0, -42.9), ylim = c(-23.3, -22.7)) +
    labs(x = "Longitude", y = "Latitude",
         title = "CET-Rio stations by mean traffic volume",
         subtitle = sprintf("%d stations | March 4-10, 2024", nrow(stations))) +
    theme_minimal(base_size = 14) +
    theme(plot.title = element_text(face = "bold"),
          legend.position = "right")

  out <- file.path(FIGS_DIR, "fig_s3_cetrio_station_map_r.png")
  ggsave(out, p, width = 10, height = 8, dpi = 300)
  cat(sprintf("Saved: %s\n", out))
  rm(cet, stations, p); gc(verbose = FALSE)
}, error = function(e) warning(sprintf("S3 failed: %s — continuing", conditionMessage(e))))

# ══════════════════════════════════════════════════════════════════════════
# Figure S4: CET-Rio 168-Hour Weekly Profiles (March 4-10, 2024)
# Ported from scripts/plot_cetrio_hourly_patterns.R
# ══════════════════════════════════════════════════════════════════════════
cat("\n══ Figure S4: CET-Rio 168-hour weekly profiles ══\n")
if (!file.exists(CETRIO)) {
  warning("S4: CET-Rio input CSV missing — skipping figure S4")
} else tryCatch({
  # ── Read CET-Rio ──
  message("Reading data...")
  cet <- fread(CETRIO, sep = ";", encoding = "UTF-8", header = FALSE,
               col.names = c("ID", "timestamp", "volume", "speed", "lat", "lon"),
               select = 1:6,
               na.strings = "")

  cet[, ts_str := as.character(timestamp)]
  cet[, hour   := as.integer(substr(ts_str, 9, 10))]
  cet[, date_str := substr(ts_str, 1, 8)]

  # Filter for the specific week: March 4 to 10, 2024
  cet <- cet[date_str >= "20240304" & date_str <= "20240310"]

  cet[, date  := as.Date(date_str, format = "%Y%m%d")]
  cet[, time  := as.POSIXct(sprintf("%s %02d:00:00", date_str, hour),
                            format = "%Y%m%d %H:%M:%S", tz = "UTC")]
  cet[, volume := as.numeric(volume)]
  cet[, speed  := as.numeric(gsub(",", ".", speed))]
  cet <- cet[!is.na(volume) & !is.na(speed)]

  # CET-Rio only has total flow + speed (no per-vehicle class splits)
  cet[, TotalFlow := volume]
  cet[, Speed     := speed]

  # Melt — same structure as ARTESP script
  df <- melt(cet,
             id.vars = c("ID", "time"),
             measure.vars = c("TotalFlow", "Speed"))

  # Compute hourly statistics across all stations
  message("Computing statistics...")
  stats <- df[, .(
    median_val = median(value, na.rm = TRUE),
    q25 = quantile(value, 0.25, na.rm = TRUE),
    q75 = quantile(value, 0.75, na.rm = TRUE)
  ), by = .(time, variable)]

  n_stations <- uniqueN(df$ID)

  # ── SAME labels & colors as ARTESP (matching TotalFlow→PC color, Speed same) ──
  var_labels <- c(
    "TotalFlow" = "Total Flow (veh/h)",
    "Speed"     = "Average Speed (km/h)"
  )
  var_colors <- c(
    "TotalFlow" = "#2166AC",
    "Speed"     = "#D6604D"
  )

  # ── Plot: SAME ggplot call as ARTESP ──
  message("Generating plot...")
  p <- ggplot(stats, aes(x = time, y = median_val, color = variable, fill = variable)) +
    geom_ribbon(aes(ymin = q25, ymax = q75), alpha = 0.2, color = NA) +
    geom_line(linewidth = 0.8) +
    facet_wrap(~variable, scales = "free_y", ncol = 1,
               labeller = labeller(variable = var_labels)) +
    geom_vline(xintercept = as.numeric(seq(as.POSIXct("2024-03-04 00:00:00", tz="UTC"),
                                           as.POSIXct("2024-03-11 00:00:00", tz="UTC"),
                                           by="1 day")),
               linetype = "dashed", color = "gray60", linewidth = 0.4) +
    scale_x_datetime(date_breaks = "1 day", date_labels = "%a\n%b %d", expand = c(0, 0)) +
    scale_color_manual(values = var_colors, guide = "none") +
    scale_fill_manual(values = var_colors, guide = "none") +
    labs(
      x = "Time (Monday March 4 to Sunday March 10, 2024)",
      y = "Value (hourly flow [veh/h] or speed [km/h])",
      title = "CET-Rio 168-Hour Weekly Profiles (March 4-10, 2024)",
      subtitle = sprintf("Solid line shows the median across %d stations; shaded area shows the interquartile range (25th to 75th percentile).", n_stations)
    ) +
    theme_minimal(base_size = 11) +
    theme(
      plot.title = element_text(face = "bold", size = 13, margin = margin(b = 5)),
      plot.subtitle = element_text(size = 10, color = "gray30", margin = margin(b = 15)),
      strip.text = element_text(face = "bold", size = 10, hjust = 0),
      panel.grid.minor = element_blank(),
      panel.grid.major.x = element_blank(),
      panel.spacing = unit(1, "lines"),
      axis.title.x = element_text(margin = margin(t = 10)),
      axis.title.y = element_text(margin = margin(r = 10))
    )

  out <- file.path(FIGS_DIR, "fig_s4_cetrio_weekly_profiles_r.png")
  ggsave(out, p, width = 10, height = 8, dpi = 300)
  cat(sprintf("Saved: %s\n", out))
  rm(cet, df, stats, p); gc(verbose = FALSE)
}, error = function(e) warning(sprintf("S4 failed: %s — continuing", conditionMessage(e))))

# ══════════════════════════════════════════════════════════════════════════
# Figure S5: TomTom probe speed coverage
# Ported from scripts/plot_tomtom_segments.R
# ══════════════════════════════════════════════════════════════════════════
cat("\n══ Figure S5: TomTom probe speed coverage ══\n")
if (!file.exists(GPKG)) {
  warning("S5: TomTom GPKG missing — skipping figure S5")
} else tryCatch({
  world_df <- map_data("world")
  map_data <- subset(world_df, region == "Brazil")

  cat("Loading TomTom segments...\n")
  seg <- read_sf(GPKG, quiet = TRUE)
  seg <- seg[!is.na(seg$harmonic_speed), ]
  seg <- st_transform(seg, 4326)

  coords <- st_coordinates(st_centroid(seg))
  segments <- data.table(
    lon = coords[, 1],
    lat = coords[, 2],
    harmonic_speed = seg$harmonic_speed
  )

  n_seg <- nrow(segments)
  cat(sprintf("Segments: %d\n", n_seg))
  cat(sprintf("Speed range: %.0f–%.0f km/h\n", min(segments$harmonic_speed), max(segments$harmonic_speed)))

  # ── Plot: SAME ggplot as ARTESP, zoomed to São Paulo city ──
  p <- ggplot() +
    geom_polygon(data = map_data, aes(x = long, y = lat, group = group),
                 fill = "transparent", color = "black", linewidth = 0.3) +
    geom_point(data = segments, aes(x = lon, y = lat, color = harmonic_speed),
               alpha = 0.35, size = 0.6) +
    scale_color_gradient(low = "#4575B4", high = "#D73027", name = "Harmonic speed\n(km/h)") +
    scale_size_continuous(range = c(1.5, 6), guide = "none") +
    coord_quickmap(xlim = c(-46.9, -46.35), ylim = c(-23.75, -23.35)) +
    labs(x = "Longitude", y = "Latitude",
         title = "TomTom probe speed coverage",
         subtitle = sprintf("%d segments | São Paulo metropolitan area", n_seg)) +
    theme_minimal(base_size = 14) +
    theme(plot.title = element_text(face = "bold"),
          legend.position = "right")

  out <- file.path(FIGS_DIR, "fig_s5_tomtom_coverage_r.png")
  ggsave(out, p, width = 10, height = 8, dpi = 300)
  cat(sprintf("Saved: %s\n", out))
  rm(seg, segments, p); gc(verbose = FALSE)
}, error = function(e) warning(sprintf("S5 failed: %s — continuing", conditionMessage(e))))

cat("\nDone. Outputs in:", FIGS_DIR, "\n")
