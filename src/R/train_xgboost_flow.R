#' Train SP-only XGBoost flow models (Mon-Wed)
train_flow_models <- function(data_path, model_dir, output_dir) {
  library(data.table); library(xgboost)
  dir.create(model_dir, showWarnings = FALSE, recursive = TRUE)
  dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

  STATIC <- c("lanes","maxspeed","nightlight","pop_density",
              "road_density_500m","road_density_1000m","road_density_5000m",
              "highway_code","surface_code","oneway_code",
              "lu_tree","lu_shrubland","lu_grassland","lu_cropland",
              "lu_builtup","lu_bare","lu_water","lu_wetland","lu_mangroves","lu_moss")
  TEMPORAL <- c("hour","day_of_week","is_weekend")
  FEATURES <- c(STATIC, TEMPORAL)

  CLASSES <- list(flow_pc = list(label = "PC", short = "pc"),
                  flow_lcv = list(label = "LCV", short = "lcv"),
                  flow_mc = list(label = "MC", short = "mc"),
                  flow_truck = list(label = "Truck", short = "truck"))

  TRAIN_PARAMS <- list(objective = "reg:squarederror", max_depth = 8,
                       learning_rate = 0.08, subsample = 0.8, colsample_bytree = 0.8,
                       min_child_weight = 5, gamma = 1, seed = 42, nthread = -1)

  fill_vals <- list(lanes = 2, maxspeed = 60, nightlight = 0, pop_density = 0,
                    road_density_500m = 0, road_density_1000m = 0, road_density_5000m = 0,
                    highway_code = 4, surface_code = 1, oneway_code = 1)
  for (c in STATIC[grep("^lu_", STATIC)]) fill_vals[[c]] <- 0
  r2_score <- function(y, yp) 1 - sum((y - yp)^2) / sum((y - mean(y))^2)

  sp <- fread(data_path); cat(sprintf("  Loaded %d rows\n", nrow(sp)))
  results <- list(); models <- list()

  for (target in names(CLASSES)) {
    cls <- CLASSES[[target]]; cat(sprintf("\n  Training %s...\n", cls$label))
    df <- copy(sp[!is.na(get(target))])
    for (c in FEATURES) { val <- fill_vals[[c]]; if (is.null(val)) val <- 0; set(df, which(is.na(df[[c]])), c, val) }
    X <- as.matrix(df[, ..FEATURES]); y <- df[[target]]
    train_mask <- df$day_of_week %in% c(0,1,2); test_mask <- df$day_of_week %in% c(3,4,5,6)
    dtrain <- xgb.DMatrix(X[train_mask,], label = y[train_mask])
    dtest  <- xgb.DMatrix(X[test_mask,], label = y[test_mask])
    model <- xgb.train(params = TRAIN_PARAMS, data = dtrain, nrounds = 500,
                       evals = list(train = dtrain, test = dtest), early_stopping_rounds = 30, verbose = 0)
    y_pred <- predict(model, dtest)
    r2 <- r2_score(y[test_mask], y_pred); mae <- mean(abs(y[test_mask] - y_pred))
    cat(sprintf("    R\u00b2=%.4f, MAE=%.2f, best_round=%d\n", r2, mae, model$best_iteration))
    model_path <- file.path(model_dir, sprintf("xgb_flow_%s_monwed.json", cls$short))
    xgb.save(model, model_path); cat(sprintf("    Saved: %s\n", model_path))
    results[[target]] <- list(r2 = r2, mae = mae, n_test = sum(test_mask), best_round = model$best_iteration)
    models[[target]] <- model
  }
  jsonlite::write_json(results, file.path(model_dir, "test_results.json"), auto_unbox = TRUE, pretty = TRUE)
  invisible(models)
}
