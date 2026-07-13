#' Predict on Thu-Sun heldout, save obs vs pred
validate_flow_models <- function(data_path, model_dir, output_dir) {
  library(data.table); library(xgboost)
  dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

  STATIC <- c("lanes","maxspeed","nightlight","pop_density",
              "road_density_500m","road_density_1000m","road_density_5000m",
              "highway_code","surface_code","oneway_code",
              "lu_tree","lu_shrubland","lu_grassland","lu_cropland",
              "lu_builtup","lu_bare","lu_water","lu_wetland","lu_mangroves","lu_moss")
  TEMPORAL <- c("hour","day_of_week","is_weekend"); FEATURES <- c(STATIC, TEMPORAL)
  TARGETS <- c("flow_pc","flow_lcv","flow_mc","flow_truck")

  sp <- fread(data_path); test <- sp[day_of_week %in% c(3,4,5,6)]
  fill_vals <- list(lanes=2,maxspeed=60,nightlight=0,pop_density=0,
                    road_density_500m=0,road_density_1000m=0,road_density_5000m=0,
                    highway_code=4,surface_code=1,oneway_code=1)
  for (c in STATIC[grep("^lu_",STATIC)]) fill_vals[[c]] <- 0
  for (c in FEATURES) { val <- fill_vals[[c]]; if(is.null(val)) val<-0; set(test,which(is.na(test[[c]])),c,val) }
  X_test <- as.matrix(test[,..FEATURES]); out <- copy(test[,.(source,hour,day_of_week,is_weekend,speed)])
  r2_score <- function(y,yp) 1-sum((y-yp)^2)/sum((y-mean(y))^2)

  for (target in TARGETS) {
    cls_short <- sub("flow_","",target); mp <- file.path(model_dir,sprintf("xgb_flow_%s_monwed.json",cls_short))
    if(!file.exists(mp)) next; m <- xgb.load(mp)
    y_obs <- test[[target]]; y_pred <- predict(m, X_test)
    out[,(paste0("obs_",cls_short)):=y_obs]; out[,(paste0("pred_",cls_short)):=y_pred]
    cat(sprintf("  %s: R\u00b2=%.4f  n=%d\n",target,r2_score(y_obs,y_pred),length(y_obs)))
  }
  fwrite(out,file.path(output_dir,"sp_test_obs_vs_pred.csv"))
  cat(sprintf("  Saved: %s/sp_test_obs_vs_pred.csv (%d rows)\n",output_dir,nrow(out)))
}
