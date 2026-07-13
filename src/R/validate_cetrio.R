#' Cross-validate on CET-Rio
validate_cetrio <- function(rj_path, model_dir, output_dir) {
  library(data.table); library(xgboost)

  STATIC <- c("lanes","maxspeed","nightlight","pop_density",
              "road_density_500m","road_density_1000m","road_density_5000m",
              "highway_code","surface_code","oneway_code",
              "lu_tree","lu_shrubland","lu_grassland","lu_cropland",
              "lu_builtup","lu_bare","lu_water","lu_wetland","lu_mangroves","lu_moss")
  TEMPORAL <- c("hour","day_of_week","is_weekend"); FEATURES <- c(STATIC, TEMPORAL)

  rj <- fread(rj_path); cat(sprintf("  CET-Rio: %d rows\n",nrow(rj)))
  fill_vals <- list(lanes=2,maxspeed=60,nightlight=0,pop_density=0,
                    road_density_500m=0,road_density_1000m=0,road_density_5000m=0,
                    highway_code=4,surface_code=1,oneway_code=1)
  for (c in STATIC[grep("^lu_",STATIC)]) fill_vals[[c]] <- 0
  for (c in FEATURES) { val <- fill_vals[[c]]; if(is.null(val)) val<-0; set(rj,which(is.na(rj[[c]])),c,val) }
  X_rj <- as.matrix(rj[,..FEATURES]); out <- copy(rj[,.(hour,day_of_week,is_weekend)])

  TARGETS <- c("flow_pc","flow_lcv","flow_mc","flow_truck")
  total_obs <- rep(0,nrow(rj)); total_pred <- rep(0,nrow(rj))
  for (target in TARGETS) {
    cls_short <- sub("flow_","",target); mp <- file.path(model_dir,sprintf("xgb_flow_%s_monwed.json",cls_short))
    if(!file.exists(mp)) next; m <- xgb.load(mp)
    y_obs <- rj[[target]]; y_pred <- predict(m,X_rj)
    out[,(paste0("obs_",cls_short)):=y_obs]; out[,(paste0("pred_",cls_short)):=y_pred]
    total_obs <- total_obs + y_obs; total_pred <- total_pred + y_pred
  }
  out[,obs_total:=total_obs]; out[,pred_total:=total_pred]
  valid <- out[obs_total>0]
  r2 <- 1 - sum((valid$obs_total-valid$pred_total)^2)/sum((valid$obs_total-mean(valid$obs_total))^2)
  cat(sprintf("  CET-Rio total flow: R\u00b2=%.4f  n=%d\n",r2,nrow(valid)))
  fwrite(out,file.path(output_dir,"cetrio_obs_vs_pred.csv"))
  cat(sprintf("  Saved: %s/cetrio_obs_vs_pred.csv (%d rows)\n",output_dir,nrow(out)))
}
