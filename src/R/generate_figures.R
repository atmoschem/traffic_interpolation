#' Paper figure helpers
library(data.table); library(ggplot2); library(patchwork)

plot_hexbin <- function(obs, pred, title="") {
  dt <- data.table(obs=obs,pred=pred)
  ggplot(dt,aes(obs,pred))+geom_hex(bins=50)+geom_abline(slope=1,intercept=0,color="red",linetype="dashed")+
    scale_fill_viridis_c(trans="log",option="inferno")+labs(title=title)+theme_minimal()
}
plot_spatial <- function(sf_data, col, title="") {
  ggplot(sf_data)+geom_sf(aes(color=.data[[col]]),size=0.3)+
    scale_color_viridis_c(option="inferno",direction=-1)+labs(title=title)+theme_void()
}
plot_error_hist <- function(errors, title="") {
  sw <- shapiro.test(sample(errors,min(length(errors),5000)))
  dt <- data.table(error=errors)
  ggplot(dt,aes(error))+geom_histogram(bins=80,fill="#2166AC",alpha=0.7,color="white")+
    annotate("text",x=Inf,y=Inf,hjust=1.1,vjust=1.5,
             label=sprintf("Shapiro-Wilk W=%.3f\np=%.2e",sw$statistic,sw$p.value),size=3)+
    labs(title=title)+theme_minimal()
}
