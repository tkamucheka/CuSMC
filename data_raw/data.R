library(usethis)
y_t <- as.matrix(read.csv("y_t.csv"))
y_sim <- t(y_t)
usethis::use_data(y_sim, overwrite = TRUE, compress = 'xz')
