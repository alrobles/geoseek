## Export the bundled xnicher rhipicephalus case-study data.
## 334 presences + 334 accessible-area background points (seeded sample
## of the 9,784 bundled M points) + their bio1/rh covariates.

load(system.file("data/rhipicephalus.rda", package = "xnicher",
                 mustWork = TRUE))

set.seed(20261011)
occ <- cbind(label = 1L, rhipicephalus[["coords_occ"]])
bg_idx <- sample(nrow(rhipicephalus[["coords_m"]]), nrow(occ))
bg <- cbind(label = 0L, rhipicephalus[["coords_m"]][bg_idx, ])
names(occ) <- names(bg) <- c("label", "lon", "lat")

pts <- rbind(occ, bg)
pts$id <- seq_len(nrow(pts))
write.csv(pts, "data/points.csv", row.names = FALSE)

env <- rbind(cbind(label = 1L, rhipicephalus[["env_occ"]]),
             cbind(label = 0L, rhipicephalus[["env_m"]][bg_idx, ]))
env$id <- seq_len(nrow(env))
write.csv(env, "data/env.csv", row.names = FALSE)

write.csv(cbind(rhipicephalus[["coords_m"]], rhipicephalus[["env_m"]]),
          "data/m_full.csv", row.names = FALSE)
