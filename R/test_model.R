# R modelio testai ir nepriklausomas Monte Carlo patikrinimas.
# Paleidimas iš repozitorijos šaknies:  Rscript R/test_model.R

source("R/model.R")

HCP <- matrix(c(-14.5, 2.45, 1.50,  -13.5, 2.30, 1.62,  -12.5, 2.15, 1.70,  -11.5, 1.95, 1.85,
                -10.5, 1.85, 1.95,   -9.5, 1.70, 2.15,   -8.5, 1.62, 2.30,   -7.5, 1.50, 2.45),
              ncol = 3, byrow = TRUE)
TOT <- matrix(c(160.5, 1.55, 2.35,  163.5, 1.75, 2.05,  165.5, 1.90, 1.90,
                167.5, 2.05, 1.75,  170.5, 2.35, 1.55), ncol = 3, byrow = TRUE)

near <- function(x, y, eps, msg) if (abs(x - y) >= eps) stop(sprintf("%s: %g != %g", msg, x, y))

r <- row_calc("hcp", -11.5, 1.95, 1.85)
near(r$pa, 0.4868, 1e-4, "P(namai -11.5)")
near(r$margin, 0.0534, 1e-4, "marža")

F <- list(hcp = fit_market("hcp", HCP), tot = fit_market("tot", TOT))
near(F$hcp$mu, 11.0, 0.05, "vid. skirtumas")
near(F$hcp$sigma, 11.31, 0.05, "σ skirtumo")
near(p_over(F$hcp, 0), 0.8346, 1e-3, "P(Fenerbahce laimi)")
near(F$tot$mu, 165.5, 0.05, "vid. suma")

M <- score_model(F)
near(M$mh, 88.25, 0.05, "namų vidurkis")
near(M$ma, 77.25, 0.05, "svečių vidurkis")
near(M$rho, 0.491, 0.01, "koreliacija")
near(sum(grid_cells(M)$p), 1, 1e-9, "tinklelio suma")

p_away <- condition_probability(M, list(list(v = "diff", op = "<", x = 0)))
p_home <- condition_probability(M, list(list(v = "diff", op = ">", x = 0)))
p_draw <- condition_probability(M, list(list(v = "diff", op = "=", x = 0)))
near(p_away + p_home + p_draw, 1, 1e-9, "laimi + pralaimi + lygiosios")
conds <- list(list(v = "home", op = ">", x = 75), list(v = "away", op = ">", x = 75), list(v = "diff", op = "<", x = 0))
p_both <- condition_probability(M, conds)
near(p_both, 0.1018, 2e-3, "abi > 75 ir laimi Partizan (sutampa su JS ir Python)")

# Bendras modelis: koeficientai iš žinomų parametrų (μh 88, μa 77, σh 11, σa 9, ρ 0.3), marža 5 %.
gen <- function(key, mu, sd, lines) {
  k <- if (key == "hcp") -lines else lines
  p <- 1 - pnorm((k - mu) / sd)
  cbind(lines, 1 / (p * 1.025), 1 / ((1 - p) * 1.025))
}
cv <- 0.3 * 11 * 9
G <- list(
  hcp  = fit_market("hcp",  gen("hcp", 11, sqrt(11^2 + 9^2 - 2 * cv), c(-16.5, -13.5, -10.5, -7.5, -4.5))),
  tot  = fit_market("tot",  gen("tot", 165, sqrt(11^2 + 9^2 + 2 * cv), c(157.5, 162.5, 165.5, 168.5, 173.5))),
  home = fit_market("home", gen("home", 88, 11, c(82.5, 86.5, 88.5, 90.5, 94.5))),
  away = fit_market("away", gen("away", 77, 9, c(72.5, 75.5, 77.5, 79.5, 82.5))))
GM <- score_model(G)
near(GM$mh, 88, 0.05, "bendras: namų vidurkis"); near(GM$ma, 77, 0.05, "bendras: svečių vidurkis")
near(GM$sh, 11, 0.05, "bendras: σ namų"); near(GM$sa, 9, 0.05, "bendras: σ svečių")
near(GM$rho, 0.3, 0.005, "bendras: ρ")
near(GM$rho_tot, 0.3, 0.005, "ρ pagal sumą"); near(GM$rho_hcp, 0.3, 0.005, "ρ pagal handikapą")
stopifnot(GM$src == "handikapas + suma + namų taškai + svečių taškai")
near(score_model(G[c("home", "away")])$rho, 0, 1e-6, "tik komandų taškai: ρ = 0")
three <- score_model(G[c("hcp", "tot", "home")])
near(three$sa, 9, 0.05, "be svečių rinkos: σ svečių"); near(three$rho, 0.3, 0.005, "be svečių rinkos: ρ")
stopifnot(is.null(score_model(G["hcp"])))

# Nepriklausomas patikrinimas: 2 mln. dvimačio normaliojo imčių per Cholesky, suapvalinta iki sveikų taškų
set.seed(1)
n <- 2e6
z1 <- rnorm(n); z2 <- rnorm(n)
hs <- round(M$mh + M$sh * z1)
as <- round(M$ma + M$sa * (M$rho * z1 + sqrt(1 - M$rho^2) * z2))
mc <- mean(hs > 75 & as > 75 & hs < as)
near(p_both, mc, 1.5e-3, "sąlyga prieš Monte Carlo")
cat(sprintf("Monte Carlo palyginimas: sąlyga %.2f%% (modelis) vs %.2f%% (Monte Carlo)\n", p_both * 100, mc * 100))

# Kėlinio dalių modelis: iš žinomų dalių sugeneruotos kėlinio linijos -> dalys atkuriamos.
FTm <- list(mh = 88, ma = 78, sh = 10, sa = 10, rho = 0.3)
truth <- share_moments(FTm, 0.27, 0.23, 80, 0.3)
pts <- list()
for (mk in list(list("hcp", c(-3.5, -1.5, 0.5, 2.5)), list("tot", c(38.5, 41.5, 44.5)), list("home", c(21.5, 24.5)), list("away", c(16.5, 18.5))))
  for (k in mk[[2]]) pts[[length(pts) + 1]] <- list(mk[[1]], k, period_p_over(truth, mk[[1]], k))
Rq <- period_share_model(FTm, pts)
near(Rq$share$h, 0.27, 1e-3, "namų kėlinio dalis")
near(Rq$share$a, 0.23, 1e-3, "svečių kėlinio dalis")
near(Rq$share$c, 80, 2, "koncentracija c")
D <- period_share_model(FTm, list())
near(D$mh, 22, 1e-12, "numatyta kėlinio dalis")
near(period_share_model(FTm, list(), kappa = 0)$rho, 0.0625 * 30 / (D$sh * D$sa), 1e-9, "κ = 0")
near(D$rho, (0.0625 * 30 + 0.3 * 0.1875 / 51 * (30 + 88 * 78)) / (D$sh * D$sa), 1e-9, "κ formulė")


# κ įvertinimas iš koeficientų: du kėliniai sugeneruoti su κ = 0.45; trečias (viena linija) netinka.
genq <- function(mh, ma, cc, k) {
  t <- share_moments(FTm, mh, ma, cc, k); out <- list()
  for (mk in list(list("hcp", c(-3.5, -1.5, 0.5, 2.5)), list("tot", c(37.5, 39.5, 41.5, 43.5, 45.5)), list("home", c(20.5, 22.5))))
    for (kk in mk[[2]]) out[[length(out) + 1]] <- list(mk[[1]], kk, period_p_over(t, mk[[1]], kk))
  out
}
rk <- fit_kappa(FTm, list(genq(0.26, 0.24, 60, 0.45), genq(0.24, 0.25, 70, 0.45), list(list("tot", 40.5, 0.5))))
near(rk$kappa, 0.45, 0.01, "κ iš koeficientų")
stopifnot(rk$quarters == 2, rk$lines == 22, !is.null(rk$se), rk$se < 0.05)
stopifnot(is.null(fit_kappa(FTm, list(list(list("tot", 40.5, 0.5), list("tot", 42.5, 0.4))))))

cat("Visi R testai praėjo.\n")
