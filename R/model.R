# Skaičiavimų modelis R kalba (tas pats kaip app/model.js ir python/model.py).
#
# Koeficientai -> tikimybės be maržos, normaliojo skirstinio įvertis kiekvienai rinkai,
# abiejų komandų taškų modelis su koreliacija, rezultatų tinklelis ir sąlygų tikimybė.
# Naudojamas tik bazinis R.
#
# Paleidimas iš repozitorijos šaknies:
#   Rscript R/model.R --hcp data/hcp.csv --tot data/tot.csv \
#     --home-name Fenerbahce --away-name Partizan \
#     --cond "home>75" --cond "away>75" --cond "diff<0"
#
# Naudojimas iš kito R kodo:  source("R/model.R")
#
# CSV failuose naudojami pirmi trys skaitiniai stulpeliai: linija, koeficientas A, koeficientas B.
# Handikapui linija yra namų komandos handikapas (pvz. -11.5), A = namai, B = svečiai.
# Sumoms ir komandų taškams A = daugiau, B = mažiau.

SIGMA0 <- c(hcp = 12, tot = 16, home = 10, away = 10)

# ---------- koeficientai -> tikimybės ----------
row_calc <- function(market, line, a, b) {
  if (!is.finite(line) || !(a > 1 && b > 1)) return(NULL)
  ia <- 1 / a; ib <- 1 / b; book <- ia + ib
  list(k = if (market == "hcp") -line else line,   # slenkstis: X > k
       pa = ia / book, pb = ib / book, margin = book - 1)
}

# ---------- normaliojo skirstinio įvertis ----------
# P(X > k) = 1 - Φ((k - μ)/σ): z = Φ⁻¹(1 - p) tiesiškai priklauso nuo k, nuolydis = 1/σ.
fit_market <- function(market, rows, sigma0 = NULL) {
  calc <- Filter(Negate(is.null), lapply(seq_len(nrow(rows)), function(i)
    row_calc(market, rows[i, 1], rows[i, 2], rows[i, 3])))
  if (!length(calc)) return(NULL)
  k <- vapply(calc, `[[`, 0, "k")
  z <- qnorm(1 - pmin(0.99, pmax(0.01, vapply(calc, `[[`, 0, "pa"))))
  n <- length(k)
  if (n >= 2) {
    sxx <- sum((k - mean(k))^2)
    if (sxx > 0) {
      slope <- sum((k - mean(k)) * (z - mean(z))) / sxx
      if (slope > 0.01) {
        sigma <- 1 / slope
        return(list(mu = mean(k) - mean(z) * sigma, sigma = sigma, n = n, assumed = FALSE))
      }
    }
  }
  sigma <- if (is.null(sigma0)) SIGMA0[[market]] else sigma0
  list(mu = mean(k - z * sigma), sigma = sigma, n = n, assumed = TRUE)
}

p_over <- function(f, k) 1 - pnorm((k - f$mu) / f$sigma)

# ---------- rezultato modelis ----------
# Namų (H) ir svečių (A) taškai – dvimatis normalusis skirstinys su μh, μa, σh, σa, ρ.
# Kiekviena rinka duoda vidurkį ir dispersiją, tiesiškai priklausančius nuo nežinomųjų:
#   handikapas  H − A:  μh − μa,   σh² + σa² − 2c        (c = ρ·σh·σa)
#   suma        H + A:  μh + μa,   σh² + σa² + 2c
#   namų taškai H:      μh,        σh²
#   svečių      A:      μa,        σa²
# Visos turimos rinkos sujungiamos svertiniu mažiausių kvadratų metodu (vidurkiai: svoris n/σ²,
# dispersijos: n/σ⁴). Labai silpnos prielaidos σh = σa ir ρ = 0 lemia tik tai, ko rinkos neapibrėžia.
SCORE_ROWS <- list(
  hcp  = list(m = c(1, -1), v = c(1, 1, -2), name = "handikapas"),
  tot  = list(m = c(1, 1),  v = c(1, 1, 2),  name = "suma"),
  home = list(m = c(1, 0),  v = c(1, 0, 0),  name = "namų taškai"),
  away = list(m = c(0, 1),  v = c(0, 1, 0),  name = "svečių taškai"))

.wls <- function(X, y, w) {
  A <- crossprod(X, w * X)
  if (rcond(A) < 1e-12) return(NULL)
  as.vector(solve(A, crossprod(X, w * y)))
}

score_model <- function(F) {
  keys <- Filter(function(k) !is.null(F[[k]]), names(SCORE_ROWS))
  if (!length(keys)) return(NULL)
  num <- function(k, f) as.numeric(F[[k]][[f]])
  sig2 <- vapply(keys, function(k) num(k, "sigma")^2, 0)
  n <- vapply(keys, function(k) num(k, "n"), 0)
  assumed <- vapply(keys, function(k) isTRUE(F[[k]]$assumed), TRUE)
  Xm <- do.call(rbind, lapply(keys, function(k) SCORE_ROWS[[k]]$m))
  mean <- .wls(Xm, vapply(keys, function(k) num(k, "mu"), 0), n / sig2)
  if (is.null(mean)) return(NULL)
  wv <- ifelse(assumed, 1e-3, 1) * n / sig2^2
  Xv <- rbind(do.call(rbind, lapply(keys, function(k) SCORE_ROWS[[k]]$v)), c(1, -1, 0), c(0, 0, 1))
  v <- .wls(Xv, c(sig2, 0, 0), c(wv, 1e-6 * max(wv), 1e-6 * max(wv)))
  sh <- sqrt(max(v[1], 1)); sa <- sqrt(max(v[2], 1))
  rho <- max(-0.95, min(0.95, v[3] / (sh * sa)))
  real <- function(k) !is.null(F[[k]]) && !isTRUE(F[[k]]$assumed)
  rho_tot <- NA_real_; rho_hcp <- NA_real_
  if (real("home") && real("away")) {
    vH <- num("home", "sigma")^2; vA <- num("away", "sigma")^2; d <- 2 * sqrt(vH * vA)
    if (real("tot")) rho_tot <- (num("tot", "sigma")^2 - vH - vA) / d
    if (real("hcp")) rho_hcp <- (vH + vA - num("hcp", "sigma")^2) / d
  }
  list(mh = mean[1], ma = mean[2], sh = sh, sa = sa, rho = rho, rho_tot = rho_tot, rho_hcp = rho_hcp,
       src = paste(vapply(keys, function(k) SCORE_ROWS[[k]]$name, ""), collapse = " + "))
}

# Visi sveiki rezultatai h : a per ±5σ su dvimačio normaliojo tankiu, sunormuoti iki 1.
grid_cells <- function(M) {
  rho <- max(-0.95, min(0.95, M$rho))
  h <- seq(max(0, floor(M$mh - 5 * M$sh)), ceiling(M$mh + 5 * M$sh))
  a <- seq(max(0, floor(M$ma - 5 * M$sa)), ceiling(M$ma + 5 * M$sa))
  g <- expand.grid(h = h, a = a)
  x <- (g$h - M$mh) / M$sh; y <- (g$a - M$ma) / M$sa
  d <- exp(-(x * x - 2 * rho * x * y + y * y) / (2 * (1 - rho * rho)))
  g$p <- d / sum(d)
  g
}

# ---------- sąlygos ----------
VARS <- list(home = function(h, a) h, away = function(h, a) a,
             total = function(h, a) h + a, diff = function(h, a) h - a)
OPS <- list(">" = `>`, ">=" = `>=`, "<" = `<`, "<=" = `<=`, "=" = `==`)

# conds: sąrašas list(v = "home", op = ">", x = 75), visos sujungtos „ir“.
condition_probability <- function(M, conds) {
  g <- grid_cells(M)
  ok <- rep(TRUE, nrow(g))
  for (c in conds) {
    if (is.null(VARS[[c$v]]) || is.null(OPS[[c$op]])) stop("Netinkama sąlyga: ", c$v, " ", c$op)
    ok <- ok & OPS[[c$op]](VARS[[c$v]](g$h, g$a), c$x)
  }
  sum(g$p[ok])
}

# ---------- kėlinio dalių modelis (kaip app/model.js periodShareModel) ----------
# Komandos kėlinio taškai = jos visų rungtynių taškų dalis: Qh = Sh·H, Qa = Sa·A, S ~ Beta(c·m, c·(1−m)),
# dalys nepriklausomos nuo H, A ir viena nuo kitos (kaip žaidėjų dalys komandos taškuose).
# E Qh = m·μh, Var Qh = (v + m²)(σh² + μh²) − m²μh² (v = m(1−m)/(c+1)), Cov(Qh, Qa) = mh·ma·ρ·σh·σa.
# Tempo dalis: Corr(Sh, Sa) = κ, tada Cov(Qh, Qa) = mh·ma·ρ·σh·σa + κ·sd(Sh)·sd(Sa)·(ρ·σh·σa + μh·μa).
# κ nustatoma ranka (numatyta 0.3). mh, ma ir c parenkami pagal kėlinio linijas (probit paklaida);
# silpna prielaida m = 25 %, c = 50.
PERIOD_KAPPA0 <- 0.3
PERIOD_SHARE0 <- 0.25
PERIOD_C0 <- 50
PERIOD_PRIOR_W <- 1e-3
SHARE_X <- list(hcp = c(1, -1), tot = c(1, 1), home = c(1, 0), away = c(0, 1))

share_moments <- function(M, mh, ma, c, kappa = 0) {
  rho <- max(-0.95, min(0.95, M$rho))
  kp <- max(-1, min(1, kappa))
  vh <- mh * (1 - mh) / (c + 1); va <- ma * (1 - ma) / (c + 1)
  sh <- sqrt((vh + mh^2) * (M$sh^2 + M$mh^2) - (mh * M$mh)^2)
  sa <- sqrt((va + ma^2) * (M$sa^2 + M$ma^2) - (ma * M$ma)^2)
  cv <- mh * ma * rho * M$sh * M$sa + kp * sqrt(vh * va) * (rho * M$sh * M$sa + M$mh * M$ma)
  list(mh = mh * M$mh, ma = ma * M$ma, sh = sh, sa = sa, rho = max(-0.95, min(0.95, cv / (sh * sa))))
}

period_p_over <- function(Q, key, k) {
  x <- SHARE_X[[key]][1]; y <- SHARE_X[[key]][2]
  sd <- sqrt(x^2 * Q$sh^2 + y^2 * Q$sa^2 + 2 * x * y * Q$rho * Q$sh * Q$sa)
  1 - pnorm((k - (x * Q$mh + y * Q$ma)) / sd)
}

.probit <- function(p) qnorm(min(0.995, max(0.005, p)))

# Toks pat deterministinis Nelder–Mead kaip app/model.js.
nelder_mead <- function(f, x0, step, iters = 600) {
  n <- length(x0)
  X <- rbind(x0, t(vapply(seq_len(n), function(i) x0 + ifelse(seq_len(n) == i, step[i], 0), numeric(n))))
  fx <- apply(X, 1, f)
  for (it in seq_len(iters)) {
    o <- order(fx); X <- X[o, , drop = FALSE]; fx <- fx[o]
    if (fx[n + 1] - fx[1] < 1e-14) break
    cen <- colMeans(X[1:n, , drop = FALSE])
    pt <- function(t) cen + t * (X[n + 1, ] - cen)
    xr <- pt(-1); fr <- f(xr)
    if (fr < fx[1]) {
      xe <- pt(-2); fe <- f(xe)
      if (fe < fr) { X[n + 1, ] <- xe; fx[n + 1] <- fe } else { X[n + 1, ] <- xr; fx[n + 1] <- fr }
    } else if (fr < fx[n]) {
      X[n + 1, ] <- xr; fx[n + 1] <- fr
    } else {
      xc <- if (fr < fx[n + 1]) pt(-0.5) else pt(0.5)
      fc <- f(xc)
      if (fc < min(fr, fx[n + 1])) { X[n + 1, ] <- xc; fx[n + 1] <- fc }
      else for (i in 2:(n + 1)) { X[i, ] <- X[1, ] + 0.5 * (X[i, ] - X[1, ]); fx[i] <- f(X[i, ]) }
    }
  }
  o <- order(fx)
  list(x = X[o[1], ], f = fx[o[1]])
}

# M = pilno laiko score_model(); pts = sąrašas list(key, k, p) iš kėlinio koeficientų.
period_share_model <- function(M, pts, kappa = PERIOD_KAPPA0, share0 = PERIOD_SHARE0, c0 = PERIOD_C0) {
  if (is.null(M)) return(NULL)
  kappa <- if (is.null(kappa) || !is.finite(kappa)) PERIOD_KAPPA0 else max(-1, min(1, kappa))
  P <- Filter(function(q) !is.null(SHARE_X[[q[[1]]]]) && is.finite(q[[2]]) && q[[3]] > 0 && q[[3]] < 1, pts)
  l0 <- log(share0); lc <- log(c0)
  at <- function(u) share_moments(M, exp(u[1]), exp(u[2]), exp(u[3]), kappa)
  err <- function(Q) sum(vapply(P, function(q) (.probit(period_p_over(Q, q[[1]], q[[2]])) - .probit(q[[3]]))^2, 0))
  obj <- function(u) {
    if (u[1] < log(0.02) || u[1] > log(0.9) || u[2] < log(0.02) || u[2] > log(0.9) || u[3] < log(2) || u[3] > log(5000)) return(1e9)
    err(at(u)) + PERIOD_PRIOR_W * ((u[1] - l0)^2 + (u[2] - l0)^2 + (u[3] - lc)^2)
  }
  u <- if (length(P)) nelder_mead(obj, c(l0, l0, lc), c(0.2, 0.2, 0.5))$x else c(l0, l0, lc)
  Q <- at(u)
  c(Q, list(rho_tot = NULL, rho_hcp = NULL, src = "pilnas laikas × kėlinio dalis",
            share = list(h = exp(u[1]), a = exp(u[2]), c = exp(u[3]), kappa = kappa, n = length(P), sse = err(Q))))
}

parse_cond <- function(text) {
  m <- regmatches(text, regexec("^\\s*(home|away|total|diff)\\s*(>=|<=|>|<|=)\\s*(-?[0-9]+(?:[.,][0-9]+)?)\\s*$", text))[[1]]
  if (!length(m)) stop("Sąlyga turi būti pvz. home>75, diff<0: ", text)
  list(v = m[2], op = m[3], x = as.numeric(sub(",", ".", m[4])))
}

# ---------- CSV ir komandinė eilutė ----------
read_rows <- function(path) {
  lines <- readLines(path, encoding = "UTF-8", warn = FALSE)
  out <- lapply(strsplit(lines, ","), function(cells) {
    nums <- suppressWarnings(as.numeric(sub(",", ".", trimws(cells))))
    nums <- nums[!is.na(nums)]
    if (length(nums) >= 3) nums[1:3] else NULL
  })
  out <- Filter(Negate(is.null), out)
  if (!length(out)) return(matrix(numeric(0), ncol = 3))
  do.call(rbind, out)
}

main <- function(args = commandArgs(trailingOnly = TRUE)) {
  opt <- list(cond = list(), home_name = "Namai", away_name = "Svečiai", top = 10)
  i <- 1
  while (i <= length(args)) {
    key <- args[i]; val <- args[i + 1]
    switch(key,
      "--hcp" = , "--tot" = , "--home" = , "--away" = { opt[[sub("--", "", key)]] <- val },
      "--home-name" = { opt$home_name <- val },
      "--away-name" = { opt$away_name <- val },
      "--cond" = { opt$cond[[length(opt$cond) + 1]] <- parse_cond(val) },
      "--top" = { opt$top <- as.integer(val) },
      stop("Nežinomas parametras: ", key))
    i <- i + 2
  }
  F <- list()
  for (k in c("hcp", "tot", "home", "away"))
    if (!is.null(opt[[k]])) F[k] <- list(fit_market(k, read_rows(opt[[k]])))
  H <- opt$home_name; A <- opt$away_name
  if (!is.null(F$hcp))
    cat(sprintf("%s laimi: %.2f%%   skirtumas %+.1f (σ %.1f)\n", H, p_over(F$hcp, 0) * 100, F$hcp$mu, F$hcp$sigma))
  if (!is.null(F$tot)) cat(sprintf("Taškų suma: %.1f (σ %.1f)\n", F$tot$mu, F$tot$sigma))
  M <- score_model(F)
  if (is.null(M)) { cat("Rezultato modeliui reikia bent dviejų rinkų (pvz. handikapo ir sumos arba abiejų komandų taškų).\n"); return(invisible()) }
  cat(sprintf("Tikėtinas rezultatas: %s %.1f : %.1f %s   σ %.1f / %.1f   ρ = %.3f   (%s)\n",
              H, M$mh, M$ma, A, M$sh, M$sa, M$rho, M$src))
  alt <- c(if (is.finite(M$rho_tot)) sprintf("pagal sumą %.3f", M$rho_tot),
           if (is.finite(M$rho_hcp)) sprintf("pagal handikapą %.3f", M$rho_hcp))
  if (length(alt)) cat("ρ atskirai:", paste(alt, collapse = " · "), "\n")
  g <- grid_cells(M); g <- g[g$h != g$a, ]; g <- head(g[order(-g$p), ], opt$top)
  cat("\nTikėtiniausi rezultatai:\n")
  for (j in seq_len(nrow(g))) cat(sprintf("  %2d. %d : %d   %.2f%%\n", j, g$h[j], g$a[j], g$p[j] * 100))
  if (length(opt$cond)) {
    p <- condition_probability(M, opt$cond)
    txt <- paste(vapply(opt$cond, function(c) sprintf("%s %s %g", c$v, c$op, c$x), ""), collapse = " ir ")
    if (p > 0) cat(sprintf("\nP(%s) = %.2f%%   teisingas koeficientas %.2f\n", txt, p * 100, 1 / p))
    else cat(sprintf("\nP(%s) = 0\n", txt))
  }
}

if (sys.nframe() == 0) main()
