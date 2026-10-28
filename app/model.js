/* Skaičiavimų modelis: koeficientai -> tikimybės, normaliojo skirstinio įvertis,
   rezultatų tinklelis ir sąlygų tikimybė. Be jokių puslapio (DOM) priklausomybių.
   Naršyklėje funkcijos tampa globaliomis, Node.js eksportuojamos per module.exports. */

// ---------- normalusis skirstinys ----------
function erf(x) { const s = Math.sign(x); x = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * x);
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
  return s * y; }
const Phi = z => 0.5 * (1 + erf(z / Math.SQRT2));
function PhiInv(p) { // Acklam
  const a = [-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.357751867269, -30.66479806614716, 2.506628277459239];
  const b = [-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572];
  const c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
  const d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416];
  const pl = 0.02425;
  if (p < pl) { const q = Math.sqrt(-2 * Math.log(p)); return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1); }
  if (p > 1 - pl) { const q = Math.sqrt(-2 * Math.log(1 - p)); return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1); }
  const q = p - 0.5, r = q * q;
  return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1);
}

// Each row -> threshold k and P(X > k) without margin. X = home−away margin for hcp, points otherwise.
function rowCalc(m, r) {
  const l = parseFloat(String(r.l).replace(",", ".")), a = parseFloat(String(r.a).replace(",", ".")), b = parseFloat(String(r.b).replace(",", "."));
  if (!isFinite(l) || !(a > 1) || !(b > 1)) return null;
  const ia = 1 / a, ib = 1 / b, book = ia + ib;
  return { k: m.key === "hcp" ? -l : l, pa: ia / book, pb: ib / book, margin: book - 1 };
}
function fit(m, rows) {
  const pts = rows.map(r => rowCalc(m, r)).filter(Boolean).map(c => ({ k: c.k, z: PhiInv(1 - Math.min(0.99, Math.max(0.01, c.pa))) }));
  if (!pts.length) return null;
  if (pts.length >= 2) {
    const n = pts.length, mk = pts.reduce((s, p) => s + p.k, 0) / n, mz = pts.reduce((s, p) => s + p.z, 0) / n;
    const sxx = pts.reduce((s, p) => s + (p.k - mk) ** 2, 0);
    if (sxx > 0) {
      const slope = pts.reduce((s, p) => s + (p.k - mk) * (p.z - mz), 0) / sxx;
      if (slope > 0.01) { const sigma = 1 / slope; return { mu: mk - mz * sigma, sigma, n, assumed: false }; }
    }
  }
  const sigma = m.sigma0;
  const mu = pts.reduce((s, p) => s + (p.k - p.z * sigma), 0) / pts.length;
  return { mu, sigma, n: pts.length, assumed: true };
}
const pOver = (f, k) => 1 - Phi((k - f.mu) / f.sigma);

// ---------- rezultato modelis ----------
// F = {hcp, tot, home, away}: kiekvienos rinkos fit() rezultatas arba null.
//
// Namų (H) ir svečių (A) taškai laikomi dvimačiu normaliuoju skirstiniu su μh, μa, σh, σa, ρ.
// Kiekviena rinka duoda vidurkį ir dispersiją, kurie tiesiškai priklauso nuo nežinomųjų:
//   handikapas  H − A:  μh − μa,   σh² + σa² − 2c        (c = ρ·σh·σa)
//   suma        H + A:  μh + μa,   σh² + σa² + 2c
//   namų taškai H:      μh,        σh²
//   svečių      A:      μa,        σa²
// Visos turimos rinkos sujungiamos svertiniu mažiausių kvadratų metodu (vidurkiai: svoris n/σ²,
// dispersijos: svoris n/σ⁴, t. y. pagal santykinę paklaidą). Kai duomenų nepakanka, labai silpnos
// prielaidos σh = σa ir ρ = 0 nulemia tik tai, ko rinkos neapibrėžia.
const SCORE_ROWS = {
  hcp: { m: [1, -1], v: [1, 1, -2], name: "handikapas" },
  tot: { m: [1, 1], v: [1, 1, 2], name: "suma" },
  home: { m: [1, 0], v: [1, 0, 0], name: "namų taškai" },
  away: { m: [0, 1], v: [0, 1, 0], name: "svečių taškai" },
};
function wls(rows) { // rows: [x[], y, w] -> mažiausių kvadratų sprendinys arba null
  const n = rows[0][0].length, A = Array.from({ length: n }, () => Array(n + 1).fill(0));
  for (const [x, y, w] of rows) for (let i = 0; i < n; i++) { for (let j = 0; j < n; j++) A[i][j] += w * x[i] * x[j]; A[i][n] += w * x[i] * y; }
  const scale = Math.max(...A.map((r, i) => Math.abs(r[i]))) || 1;
  for (let c = 0; c < n; c++) { // Gauso eliminacija su pagrindinio elemento parinkimu
    let p = c; for (let r = c + 1; r < n; r++) if (Math.abs(A[r][c]) > Math.abs(A[p][c])) p = r;
    if (Math.abs(A[p][c]) < 1e-12 * scale) return null;
    [A[c], A[p]] = [A[p], A[c]];
    for (let r = 0; r < n; r++) if (r !== c) { const f = A[r][c] / A[c][c]; for (let j = c; j <= n; j++) A[r][j] -= f * A[c][j]; }
  }
  return A.map((r, i) => r[n] / r[i]);
}
function scoreModel(F) {
  const keys = Object.keys(SCORE_ROWS).filter(k => F[k]);
  if (!keys.length) return null; // dar nėra nė vienos rinkos (pvz. tuščias kėlinys)
  const mean = wls(keys.map(k => [SCORE_ROWS[k].m, F[k].mu, F[k].n / F[k].sigma ** 2]));
  if (!mean) return null;
  // Rinkos su numatyta σ (viena linija) beveik nesveria; prielaidos sveria dar mažiau.
  const vRows = keys.map(k => [SCORE_ROWS[k].v, F[k].sigma ** 2, (F[k].assumed ? 1e-3 : 1) * F[k].n / F[k].sigma ** 4]);
  const wMax = Math.max(...vRows.map(r => r[2]));
  vRows.push([[1, -1, 0], 0, 1e-6 * wMax], [[0, 0, 1], 0, 1e-6 * wMax]);
  let [vh, va, c] = wls(vRows);
  vh = Math.max(vh, 1); va = Math.max(va, 1);
  const sh = Math.sqrt(vh), sa = Math.sqrt(va), rho = Math.max(-0.95, Math.min(0.95, c / (sh * sa)));
  // Atskiri ρ įverčiai palyginimui: iš komandų taškų σ ir sumos arba handikapo σ.
  let rho_tot = null, rho_hcp = null;
  const real = k => F[k] && !F[k].assumed;
  if (real("home") && real("away")) {
    const vH = F.home.sigma ** 2, vA = F.away.sigma ** 2, d = 2 * Math.sqrt(vH * vA);
    if (real("tot")) rho_tot = (F.tot.sigma ** 2 - vH - vA) / d;
    if (real("hcp")) rho_hcp = (vH + vA - F.hcp.sigma ** 2) / d;
  }
  return { mh: mean[0], ma: mean[1], sh, sa, rho, rho_tot, rho_hcp, src: keys.map(k => SCORE_ROWS[k].name).join(" + ") };
}

// Visi sveiki rezultatai h : a per ±5σ su tikimybėmis (suma = 1).
function gridCells(M) {
  const rho = Math.max(-0.95, Math.min(0.95, M.rho)), cells = []; let tot = 0;
  for (let h = Math.max(0, Math.floor(M.mh - 5 * M.sh)); h <= Math.ceil(M.mh + 5 * M.sh); h++)
    for (let a = Math.max(0, Math.floor(M.ma - 5 * M.sa)); a <= Math.ceil(M.ma + 5 * M.sa); a++) {
      const x = (h - M.mh) / M.sh, y = (a - M.ma) / M.sa;
      const d = Math.exp(-(x * x - 2 * rho * x * y + y * y) / (2 * (1 - rho * rho)));
      cells.push([h, a, d]); tot += d;
    }
  cells.forEach(c => c[2] /= tot);
  return cells;
}

// ---------- sąlygos ----------
const VARS = [["home", h => h], ["away", (h, a) => a], ["total", (h, a) => h + a], ["diff", (h, a) => h - a]];
const OPS = { ">": (x, y) => x > y, ">=": (x, y) => x >= y, "<": (x, y) => x < y, "<=": (x, y) => x <= y, "=": (x, y) => x === y };

// conds = [{v: "home"|"away"|"total"|"diff", op: ">"|">="|"<"|"<="|"=", x: skaičius}], visos sujungtos „ir“.
function conditionProbability(M, conds) {
  const cs = conds.map(c => ({ f: VARS.find(v => v[0] === c.v)?.[1], op: OPS[c.op], x: Number(c.x) }));
  if (cs.some(c => !c.f || !c.op || !isFinite(c.x))) throw new Error("Netinkama sąlyga");
  let p = 0;
  for (const [h, a, w] of gridCells(M)) if (cs.every(c => c.op(c.f(h, a), c.x))) p += w;
  return p;
}

// ---------- kėlinio dalių modelis ----------
// Kaip žaidėjų dalys komandos taškuose (event_probability_players): kai yra pilno laiko rezultato modelis M,
// komandos kėlinio taškai laikomi jos visų rungtynių taškų dalimi:  Qh = Sh·H,  Qa = Sa·A.
// Dalis S ~ Beta(c·m, c·(1−m)) (vidurkis m, dispersija m(1−m)/(c+1)), nepriklausoma nuo H, A ir kitos komandos dalies.
// Iš to kėlinio taškų vidurkiai, σ ir koreliacija:
//   E Qh = m·μh,   Var Qh = (v + m²)(σh² + μh²) − m²μh²,   Cov(Qh, Qa) = mh·ma·ρ·σh·σa
// ir kėlinys skaičiuojamas tuo pačiu dvimačiu normaliuoju modeliu kaip pilnas laikas.
// Tempo dalis: namų ir svečių dalys koreliuotos, Corr(Sh, Sa) = κ (greitame kėlinyje abi dalys kartu didesnės). Tada
//   Cov(Qh, Qa) = mh·ma·ρ·σh·σa + κ·sd(Sh)·sd(Sa)·(ρ·σh·σa + μh·μa).   κ = 0 – dalys nepriklausomos.
// κ nustatoma ranka (numatyta 0.3). Dalys mh, ma ir c parenkamos, kad modelio tikimybės geriausiai atkurtų kėlinio
// linijas (probit paklaida, kaip žaidėjams). Labai silpna prielaida (m = 25 %, c = 50) lemia tik tai, ko linijos
// neapibrėžia; be kėlinio linijų imama ji pati.
const PERIOD_SHARE0 = 0.25, PERIOD_C0 = 50, PERIOD_PRIOR_W = 1e-3, PERIOD_KAPPA0 = 0.3;
const SHARE_X = { hcp: [1, -1], tot: [1, 1], home: [1, 0], away: [0, 1] };
function shareMoments(M, mh, ma, c, kappa = 0) {
  const rho = Math.max(-0.95, Math.min(0.95, M.rho)), k = Math.max(-1, Math.min(1, kappa));
  const vh = mh * (1 - mh) / (c + 1), va = ma * (1 - ma) / (c + 1);
  const sh = Math.sqrt((vh + mh * mh) * (M.sh ** 2 + M.mh ** 2) - (mh * M.mh) ** 2);
  const sa = Math.sqrt((va + ma * ma) * (M.sa ** 2 + M.ma ** 2) - (ma * M.ma) ** 2);
  const cov = mh * ma * rho * M.sh * M.sa + k * Math.sqrt(vh * va) * (rho * M.sh * M.sa + M.mh * M.ma);
  return { mh: mh * M.mh, ma: ma * M.ma, sh, sa, rho: Math.max(-0.95, Math.min(0.95, cov / (sh * sa))) };
}
// P(X > k) kėlinio modelyje, X pagal rinką: handikapas H − A, suma H + A, namų H, svečių A.
function periodPOver(Q, key, k) {
  const [x, y] = SHARE_X[key];
  const sd = Math.sqrt(x * x * Q.sh ** 2 + y * y * Q.sa ** 2 + 2 * x * y * Q.rho * Q.sh * Q.sa);
  return 1 - Phi((k - (x * Q.mh + y * Q.ma)) / sd);
}
const probitC = p => PhiInv(Math.min(0.995, Math.max(0.005, p)));
// Nelder–Mead minimizavimas (deterministinis, toks pat Python ir R versijose).
function nelderMead(f, x0, step, iters = 600) {
  const n = x0.length;
  let S = [x0.slice(), ...x0.map((_, i) => x0.map((v, j) => v + (i === j ? step[i] : 0)))].map(x => [x, f(x)]);
  for (let it = 0; it < iters; it++) {
    S.sort((a, b) => a[1] - b[1]);
    if (S[n][1] - S[0][1] < 1e-14) break;
    const cen = x0.map((_, j) => S.slice(0, n).reduce((s, p) => s + p[0][j], 0) / n);
    const pt = t => cen.map((c, j) => c + t * (S[n][0][j] - c));
    const xr = pt(-1), fr = f(xr);
    if (fr < S[0][1]) { const xe = pt(-2), fe = f(xe); S[n] = fe < fr ? [xe, fe] : [xr, fr]; }
    else if (fr < S[n - 1][1]) S[n] = [xr, fr];
    else {
      const xc = fr < S[n][1] ? pt(-0.5) : pt(0.5), fc = f(xc);
      if (fc < Math.min(fr, S[n][1])) S[n] = [xc, fc];
      else S = S.map((p, i) => { if (!i) return p; const x = p[0].map((v, j) => S[0][0][j] + 0.5 * (v - S[0][0][j])); return [x, f(x)]; });
    }
  }
  S.sort((a, b) => a[1] - b[1]);
  return S[0];
}
// M = pilno laiko scoreModel(); pts = [{key: "hcp"|"tot"|"home"|"away", k, p}] (kėlinio rinkų P(X > k) be maržos);
// kappa = dalių (tempo) koreliacija κ.
const cleanPts = pts => (pts || []).filter(q => SHARE_X[q.key] && isFinite(q.k) && q.p > 0 && q.p < 1);
// Dalių mh, ma ir c parinkimas esant duotai κ. Grąžina {u: [ln mh, ln ma, ln c], f: tikslo funkcija su prielaida}.
function shareFit(M, P, kappa, share0 = PERIOD_SHARE0, c0 = PERIOD_C0) {
  const l0 = Math.log(share0), lc = Math.log(c0);
  if (!P.length) return { u: [l0, l0, lc], f: 0 };
  const obj = u => {
    if (u[0] < Math.log(0.02) || u[0] > Math.log(0.9) || u[1] < Math.log(0.02) || u[1] > Math.log(0.9) || u[2] < Math.log(2) || u[2] > Math.log(5000)) return 1e9;
    const Q = shareMoments(M, Math.exp(u[0]), Math.exp(u[1]), Math.exp(u[2]), kappa); let s = 0;
    for (const q of P) s += (probitC(periodPOver(Q, q.key, q.k)) - probitC(q.p)) ** 2;
    return s + PERIOD_PRIOR_W * ((u[0] - l0) ** 2 + (u[1] - l0) ** 2 + (u[2] - lc) ** 2);
  };
  const [u, f] = nelderMead(obj, [l0, l0, lc], [0.2, 0.2, 0.5]);
  return { u, f };
}
function periodShareModel(M, pts, kappa = PERIOD_KAPPA0, share0 = PERIOD_SHARE0, c0 = PERIOD_C0) {
  if (!M) return null;
  kappa = Number.isFinite(Number(kappa)) ? Math.max(-1, Math.min(1, Number(kappa))) : PERIOD_KAPPA0;
  const P = cleanPts(pts);
  const { u } = shareFit(M, P, kappa, share0, c0);
  let sse = 0; const Q = shareMoments(M, Math.exp(u[0]), Math.exp(u[1]), Math.exp(u[2]), kappa);
  for (const q of P) sse += (probitC(periodPOver(Q, q.key, q.k)) - probitC(q.p)) ** 2;
  return { ...Q, rho_tot: null, rho_hcp: null, src: "pilnas laikas × kėlinio dalis",
    share: { h: Math.exp(u[0]), a: Math.exp(u[1]), c: Math.exp(u[2]), kappa, n: P.length, sse } };
}

// ---------- κ įvertinimas iš koeficientų ----------
// κ matoma tik iš sklaidų: Var(suma) − Var(handikapas) = 4·Cov. Todėl kėlinys tinka, kai jame yra bent dvi skirtingos
// sumos linijos ir bent dvi skirtingos handikapo arba vienos komandos taškų linijos.
const nLines = (P, key) => new Set(P.filter(q => q.key === key).map(q => q.k)).size;
const kappaEligible = pts => { const P = cleanPts(pts); return nLines(P, "tot") >= 2 && ["hcp", "home", "away"].some(k => nLines(P, k) >= 2); };
function goldenMin(f, lo, hi, it = 30) {
  const r = (Math.sqrt(5) - 1) / 2; let x1 = hi - r * (hi - lo), x2 = lo + r * (hi - lo), f1 = f(x1), f2 = f(x2);
  for (let i = 0; i < it; i++) {
    if (f1 < f2) { hi = x2; x2 = x1; f2 = f1; x1 = hi - r * (hi - lo); f1 = f(x1); }
    else { lo = x1; x1 = x2; f1 = f2; x2 = lo + r * (hi - lo); f2 = f(x2); }
  }
  return (lo + hi) / 2;
}
// Bendras κ visiems tinkamiems kėliniams: kiekvienam κ kiekvieno kėlinio dalys ir c parenkami atskirai, o κ – ta, kuriai
// bendra probit paklaida mažiausia (tinklelis kas 0.1 intervale [−0.5, 0.95], tada auksinis pjūvis).
// groups = [pts, ...] (po vieną kiekvienam kėliniui). Paklaida (se) – iš paklaidos funkcijos kreivumo.
const KAPPA_LO = -0.5, KAPPA_HI = 0.95;
function fitKappa(M, groups) {
  if (!M) return null;
  const G = (groups || []).map(cleanPts).filter(kappaEligible);
  if (!G.length) return null;
  const f = k => G.reduce((s, P) => s + shareFit(M, P, k).f, 0);
  let best = KAPPA_LO, bv = Infinity;
  for (let i = 0; i <= 14; i++) { const k = Math.min(KAPPA_HI, KAPPA_LO + 0.1 * i), v = f(k); if (v < bv) { bv = v; best = k; } }
  const kappa = goldenMin(f, Math.max(KAPPA_LO, best - 0.1), Math.min(KAPPA_HI, best + 0.1));
  const N = G.reduce((s, P) => s + P.length, 0), dof = N - 3 * G.length - 1, fk = f(kappa);
  let se = null;
  if (dof > 0) {
    const h = 0.02, d2 = (f(Math.min(1, kappa + h)) - 2 * fk + f(Math.max(-1, kappa - h))) / (h * h);
    if (d2 > 0) se = Math.sqrt(2 * (fk / dof) / d2);
  }
  return { kappa, se, quarters: G.length, lines: N };
}

// ---------- CSV įkėlimas ----------
// Kiekvienoje eilutėje imami pirmi trys skaitiniai stulpeliai: linija, koeficientas A, koeficientas B
// (handikapui A = namai, B = svečiai; sumoms A = daugiau, B = mažiau).
// Jei yra stulpelis „market“/„rinka“, eilutės paskirstomos rinkoms pagal jo reikšmę.
// Skirtukas aptinkamas automatiškai (, ; arba tab); su ; ir tab leidžiamas dešimtainis kablelis.
const MARKET_ALIASES = [
  ["hcp", /^(hcp|handicap|handikap|handikapas|spread|fora)/i],
  ["tot", /^(tot|total|totals|suma|taškų suma|tasku suma|over|o\/u)/i],
  ["home", /^(home|namai|namų|namu|namų komandos|team1|1)/i],
  ["away", /^(away|svečiai|sveciai|svečių|veciu|team2|2)/i],
];
function parseOddsCSV(text) {
  const lines = String(text).replace(/^﻿/, "").split(/\r?\n/).filter(l => l.trim());
  if (!lines.length) return { byMarket: {}, unassigned: [], count: 0 };
  const head = lines[0];
  const delim = [["\t", (head.match(/\t/g) || []).length], [";", (head.match(/;/g) || []).length], [",", (head.match(/,/g) || []).length]]
    .sort((x, y) => y[1] - x[1])[0][0];
  const split = l => l.split(delim).map(c => c.trim().replace(/^"(.*)"$/, "$1").trim());
  const num = c => { const v = delim === "," ? c : c.replace(",", "."); return /^[-+]?\d+(\.\d+)?$/.test(v) ? Number(v) : NaN; };
  let marketCol = -1, start = 0;
  const first = split(head);
  if (first.filter(c => isFinite(num(c))).length < 3) {
    start = 1;
    marketCol = first.findIndex(c => /^(market|rinka|turgus|tipas|type)$/i.test(c));
  }
  const byMarket = {}, unassigned = [];
  for (const line of lines.slice(start)) {
    const cells = split(line);
    const nums = cells.filter((c, i) => i !== marketCol).map(num).filter(Number.isFinite);
    if (nums.length < 3) continue;
    const row = { l: String(nums[0]), a: nums[1].toFixed(2), b: nums[2].toFixed(2) };
    const mv = marketCol >= 0 ? cells[marketCol] || "" : "";
    const key = mv && MARKET_ALIASES.find(([, re]) => re.test(mv))?.[0];
    if (key) (byMarket[key] ||= []).push(row); else unassigned.push(row);
  }
  const count = unassigned.length + Object.values(byMarket).reduce((s, r) => s + r.length, 0);
  return { byMarket, unassigned, count };
}

if (typeof module !== "undefined") {
  module.exports = { erf, Phi, PhiInv, rowCalc, fit, pOver, scoreModel, gridCells, conditionProbability, VARS, OPS, parseOddsCSV,
    shareMoments, periodPOver, nelderMead, periodShareModel, PERIOD_SHARE0, PERIOD_C0, PERIOD_KAPPA0,
    kappaEligible, fitKappa };
}
