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
  module.exports = { erf, Phi, PhiInv, rowCalc, fit, pOver, scoreModel, gridCells, conditionProbability, VARS, OPS, parseOddsCSV };
}
