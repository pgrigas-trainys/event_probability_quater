// Skaičiavimų patikra: node tests/model.test.js
const assert = require("node:assert/strict");
const M = require("../app/model.js");

const HCP = { key: "hcp", sigma0: 12 };
const TOT = { key: "tot", sigma0: 16 };
const rows = list => list.map(([l, a, b]) => ({ l, a, b }));

// Fenerbahce – Partizan, Betsson handikapai
const hcp = rows([[-14.5, 2.45, 1.50], [-13.5, 2.30, 1.62], [-12.5, 2.15, 1.70], [-11.5, 1.95, 1.85],
                  [-10.5, 1.85, 1.95], [-9.5, 1.70, 2.15], [-8.5, 1.62, 2.30], [-7.5, 1.50, 2.45]]);
const tot = rows([[160.5, 1.55, 2.35], [163.5, 1.75, 2.05], [165.5, 1.90, 1.90], [167.5, 2.05, 1.75], [170.5, 2.35, 1.55]]);

const near = (x, y, eps, msg) => assert.ok(Math.abs(x - y) < eps, `${msg}: ${x} != ${y}`);

// Normalusis skirstinys
near(M.Phi(1.96), 0.975, 1e-4, "Phi(1.96)");
near(M.PhiInv(0.975), 1.96, 1e-3, "PhiInv(0.975)");

// Koeficientai -> tikimybės be maržos
const r = M.rowCalc(HCP, hcp[3]);
near(r.pa, 0.4868, 1e-4, "P(namai -11.5)");
near(r.margin, 0.0534, 1e-4, "marža");

// Normaliojo skirstinio pritaikymas
const F = { hcp: M.fit(HCP, hcp), tot: M.fit(TOT, tot), home: null, away: null };
near(F.hcp.mu, 11.0, 0.05, "vid. skirtumas");
near(F.hcp.sigma, 11.31, 0.05, "σ skirtumo");
near(M.pOver(F.hcp, 0), 0.8346, 1e-3, "P(Fenerbahce laimi)");
near(F.tot.mu, 165.5, 0.05, "vid. suma");

// Rezultato modelis ir tinklelis
const SM = M.scoreModel(F);
near(SM.mh, 88.25, 0.05, "namų vidurkis");
near(SM.ma, 77.25, 0.05, "svečių vidurkis");
near(SM.rho, 0.491, 0.01, "koreliacija");
const cells = M.gridCells(SM);
near(cells.reduce((s, c) => s + c[2], 0), 1, 1e-9, "tinklelio suma");

// Bendras modelis iš visų keturių rinkų: koeficientai sugeneruoti iš žinomų parametrų
// (μh 88, μa 77, σh 11, σa 9, ρ 0.3) su 5 % marža; modelis turi juos atkurti.
const TRUE = { mh: 88, ma: 77, sh: 11, sa: 9, rho: 0.3 };
const cov = TRUE.rho * TRUE.sh * TRUE.sa;
const gen = (key, mu, sd, lines) => lines.map(l => {
  const k = key === "hcp" ? -l : l, p = 1 - M.Phi((k - mu) / sd);
  return { l, a: 1 / (p * 1.025), b: 1 / ((1 - p) * 1.025) };
});
const mk = key => ({ key, sigma0: { hcp: 12, tot: 16, home: 10, away: 10 }[key] });
const G = {
  hcp: M.fit(mk("hcp"), gen("hcp", TRUE.mh - TRUE.ma, Math.sqrt(TRUE.sh ** 2 + TRUE.sa ** 2 - 2 * cov), [-16.5, -13.5, -10.5, -7.5, -4.5])),
  tot: M.fit(mk("tot"), gen("tot", TRUE.mh + TRUE.ma, Math.sqrt(TRUE.sh ** 2 + TRUE.sa ** 2 + 2 * cov), [157.5, 162.5, 165.5, 168.5, 173.5])),
  home: M.fit(mk("home"), gen("home", TRUE.mh, TRUE.sh, [82.5, 86.5, 88.5, 90.5, 94.5])),
  away: M.fit(mk("away"), gen("away", TRUE.ma, TRUE.sa, [72.5, 75.5, 77.5, 79.5, 82.5])),
};
const GM = M.scoreModel(G);
near(GM.mh, 88, 0.05, "bendras: namų vidurkis"); near(GM.ma, 77, 0.05, "bendras: svečių vidurkis");
near(GM.sh, 11, 0.05, "bendras: σ namų"); near(GM.sa, 9, 0.05, "bendras: σ svečių");
near(GM.rho, 0.3, 0.005, "bendras: ρ"); near(GM.rho_tot, 0.3, 0.005, "ρ pagal sumą"); near(GM.rho_hcp, 0.3, 0.005, "ρ pagal handikapą");
assert.equal(GM.src, "handikapas + suma + namų taškai + svečių taškai");
// Be bendrų rinkų koreliacija neįvertinama (0), be komandų rinkų σ laikomos lygiomis.
const onlyTeams = M.scoreModel({ home: G.home, away: G.away });
near(onlyTeams.rho, 0, 1e-6, "tik komandų taškai: ρ = 0"); near(onlyTeams.sh, 11, 0.05, "tik komandų taškai: σ namų");
const noTeams = M.scoreModel({ hcp: G.hcp, tot: G.tot });
near(noTeams.sh, noTeams.sa, 1e-6, "be komandų rinkų σ vienodos");
// Suma + vienos komandos taškai: likusi komanda ir ρ įvertinami iš likusių rinkų.
const three = M.scoreModel({ hcp: G.hcp, tot: G.tot, home: G.home });
near(three.sa, 9, 0.05, "be svečių rinkos: σ svečių"); near(three.rho, 0.3, 0.005, "be svečių rinkos: ρ");
assert.equal(M.scoreModel({ hcp: G.hcp }), null, "vien handikapo nepakanka");

// Sąlygos
const pAway = M.conditionProbability(SM, [{ v: "diff", op: "<", x: 0 }]);
const pHome = M.conditionProbability(SM, [{ v: "diff", op: ">", x: 0 }]);
const pDraw = M.conditionProbability(SM, [{ v: "diff", op: "=", x: 0 }]);
near(pAway + pHome + pDraw, 1, 1e-9, "laimi + pralaimi + lygiosios");
const pBoth = M.conditionProbability(SM, [{ v: "home", op: ">", x: 75 }, { v: "away", op: ">", x: 75 }, { v: "diff", op: "<", x: 0 }]);
assert.ok(pBoth < pAway, "papildomos sąlygos mažina tikimybę");
near(pBoth, 0.1018, 2e-3, "abi > 75 ir laimi Partizan");
assert.throws(() => M.conditionProbability(SM, [{ v: "nėra", op: ">", x: 1 }]));

// CSV įkėlimas
const fs = require("node:fs");
const path = require("node:path");
const own = M.parseOddsCSV(fs.readFileSync(path.join(__dirname, "../data/fenerbahce_partizan_betsson_handicap.csv"), "utf8"));
assert.equal(own.count, 8, "esamas handikapo failas");
assert.deepEqual(own.byMarket.hcp?.[3], { l: "-11.5", a: "1.95", b: "1.85" }, "market=Handicap -> hcp");

const semi = M.parseOddsCSV("Linija;Daugiau;Mažiau\n165,5;1,90;1,90\n167,5;2,05;1,75\n");
assert.equal(semi.unassigned.length, 2, "kabliataškis ir dešimtainis kablelis");
assert.deepEqual(semi.unassigned[0], { l: "165.5", a: "1.90", b: "1.90" });

const combo = M.parseOddsCSV(["rinka,linija,a,b", "handikapas,-11.5,1.95,1.85", "taškų suma,165.5,1.90,1.90",
  "namai,88.5,1.85,1.95", "svečiai,77.5,1.90,1.90", "tekstas be skaičių"].join("\n"));
assert.equal(combo.byMarket.hcp.length, 1); assert.equal(combo.byMarket.tot.length, 1);
assert.equal(combo.byMarket.home.length, 1); assert.equal(combo.byMarket.away.length, 1);
assert.equal(combo.count, 4, "eilutės be skaičių praleidžiamos");

assert.equal(M.scoreModel({ hcp: null, tot: null, home: null, away: null }), null, "be rinkų modelio nėra");

// Kėlinys: viena linija -> naudojama to kėlinio numatyta σ (perpus mažesnė), tinklelyje galimos lygiosios.
const Q = M.fit({ key: "hcp", sigma0: 6 }, rows([[-2.5, 1.90, 1.90]]));
assert.equal(Q.sigma, 6); assert.ok(Q.assumed);
near(Q.mu, 2.5, 1e-9, "kėlinio vidurkis");
const QSM = M.scoreModel({ hcp: M.fit({ key: "hcp", sigma0: 6 }, rows([[-3.5, 2.0, 1.8], [-1.5, 1.6, 2.3]])),
  tot: M.fit({ key: "tot", sigma0: 8 }, rows([[40.5, 1.8, 2.0], [42.5, 2.0, 1.8]])), home: null, away: null });
const pd = M.conditionProbability(QSM, [{ v: "diff", op: "=", x: 0 }]);
const ph = M.conditionProbability(QSM, [{ v: "diff", op: ">", x: 0 }]), pa = M.conditionProbability(QSM, [{ v: "diff", op: "<", x: 0 }]);
assert.ok(pd > 0.03 && pd < 0.15, `kėlinio lygiosios ${pd}`);
near(ph + pd + pa, 1, 1e-9, "pergalė + lygiosios + pralaimėjimas");

// Kėlinio dalių modelis (kaip žaidėjų dalys): iš žinomų dalių sugeneruotos kėlinio linijos -> dalys atkuriamos.
{
  const FT = { mh: 88, ma: 78, sh: 10, sa: 10, rho: 0.3 };
  const truth = M.shareMoments(FT, 0.27, 0.23, 80), pts = [];
  for (const [key, ks] of [["hcp", [-3.5, -1.5, 0.5, 2.5]], ["tot", [38.5, 41.5, 44.5]], ["home", [21.5, 24.5]], ["away", [16.5, 18.5]]])
    for (const k of ks) pts.push({ key, k, p: M.periodPOver(truth, key, k) });
  const R = M.periodShareModel(FT, pts);
  near(R.share.h, 0.27, 1e-3, "namų kėlinio dalis"); near(R.share.a, 0.23, 1e-3, "svečių kėlinio dalis");
  near(R.share.c, 80, 2, "koncentracija c"); near(R.mh, 23.76, 0.02, "kėlinio namų vidurkis");
  const D = M.periodShareModel(FT, []); // be kėlinio koeficientų: numatyta 25 %, c = 50
  assert.equal(D.share.h, 0.25); assert.equal(D.mh, 22); assert.equal(D.ma, 19.5);
  near(D.sh, Math.sqrt((0.1875 / 51 + 0.0625) * (100 + 88 * 88) - 22 * 22), 1e-9, "kėlinio σ");
  assert.equal(M.periodShareModel(null, pts), null);
  const one = M.periodShareModel(FT, [{ key: "tot", k: 40.5, p: 0.5 }]); // viena linija: suma atkuriama, dalys beveik lygios
  near(one.mh + one.ma, 40.5, 0.05, "kėlinio suma iš vienos linijos"); near(one.share.h, one.share.a, 0.01, "lygios dalys");
}

console.log("Visi skaičiavimų testai praėjo.");
