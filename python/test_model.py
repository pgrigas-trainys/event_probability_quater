"""Python modelio testai ir palyginimas su SciPy.

Paleidimas iš repozitorijos šaknies:  python3 python/test_model.py
SciPy dalis praleidžiama, jei SciPy neįdiegta (pip install scipy).
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model as m  # noqa: E402

HCP = [(-14.5, 2.45, 1.50), (-13.5, 2.30, 1.62), (-12.5, 2.15, 1.70), (-11.5, 1.95, 1.85),
       (-10.5, 1.85, 1.95), (-9.5, 1.70, 2.15), (-8.5, 1.62, 2.30), (-7.5, 1.50, 2.45)]
TOT = [(160.5, 1.55, 2.35), (163.5, 1.75, 2.05), (165.5, 1.90, 1.90), (167.5, 2.05, 1.75), (170.5, 2.35, 1.55)]


def near(x, y, eps, msg):
    assert abs(x - y) < eps, f"{msg}: {x} != {y}"


def test_model():
    r = m.row_calc("hcp", -11.5, 1.95, 1.85)
    near(r.pa, 0.4868, 1e-4, "P(namai -11.5)")
    near(r.margin, 0.0534, 1e-4, "marža")

    F = {"hcp": m.fit("hcp", HCP), "tot": m.fit("tot", TOT), "home": None, "away": None}
    near(F["hcp"].mu, 11.0, 0.05, "vid. skirtumas")
    near(F["hcp"].sigma, 11.31, 0.05, "σ skirtumo")
    near(m.p_over(F["hcp"], 0), 0.8346, 1e-3, "P(Fenerbahce laimi)")
    near(F["tot"].mu, 165.5, 0.05, "vid. suma")

    M = m.score_model(F)
    near(M.mh, 88.25, 0.05, "namų vidurkis")
    near(M.ma, 77.25, 0.05, "svečių vidurkis")
    near(M.rho, 0.491, 0.01, "koreliacija")
    near(sum(c[2] for c in m.grid_cells(M)), 1, 1e-9, "tinklelio suma")

    p_away = m.condition_probability(M, [("diff", "<", 0)])
    p_home = m.condition_probability(M, [("diff", ">", 0)])
    p_draw = m.condition_probability(M, [("diff", "=", 0)])
    near(p_away + p_home + p_draw, 1, 1e-9, "laimi + pralaimi + lygiosios")
    p_both = m.condition_probability(M, [("home", ">", 75), ("away", ">", 75), ("diff", "<", 0)])
    near(p_both, 0.1018, 2e-3, "abi > 75 ir laimi Partizan (sutampa su JS)")
    return F, M


def test_against_scipy(F, M):
    try:
        import numpy as np
        from scipy import stats
    except ImportError:
        print("SciPy neįdiegta: palyginimas praleistas.")
        return
    # Φ ir Φ⁻¹
    near(m.N.cdf(1.2345), stats.norm.cdf(1.2345), 1e-9, "Φ")
    near(m.N.inv_cdf(0.3), stats.norm.ppf(0.3), 1e-9, "Φ⁻¹")

    # Rezultato tinklelis prieš SciPy dvimatį normalųjį: tikimybė langeliui h : a
    # integruojama per [h−0.5, h+0.5] × [a−0.5, a+0.5].
    cov = [[M.sh ** 2, M.rho * M.sh * M.sa], [M.rho * M.sh * M.sa, M.sa ** 2]]
    mvn = stats.multivariate_normal(mean=[M.mh, M.ma], cov=cov)

    def box(h0, h1, a0, a1):
        return mvn.cdf([h1, a1]) - mvn.cdf([h0, a1]) - mvn.cdf([h1, a0]) + mvn.cdf([h0, a0])

    grid = {(h, a): p for h, a, p in m.grid_cells(M)}
    for h, a in [(88, 77), (95, 70), (80, 82)]:
        near(grid[(h, a)], box(h - 0.5, h + 0.5, a - 0.5, a + 0.5), 2e-5, f"langelis {h}:{a}")

    # Sąlyga „abi > 75 ir laimi svečiai“ Monte Carlo su SciPy (2 mln. imčių, suapvalinta iki sveikų taškų)
    rng = np.random.default_rng(1)
    xs = np.rint(mvn.rvs(size=2_000_000, random_state=rng))
    mc = np.mean((xs[:, 0] > 75) & (xs[:, 1] > 75) & (xs[:, 0] < xs[:, 1]))
    p = m.condition_probability(M, [("home", ">", 75), ("away", ">", 75), ("diff", "<", 0)])
    near(p, mc, 1.5e-3, "sąlyga prieš Monte Carlo")
    print(f"SciPy palyginimas: sąlyga {p * 100:.2f}% (modelis) vs {mc * 100:.2f}% (Monte Carlo)")


def test_joint_model():
    """Bendras modelis: koeficientai iš žinomų parametrų (μh 88, μa 77, σh 11, σa 9, ρ 0.3), marža 5 %."""
    mh, ma, sh, sa, rho = 88, 77, 11, 9, 0.3
    cov = rho * sh * sa

    def gen(key, mu, sd, lines):
        out = []
        for l in lines:
            k = -l if key == "hcp" else l
            p = 1 - m.N.cdf((k - mu) / sd)
            out.append((l, 1 / (p * 1.025), 1 / ((1 - p) * 1.025)))
        return out

    G = {
        "hcp": m.fit("hcp", gen("hcp", mh - ma, math.sqrt(sh**2 + sa**2 - 2 * cov), [-16.5, -13.5, -10.5, -7.5, -4.5])),
        "tot": m.fit("tot", gen("tot", mh + ma, math.sqrt(sh**2 + sa**2 + 2 * cov), [157.5, 162.5, 165.5, 168.5, 173.5])),
        "home": m.fit("home", gen("home", mh, sh, [82.5, 86.5, 88.5, 90.5, 94.5])),
        "away": m.fit("away", gen("away", ma, sa, [72.5, 75.5, 77.5, 79.5, 82.5])),
    }
    M = m.score_model(G)
    near(M.mh, 88, 0.05, "bendras: namų vidurkis")
    near(M.ma, 77, 0.05, "bendras: svečių vidurkis")
    near(M.sh, 11, 0.05, "bendras: σ namų")
    near(M.sa, 9, 0.05, "bendras: σ svečių")
    near(M.rho, 0.3, 0.005, "bendras: ρ")
    near(M.rho_tot, 0.3, 0.005, "ρ pagal sumą")
    near(M.rho_hcp, 0.3, 0.005, "ρ pagal handikapą")
    assert M.src == "handikapas + suma + namų taškai + svečių taškai"
    near(m.score_model({"home": G["home"], "away": G["away"]}).rho, 0, 1e-6, "tik komandų taškai: ρ = 0")
    three = m.score_model({"hcp": G["hcp"], "tot": G["tot"], "home": G["home"]})
    near(three.sa, 9, 0.05, "be svečių rinkos: σ svečių")
    near(three.rho, 0.3, 0.005, "be svečių rinkos: ρ")
    assert m.score_model({"hcp": G["hcp"]}) is None


def test_quarter_sigma0():
    # Kėliniui su viena linija naudojama perduota (mažesnė) numatyta σ.
    one = [(-2.5, 1.90, 1.90)]
    assert m.fit("hcp", one).sigma == 12
    q = m.fit("hcp", one, sigma0=6)
    assert q.sigma == 6 and q.assumed
    near(q.mu, 2.5, 1e-9, "kėlinio vidurkis")


def test_period_share():
    FT = m.ScoreModel(mh=88, ma=78, sh=10, sa=10, rho=0.3, src="")
    truth = m.share_moments(FT, 0.27, 0.23, 80)
    pts = [(key, k, m.period_p_over(truth, key, k)) for key, ks in
           [("hcp", [-3.5, -1.5, 0.5, 2.5]), ("tot", [38.5, 41.5, 44.5]), ("home", [21.5, 24.5]), ("away", [16.5, 18.5])] for k in ks]
    Q, s = m.period_share_model(FT, pts)
    near(s["h"], 0.27, 1e-3, "namų kėlinio dalis")
    near(s["a"], 0.23, 1e-3, "svečių kėlinio dalis")
    near(s["c"], 80, 2, "koncentracija c")
    Q0, s0 = m.period_share_model(FT, [])
    assert s0["h"] == 0.25 and Q0.mh == 22
    assert m.period_share_model(None, pts) is None
    # Monte Carlo: dalys iš Beta, taškai iš dvimačio normaliojo -> momentai sutampa su formulėmis.
    if m.np is not None:
        rng = m.np.random.default_rng(1)
        n = 400_000
        z1, z2 = rng.standard_normal(n), rng.standard_normal(n)
        H = 88 + 10 * z1
        A = 78 + 10 * (0.3 * z1 + math.sqrt(1 - 0.09) * z2)
        qh = rng.beta(80 * 0.27, 80 * 0.73, n) * H
        qa = rng.beta(80 * 0.23, 80 * 0.77, n) * A
        near(float(qh.std()), truth.sh, 0.03, "MC kėlinio σ namų")
        near(float(qa.std()), truth.sa, 0.03, "MC kėlinio σ svečių")
        near(float(m.np.corrcoef(qh, qa)[0, 1]), truth.rho, 0.01, "MC kėlinio ρ")
        print(f"Kėlinio Monte Carlo: σ {qh.std():.3f} vs {truth.sh:.3f}, ρ {m.np.corrcoef(qh, qa)[0, 1]:.4f} vs {truth.rho:.4f}")


if __name__ == "__main__":
    F, M = test_model()
    test_joint_model()
    test_against_scipy(F, M)
    test_quarter_sigma0()
    test_period_share()
    print("Visi Python testai praėjo.")
