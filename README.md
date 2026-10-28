# event_probability_quater

Krepšinio rungtynių tikimybės iš bukmekerių koeficientų (kopija iš `event_probability`).

Viršuje pasirenkamas laikotarpis: **Pilnas laikas** arba **1–4 kėlinys**. Kiekvienas kėlinys turi savo
koeficientus (handikapas, taškų suma, komandų taškai) ir sąlygas, o skiltys „Koeficientai“,
„Tikslus rezultatas“ ir „Sąlygos“ veikia taip pat kaip pilnam laikui. Kėliniui lygiosios galimos,
todėl jos rodomos santraukoje ir tikslaus rezultato sąraše.

## Kaip skaičiuojamas kėlinys

- **Yra pilno laiko koeficientai** (bent tiek, kad būtų pilno laiko rezultato modelis): kėlinio taškai
  skaičiuojami kaip komandos visų rungtynių taškų dalis, taip pat kaip `event_probability_players`
  skaičiuoja žaidėjo dalį komandos taškuose. Dalis S ~ Beta(c·m, c·(1−m)), kėlinio taškai = S × komandos taškai.
  Dalys (namų ir svečių) ir koncentracija c parenkamos pagal to kėlinio koeficientus; jei kėlinio
  koeficientų nėra, imama 25 % ir c = 50.
  Namų ir svečių dalys koreliuotos per bendrą kėlinio tempą: koreliacija κ keičiama ranka puslapio viršuje
  (numatyta 0.3; κ = 0 reiškia nepriklausomas dalis, tada lieka tik iš pilno laiko paveldėta koreliacija).
- **Pilno laiko koeficientų nėra**: kėlinys skaičiuojamas iš savo koeficientų taip pat kaip pilnas laikas.

Tas pats modelis yra JavaScript (`app/model.js`), Python (`python/model.py`) ir R (`R/model.R`) versijose.
