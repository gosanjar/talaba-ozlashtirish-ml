"""Talabalar bazasi (SQLite, normallashgan tuzilma).

Tuzilma:  fakultetlar -> yonalishlar -> guruhlar (kurs = bosqich)
          fanlar (har biri o'z baholash tizimiga ega) + reja (qaysi yo'nalish/bosqichda qaysi fan)
          talabalar (guruhga bog'langan) + baholar (talaba x fan: joriy, oraliq, yakuniy — barchasi 0–100 ball)
          tizimlar (baholash tizimlari: chegaralar JSON) + sozlamalar (faol tizim)

Birinchi ishga tushishda ~1500 ta namunaviy talaba yaratiladi. Ular real qonuniyatlarga o'xshatib generatsiya
qilingan (yashirin omillar: qobiliyat, motivatsiya, ijtimoiy yuk). Haqiqiy tadqiqotda HEMIS ma'lumoti bilan almashtiring.
"""
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "talabalar.db"
QABUL_YILI_1KURS = 2026

STUDENT_TEXT = ["fio", "jinsi", "hudud", "tolov", "kontrakt_qarz", "stipendiya", "yashash", "ishlaydi", "ota_ona",
                "togarak", "olimpiada"]
STUDENT_NUM = ["yosh", "ish_soati", "yol_vaqti", "kirish_bali", "oldingi_ball", "oldingi_qarzlar", "nb_sababli",
               "nb_sababsiz", "topshiriq", "lms_kirish", "kutubxona", "mustaqil_soat"]
STUDENT_COLS = STUDENT_TEXT + STUDENT_NUM

SCHEMA = f"""
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS tizimlar (id INTEGER PRIMARY KEY, nomi TEXT UNIQUE NOT NULL, tavsif TEXT, chegaralar TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS sozlamalar (kalit TEXT PRIMARY KEY, qiymat TEXT);
CREATE TABLE IF NOT EXISTS fakultetlar (id INTEGER PRIMARY KEY, nomi TEXT UNIQUE NOT NULL);
CREATE TABLE IF NOT EXISTS yonalishlar (id INTEGER PRIMARY KEY, fakultet_id INTEGER NOT NULL REFERENCES fakultetlar(id),
  kod TEXT NOT NULL, nomi TEXT UNIQUE NOT NULL);
CREATE TABLE IF NOT EXISTS guruhlar (id INTEGER PRIMARY KEY, yonalish_id INTEGER NOT NULL REFERENCES yonalishlar(id),
  nomi TEXT UNIQUE NOT NULL, kurs INTEGER NOT NULL CHECK (kurs BETWEEN 1 AND 6));
CREATE TABLE IF NOT EXISTS fanlar (id INTEGER PRIMARY KEY, nomi TEXT UNIQUE NOT NULL, kredit INTEGER DEFAULT 6,
  tizim_id INTEGER NOT NULL REFERENCES tizimlar(id));
CREATE TABLE IF NOT EXISTS reja (yonalish_id INTEGER NOT NULL REFERENCES yonalishlar(id), kurs INTEGER NOT NULL,
  fan_id INTEGER NOT NULL REFERENCES fanlar(id), PRIMARY KEY (yonalish_id, kurs, fan_id));
CREATE TABLE IF NOT EXISTS talabalar (id INTEGER PRIMARY KEY AUTOINCREMENT,
  guruh_id INTEGER NOT NULL REFERENCES guruhlar(id),
  {", ".join(f"{c} TEXT" for c in STUDENT_TEXT)}, {", ".join(f"{c} REAL" for c in STUDENT_NUM)},
  created_at TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS baholar (talaba_id INTEGER NOT NULL REFERENCES talabalar(id) ON DELETE CASCADE,
  fan_id INTEGER NOT NULL REFERENCES fanlar(id), joriy REAL, oraliq REAL, yakuniy REAL,
  PRIMARY KEY (talaba_id, fan_id));
"""

# ---------------------------------------------------------------- boshlang'ich (seed) ma'lumotlar
TIZIMLAR = [
    ("5 baholik tizim (100 ball)", "O'zbekiston OTMlarida qo'llaniladigan an'anaviy tizim",
     [{"min": 86, "baho": "5", "nomi": "A'lo", "otish": True}, {"min": 71, "baho": "4", "nomi": "Yaxshi", "otish": True},
      {"min": 55, "baho": "3", "nomi": "Qoniqarli", "otish": True}, {"min": 0, "baho": "2", "nomi": "Qoniqarsiz", "otish": False}]),
    ("ECTS (A–F)", "Yevropa kredit-transfer tizimi",
     [{"min": 90, "baho": "A", "nomi": "A'lo", "otish": True}, {"min": 80, "baho": "B", "nomi": "Juda yaxshi", "otish": True},
      {"min": 70, "baho": "C", "nomi": "Yaxshi", "otish": True}, {"min": 60, "baho": "D", "nomi": "Qoniqarli", "otish": True},
      {"min": 50, "baho": "E", "nomi": "Yetarli", "otish": True}, {"min": 0, "baho": "F", "nomi": "O'tmadi", "otish": False}]),
    ("GPA 4.0", "Kredit-modul tizimi (AQSh uslubi)",
     [{"min": 90, "baho": "4.0", "nomi": "A", "otish": True}, {"min": 80, "baho": "3.0", "nomi": "B", "otish": True},
      {"min": 70, "baho": "2.0", "nomi": "C", "otish": True}, {"min": 60, "baho": "1.0", "nomi": "D", "otish": True},
      {"min": 0, "baho": "0.0", "nomi": "F", "otish": False}]),
]

STRUKTURA = {
    "Kompyuter injiniringi fakulteti": [("KI", "Kompyuter injiniringi"), ("AX", "Axborot xavfsizligi")],
    "Dasturiy injiniring fakulteti": [("DI", "Dasturiy injiniring"), ("SI", "Sun'iy intellekt")],
    "Raqamli iqtisodiyot fakulteti": [("RI", "Raqamli iqtisodiyot"), ("AT", "Axborot tizimlari")],
}
# fan: (qiyinlik, tur)   tur: mat / dast / til / gum
FANLAR = {
    "Oliy matematika": (-4, "mat"), "Fizika": (-3, "mat"), "Ingliz tili": (3, "til"), "Dasturlash asoslari": (0, "dast"),
    "Diskret matematika": (-3, "mat"), "O'zbekiston tarixi": (5, "gum"),
    "Algoritmlar va ma'lumotlar tuzilmalari": (-3, "dast"), "Ma'lumotlar bazasi": (1, "dast"),
    "Ehtimollar nazariyasi": (-4, "mat"), "Obyektga yo'naltirilgan dasturlash": (-1, "dast"), "Kompyuter tarmoqlari": (0, "dast"),
    "Operatsion tizimlar": (-1, "dast"), "Falsafa": (5, "gum"), "Kompyuter arxitekturasi": (-2, "mat"),
    "Mikroprotsessorlar": (-3, "mat"), "Raqamli elektronika": (-2, "mat"), "Kriptografiya": (-4, "mat"),
    "Tarmoq xavfsizligi": (-1, "dast"), "Kiberxavfsizlik asoslari": (1, "dast"), "Dasturiy ta'minot muhandisligi": (0, "dast"),
    "Web dasturlash": (2, "dast"), "Mobil ilovalar": (1, "dast"), "Mashinali o'qitish": (-3, "mat"),
    "Sun'iy intellekt asoslari": (-1, "dast"), "Kompyuter ko'rish": (-3, "mat"), "Ekonometrika": (-2, "mat"),
    "Raqamli marketing": (4, "gum"), "Mikroiqtisodiyot": (1, "gum"), "Axborot tizimlarini loyihalash": (0, "dast"),
    "ERP tizimlari": (2, "dast"), "Biznes tahlil": (2, "gum"), "Loyiha boshqaruvi": (3, "gum"), "Malakaviy amaliyot": (6, "gum"),
    "Ma'lumotlar tahlili": (-1, "mat"),
}
REJA_UMUMIY = {
    1: ["Oliy matematika", "Fizika", "Ingliz tili", "Dasturlash asoslari", "Diskret matematika", "O'zbekiston tarixi"],
    2: ["Algoritmlar va ma'lumotlar tuzilmalari", "Ma'lumotlar bazasi", "Ehtimollar nazariyasi", "Obyektga yo'naltirilgan dasturlash", "Kompyuter tarmoqlari"],
    3: ["Operatsion tizimlar", "Falsafa"],
    4: ["Loyiha boshqaruvi", "Malakaviy amaliyot"],
}
REJA_MAXSUS = {
    "KI": {2: ["Raqamli elektronika"], 3: ["Kompyuter arxitekturasi", "Mikroprotsessorlar", "Raqamli elektronika", "Ma'lumotlar tahlili"], 4: ["Kompyuter ko'rish", "Mikroprotsessorlar", "Web dasturlash", "Ma'lumotlar tahlili"]},
    "AX": {2: ["Kiberxavfsizlik asoslari"], 3: ["Kriptografiya", "Tarmoq xavfsizligi", "Kiberxavfsizlik asoslari", "Web dasturlash"], 4: ["Kriptografiya", "Tarmoq xavfsizligi", "Mobil ilovalar", "Ma'lumotlar tahlili"]},
    "DI": {2: ["Web dasturlash"], 3: ["Dasturiy ta'minot muhandisligi", "Web dasturlash", "Mobil ilovalar", "Ma'lumotlar tahlili"], 4: ["Dasturiy ta'minot muhandisligi", "Mobil ilovalar", "Mashinali o'qitish", "ERP tizimlari"]},
    "SI": {2: ["Ma'lumotlar tahlili"], 3: ["Mashinali o'qitish", "Sun'iy intellekt asoslari", "Ma'lumotlar tahlili", "Kompyuter ko'rish"], 4: ["Mashinali o'qitish", "Kompyuter ko'rish", "Sun'iy intellekt asoslari", "Web dasturlash"]},
    "RI": {2: ["Mikroiqtisodiyot"], 3: ["Ekonometrika", "Raqamli marketing", "Mikroiqtisodiyot", "Biznes tahlil"], 4: ["Ekonometrika", "Raqamli marketing", "ERP tizimlari", "Biznes tahlil"]},
    "AT": {2: ["Biznes tahlil"], 3: ["Axborot tizimlarini loyihalash", "ERP tizimlari", "Web dasturlash", "Biznes tahlil"], 4: ["Axborot tizimlarini loyihalash", "ERP tizimlari", "Ma'lumotlar tahlili", "Mobil ilovalar"]},
}
YON_EFFECT = {"KI": 1, "AX": 0, "DI": 2, "SI": 3, "RI": -2, "AT": -1}
M_NAMES = ["Jasur", "Sardor", "Bekzod", "Javohir", "Otabek", "Diyorbek", "Asilbek", "Shohruh", "Abdulloh", "Sherzod",
           "Ulug'bek", "Doston", "Islom", "Azizbek", "Behruz", "Firdavs", "Muhammadali", "Kamron", "Jahongir", "Ozodbek",
           "Bobur", "Sanjar", "Akmal", "Rustam", "Eldor", "Nodirbek", "Shoxjahon", "Temur", "Alisher", "Farrux"]
F_NAMES = ["Madina", "Dilnoza", "Sevara", "Nilufar", "Malika", "Zarina", "Shahzoda", "Gulnoza", "Mohinur", "Aziza",
           "Sabina", "Kamola", "Nodira", "Dildora", "Munisa", "Robiya", "Farangiz", "Laylo", "Shohista", "Mehribon",
           "Gulruh", "Marjona", "Durdona", "Feruza", "Iroda", "Charos", "Zilola", "Oydin", "Hilola", "Barno"]
SURNAMES = ["Karimov", "Rahimov", "Tursunov", "Aliyev", "Yusupov", "Normatov", "Ergashev", "Qodirov", "Saidov",
            "Xolmatov", "Abdullayev", "Mirzayev", "Sobirov", "Islomov", "Toshpo'latov", "Hasanov", "Nazarov",
            "Raximberdiyev", "Jo'rayev", "Umarov", "Ismoilov", "Bozorov", "Ortiqov", "Zokirov", "Valiyev", "Haydarov",
            "Sharipov", "Musayev", "Qosimov", "Fayzullayev", "Olimov", "Bakirov", "Salimov", "Yo'ldoshev", "Rustamov"]
REGIONS = ["Toshkent sh.", "Toshkent vil.", "Samarqand", "Farg'ona", "Andijon", "Namangan", "Buxoro", "Qashqadaryo",
           "Surxondaryo", "Xorazm", "Jizzax", "Sirdaryo", "Navoiy", "Qoraqalpog'iston"]


def guruh_nomi(kod, kurs, n=1):
    return f"{kod}{(QABUL_YILI_1KURS - kurs + 1) % 100:02d}-{n}"


def conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA foreign_keys = ON")
    return c


def q(sql, args=()):
    with conn() as c:
        return [dict(r) for r in c.execute(sql, args).fetchall()]


def exe(sql, args=()):
    with conn() as c:
        cur = c.execute(sql, args)
        return cur.lastrowid


def _seed(c, n_students=1500, seed=11, unknown_share=0.15):
    rng = np.random.default_rng(seed)
    sig = lambda x: 1 / (1 + np.exp(-x))
    for nomi, tavsif, ch in TIZIMLAR:
        c.execute("INSERT INTO tizimlar (nomi, tavsif, chegaralar) VALUES (?,?,?)", (nomi, tavsif, json.dumps(ch, ensure_ascii=False)))
    c.execute("INSERT INTO sozlamalar VALUES ('faol_tizim', '1')")
    fan_id = {}
    for nomi in FANLAR:
        fan_id[nomi] = c.execute("INSERT INTO fanlar (nomi, kredit, tizim_id) VALUES (?,?,1)", (nomi, int(rng.choice([4, 5, 6])))).lastrowid
    groups = []
    for fak, ys in STRUKTURA.items():
        fid = c.execute("INSERT INTO fakultetlar (nomi) VALUES (?)", (fak,)).lastrowid
        for kod, nom in ys:
            yid = c.execute("INSERT INTO yonalishlar (fakultet_id, kod, nomi) VALUES (?,?,?)", (fid, kod, nom)).lastrowid
            for kurs in (1, 2, 3, 4):
                fanlar = list(dict.fromkeys(REJA_UMUMIY[kurs] + REJA_MAXSUS[kod].get(kurs, [])))[:6]
                for f in fanlar:
                    c.execute("INSERT INTO reja VALUES (?,?,?)", (yid, kurs, fan_id[f]))
                for gn in (1, 2):
                    gid = c.execute("INSERT INTO guruhlar (yonalish_id, nomi, kurs) VALUES (?,?,?)",
                                    (yid, guruh_nomi(kod, kurs, gn), kurs)).lastrowid
                    groups.append((gid, kod, kurs, fanlar))
    order = rng.permutation(np.resize(np.arange(len(groups)), n_students))
    used = set()
    for i in range(n_students):
        gid, kod, kurs, fanlar = groups[order[i]]
        a, m, s = rng.normal(), rng.normal(), rng.normal()          # qobiliyat, motivatsiya, ijtimoiy yuk
        apt = {t: a + rng.normal(0, 0.4) for t in ("mat", "dast", "til", "gum")}
        sex = str(rng.choice(["Erkak", "Ayol"], p=[0.55, 0.45]))
        for _ in range(60):
            fio = f"{rng.choice(SURNAMES)}{'a' if sex == 'Ayol' else ''} {rng.choice(M_NAMES if sex == 'Erkak' else F_NAMES)}"
            if fio not in used: break
        else:
            fio = f"{fio} ({i})"
        used.add(fio)
        hudud = str(rng.choice(REGIONS, p=np.r_[0.2, 0.08, np.full(12, 0.72 / 12)]))
        p_oliy = sig(0.3 * a - 0.5 * s + 0.2) * 0.6
        ota_ona = str(rng.choice(["Oliy", "O'rta maxsus", "O'rta"], p=[p_oliy, 0.35, 0.65 - p_oliy]))
        kirish = float(np.clip(135 + 22 * a + 4 * (ota_ona == "Oliy") + rng.normal(0, 9), 60, 189))
        tolov = "Grant" if rng.random() < sig((kirish - 160) / 7) else "Kontrakt"
        qarz = "Ha" if tolov == "Kontrakt" and rng.random() < sig(-1.6 + 0.9 * s - 0.4 * m) else "Yo'q"
        ishlaydi = "Ha" if rng.random() < sig(-2.1 + 0.9 * s + 0.6 * (tolov == "Kontrakt") + 0.35 * kurs) else "Yo'q"
        ish_soati = int(np.clip(rng.normal(18 + 6 * s, 7), 4, 45)) if ishlaydi == "Ha" else 0
        yashash = str(rng.choice(["Uyda", "Yotoqxona", "Ijara"], p=[0.45, 0.3, 0.25] if hudud.startswith("Toshkent") else [0.15, 0.45, 0.4]))
        yol = int(np.clip(rng.normal(55 if yashash == "Uyda" else 25, 18), 5, 120))
        oq = 0 if kurs == 1 else int(np.clip(rng.poisson(np.exp(-0.9 - 0.9 * a - 0.6 * m + 0.3 * s)), 0, 6))
        nb = int(np.clip(rng.poisson(np.exp(1.6 - 0.75 * m + 0.4 * (ishlaydi == "Ha") + 0.004 * yol + 0.15 * s)) * 2, 0, 90))
        nbs = int(rng.poisson(2.5 + 1.5 * (s > 1)) * 2)
        top = float(np.clip(100 * sig(1.0 + 1.1 * m + 0.4 * a - 0.02 * ish_soati + rng.normal(0, 0.4)), 5, 100))
        lms = int(np.clip(rng.poisson(np.exp(2.9 + 0.45 * m + 0.15 * a)), 0, 150))
        kut = int(np.clip(rng.poisson(np.exp(0.6 + 0.5 * m + 0.2 * a + 0.3 * (yashash == "Yotoqxona"))), 0, 30))
        mus = float(np.clip(rng.normal(8 + 3.5 * m + 1.5 * a - 0.15 * ish_soati, 3), 0, 40))
        tog = "Ha" if rng.random() < sig(-1.0 + 0.7 * m + 0.3 * a) else "Yo'q"
        oli = "Ha" if rng.random() < sig(-3.3 + 1.4 * a + 0.5 * m) else "Yo'q"
        old = float(np.clip(70 + 9 * a + 5 * m - 3 * oq + rng.normal(0, 4), 30, 99)) if kurs > 1 else float(np.clip(70 + 9 * a + 4 * m + rng.normal(0, 6), 35, 99))
        stip = "Ha" if tolov == "Grant" and old >= 71 else "Yo'q"
        vals = dict(fio=fio, jinsi=sex, hudud=hudud, tolov=tolov, kontrakt_qarz=qarz, stipendiya=stip, yashash=yashash,
                    ishlaydi=ishlaydi, ota_ona=ota_ona, togarak=tog, olimpiada=oli, yosh=int(17 + kurs + rng.integers(0, 3)),
                    ish_soati=ish_soati, yol_vaqti=yol, kirish_bali=round(kirish, 1), oldingi_ball=round(old, 1),
                    oldingi_qarzlar=oq, nb_sababli=nbs, nb_sababsiz=nb, topshiriq=round(top), lms_kirish=lms,
                    kutubxona=kut, mustaqil_soat=round(mus, 1))
        tid = c.execute(f"INSERT INTO talabalar (guruh_id, {', '.join(STUDENT_COLS)}) VALUES (?, {', '.join('?' * len(STUDENT_COLS))})",
                        [gid] + [vals[k] for k in STUDENT_COLS]).lastrowid
        known = rng.random() > unknown_share
        effort = 0.12 * (top - 70) + 0.05 * (lms - 20) + 0.25 * (mus - 8) + 1.5 * (tog == "Ha")
        nb_out = nb >= 40 and rng.random() < 0.6
        for f in fanlar:
            dif, tur = FANLAR[f]
            base = 76 + dif + YON_EFFECT[kod] + 8 * (apt[tur] + rng.normal(0, 0.3)) + 4.5 * m - 1.2 * s
            joriy = np.clip(base + 0.6 * effort - 0.12 * nb + rng.normal(0, 5), 5, 100)
            oraliq = np.clip(base + 0.3 * effort - 0.18 * nb + rng.normal(0, 5.5), 5, 100)
            yak = (0.35 * oraliq + 0.2 * joriy + 0.45 * (base + 0.5 * effort) - 0.08 * nb - 0.12 * max(0, ish_soati - 20)
                   - 3.5 * (qarz == "Ha") - 1.5 * oq + 2.5 * (oli == "Ha") + rng.normal(0, 3.5))
            if nb_out and rng.random() < 0.8:                    # NB limiti oshgan: imtihonga kiritilmaslik
                yak = min(yak, rng.uniform(25, 54))
            c.execute("INSERT INTO baholar VALUES (?,?,?,?,?)", (tid, fan_id[f], round(float(joriy)), round(float(oraliq)),
                                                                 round(float(np.clip(yak, 0, 100))) if known else None))


def init_db():
    DB_PATH.parent.mkdir(exist_ok=True)
    with conn() as c:
        c.executescript(SCHEMA)
        if c.execute("SELECT COUNT(*) FROM tizimlar").fetchone()[0] == 0:
            _seed(c)


# ---------------------------------------------------------------- o'qish
def tizimlar():
    out = q("SELECT * FROM tizimlar ORDER BY id")
    for t in out:
        t["chegaralar"] = sorted(json.loads(t["chegaralar"]), key=lambda x: -x["min"])
    return out


def faol_tizim_id():
    r = q("SELECT qiymat FROM sozlamalar WHERE kalit='faol_tizim'")
    return int(r[0]["qiymat"]) if r else 1


def load_students() -> pd.DataFrame:
    with conn() as c:
        return pd.read_sql("""SELECT t.*, g.nomi AS guruh, g.kurs, y.nomi AS yonalish, y.kod AS yonalish_kod, y.id AS yonalish_id,
                                     f.nomi AS fakultet, f.id AS fakultet_id
                              FROM talabalar t JOIN guruhlar g ON g.id=t.guruh_id JOIN yonalishlar y ON y.id=g.yonalish_id
                              JOIN fakultetlar f ON f.id=y.fakultet_id ORDER BY t.id""", c)


def load_grades() -> pd.DataFrame:
    with conn() as c:
        return pd.read_sql("""SELECT b.talaba_id, b.fan_id, f.nomi AS fan, f.kredit, f.tizim_id, b.joriy, b.oraliq, b.yakuniy
                              FROM baholar b JOIN fanlar f ON f.id=b.fan_id""", c)


def tree():
    fak = q("SELECT * FROM fakultetlar ORDER BY nomi")
    yon = q("SELECT y.*, (SELECT COUNT(*) FROM guruhlar g JOIN talabalar t ON t.guruh_id=g.id WHERE g.yonalish_id=y.id) AS talabalar FROM yonalishlar y ORDER BY nomi")
    gr = q("SELECT g.*, (SELECT COUNT(*) FROM talabalar t WHERE t.guruh_id=g.id) AS talabalar FROM guruhlar g ORDER BY kurs, nomi")
    rj = q("SELECT r.*, f.nomi FROM reja r JOIN fanlar f ON f.id=r.fan_id ORDER BY f.nomi")
    for f in fak:
        f["yonalishlar"] = [y for y in yon if y["fakultet_id"] == f["id"]]
        for y in f["yonalishlar"]:
            y["guruhlar"] = [g for g in gr if g["yonalish_id"] == y["id"]]
            y["reja"] = {k: [{"id": r["fan_id"], "nomi": r["nomi"]} for r in rj if r["yonalish_id"] == y["id"] and r["kurs"] == k] for k in range(1, 7)}
    return fak


def fanlar():
    return q("""SELECT f.*, t.nomi AS tizim, (SELECT COUNT(*) FROM baholar b WHERE b.fan_id=f.id) AS baholar_soni,
                (SELECT COUNT(*) FROM reja r WHERE r.fan_id=f.id) AS rejada FROM fanlar f JOIN tizimlar t ON t.id=f.tizim_id ORDER BY f.nomi""")


def group_subjects(guruh_id):
    return q("""SELECT f.id, f.nomi, f.tizim_id FROM guruhlar g JOIN reja r ON r.yonalish_id=g.yonalish_id AND r.kurs=g.kurs
                JOIN fanlar f ON f.id=r.fan_id WHERE g.id=? ORDER BY f.nomi""", (guruh_id,))


def student_grades(sid):
    return q("""SELECT b.fan_id, f.nomi AS fan, f.tizim_id, b.joriy, b.oraliq, b.yakuniy FROM baholar b JOIN fanlar f ON f.id=b.fan_id
                WHERE b.talaba_id=? ORDER BY f.nomi""", (sid,))


# ---------------------------------------------------------------- yozish
def save_student(rec: dict, grades: list, sid=None) -> int:
    with conn() as c:
        if sid is None:
            sid = c.execute(f"INSERT INTO talabalar (guruh_id, {', '.join(STUDENT_COLS)}) VALUES (?, {', '.join('?' * len(STUDENT_COLS))})",
                            [rec["guruh_id"]] + [rec.get(k) for k in STUDENT_COLS]).lastrowid
        else:
            c.execute(f"UPDATE talabalar SET guruh_id=?, {', '.join(f'{k}=?' for k in STUDENT_COLS)} WHERE id=?",
                      [rec["guruh_id"]] + [rec.get(k) for k in STUDENT_COLS] + [sid])
            c.execute("DELETE FROM baholar WHERE talaba_id=?", (sid,))
        for g in grades:
            c.execute("INSERT INTO baholar VALUES (?,?,?,?,?)", (sid, g["fan_id"], g["joriy"], g["oraliq"], g.get("yakuniy")))
    return sid


def delete_student(sid):
    exe("DELETE FROM talabalar WHERE id=?", (sid,))


# ---------------------------------------------------------------- boshqaruv (admin)
ADMIN_TABLES = {
    "fakultetlar": ["nomi"],
    "yonalishlar": ["fakultet_id", "kod", "nomi"],
    "guruhlar": ["yonalish_id", "nomi", "kurs"],
    "fanlar": ["nomi", "kredit", "tizim_id"],
}


def admin_save(table, data: dict, rid=None):
    cols = ADMIN_TABLES[table]
    vals = [data.get(c) for c in cols]
    if rid is None:
        return exe(f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})", vals)
    exe(f"UPDATE {table} SET {', '.join(f'{c}=?' for c in cols)} WHERE id=?", vals + [rid])
    return rid


def admin_delete(table, rid):
    if table == "yonalishlar":
        with conn() as c:
            if c.execute("SELECT COUNT(*) FROM guruhlar WHERE yonalish_id=?", (rid,)).fetchone()[0]:
                raise sqlite3.IntegrityError("FOREIGN KEY constraint failed")
            c.execute("DELETE FROM reja WHERE yonalish_id=?", (rid,))
            c.execute("DELETE FROM yonalishlar WHERE id=?", (rid,))
        return
    if table == "fanlar":
        with conn() as c:
            if c.execute("SELECT COUNT(*) FROM baholar WHERE fan_id=?", (rid,)).fetchone()[0]:
                raise sqlite3.IntegrityError("FOREIGN KEY constraint failed")
            c.execute("DELETE FROM reja WHERE fan_id=?", (rid,))
            c.execute("DELETE FROM fanlar WHERE id=?", (rid,))
        return
    exe(f"DELETE FROM {table} WHERE id=?", (rid,))


def reja_add(yonalish_id, kurs, fan_id):
    exe("INSERT OR IGNORE INTO reja VALUES (?,?,?)", (yonalish_id, kurs, fan_id))


def reja_remove(yonalish_id, kurs, fan_id):
    exe("DELETE FROM reja WHERE yonalish_id=? AND kurs=? AND fan_id=?", (yonalish_id, kurs, fan_id))


def tizim_save(data: dict, rid=None):
    ch = json.dumps(sorted(data["chegaralar"], key=lambda x: -x["min"]), ensure_ascii=False)
    if rid is None:
        return exe("INSERT INTO tizimlar (nomi, tavsif, chegaralar) VALUES (?,?,?)", (data["nomi"], data.get("tavsif"), ch))
    exe("UPDATE tizimlar SET nomi=?, tavsif=?, chegaralar=? WHERE id=?", (data["nomi"], data.get("tavsif"), ch, rid))
    return rid


def tizim_delete(rid):
    if rid == faol_tizim_id():
        raise sqlite3.IntegrityError("faol tizim")
    exe("DELETE FROM tizimlar WHERE id=?", (rid,))


def set_faol_tizim(rid):
    exe("INSERT OR REPLACE INTO sozlamalar VALUES ('faol_tizim', ?)", (str(int(rid)),))


def data_signature():
    r = q("""SELECT (SELECT COUNT(*) FROM talabalar) a, (SELECT MAX(id) FROM talabalar) b, (SELECT COUNT(*) FROM baholar) c,
                    (SELECT TOTAL(yakuniy) + TOTAL(oraliq) + TOTAL(joriy) FROM baholar) d,
                    (SELECT TOTAL(nb_sababsiz) + TOTAL(topshiriq) + TOTAL(kirish_bali) FROM talabalar) e,
                    (SELECT GROUP_CONCAT(id || ':' || tizim_id || ':' || kredit) FROM fanlar) f""")[0]
    return repr(tuple(r.values()))
