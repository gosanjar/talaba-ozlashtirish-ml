"""Talabalar o'zlashtirishini tahlil qilish tizimi — FastAPI server.

Ishga tushirish:  uvicorn app.main:app --port 8000
Tashqi (ngrok) so'rovlar parol bilan himoyalanadi: login/parol data/auth.txt faylida.
"""
import base64
import io
import secrets
import sqlite3
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent))
import db  # noqa: E402
import ml  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ENGINE_PATH = ROOT / "data" / "engine.joblib"
AUTH_PATH = ROOT / "data" / "auth.txt"
STATIC = Path(__file__).resolve().parent / "static"

OPTIONS = {
    "jinsi": ["Erkak", "Ayol"], "tolov": ["Grant", "Kontrakt"], "kontrakt_qarz": ["Yo'q", "Ha"], "stipendiya": ["Yo'q", "Ha"],
    "yashash": ["Uyda", "Yotoqxona", "Ijara"], "ishlaydi": ["Yo'q", "Ha"], "ota_ona": ["Oliy", "O'rta maxsus", "O'rta"],
    "togarak": ["Yo'q", "Ha"], "olimpiada": ["Yo'q", "Ha"], "hudud": db.REGIONS,
}
NUM_RANGES = {
    "yosh": (16, 40), "ish_soati": (0, 60), "yol_vaqti": (0, 180), "kirish_bali": (0, 189), "oldingi_ball": (0, 100),
    "oldingi_qarzlar": (0, 20), "nb_sababli": (0, 200), "nb_sababsiz": (0, 200), "topshiriq": (0, 100), "lms_kirish": (0, 300),
    "kutubxona": (0, 60), "mustaqil_soat": (0, 60),
}


# =============================================================================== holat
class State:
    def __init__(self):
        self.engine: Optional[ml.Engine] = None
        self.lock = threading.RLock()
        self.version = 0
        self.cache = None                      # (version, DataFrame D, preds_by_id, S)
        self.job = {"busy": False, "kind": None, "message": "", "error": None, "started": None, "finished": None}
        self.pending = None                    # navbatdagi ish: "full" | "refit"


ST = State()


def grading():
    return ml.Grading(db.tizimlar(), db.faol_tizim_id())


def _run_jobs():
    while True:
        with ST.lock:
            kind, ST.pending = ST.pending, None
            if kind is None:
                ST.job["busy"] = False
                return
            ST.job.update(busy=True, kind=kind, error=None, started=time.time(),
                          message="To'liq o'qitish boshlandi…" if kind == "full" else "Model yangi ma'lumot bilan qayta o'qitilmoqda…")
        try:
            students, grades, G = db.load_students(), db.load_grades(), grading()
            sig = (db.data_signature(), G.signature())
            if kind == "full" or ST.engine is None:
                eng = ml.Engine()
                eng.train_full(students, grades, G, status=lambda m: ST.job.update(message=m))
            else:
                eng = joblib.load(ENGINE_PATH) if ENGINE_PATH.exists() else ST.engine
                eng.refit(students, grades, G)
                eng.metrics["refit_at"] = time.strftime("%Y-%m-%d %H:%M")
            eng.data_sig = sig
            eng.engine_version = ml.ENGINE_VERSION
            joblib.dump(eng, ENGINE_PATH)
            with ST.lock:
                ST.engine = eng
                ST.version += 1
            ST.job.update(message="Tayyor", finished=time.time())
        except Exception as e:  # noqa: BLE001
            ST.job.update(error=str(e), message="Xatolik: " + str(e), finished=time.time())


def schedule(kind):
    """Fon rejimida o'qitish. 'full' 'refit'dan ustun turadi; ish davomida kelgan so'rovlar navbatga qo'shiladi."""
    with ST.lock:
        if ST.pending != "full":
            ST.pending = kind
        if not ST.job["busy"]:
            ST.job["busy"] = True
            threading.Thread(target=_run_jobs, daemon=True).start()


def invalidate(drop_oof=()):
    with ST.lock:
        ST.version += 1
        if ST.engine is not None:
            for i in drop_oof:
                ST.engine.oof.pop(int(i), None)


def need_engine():
    if ST.engine is None:
        raise HTTPException(503, ST.job.get("message") or "Model o'qitilmoqda, biroz kuting…")
    return ST.engine


def data():
    """Barcha talabalar bo'yicha bashoratlar (versiya bo'yicha keshlanadi)."""
    eng = need_engine()
    with ST.lock:
        if ST.cache and ST.cache[0] == ST.version:
            return ST.cache[1], ST.cache[2], ST.cache[3]
        v = ST.version
    students, grades = db.load_students(), db.load_grades()
    preds, S = eng.predict(students, grades)
    P = pd.DataFrame([{k: v2 for k, v2 in p.items() if k not in ("subjects", "toifa_proba", "flags")} for p in preds])
    P["flags_n"] = [len(p["flags"]) for p in preds]
    keep = ["id", "fio", "guruh_id", "guruh", "kurs", "yonalish", "yonalish_kod", "yonalish_id", "fakultet", "fakultet_id",
            "yakuniy_ortacha", "qarz_soni", "oraliq_ortacha", "joriy_ortacha", "past_fanlar", "fanlar_soni", "trend"] + db.STUDENT_COLS[1:]
    D = S[keep].merge(P, on="id")
    D["yakuniy_ortacha"] = D["yakuniy_ortacha"].astype(float)
    D["natija"] = D["yakuniy_ortacha"].fillna(D["ball"])          # ma'lum bo'lsa haqiqiy, aks holda bashorat
    D["natija_manba"] = np.where(D["yakuniy_ortacha"].notna(), "haqiqiy", "bashorat")
    G = eng.G
    D["natija_toifa"] = [G.categories()[G.cat_index(b)] for b in D["natija"]]
    D["xavf"] = np.select([D["chet_proba"] >= 0.5, D["fail_proba"] >= 0.5, D["fail_proba"] >= 0.25], ["yuqori", "o'rta", "past"], "yo'q")
    by_id = {p["id"]: p for p in preds}
    with ST.lock:
        if ST.version == v:
            ST.cache = (v, D, by_id, S)
    return D, by_id, S


def filt(D, fakultet_id=None, yonalish_id=None, guruh_id=None, kurs=None):
    if fakultet_id: D = D[D["fakultet_id"] == fakultet_id]
    if yonalish_id: D = D[D["yonalish_id"] == yonalish_id]
    if guruh_id: D = D[D["guruh_id"] == guruh_id]
    if kurs: D = D[D["kurs"] == kurs]
    return D


def r1(v, n=1):
    if v is None or (isinstance(v, float) and np.isnan(v)): return None
    return round(float(v), n)


def clean(v):
    if isinstance(v, dict): return {k: clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)): return [clean(x) for x in v]
    if isinstance(v, (np.integer,)): return int(v)
    if isinstance(v, (np.floating, float)): return None if np.isnan(v) else round(float(v), 3)
    if isinstance(v, np.bool_): return bool(v)
    return v


def unit_stats(d: pd.DataFrame):
    if len(d) == 0:
        return {"n": 0}
    G = ST.engine.G
    cats = G.categories()
    return {
        "n": int(len(d)), "ortacha": r1(d["natija"].mean()), "bashorat": r1(d["ball"].mean()),
        "oraliq": r1(d["oraliq_ortacha"].mean()), "nb": r1(d["nb_sababsiz"].mean()),
        "topshiriq": r1(d["topshiriq"].mean()),
        "otmaydi": r1((d["fail_proba"] >= 0.5).mean() * 100), "chetlash": r1((d["chet_proba"] >= 0.5).mean() * 100),
        "grant": r1((d["tolov"] == "Grant").mean() * 100), "qarzdor": r1((d["kontrakt_qarz"] == "Ha").mean() * 100),
        "alo_dav": int(((d["alo_proba"] >= 0.5)).sum()), "anomal": int(d["anomaly"].sum()),
        "toifa": {c: int((d["natija_toifa"] == c).sum()) for c in cats},
        "klaster": {n: int((d["cluster_name"] == n).sum()) for n in ST.engine.cluster_names.values()},
        "haqiqiy_ulush": r1((d["natija_manba"] == "haqiqiy").mean() * 100),
    }


def student_brief(r):
    return {"id": int(r["id"]), "fio": r["fio"], "guruh": r["guruh"], "kurs": int(r["kurs"]), "yonalish_kod": r["yonalish_kod"],
            "tolov": r["tolov"], "ball": r1(r["ball"]), "natija": r1(r["natija"]), "natija_manba": r["natija_manba"],
            "yakuniy": r1(r["yakuniy_ortacha"]), "baho": r["baho"], "toifa": r["toifa"], "fail_proba": r1(r["fail_proba"], 3),
            "chet_proba": r1(r["chet_proba"], 3), "alo_proba": r1(r["alo_proba"], 3), "pasayish_proba": r1(r["pasayish_proba"], 3),
            "cluster": int(r["cluster"]), "cluster_name": r["cluster_name"], "anomaly": bool(r["anomaly"]),
            "nb": r1(r["nb_sababsiz"], 0), "oraliq": r1(r["oraliq_ortacha"]), "xavf": r["xavf"], "qarz_kutilgan": r1(r["qarz_kutilgan"], 2)}


# =============================================================================== maslahatlar
def advice_for(row: pd.Series, p: dict):
    out = []
    G = ST.engine.G
    if p["chet_proba"] >= 0.5:
        out.append(("danger", f"Chetlashtirish xavfi yuqori ({p['chet_proba']*100:.0f}%): 3 va undan ortiq fandan qarz kutilmoqda. "
                              "Dekanat, kurator va ota-ona bilan uchrashuv, individual reja tuzish tavsiya etiladi."))
    elif p["fail_proba"] >= 0.5:
        out.append(("danger", f"Akademik qarz ehtimoli {p['fail_proba']*100:.0f}%. Qarz kutilayotgan fanlar bo'yicha qo'shimcha konsultatsiya kerak."))
    weak = [s for s in p["subjects"] if s["fail_p"] >= 0.35]
    if weak:
        out.append(("warn", "Xavfli fanlar: " + ", ".join(f"{s['fan']} ({s['pred']:.0f} ball, qarz ehtimoli {s['fail_p']*100:.0f}%)" for s in weak[:4])
                    + ". Shu fan o'qituvchilari bilan qayta topshirish rejasini kelishing."))
    nb = row.get("nb_sababsiz", 0) or 0
    if nb >= 36:
        out.append(("danger", f"Sababsiz NB {nb:.0f} soat: limitga yetgan. Imtihonga kiritilmaslik xavfi bor, darhol kurator nazoratiga oling."))
    elif nb >= 20:
        out.append(("warn", f"Sababsiz NB {nb:.0f} soat: davomatni haftalik nazorat qiling."))
    if (row.get("topshiriq") or 100) < 55:
        out.append(("warn", f"Topshiriqlar atigi {row['topshiriq']:.0f}% bajarilgan. Muddatlar jadvali va mustaqil ish nazorati kerak."))
    if (row.get("ish_soati") or 0) >= 25:
        out.append(("info", f"Talaba haftasiga {row['ish_soati']:.0f} soat ishlaydi. Ish yuki o'qishga ta'sir qilmoqda, moslashuvchan grafik muhokama qiling."))
    if row.get("kontrakt_qarz") == "Ha":
        out.append(("info", "Kontrakt qarzdorligi bor: bo'lib to'lash yoki imtiyozlar bo'yicha maslahat bering (moliyaviy stress o'zlashtirishga ta'sir qiladi)."))
    if p["alo_proba"] >= 0.5 and row.get("tolov") == "Kontrakt":
        out.append(("good", f"A'lo natija ehtimoli {p['alo_proba']*100:.0f}%: grantga o'tkazish uchun da'vogar. Olimpiada va ilmiy to'garaklarga jalb qiling."))
    if row.get("tolov") == "Grant" and p["pasayish_proba"] >= 0.3:
        out.append(("warn", f"Grant talabasi: o'rtacha ball {ml.GRANT_SAQLASH} dan pastga tushish ehtimoli {p['pasayish_proba']*100:.0f}%. Stipendiya xavf ostida."))
    tr = row.get("trend")
    if tr is not None and not pd.isna(tr) and tr <= -10:
        out.append(("warn", f"Oldingi semestrga nisbatan {tr:.0f} ball pasayish bor. Sababini aniqlash uchun suhbat o'tkazing."))
    if p["cluster_name"] == "Faol, lekin bahosi past":
        out.append(("info", "Talaba faol, lekin natija past: o'qish usullari, tyutorlik yoki juftlik (peer) o'qitish samarali bo'ladi."))
    if p["anomaly"]:
        out.append(("info", "Ko'rsatkichlar g'ayrioddiy kombinatsiyada: ma'lumotlarni tekshiring (kiritishda xato bo'lishi mumkin)."))
    if not out:
        out.append(("good", f"Xavf belgilari yo'q. Kutilayotgan natija {p['ball']:.0f} ball ({G.grade(p['ball'])['nomi']})."))
    return [{"tur": t, "matn": m} for t, m in out]


# =============================================================================== ilova
app = FastAPI(title="Talabalar o'zlashtirishini tahlil qilish tizimi")


def _credentials():
    if not AUTH_PATH.exists():
        AUTH_PATH.write_text(f"admin:{secrets.token_urlsafe(9)}\n")
    return AUTH_PATH.read_text().strip()


@app.middleware("http")
async def basic_auth(request: Request, call_next):
    host = (request.headers.get("host") or "").split(":")[0]
    if host not in ("localhost", "127.0.0.1", "[::1]", "testserver"):
        expected = "Basic " + base64.b64encode(_credentials().encode()).decode()
        if not secrets.compare_digest(request.headers.get("authorization", ""), expected):
            return Response("Kirish uchun login va parol kerak", status_code=401,
                            headers={"WWW-Authenticate": 'Basic realm="Talabalar tizimi", charset="UTF-8"'})
    return await call_next(request)


@app.exception_handler(sqlite3.IntegrityError)
async def integrity(_, e: sqlite3.IntegrityError):
    s = str(e)
    if "UNIQUE" in s: msg = "Bunday nomli yozuv allaqachon mavjud"
    elif "FOREIGN KEY" in s: msg = "O'chirib bo'lmaydi: bu yozuvga bog'langan ma'lumotlar bor (avval ularni o'chiring yoki ko'chiring)"
    elif "CHECK" in s: msg = "Qiymat ruxsat etilgan oraliqda emas (kurs 1–6)"
    elif "faol tizim" in s: msg = "Faol baholash tizimini o'chirib bo'lmaydi"
    elif "NOT NULL" in s: msg = "Majburiy maydon to'ldirilmagan"
    else: msg = s
    return JSONResponse({"detail": msg}, status_code=400)


from fastapi.exceptions import RequestValidationError  # noqa: E402


@app.exception_handler(RequestValidationError)
async def validation(_, e: RequestValidationError):
    msgs = []
    for err in e.errors():
        f = str(err["loc"][-1])
        msgs.append(f"{ml.LABELS.get(f, f)}: {err['msg']}")
    return JSONResponse({"detail": "; ".join(msgs[:4])}, status_code=400)


@app.on_event("startup")
def startup():
    db.init_db()
    _credentials()
    G = grading()
    sig = (db.data_signature(), G.signature())
    if ENGINE_PATH.exists():
        try:
            eng = joblib.load(ENGINE_PATH)
            ST.engine = eng
            if getattr(eng, "data_sig", None) == sig and getattr(eng, "engine_version", 0) == ml.ENGINE_VERSION:
                return
            if getattr(eng, "data_sig", (None, None))[1] != G.signature() and getattr(eng, "G", None) and eng.G.signature() != G.signature():
                schedule("full")
            else:
                schedule("refit")
            return
        except Exception:  # noqa: BLE001
            ST.engine = None
    schedule("full")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-store"})


# ------------------------------------------------------------------------------- umumiy
@app.get("/api/status")
def status():
    j = dict(ST.job)
    j["ready"] = ST.engine is not None
    j["elapsed"] = round(time.time() - j["started"], 1) if j.get("busy") and j.get("started") else None
    if ST.engine is not None:
        j["trained_at"] = ST.engine.metrics.get("trained_at")
        j["refit_at"] = ST.engine.metrics.get("refit_at")
    return j


@app.get("/api/meta")
def meta():
    G = grading()
    eng = ST.engine
    return {
        "tizimlar": db.tizimlar(), "faol_tizim": G.faol_id, "kategoriyalar": G.categories(),
        "tuzilma": db.tree(), "fanlar": db.fanlar(), "options": OPTIONS, "ranges": NUM_RANGES,
        "labels": ml.LABELS, "text_cols": db.STUDENT_TEXT, "num_cols": db.STUDENT_NUM,
        "klasterlar": list(eng.cluster_names.values()) if eng else [],
        "klaster_map": {str(k): v for k, v in eng.cluster_names.items()} if eng else {},
        "grant_chegara": ml.GRANT_CHEGARA, "grant_saqlash": ml.GRANT_SAQLASH, "status": status(),
    }


# ------------------------------------------------------------------------------- tahlil (bosh panel + tuzilma)
@app.get("/api/tahlil")
def tahlil(fakultet_id: int = 0, yonalish_id: int = 0, guruh_id: int = 0, kurs: int = 0):
    D, by_id, _ = data()
    d = filt(D, fakultet_id, yonalish_id, guruh_id, kurs)
    eng = ST.engine
    G = eng.G
    # keyingi daraja
    if guruh_id:
        level, children = "talaba", []
    elif yonalish_id:
        level, key, name = "guruh", "guruh_id", "guruh"
    elif fakultet_id:
        level, key, name = "yonalish", "yonalish_id", "yonalish"
    else:
        level, key, name = "fakultet", "fakultet_id", "fakultet"
    if level != "talaba":
        children = []
        for k, grp in d.groupby(key):
            s = unit_stats(grp)
            s.update(id=int(k), nomi=grp[name].iloc[0])
            if level == "guruh": s["kurs"] = int(grp["kurs"].iloc[0])
            if level == "yonalish": s["kod"] = grp["yonalish_kod"].iloc[0]
            children.append(s)
        children.sort(key=lambda x: (x.get("kurs", 0), x["nomi"]))
    by_kurs = [dict(unit_stats(g), kurs=int(k)) for k, g in d.groupby("kurs")]
    # fanlar kesimi
    grades = db.load_grades()
    gd = grades[grades["talaba_id"].isin(d["id"])]
    subj = []
    pred_map = {}
    for sid in d["id"]:
        for s in by_id[sid]["subjects"]:
            pred_map[(sid, s["fan_id"])] = s
    for fid, g in gd.groupby("fan_id"):
        pm = G.pass_min(int(g["tizim_id"].iloc[0]))
        res = g["yakuniy"].where(g["yakuniy"].notna(), [pred_map.get((t, fid), {}).get("pred", np.nan) for t in g["talaba_id"]])
        dist = {c: 0 for c in G.categories()}
        for v in res.dropna():
            dist[G.categories()[G.cat_index(v)]] += 1
        fp = [pred_map.get((t, fid), {}).get("fail_p", 0) for t in g["talaba_id"]]
        subj.append({"fan_id": int(fid), "fan": g["fan"].iloc[0], "n": int(len(g)), "oraliq": r1(g["oraliq"].mean()),
                     "joriy": r1(g["joriy"].mean()), "yakuniy": r1(g["yakuniy"].mean()), "natija": r1(res.mean()),
                     "qarz_foiz": r1((res < pm).mean() * 100), "qarz_bashorat": r1(np.mean(fp) * 100), "dist": dist})
    subj.sort(key=lambda s: s["natija"] or 0)
    risk = d.sort_values(["chet_proba", "fail_proba"], ascending=False).head(12)
    top = d.sort_values("natija", ascending=False).head(8)
    nb_bins = [0, 6, 12, 20, 28, 36, 50, 200]
    nb_hist = pd.cut(d["nb_sababsiz"], nb_bins, right=False).value_counts(sort=False)
    ball_bins = list(range(0, 101, 5))
    ball_hist = pd.cut(d["natija"].clip(0, 99.9), ball_bins, right=False).value_counts(sort=False)
    tolov = []
    for t, g in d.groupby("tolov"):
        tolov.append({"tolov": t, "n": int(len(g)), "ortacha": r1(g["natija"].mean()),
                      "otmaydi": r1((g["fail_proba"] >= 0.5).mean() * 100), "toifa": {c: int((g["natija_toifa"] == c).sum()) for c in G.categories()}})
    sample = d if len(d) <= 900 else d.sample(900, random_state=1)
    return clean({
        "level": level, "stats": unit_stats(d), "children": children, "by_kurs": by_kurs, "fanlar": subj,
        "risk": [student_brief(r) for _, r in risk.iterrows()], "top": [student_brief(r) for _, r in top.iterrows()],
        "nb_hist": {"labels": [f"{a}–{b-1}" if b < 200 else f"{a}+" for a, b in zip(nb_bins, nb_bins[1:])], "values": nb_hist.tolist()},
        "ball_hist": {"labels": [f"{a}" for a in ball_bins[:-1]], "values": ball_hist.tolist()},
        "tolov": tolov, "kategoriyalar": G.categories(),
        "scatter": [{"id": int(r["id"]), "x": r["x"], "y": r["y"], "c": int(r["cluster"]), "fio": r["fio"], "b": r1(r["natija"])} for _, r in sample.iterrows()],
        "students": [student_brief(r) for _, r in d.sort_values("fio").iterrows()] if guruh_id else [],
    })


# ------------------------------------------------------------------------------- talabalar ro'yxati
@app.get("/api/students")
def students(fakultet_id: int = 0, yonalish_id: int = 0, guruh_id: int = 0, kurs: int = 0, q: str = "", toifa: str = "",
             klaster: str = "", xavf: str = "", tolov: str = "", maxsus: str = "", sort: str = "fio", desc: int = 0,
             page: int = 1, size: int = 50):
    D, _, _ = data()
    d = filt(D, fakultet_id, yonalish_id, guruh_id, kurs)
    if q: d = d[d["fio"].str.lower().str.contains(q.lower(), regex=False) | (d["id"].astype(str) == q)]
    if toifa: d = d[d["toifa"] == toifa]
    if klaster: d = d[d["cluster_name"] == klaster]
    if xavf: d = d[d["xavf"] == xavf]
    if tolov: d = d[d["tolov"] == tolov]
    if maxsus == "anomal": d = d[d["anomaly"]]
    elif maxsus == "grant_dav": d = d[(d["tolov"] == "Kontrakt") & (d["alo_proba"] >= 0.5)]
    elif maxsus == "grant_xavf": d = d[(d["tolov"] == "Grant") & (d["pasayish_proba"] >= 0.3)]
    elif maxsus == "chetlash": d = d[d["chet_proba"] >= 0.5]
    elif maxsus == "nb": d = d[d["nb_sababsiz"] >= 36]
    elif maxsus == "nomalum": d = d[d["natija_manba"] == "bashorat"]
    col = {"fio": "fio", "ball": "ball", "natija": "natija", "fail": "fail_proba", "chet": "chet_proba", "nb": "nb_sababsiz",
           "guruh": "guruh", "oraliq": "oraliq_ortacha", "alo": "alo_proba"}.get(sort, "fio")
    d = d.sort_values(col, ascending=not desc, na_position="last")
    total = len(d)
    size = max(10, min(size, 500))
    page = max(1, page)
    rows = d.iloc[(page - 1) * size: page * size]
    return clean({"total": total, "page": page, "size": size, "items": [student_brief(r) for _, r in rows.iterrows()]})


@app.get("/api/export.csv")
def export(fakultet_id: int = 0, yonalish_id: int = 0, guruh_id: int = 0, kurs: int = 0):
    D, _, _ = data()
    d = filt(D, fakultet_id, yonalish_id, guruh_id, kurs)
    cols = ["id", "fio", "fakultet", "yonalish", "guruh", "kurs", "tolov", "kontrakt_qarz", "nb_sababsiz", "oraliq_ortacha",
            "yakuniy_ortacha", "ball", "baho", "toifa", "fail_proba", "chet_proba", "alo_proba", "pasayish_proba",
            "qarz_kutilgan", "cluster_name", "anomaly", "xavf"]
    buf = io.StringIO()
    d[cols].to_csv(buf, index=False)
    return StreamingResponse(iter(["﻿" + buf.getvalue()]), media_type="text/csv",
                             headers={"Content-Disposition": "attachment; filename=talabalar_bashorat.csv"})


# ------------------------------------------------------------------------------- talaba profili
def _profile_payload(row: pd.Series, p: dict, S_row: pd.DataFrame, exclude_id=None, D=None):
    eng = ST.engine
    sims = eng.similar(S_row, exclude_id=exclude_id)
    sim = []
    if D is not None:
        Di = D.set_index("id")
        for i in sims:
            if i in Di.index:
                r = Di.loc[i]
                sim.append({"id": int(i), "fio": r["fio"], "guruh": r["guruh"], "natija": r1(r["natija"]), "natija_manba": r["natija_manba"],
                            "cluster_name": r["cluster_name"]})
    return {"prediction": p, "explain": eng.explain(S_row), "advice": advice_for(row, p), "similar": sim,
            "base_value": r1(float(np.ravel(eng.explainer.expected_value)[0]))}


@app.get("/api/students/{sid}")
def student(sid: int):
    D, by_id, S = data()
    if sid not in by_id:
        raise HTTPException(404, "Talaba topilmadi")
    rec = db.q("SELECT * FROM talabalar WHERE id=?", (sid,))[0]
    row = D[D["id"] == sid].iloc[0]
    S_row = S[S["id"] == sid]
    p = by_id[sid]
    grades = db.student_grades(sid)
    G = ST.engine.G
    pmap = {s["fan_id"]: s for s in p["subjects"]}
    for g in grades:
        g["pass_min"] = G.pass_min(g["tizim_id"])
        g["joriy_baho"] = G.grade(g["joriy"], g["tizim_id"])["baho"] if g["joriy"] is not None else None
        g["oraliq_baho"] = G.grade(g["oraliq"], g["tizim_id"])["baho"] if g["oraliq"] is not None else None
        gy = G.grade(g["yakuniy"], g["tizim_id"]) if g["yakuniy"] is not None else None
        g["yakuniy_baho"], g["yakuniy_nomi"] = (gy["baho"], gy["nomi"]) if gy else (None, None)
        g["bashorat"] = pmap.get(g["fan_id"])
    grp = D[D["guruh_id"] == row["guruh_id"]].sort_values("natija", ascending=False)
    kurs_d = D[(D["yonalish_id"] == row["yonalish_id"]) & (D["kurs"] == row["kurs"])]
    payload = _profile_payload(row, p, S_row, exclude_id=sid, D=D)
    payload.update({
        "talaba": rec, "tuzilma": {k: (row[k].item() if hasattr(row[k], "item") else row[k]) for k in
                                   ["guruh", "guruh_id", "kurs", "yonalish", "yonalish_id", "yonalish_kod", "fakultet", "fakultet_id"]},
        "baholar": grades,
        "haqiqiy": {"yakuniy": r1(row["yakuniy_ortacha"]), "qarz_soni": None if pd.isna(row["qarz_soni"]) else int(row["qarz_soni"]),
                    "baho": G.grade(row["yakuniy_ortacha"])["nomi"] if pd.notna(row["yakuniy_ortacha"]) else None},
        "agregat": {k: r1(row[k]) for k in ["oraliq_ortacha", "joriy_ortacha", "past_fanlar", "trend"]},
        "reyting": {"guruhda": int(list(grp["id"]).index(sid)) + 1, "guruh_n": int(len(grp)),
                    "guruh_ortacha": r1(grp["natija"].mean()), "kurs_ortacha": r1(kurs_d["natija"].mean()),
                    "kursda": int((kurs_d["natija"] > row["natija"]).sum()) + 1, "kurs_n": int(len(kurs_d)),
                    "umumiy_ortacha": r1(D["natija"].mean())},
    })
    return clean(payload)


# ------------------------------------------------------------------------------- qo'shish / tahrirlash / what-if
class Baho(BaseModel):
    fan_id: int
    joriy: float = Field(ge=0, le=100)
    oraliq: float = Field(ge=0, le=100)
    yakuniy: Optional[float] = Field(default=None, ge=0, le=100)


class Student(BaseModel):
    guruh_id: int
    fio: str = Field(min_length=3, max_length=120)
    jinsi: str = "Erkak"
    hudud: str = "Toshkent sh."
    tolov: str = "Kontrakt"
    kontrakt_qarz: str = "Yo'q"
    stipendiya: str = "Yo'q"
    yashash: str = "Uyda"
    ishlaydi: str = "Yo'q"
    ota_ona: str = "O'rta"
    togarak: str = "Yo'q"
    olimpiada: str = "Yo'q"
    yosh: float = Field(19, ge=14, le=60)
    ish_soati: float = Field(0, ge=0, le=80)
    yol_vaqti: float = Field(30, ge=0, le=300)
    kirish_bali: float = Field(140, ge=0, le=200)
    oldingi_ball: float = Field(70, ge=0, le=100)
    oldingi_qarzlar: float = Field(0, ge=0, le=30)
    nb_sababli: float = Field(4, ge=0, le=300)
    nb_sababsiz: float = Field(10, ge=0, le=300)
    topshiriq: float = Field(70, ge=0, le=100)
    lms_kirish: float = Field(20, ge=0, le=500)
    kutubxona: float = Field(2, ge=0, le=100)
    mustaqil_soat: float = Field(7, ge=0, le=80)
    baholar: list[Baho] = []


def _frames_for(s: Student, sid=-1):
    g = db.q("""SELECT g.nomi AS guruh, g.kurs, y.nomi AS yonalish, y.kod AS yonalish_kod, y.id AS yonalish_id,
                       f.nomi AS fakultet, f.id AS fakultet_id FROM guruhlar g JOIN yonalishlar y ON y.id=g.yonalish_id
                JOIN fakultetlar f ON f.id=y.fakultet_id WHERE g.id=?""", (s.guruh_id,))
    if not g:
        raise HTTPException(400, "Guruh topilmadi")
    rec = s.model_dump(exclude={"baholar"})
    rec.update(g[0], id=sid)
    students = pd.DataFrame([rec])
    fan = {f["id"]: f for f in db.fanlar()}
    rows = []
    for b in s.baholar:
        if b.fan_id not in fan:
            raise HTTPException(400, f"Fan topilmadi: {b.fan_id}")
        rows.append({"talaba_id": sid, "fan_id": b.fan_id, "fan": fan[b.fan_id]["nomi"], "kredit": fan[b.fan_id]["kredit"],
                     "tizim_id": fan[b.fan_id]["tizim_id"], "joriy": b.joriy, "oraliq": b.oraliq,
                     "yakuniy": np.nan if b.yakuniy is None else b.yakuniy})
    grades = pd.DataFrame(rows, columns=["talaba_id", "fan_id", "fan", "kredit", "tizim_id", "joriy", "oraliq", "yakuniy"])
    grades["yakuniy"] = grades["yakuniy"].astype(float)
    return students, grades, rec


@app.post("/api/preview")
def preview(s: Student):
    eng = need_engine()
    students, grades, rec = _frames_for(s)
    preds, S = eng.predict(students, grades, use_oof=False)
    D, _, _ = data()
    row = S.iloc[0]
    payload = _profile_payload(row, preds[0], S, D=D)
    payload["agregat"] = {k: r1(row[k]) for k in ["oraliq_ortacha", "joriy_ortacha", "past_fanlar", "trend"]}
    return clean(payload)


def _validate(s: Student):
    for k, opts in OPTIONS.items():
        if k != "hudud" and getattr(s, k) not in opts:
            raise HTTPException(400, f"{ml.LABELS.get(k, k)}: noto'g'ri qiymat")
    ids = [b.fan_id for b in s.baholar]
    if len(ids) != len(set(ids)):
        raise HTTPException(400, "Bir fan ikki marta kiritilgan")


@app.post("/api/students")
def create_student(s: Student):
    _validate(s)
    sid = db.save_student(s.model_dump(exclude={"baholar"}), [b.model_dump() for b in s.baholar])
    invalidate()
    if any(b.yakuniy is not None for b in s.baholar):
        schedule("refit")
    return {"id": sid}


@app.put("/api/students/{sid}")
def update_student(sid: int, s: Student):
    _validate(s)
    if not db.q("SELECT id FROM talabalar WHERE id=?", (sid,)):
        raise HTTPException(404, "Talaba topilmadi")
    old = db.student_grades(sid)
    db.save_student(s.model_dump(exclude={"baholar"}), [b.model_dump() for b in s.baholar], sid)
    invalidate(drop_oof=[sid])
    if any(b.yakuniy is not None for b in s.baholar) or any(g["yakuniy"] is not None for g in old):
        schedule("refit")
    return {"id": sid}


@app.delete("/api/students/{sid}")
def remove_student(sid: int):
    had = any(g["yakuniy"] is not None for g in db.student_grades(sid))
    db.delete_student(sid)
    invalidate(drop_oof=[sid])
    if had:
        schedule("refit")
    return {"ok": True}


@app.get("/api/guruhlar/{gid}/fanlar")
def guruh_fanlar(gid: int):
    return db.group_subjects(gid)


# ------------------------------------------------------------------------------- analitika
FACTOR_CAT = ["tolov", "kontrakt_qarz", "stipendiya", "yashash", "ishlaydi", "jinsi", "ota_ona", "togarak", "olimpiada", "hudud"]
FACTOR_NUM = {"nb_sababsiz": [0, 6, 12, 20, 30, 40, 200], "topshiriq": [0, 40, 55, 70, 85, 101], "ish_soati": [0, 1, 15, 25, 35, 100],
              "lms_kirish": [0, 10, 20, 30, 45, 500], "mustaqil_soat": [0, 3, 6, 9, 12, 100], "kirish_bali": [0, 110, 130, 150, 170, 200],
              "yol_vaqti": [0, 20, 40, 60, 90, 400], "oldingi_qarzlar": [0, 1, 2, 3, 30]}


@app.get("/api/analitika")
def analitika(fakultet_id: int = 0, yonalish_id: int = 0, guruh_id: int = 0, kurs: int = 0):
    D, by_id, _ = data()
    d = filt(D, fakultet_id, yonalish_id, guruh_id, kurs)
    eng = ST.engine

    def grp_stats(g, label):
        return {"guruh": str(label), "n": int(len(g)), "ortacha": r1(g["natija"].mean()),
                "otmaydi": r1((g["fail_proba"] >= 0.5).mean() * 100), "chetlash": r1((g["chet_proba"] >= 0.5).mean() * 100),
                "alo": r1((g["natija"] >= ml.GRANT_CHEGARA).mean() * 100)}

    factors = []
    for c in FACTOR_CAT:
        rows = [grp_stats(g, k) for k, g in d.groupby(c) if len(g) >= 5]
        rows.sort(key=lambda r: -(r["ortacha"] or 0))
        factors.append({"key": c, "label": ml.LABELS.get(c, c), "tur": "cat", "rows": rows,
                        "farq": r1(max(r["ortacha"] for r in rows) - min(r["ortacha"] for r in rows)) if len(rows) > 1 else 0})
    for c, bins in FACTOR_NUM.items():
        cut = pd.cut(d[c], bins, right=False)
        rows = []
        for iv, g in d.groupby(cut, observed=True):
            if len(g) >= 5:
                lab = f"{int(iv.left)}–{int(iv.right) - 1}" if iv.right < bins[-1] else f"{int(iv.left)}+"
                rows.append(grp_stats(g, lab))
        factors.append({"key": c, "label": ml.LABELS.get(c, c), "tur": "num", "rows": rows,
                        "korr": r1(d[[c, "natija"]].corr().iloc[0, 1], 3) if len(d) > 2 else None,
                        "farq": r1(max(r["ortacha"] for r in rows) - min(r["ortacha"] for r in rows)) if len(rows) > 1 else 0})
    factors.sort(key=lambda f: -(f["farq"] or 0))
    num_cols = ["oraliq_ortacha", "joriy_ortacha", "topshiriq", "nb_sababsiz", "lms_kirish", "mustaqil_soat", "oldingi_ball",
                "kirish_bali", "ish_soati", "kutubxona", "yol_vaqti", "natija"]
    corr = d[num_cols].corr().round(2)
    lists = {
        "chetlash": d[d["chet_proba"] >= 0.5].sort_values("chet_proba", ascending=False),
        "grant_dav": d[(d["tolov"] == "Kontrakt") & (d["alo_proba"] >= 0.5)].sort_values("alo_proba", ascending=False),
        "grant_xavf": d[(d["tolov"] == "Grant") & (d["pasayish_proba"] >= 0.3)].sort_values("pasayish_proba", ascending=False),
        "anomal": d[d["anomaly"]].sort_values("anomaly_score", ascending=False),
        "nb": d[d["nb_sababsiz"] >= 36].sort_values("nb_sababsiz", ascending=False),
    }
    # bashorat qilingan qarzlar bo'yicha fanlar
    fan_risk = {}
    for sid in d["id"]:
        for s in by_id[sid]["subjects"]:
            f = fan_risk.setdefault(s["fan"], {"fan": s["fan"], "n": 0, "qarz_kutilgan": 0.0, "pred": 0.0})
            f["n"] += 1; f["qarz_kutilgan"] += s["fail_p"]; f["pred"] += s["pred"]
    fan_risk = sorted(({"fan": f["fan"], "n": f["n"], "qarz_kutilgan": r1(f["qarz_kutilgan"]), "qarz_foiz": r1(f["qarz_kutilgan"] / f["n"] * 100),
                        "pred": r1(f["pred"] / f["n"])} for f in fan_risk.values()), key=lambda f: -f["qarz_foiz"])
    return clean({
        "n": int(len(d)), "factors": factors,
        "corr": {"cols": [ml.LABELS.get(c, "Yakuniy natija" if c == "natija" else c) for c in num_cols], "values": corr.values.tolist()},
        "lists": {k: {"n": int(len(v)), "items": [student_brief(r) for _, r in v.head(30).iterrows()]} for k, v in lists.items()},
        "fan_risk": fan_risk,
        "qarz_kutilgan_jami": r1(d["qarz_kutilgan"].sum()),
        "global_imp": [{"feature": k, "label": ml.LABELS.get(k, k), "value": r1(v, 3)} for k, v in eng.global_imp.head(15).items()],
    })


# ------------------------------------------------------------------------------- klasterlar va modellar
@app.get("/api/klasterlar")
def klasterlar():
    D, _, _ = data()
    eng = ST.engine
    prof = []
    for k, name in eng.cluster_names.items():
        g = D[D["cluster"] == k]
        prof.append({"id": int(k), "nomi": name, "n": int(len(g)), "ortacha": r1(g["natija"].mean()),
                     **{c: r1(g[c].mean()) for c in ml.CLUSTER_FEATS},
                     "otmaydi": r1((g["fail_proba"] >= 0.5).mean() * 100), "grant": r1((g["tolov"] == "Grant").mean() * 100),
                     "ishlaydi": r1((g["ishlaydi"] == "Ha").mean() * 100), "qarzdor": r1((g["kontrakt_qarz"] == "Ha").mean() * 100)})
    sample = D if len(D) <= 1500 else D.sample(1500, random_state=1)
    return clean({"profiles": sorted(prof, key=lambda p: -(p["ortacha"] or 0)), "silhouette": r1(eng.silhouette, 3), "k_scan": eng.k_scan,
                  "features": [{"key": c, "label": ml.LABELS.get(c, c)} for c in ml.CLUSTER_FEATS],
                  "scatter": [{"id": int(r["id"]), "x": r["x"], "y": r["y"], "c": int(r["cluster"]), "fio": r["fio"], "b": r1(r["natija"])}
                              for _, r in sample.iterrows()]})


@app.get("/api/models")
def models():
    eng = need_engine()
    m = dict(eng.metrics)
    m["global_imp"] = [{"feature": k, "label": ml.LABELS.get(k, k), "value": r1(v, 3)} for k, v in eng.global_imp.head(20).items()]
    m["n_total"] = eng.n_total
    m["silhouette"] = r1(eng.silhouette, 3)
    m["tizim"] = eng.G.faol["nomi"]
    return clean(m)


@app.post("/api/retrain")
def retrain():
    schedule("full")
    return status()


# ------------------------------------------------------------------------------- boshqaruv (admin)
class Fakultet(BaseModel):
    nomi: str = Field(min_length=2, max_length=120)


class Yonalish(BaseModel):
    fakultet_id: int
    kod: str = Field(min_length=1, max_length=10)
    nomi: str = Field(min_length=2, max_length=120)


class Guruh(BaseModel):
    yonalish_id: int
    nomi: str = Field(min_length=1, max_length=40)
    kurs: int = Field(ge=1, le=6)


class Fan(BaseModel):
    nomi: str = Field(min_length=2, max_length=120)
    kredit: int = Field(6, ge=1, le=30)
    tizim_id: int


class Chegara(BaseModel):
    min: float = Field(ge=0, le=100)
    baho: str = Field(min_length=1, max_length=10)
    nomi: str = Field(min_length=1, max_length=40)
    otish: bool = True


class Tizim(BaseModel):
    nomi: str = Field(min_length=2, max_length=120)
    tavsif: Optional[str] = None
    chegaralar: list[Chegara]


class Reja(BaseModel):
    yonalish_id: int
    kurs: int = Field(ge=1, le=6)
    fan_id: int


ADMIN_MODELS = {"fakultetlar": Fakultet, "yonalishlar": Yonalish, "guruhlar": Guruh, "fanlar": Fan}


def _admin_body(table, body: dict):
    try:
        return ADMIN_MODELS[table](**body).model_dump()
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, "Maydonlar noto'g'ri to'ldirilgan: " + str(e).split("\n")[1] if "\n" in str(e) else str(e))


@app.post("/api/admin/reja")
def reja_add(r: Reja):
    db.reja_add(r.yonalish_id, r.kurs, r.fan_id)
    return {"ok": True}


@app.delete("/api/admin/reja")
def reja_remove(yonalish_id: int, kurs: int, fan_id: int):
    db.reja_remove(yonalish_id, kurs, fan_id)
    return {"ok": True}


def _check_tizim(t: Tizim):
    ch = t.chegaralar
    if len(ch) < 2: raise HTTPException(400, "Kamida 2 ta daraja bo'lishi kerak")
    if not any(c.min == 0 for c in ch): raise HTTPException(400, "Eng past daraja 0 balldan boshlanishi kerak")
    if len({c.min for c in ch}) != len(ch): raise HTTPException(400, "Darajalar chegaralari takrorlanmasligi kerak")
    if not any(c.otish for c in ch) or all(c.otish for c in ch):
        raise HTTPException(400, "Kamida bitta o'tuvchi va bitta o'tmaydigan daraja bo'lishi kerak")
    s = sorted(ch, key=lambda c: c.min)
    first_pass = next(i for i, c in enumerate(s) if c.otish)
    if not all(c.otish for c in s[first_pass:]):
        raise HTTPException(400, "O'tuvchi darajalar ketma-ket va yuqorida bo'lishi kerak")


@app.post("/api/admin/tizimlar")
def tizim_create(t: Tizim):
    _check_tizim(t)
    rid = db.tizim_save(t.model_dump())
    return {"id": rid}


@app.put("/api/admin/tizimlar/{rid}")
def tizim_update(rid: int, t: Tizim):
    _check_tizim(t)
    used = db.q("SELECT COUNT(*) n FROM fanlar WHERE tizim_id=?", (rid,))[0]["n"]
    db.tizim_save(t.model_dump(), rid)
    invalidate()
    if used or rid == db.faol_tizim_id():
        schedule("full")
    return {"id": rid, "retrain": bool(used or rid == db.faol_tizim_id())}


@app.delete("/api/admin/tizimlar/{rid}")
def tizim_delete(rid: int):
    db.tizim_delete(rid)
    return {"ok": True}


@app.put("/api/admin/faol_tizim/{rid}")
def faol_tizim(rid: int):
    if not db.q("SELECT id FROM tizimlar WHERE id=?", (rid,)): raise HTTPException(404, "Tizim topilmadi")
    db.set_faol_tizim(rid)
    invalidate()
    schedule("full")
    return status()


# umumiy CRUD marshrutlari oxirida turadi (aniq marshrutlar /reja, /tizimlar ustun bo'lishi uchun)
@app.post("/api/admin/{table}")
async def admin_create(table: str, request: Request):
    if table not in ADMIN_MODELS: raise HTTPException(404)
    rid = db.admin_save(table, _admin_body(table, await request.json()))
    invalidate()
    return {"id": rid}


@app.put("/api/admin/{table}/{rid}")
async def admin_update(table: str, rid: int, request: Request):
    if table not in ADMIN_MODELS: raise HTTPException(404)
    body = _admin_body(table, await request.json())
    old = db.q(f"SELECT * FROM {table} WHERE id=?", (rid,))
    if not old: raise HTTPException(404, "Topilmadi")
    db.admin_save(table, body, rid)
    invalidate()
    if table == "fanlar" and (old[0]["tizim_id"] != body["tizim_id"] or old[0]["kredit"] != body["kredit"]):
        schedule("full")
    elif table == "guruhlar" and old[0]["kurs"] != body["kurs"]:
        schedule("refit")
    return {"id": rid}


@app.delete("/api/admin/{table}/{rid}")
def admin_delete(table: str, rid: int):
    if table not in ADMIN_MODELS: raise HTTPException(404)
    db.admin_delete(table, rid)
    invalidate()
    return {"ok": True}
