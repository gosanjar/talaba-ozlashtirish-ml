"""Bazadagi ma'lumotlarni ochiq CSV/JSON datasetga eksport qiladi: python tools/export_dataset.py -> dataset/"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))
import db  # noqa: E402

OUT = ROOT / "dataset"
OUT.mkdir(exist_ok=True)
db.init_db()

s = db.load_students()
cols = ["id", "fio", "fakultet", "yonalish_kod", "yonalish", "guruh", "kurs"] + db.STUDENT_COLS[1:]
s[cols].rename(columns={"id": "talaba_id"}).to_csv(OUT / "talabalar.csv", index=False)

g = db.load_grades()
g[["talaba_id", "fan_id", "fan", "kredit", "tizim_id", "joriy", "oraliq", "yakuniy"]].sort_values(["talaba_id", "fan"]).to_csv(OUT / "baholar.csv", index=False)

f = db.fanlar()
import pandas as pd  # noqa: E402
pd.DataFrame(f)[["id", "nomi", "kredit", "tizim_id", "tizim"]].to_csv(OUT / "fanlar.csv", index=False)
(OUT / "baholash_tizimlari.json").write_text(json.dumps(db.tizimlar(), ensure_ascii=False, indent=2))
tz = [{"fakultet": fk["nomi"], "yonalishlar": [{"kod": y["kod"], "nomi": y["nomi"], "guruhlar": [{"nomi": gr["nomi"], "kurs": gr["kurs"]} for gr in y["guruhlar"]],
        "reja": {k: [x["nomi"] for x in v] for k, v in y["reja"].items() if v}} for y in fk["yonalishlar"]]} for fk in db.tree()]
(OUT / "tuzilma.json").write_text(json.dumps(tz, ensure_ascii=False, indent=2))
print(f"talabalar: {len(s)}, baholar: {len(g)}, fanlar: {len(f)} -> {OUT}")
