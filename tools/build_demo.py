"""GitHub Pages uchun statik demo yasaydi: python tools/build_demo.py -> docs/

Server API javoblari oldindan hisoblanib JSON fayllarga yoziladi, frontend esa DEMO rejimida
ularni o'qiydi (qo'shish/tahrirlash/simulyator faqat serverli versiyada ishlaydi).
"""
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import app.main as M  # noqa: E402

OUT = ROOT / "docs"
API = OUT / "api"


def default(o):
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.floating): return None if np.isnan(o) else float(o)
    if isinstance(o, np.bool_): return bool(o)
    if isinstance(o, np.ndarray): return o.tolist()
    raise TypeError(type(o))


def dump(rel, obj):
    p = API / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":"), default=default))


def main():
    M.startup()
    while M.ST.job["busy"]:
        print("…", M.ST.job["message"]); time.sleep(2)
    if M.ST.engine is None:
        raise SystemExit("Model tayyor emas: " + str(M.ST.job))
    if API.exists(): shutil.rmtree(API)
    t0 = time.time()
    st = M.status(); st.update(busy=False, demo=True)
    dump("status.json", st)
    meta = M.meta(); meta["status"] = st
    dump("meta.json", meta)
    D, by_id, S = M.data()
    units = [("all", {})] + [(f"f{f['id']}", {"fakultet_id": f["id"]}) for f in meta["tuzilma"]] + \
            [(f"y{y['id']}", {"yonalish_id": y["id"]}) for f in meta["tuzilma"] for y in f["yonalishlar"]]
    kurslar = sorted(D["kurs"].unique().tolist())
    for key, p in units:
        for k in [0] + kurslar:
            if len(M.filt(D, **p, kurs=k)) == 0: continue
            dump(f"tahlil/{key}_k{k}.json", M.tahlil(**p, kurs=k))
            dump(f"analitika/{key}_k{k}.json", M.analitika(**p, kurs=k))
    for f in meta["tuzilma"]:
        for y in f["yonalishlar"]:
            for g in y["guruhlar"]:
                dump(f"guruhlar/{g['id']}.json", M.guruh_fanlar(g["id"]))
                if g["talabalar"]:
                    dump(f"tahlil/g{g['id']}.json", M.tahlil(guruh_id=g["id"]))
                    dump(f"analitika/g{g['id']}.json", M.analitika(guruh_id=g["id"]))
    rows = []
    for _, r in D.iterrows():
        b = M.student_brief(r)
        b.update(fakultet_id=int(r["fakultet_id"]), yonalish_id=int(r["yonalish_id"]), guruh_id=int(r["guruh_id"]))
        rows.append(M.clean(b))
    dump("talabalar.json", rows)
    for i, sid in enumerate(D["id"]):
        dump(f"talaba/{int(sid)}.json", M.student(int(sid)))
        if i % 300 == 0: print(f"profil {i}/{len(D)}")
    dump("klasterlar.json", M.klasterlar())
    dump("models.json", M.models())
    html = (ROOT / "app" / "static" / "index.html").read_text()
    html = html.replace("<script>\n/* ===", "<script>window.DEMO_MODE=true</script>\n<script>\n/* ===", 1)
    assert "DEMO_MODE=true" in html
    (OUT / "index.html").write_text(html)
    (OUT / ".nojekyll").write_text("")
    size = sum(p.stat().st_size for p in API.rglob("*.json")) / 1e6
    print(f"Tayyor: {sum(1 for _ in API.rglob('*.json'))} fayl, {size:.1f} MB, {time.time() - t0:.0f} s -> {OUT}")


if __name__ == "__main__":
    main()
