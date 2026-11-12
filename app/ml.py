"""ML dvigateli.

Talaba darajasida (har bir talaba = 1 qator):
  1) Regressiya        — semestr yakuniy o'rtacha bali (kredit bo'yicha vaznli, 0–100)
  2) Klassifikatsiya   — toifa (faol baholash tizimi bo'yicha: A'lo/Yaxshi/... yoki A–F)
  3) Klassifikatsiya   — o'tadi / o'tmaydi (kamida bitta fandan qarz)
  4) Klassifikatsiya   — chetlashtirish xavfi (3 va undan ko'p fandan qarz)
Fan darajasida (talaba x fan = 1 qator, fanlar soniga bog'liq emas — yangi fan qo'shilsa ham ishlaydi):
  5) Regressiya        — har bir fan bo'yicha yakuniy ball va shu fandan qarzdor bo'lish ehtimoli
Qo'shimcha: K-Means klasterlash, IsolationForest anomaliyalar, SHAP tushuntirish, learning curve, ablation.
Aniqlikni oshirish: hosila belgilar, XGBoost tuning (RandomizedSearchCV), Stacking/Voting ansambllar.
"""
import time

import numpy as np
import pandas as pd
import shap
from scipy.stats import norm
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor, IsolationForest,
                              RandomForestClassifier, RandomForestRegressor, StackingRegressor, VotingClassifier)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, RidgeCV
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error, r2_score, recall_score, roc_auc_score,
                             silhouette_score)
from sklearn.model_selection import GroupKFold, KFold, RandomizedSearchCV, StratifiedKFold, cross_val_predict
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier, XGBRegressor

import db

SEED = 42
GRANT_CHEGARA = 86      # "A'lo" daraja (grantga da'vogarlik)
GRANT_SAQLASH = 71      # grant/stipendiyani saqlash uchun minimal o'rtacha

NUM_BASE = ["kurs"] + db.STUDENT_NUM
CAT = ["jinsi", "tolov", "kontrakt_qarz", "stipendiya", "yashash", "ishlaydi", "ota_ona", "togarak", "olimpiada",
       "hudud", "yonalish_kod"]
AGG = ["joriy_ortacha", "oraliq_ortacha", "oraliq_min", "oraliq_max", "oraliq_std", "past_fanlar", "fanlar_soni"]
ENG = ["trend", "joriy_oraliq_farq", "nb_jami", "ish_yuki", "faollik", "nb_limit"]
NUM = NUM_BASE + AGG + ENG
SUBJ_EXTRA = ["joriy", "oraliq", "fan_qiyinlik", "fan_yak_farq", "oraliq_vs_ortacha", "kredit"]
CLUSTER_FEATS = ["oraliq_ortacha", "nb_sababsiz", "topshiriq", "lms_kirish", "oldingi_ball", "mustaqil_soat"]

LABELS = {
    "kurs": "Bosqich (kurs)", "yosh": "Yoshi", "ish_soati": "Ish soati (haftasiga)", "yol_vaqti": "Yo'lga ketadigan vaqt (daq.)",
    "kirish_bali": "Kirish imtihoni bali", "oldingi_ball": "Oldingi semestr bali", "oldingi_qarzlar": "Oldingi akademik qarzlar",
    "nb_sababli": "NB sababli (soat)", "nb_sababsiz": "NB sababsiz (soat)", "topshiriq": "Topshiriqlar bajarilishi, %",
    "lms_kirish": "LMS faolligi (oyiga kirish)", "kutubxona": "Kutubxonaga tashrif (oyiga)", "mustaqil_soat": "Mustaqil ta'lim (soat/hafta)",
    "joriy_ortacha": "Joriy nazorat o'rtachasi", "oraliq_ortacha": "Oraliq nazorat o'rtachasi", "oraliq_min": "Eng past oraliq ball",
    "oraliq_max": "Eng yuqori oraliq ball", "oraliq_std": "Fanlar orasidagi tafovut", "past_fanlar": "Oraliqdan o'tmagan fanlar soni",
    "fanlar_soni": "Fanlar soni", "trend": "O'zgarish (oraliq − oldingi semestr)", "joriy_oraliq_farq": "Joriy va oraliq farqi",
    "nb_jami": "NB jami (soat)", "ish_yuki": "Ortiqcha ish yuki (20+ soat)", "faollik": "Faollik indeksi", "nb_limit": "NB limitiga yaqin",
    "jinsi": "Jinsi", "tolov": "To'lov shakli", "kontrakt_qarz": "Kontrakt qarzdorligi", "stipendiya": "Stipendiya", "yashash": "Yashash joyi",
    "ishlaydi": "Ishlaydi", "ota_ona": "Ota-ona ma'lumoti", "togarak": "To'garakda qatnashadi", "olimpiada": "Olimpiada ishtirokchisi",
    "hudud": "Hudud", "yonalish_kod": "Yo'nalish", "joriy": "Fan: joriy nazorat", "oraliq": "Fan: oraliq nazorat",
    "fan_qiyinlik": "Fan qiyinligi", "fan_yak_farq": "Fan bo'yicha yakuniy farq", "oraliq_vs_ortacha": "Fan oralig'i va o'rtacha farqi", "kredit": "Kredit",
}


# ======================================================================= baholash tizimlari
class Grading:
    def __init__(self, tizimlar, faol_id):
        self.t = {t["id"]: t for t in tizimlar}
        self.faol_id = faol_id if faol_id in self.t else min(self.t)

    @property
    def faol(self):
        return self.t[self.faol_id]

    def signature(self):
        return repr(sorted((k, v["chegaralar"]) for k, v in self.t.items())) + f"|{self.faol_id}"

    def pass_min(self, tid):
        cuts = self.t.get(tid, self.faol)["chegaralar"]
        passing = [c["min"] for c in cuts if c["otish"]]
        return min(passing) if passing else 0

    def grade(self, ball, tid=None):
        if ball is None or (isinstance(ball, float) and np.isnan(ball)): return None
        cuts = self.t.get(tid, self.faol)["chegaralar"]
        for c in cuts:
            if ball >= c["min"]:
                return c
        return cuts[-1]

    def categories(self):                       # pastdan yuqoriga: indeks 0 = eng past
        return [c["nomi"] for c in reversed(self.faol["chegaralar"])]

    def cat_index(self, ball):
        idx = 0
        for i, c in enumerate(reversed(self.faol["chegaralar"])):
            if ball >= c["min"]: idx = i
        return idx


# ======================================================================= belgilar
def fan_statistics(g):
    gm = g["oraliq"].mean()
    by = g.groupby("fan_id")
    n = by.size()
    k = 20                                                     # kam ma'lumotli fanlar uchun silliqlash
    qiy = (by["oraliq"].mean() - gm) * n / (n + k)
    yk = g.dropna(subset=["yakuniy"])
    d = yk["yakuniy"] - yk["oraliq"]
    yd = d.groupby(yk["fan_id"])
    yak_farq = (yd.mean() - d.mean()) * yd.size() / (yd.size() + k)
    return {"qiyinlik": qiy.to_dict(), "yak_farq": yak_farq.to_dict()}


def build_frames(students: pd.DataFrame, grades: pd.DataFrame, G: Grading, fan_stats=None):
    g = grades.copy()
    g["pass_min"] = g["tizim_id"].map(G.pass_min)
    agg = g.groupby("talaba_id").agg(joriy_ortacha=("joriy", "mean"), oraliq_ortacha=("oraliq", "mean"),
                                     oraliq_min=("oraliq", "min"), oraliq_max=("oraliq", "max"),
                                     oraliq_std=("oraliq", "std"), fanlar_soni=("fan_id", "count"))
    agg["past_fanlar"] = (g["oraliq"] < g["pass_min"]).groupby(g["talaba_id"]).sum()
    known = g.groupby("talaba_id")["yakuniy"].apply(lambda s: bool(s.notna().all()) and len(s) > 0)
    gk = g[g["talaba_id"].isin(known[known].index)]
    w = gk["kredit"].clip(lower=1)
    yak = (gk["yakuniy"] * w).groupby(gk["talaba_id"]).sum() / w.groupby(gk["talaba_id"]).sum()
    qarz = (gk["yakuniy"] < gk["pass_min"]).groupby(gk["talaba_id"]).sum()
    S = students.set_index("id").join(agg, how="left").join(yak.rename("yakuniy_ortacha")).join(qarz.rename("qarz_soni"))
    S["oraliq_std"] = S["oraliq_std"].fillna(0)
    S[["fanlar_soni", "past_fanlar"]] = S[["fanlar_soni", "past_fanlar"]].fillna(0)
    S["trend"] = S["oraliq_ortacha"] - S["oldingi_ball"]
    S["joriy_oraliq_farq"] = S["joriy_ortacha"] - S["oraliq_ortacha"]
    S["nb_jami"] = S["nb_sababsiz"] + S["nb_sababli"]
    S["ish_yuki"] = (S["ish_soati"] - 20).clip(lower=0)
    S["faollik"] = S["topshiriq"] / 100 + np.log1p(S["lms_kirish"]) / 5 + S["mustaqil_soat"] / 20 + S["kutubxona"] / 10
    S["nb_limit"] = (S["nb_sababsiz"] >= 36).astype(int)
    S = S.reset_index()
    if fan_stats is None:
        fan_stats = fan_statistics(g)
    F = g.merge(S[["id"] + NUM_BASE + CAT + AGG + ENG].rename(columns={"id": "talaba_id"}), on="talaba_id", how="left")
    F["fan_qiyinlik"] = F["fan_id"].map(fan_stats["qiyinlik"]).fillna(0)
    F["fan_yak_farq"] = F["fan_id"].map(fan_stats["yak_farq"]).fillna(0)
    F["oraliq_vs_ortacha"] = F["oraliq"] - F["oraliq_ortacha"]
    return S, F, fan_stats


def preprocessor(num, cat):
    parts = [("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num)]
    if cat:
        parts.append(("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                                       ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat))
    return ColumnTransformer(parts)


def P(model, num=NUM, cat=CAT):
    return Pipeline([("pre", preprocessor(num, cat)), ("m", model)])


XGB_REG_DEFAULT = dict(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.85, colsample_bytree=0.8)


def reg_candidates(xgb_params):
    xgb = XGBRegressor(**xgb_params, random_state=SEED, n_jobs=1)
    hgb = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=SEED)
    return {
        "Ridge regressiya": Ridge(alpha=3.0),
        "Random Forest": RandomForestRegressor(n_estimators=250, min_samples_leaf=2, max_features=0.5, random_state=SEED, n_jobs=1),
        "HistGradientBoosting": hgb,
        "XGBoost (sozlangan)": xgb,
        "Stacking ansambl": StackingRegressor([("ridge", Ridge(alpha=3.0)), ("xgb", clone(xgb)), ("hgb", clone(hgb))],
                                              final_estimator=RidgeCV(), cv=3, n_jobs=1),
    }


def clf_candidates():
    xgb = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.85, colsample_bytree=0.8,
                        random_state=SEED, n_jobs=1)
    lr = LogisticRegression(max_iter=4000, C=0.5, class_weight="balanced")
    rf = RandomForestClassifier(n_estimators=250, min_samples_leaf=2, class_weight="balanced", random_state=SEED, n_jobs=1)
    return {
        "Logistik regressiya": lr, "Random Forest": rf,
        "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=250, learning_rate=0.05, class_weight="balanced", random_state=SEED),
        "XGBoost": xgb,
        "Soft-voting ansambl": VotingClassifier([("lr", clone(lr)), ("rf", clone(rf)), ("xgb", clone(xgb))], voting="soft"),
    }


def _reg_row(name, y, p, baseline=False):
    r = {"model": name, "MAE": round(float(mean_absolute_error(y, p)), 3),
         "RMSE": round(float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2))), 3), "R2": round(float(r2_score(y, p)), 3)}
    if baseline: r["baseline"] = True
    return r


def _clf_row(name, y, pred, proba, binary, baseline=False):
    r = {"model": name, "Accuracy": round(float(accuracy_score(y, pred)), 3),
         "F1 (macro)": round(float(f1_score(y, pred, average="macro")), 3)}
    if binary:
        r["Recall"] = round(float(recall_score(y, pred)), 3)
        r["ROC-AUC"] = round(float(roc_auc_score(y, proba[:, 1] if proba is not None else pred)), 3)
    if baseline: r["baseline"] = True
    return r


N_CLUSTERS = 5
ENGINE_VERSION = 2                     # kod o'zgarsa, saqlangan model qayta o'qitiladi


def name_clusters(cen: pd.DataFrame):
    """Klaster markazlari (z-ballar) bo'yicha tushunarli nom berish: natija o'qi va faollik o'qi."""
    score = (cen["oraliq_ortacha"] + 0.5 * cen["oldingi_ball"]) / 1.5
    act = (cen["topshiriq"] + 0.5 * cen["lms_kirish"] + 0.5 * cen["mustaqil_soat"] - cen["nb_sababsiz"]) / 2.5
    tot = score + act
    names, left = {}, set(cen.index)
    top = int(tot.idxmax()); names[top] = "A'lochi va faol"; left.discard(top)
    low = int(tot.loc[list(left)].idxmin()); names[low] = "Xavf guruhi"; left.discard(low)
    for k in sorted(left, key=lambda k: -tot[k]):
        d = act[k] - score[k]
        if d > 0.3: n = "Faol, lekin bahosi past"
        elif d < -0.3: n = "Qobiliyatli, lekin intizomi past"
        elif tot[k] > 0: n = "Yaxshi, barqaror"
        else: n = "O'rtacha, sust"
        while n in names.values(): n += " (2)" if not n.endswith(")") else ""
        names[int(k)] = n
    return names


# ======================================================================= dvigatel
class Engine:
    def __init__(self):
        self.ready = False
        self.oof = {}

    # ------------------------------------------------------------- to'liq o'qitish (CV, tuning, taqqoslash)
    def train_full(self, students, grades, G: Grading, status=lambda s: None):
        t0 = time.time()
        self.G = G
        S, F, self.fan_stats = build_frames(students, grades, G)
        lab = S[S["yakuniy_ortacha"].notna()].reset_index(drop=True)
        if len(lab) < 50:
            raise ValueError("O'qitish uchun yakuniy natijasi ma'lum kamida 50 ta talaba kerak")
        X, y = lab[NUM + CAT], lab["yakuniy_ortacha"]
        ids = lab["id"].to_numpy()
        self.metrics = {}
        kf = KFold(5, shuffle=True, random_state=SEED)

        # --- 1. XGBoost tuning
        status("1/7 XGBoost giperparametrlari sozlanmoqda (RandomizedSearchCV)…")
        search = RandomizedSearchCV(P(XGBRegressor(random_state=SEED, n_jobs=1)), {
            "m__n_estimators": [200, 300, 400, 600, 800], "m__max_depth": [2, 3, 4, 5, 6],
            "m__learning_rate": [0.02, 0.03, 0.05, 0.08], "m__subsample": [0.7, 0.8, 0.9, 1.0],
            "m__colsample_bytree": [0.5, 0.7, 0.9], "m__min_child_weight": [1, 3, 5], "m__reg_lambda": [0.5, 1, 3, 10]},
            n_iter=16, scoring="neg_mean_absolute_error", cv=3, random_state=SEED, n_jobs=-1)
        search.fit(X, y)
        self.xgb_params = {k.replace("m__", ""): v for k, v in search.best_params_.items()}
        pdef = cross_val_predict(P(XGBRegressor(**XGB_REG_DEFAULT, random_state=SEED, n_jobs=1)), X, y, cv=3, n_jobs=-1)
        self.metrics["tuning"] = {"params": self.xgb_params, "mae_default": round(float(mean_absolute_error(y, pdef)), 3),
                                  "mae_tuned": round(float(-search.best_score_), 3), "iterations": 16}

        # --- 2. Regressiya
        status("2/7 Regressiya modellari taqqoslanmoqda (5-fold CV)…")
        rows, preds = [], {}
        rows.append(_reg_row("Bazaviy: o'rtacha qiymat", y, cross_val_predict(DummyRegressor(), X, y, cv=kf), True))
        rows.append(_reg_row("Bazaviy: faqat oraliq o'rtachasi", y, cross_val_predict(LinearRegression(), lab[["oraliq_ortacha"]], y, cv=kf), True))
        for name, m in reg_candidates(self.xgb_params).items():
            preds[name] = cross_val_predict(P(m), X, y, cv=kf, n_jobs=-1)
            rows.append(_reg_row(name, y, preds[name]))
        best = min((r for r in rows if not r.get("baseline")), key=lambda r: r["MAE"])
        self.reg_name = best["model"]
        self.sigma = float(np.std(y - preds[self.reg_name]))
        self.metrics["reg"] = {"best": self.reg_name, "rows": rows, "sigma": round(self.sigma, 2),
                               "scatter": [[round(float(a), 1), round(float(b), 1)] for a, b in zip(y[:500], preds[self.reg_name][:500])]}
        oof_reg = preds[self.reg_name]

        # --- 3. Klassifikatsiyalar
        cats = G.categories()
        y_cat = y.map(G.cat_index).astype(int)
        y_fail = (lab["qarz_soni"] > 0).astype(int)
        y_chet = (lab["qarz_soni"] >= 3).astype(int)
        status("3/7 Klassifikatsiya: toifalar…")
        self.clf_name, oof_cat = self._cv_clf("toifa", X, y_cat, len(cats), ("Bazaviy: oraliq bo'yicha toifa", lab["oraliq_ortacha"].map(G.cat_index).astype(int)))
        self.metrics["toifa"]["categories"] = cats
        self.metrics["toifa"]["distribution"] = {cats[i]: int((y_cat == i).sum()) for i in range(len(cats))}
        status("4/7 Klassifikatsiya: o'tadi / o'tmaydi, chetlashtirish xavfi…")
        self.bin_name, oof_fail = self._cv_clf("otmaydi", X, y_fail, 2, ("Bazaviy: oraliqda o'tmagan fan bor", (lab["past_fanlar"] > 0).astype(int)))
        self.chet_name, oof_chet = self._cv_clf("chetlash", X, y_chet, 2, ("Bazaviy: oraliqda 3+ o'tmagan fan", (lab["past_fanlar"] >= 3).astype(int)))

        # --- 4. Fan darajasidagi model
        status("5/7 Har bir fan bo'yicha model (talaba × fan)…")
        Fk = F[F["talaba_id"].isin(ids) & F["yakuniy"].notna()].reset_index(drop=True)
        XS, yS = Fk[NUM + CAT + SUBJ_EXTRA], Fk["yakuniy"]
        gkf = GroupKFold(5)
        srows = [_reg_row("Bazaviy: fan oraliq + joriy", yS, cross_val_predict(LinearRegression(), Fk[["oraliq", "joriy"]], yS, cv=gkf, groups=Fk["talaba_id"]), True)]
        ps = cross_val_predict(P(XGBRegressor(**self.xgb_params, random_state=SEED, n_jobs=1), NUM + SUBJ_EXTRA, CAT), XS, yS,
                               cv=gkf, groups=Fk["talaba_id"], n_jobs=-1)
        srows.append(_reg_row("XGBoost (fan darajasi)", yS, ps))
        self.subj_sigma = float(np.std(yS - ps))
        pm = Fk["pass_min"].to_numpy()
        fail_true = (yS.to_numpy() < pm).astype(int)
        fail_p = norm.cdf((pm - ps) / self.subj_sigma)
        per_fan = []
        for fid, grp in Fk.assign(pred=ps, fp=fail_p, ft=fail_true).groupby("fan_id"):
            per_fan.append({"fan_id": int(fid), "fan": grp["fan"].iloc[0], "n": int(len(grp)),
                            "MAE": round(float(mean_absolute_error(grp["yakuniy"], grp["pred"])), 2),
                            "qarz_haqiqiy": round(float(grp["ft"].mean()), 3), "qarz_bashorat": round(float(grp["fp"].mean()), 3)})
        self.metrics["fan"] = {"rows": srows, "sigma": round(self.subj_sigma, 2), "n_rows": int(len(Fk)),
                               "auc_qarz": round(float(roc_auc_score(fail_true, fail_p)), 3) if len(set(fail_true)) > 1 else None,
                               "per_fan": sorted(per_fan, key=lambda r: -r["n"])}

        # --- 5. Aniqlikni oshirish tahlili
        status("6/7 Belgilar guruhlari ta'siri (ablation) va o'rganish egri chizig'i…")
        xgbp = lambda: XGBRegressor(**self.xgb_params, random_state=SEED, n_jobs=1)
        act = ["nb_sababli", "nb_sababsiz", "topshiriq", "lms_kirish", "kutubxona", "mustaqil_soat"]
        abl = []
        for name, num, cat in (("Faqat baholar (joriy + oraliq)", AGG, []), ("+ davomat va faollik", AGG + act, []),
                               ("+ ijtimoiy-iqtisodiy omillar", NUM_BASE + AGG, CAT), ("+ hosila belgilar (to'liq)", NUM, CAT)):
            p = cross_val_predict(P(xgbp(), num, cat), lab[num + cat], y, cv=kf, n_jobs=-1)
            abl.append({"belgilar": name, "n_features": len(num) + len(cat), "MAE": round(float(mean_absolute_error(y, p)), 3),
                        "R2": round(float(r2_score(y, p)), 3)})
        self.metrics["ablation"] = abl
        rng = np.random.default_rng(SEED)
        perm = rng.permutation(len(lab))
        test_idx, pool = perm[: len(lab) // 5], perm[len(lab) // 5:]
        lc = []
        for n in [s for s in (50, 100, 200, 400, 700) if s < len(pool)] + [len(pool)]:
            maes, aucs = [], []
            for _ in range(3 if n < len(pool) else 1):
                tr = rng.choice(pool, n, replace=False) if n < len(pool) else pool
                m = P(xgbp()).fit(X.iloc[tr], y.iloc[tr])
                maes.append(mean_absolute_error(y.iloc[test_idx], m.predict(X.iloc[test_idx])))
                if y_fail.iloc[tr].nunique() > 1:
                    c = P(LogisticRegression(max_iter=3000, C=0.5, class_weight="balanced")).fit(X.iloc[tr], y_fail.iloc[tr])
                    aucs.append(roc_auc_score(y_fail.iloc[test_idx], c.predict_proba(X.iloc[test_idx])[:, 1]))
            lc.append({"n": int(n), "MAE": round(float(np.mean(maes)), 3), "AUC": round(float(np.mean(aucs)), 3) if aucs else None})
        self.metrics["learning_curve"] = lc

        self.oof = {int(i): {"ball": float(oof_reg[j]), "cat": oof_cat[j].tolist(), "fail": float(oof_fail[j][1]),
                             "chet": float(oof_chet[j][1])} for j, i in enumerate(ids)}
        self.oof_sig = G.signature()
        status("7/7 Yakuniy modellar barcha ma'lumotda o'qitilmoqda…")
        self._fit_final(S, F)
        self.metrics.update(trained_at=time.strftime("%Y-%m-%d %H:%M"), n_labeled=int(len(lab)), n_subject_rows=int(len(Fk)),
                            train_seconds=round(time.time() - t0, 1))
        self.ready = True

    def _cv_clf(self, key, X, y, n_classes, baseline):
        present = sorted(y.unique())
        ye = y.map({c: i for i, c in enumerate(present)})
        skf = StratifiedKFold(max(2, min(5, int(ye.value_counts().min()))), shuffle=True, random_state=SEED)
        binary = n_classes == 2
        rows, probs = [_clf_row(baseline[0], y, baseline[1], None, binary, True)], {}
        for name, m in clf_candidates().items():
            pr = cross_val_predict(P(m), X, ye, cv=skf, method="predict_proba", n_jobs=-1)
            full = np.zeros((len(X), n_classes)); full[:, present] = pr
            probs[name] = full
            rows.append(_clf_row(name, y, full.argmax(1), full, binary))
        score = "ROC-AUC" if binary else "F1 (macro)"
        best = max((r for r in rows if not r.get("baseline")), key=lambda r: r[score])
        self.metrics[key] = {"best": best["model"], "rows": rows, "score": score, "rate": round(float(y.mean()), 3) if binary else None}
        return best["model"], probs[best["model"]]

    # ------------------------------------------------------------- yakuniy modellar (tez qayta o'qitish ham shu)
    def _fit_final(self, S, F):
        lab = S[S["yakuniy_ortacha"].notna()]
        X, y = lab[NUM + CAT], lab["yakuniy_ortacha"]
        self.reg = P(reg_candidates(self.xgb_params)[self.reg_name]).fit(X, y)
        y_cat = y.map(self.G.cat_index).astype(int)
        self.clf_classes = sorted(y_cat.unique())
        self.clf = P(clf_candidates()[self.clf_name]).fit(X, y_cat.map({c: i for i, c in enumerate(self.clf_classes)}))
        self.bin = P(clf_candidates()[self.bin_name]).fit(X, (lab["qarz_soni"] > 0).astype(int))
        y_chet = (lab["qarz_soni"] >= 3).astype(int)
        self.chet = P(clf_candidates()[self.chet_name]).fit(X, y_chet) if y_chet.nunique() > 1 else None
        Fk = F[F["talaba_id"].isin(lab["id"]) & F["yakuniy"].notna()]
        self.subj = P(XGBRegressor(**self.xgb_params, random_state=SEED, n_jobs=-1), NUM + SUBJ_EXTRA, CAT).fit(Fk[NUM + CAT + SUBJ_EXTRA], Fk["yakuniy"])
        self.explain_pipe = P(XGBRegressor(**self.xgb_params, random_state=SEED, n_jobs=-1)).fit(X, y)
        pre = self.explain_pipe.named_steps["pre"]
        oh = pre.named_transformers_["cat"].named_steps["oh"]
        self.fmap = np.array(NUM + [f for f, c in zip(CAT, oh.categories_) for _ in c])
        self.explainer = shap.TreeExplainer(self.explain_pipe.named_steps["m"])
        Xt = pre.transform(X.sample(min(600, len(X)), random_state=SEED))
        self.global_imp = (pd.Series(np.abs(np.asarray(self.explainer.shap_values(Xt))).mean(0), index=self.fmap)
                           .groupby(level=0).sum().sort_values(ascending=False))
        self._fit_unsupervised(S)
        self.n_labeled, self.n_total = int(len(lab)), int(len(S))

    def _cluster_matrix(self, S):
        C = S[CLUSTER_FEATS].astype(float).copy()
        C["nb_sababsiz"] = np.log1p(C["nb_sababsiz"]); C["lms_kirish"] = np.log1p(C["lms_kirish"])
        return C.fillna(C.median()).fillna(0)

    def _fit_unsupervised(self, S):
        C = self._cluster_matrix(S)
        self.c_scaler = StandardScaler().fit(C)
        Z = self.c_scaler.transform(C)
        self.km = KMeans(N_CLUSTERS, n_init=20, random_state=SEED).fit(Z)
        self.pca = PCA(2, random_state=SEED).fit(Z)
        ss = dict(sample_size=min(1500, len(Z)), random_state=SEED)
        self.silhouette = float(silhouette_score(Z, self.km.labels_, **ss))
        self.k_scan = [{"k": k, "silhouette": round(float(silhouette_score(Z, KMeans(k, n_init=10, random_state=SEED).fit_predict(Z), **ss)), 3),
                        } for k in range(2, 8)]
        self.cluster_names = name_clusters(pd.DataFrame(self.km.cluster_centers_, columns=CLUSTER_FEATS))
        self.cluster_centers = pd.DataFrame(self.km.cluster_centers_, columns=CLUSTER_FEATS).round(2).to_dict("records")
        self.nn = NearestNeighbors(n_neighbors=7).fit(Z)
        self.nn_ids = S["id"].to_numpy()
        self.iso_pre = preprocessor(NUM, [])
        self.iso = IsolationForest(n_estimators=300, contamination=0.04, random_state=SEED).fit(self.iso_pre.fit_transform(S[NUM]))

    def refit(self, students, grades, G):
        """Tez qayta o'qitish: tanlangan algoritmlar va parametrlar bilan, CV'siz."""
        self.G = G
        S, F, self.fan_stats = build_frames(students, grades, G)
        self._fit_final(S, F)

    # ------------------------------------------------------------- bashorat
    def predict(self, students, grades, use_oof=True):
        S, F, _ = build_frames(students, grades, self.G, self.fan_stats)
        X = S[NUM + CAT]
        ball = np.clip(self.reg.predict(X), 0, 100)
        cats = self.G.categories()
        pc = np.zeros((len(X), len(cats)))
        cls = [c for c in self.clf_classes if c < len(cats)]
        pc[:, cls] = self.clf.predict_proba(X)[:, :len(cls)]
        pf = self.bin.predict_proba(X)[:, 1]
        pch = self.chet.predict_proba(X)[:, 1] if self.chet is not None else np.zeros(len(X))
        Z = self.c_scaler.transform(self._cluster_matrix(S))
        cl, xy = self.km.predict(Z), self.pca.transform(Z)
        A = self.iso_pre.transform(S[NUM])
        an_score, an_flag = -self.iso.score_samples(A), self.iso.predict(A) == -1
        Fp = F.copy()
        if len(Fp):
            Fp["pred"] = np.clip(self.subj.predict(F[NUM + CAT + SUBJ_EXTRA]), 0, 100)
            Fp["fail_p"] = norm.cdf((Fp["pass_min"] - Fp["pred"]) / self.subj_sigma)
        by_student = dict(tuple(Fp.groupby("talaba_id"))) if len(Fp) else {}
        oof_ok = use_oof and getattr(self, "oof_sig", None) == self.G.signature()
        out = []
        for i in range(len(S)):
            r = S.iloc[i]
            sid = int(r["id"]) if pd.notna(r["id"]) else None
            o = self.oof.get(sid) if oof_ok and sid is not None else None
            b = float(np.clip(o["ball"], 0, 100)) if o else float(ball[i])
            cp = np.array(o["cat"]) if o and len(o["cat"]) == len(cats) else pc[i]
            f = o["fail"] if o else float(pf[i])
            ch = o["chet"] if o else float(pch[i])
            subj = []
            sg = by_student.get(r["id"])
            if sg is not None:
                for _, g in sg.sort_values("fan").iterrows():
                    gr = self.G.grade(float(g["pred"]), g["tizim_id"])
                    subj.append({"fan_id": int(g["fan_id"]), "fan": g["fan"], "pred": round(float(g["pred"]), 1),
                                 "fail_p": round(float(g["fail_p"]), 3), "pass_min": float(g["pass_min"]),
                                 "baho": gr["baho"], "baho_nomi": gr["nomi"]})
            flags = []
            if r["nb_sababsiz"] >= 36: flags.append(f"Sababsiz NB {int(r['nb_sababsiz'])} soat: limitga yaqin yoki oshgan")
            if pd.notna(r["trend"]) and r["trend"] <= -15: flags.append(f"Keskin pasayish: oldingi semestrga nisbatan {r['trend']:.0f} ball")
            if pd.notna(r["joriy_oraliq_farq"]) and abs(r["joriy_oraliq_farq"]) >= 15: flags.append("Joriy va oraliq nazorat natijalari keskin farq qiladi")
            if an_flag[i]: flags.append("Ko'rsatkichlar g'ayrioddiy kombinatsiyada (IsolationForest)")
            gr = self.G.grade(b)
            out.append({"id": sid, "ball": round(b, 1), "baho": gr["baho"], "toifa": cats[int(np.argmax(cp))],
                        "toifa_proba": {cats[j]: round(float(cp[j]), 3) for j in range(len(cats))},
                        "fail_proba": round(f, 3), "otadi": f < 0.5, "chet_proba": round(ch, 3),
                        "qarz_kutilgan": round(float(sum(s["fail_p"] for s in subj)), 2),
                        "alo_proba": round(float(norm.cdf((b - GRANT_CHEGARA) / self.sigma)), 3),
                        "pasayish_proba": round(float(norm.cdf((GRANT_SAQLASH - b) / self.sigma)), 3),
                        "cluster": int(cl[i]), "cluster_name": self.cluster_names[int(cl[i])],
                        "x": round(float(xy[i, 0]), 3), "y": round(float(xy[i, 1]), 3),
                        "anomaly": bool(an_flag[i]), "anomaly_score": round(float(an_score[i]), 3), "flags": flags,
                        "subjects": subj, "oof": bool(o)})
        return out, S

    def explain(self, S_row: pd.DataFrame, top=12):
        Xt = self.explain_pipe.named_steps["pre"].transform(S_row[NUM + CAT])
        sv = np.asarray(self.explainer.shap_values(Xt))[0]
        s = pd.Series(sv, index=self.fmap).groupby(level=0).sum()
        s = s.reindex(s.abs().sort_values(ascending=False).index).head(top)
        return [{"feature": f, "label": LABELS.get(f, f), "value": _fmt(S_row.iloc[0][f]), "impact": round(float(v), 2)} for f, v in s.items()]

    def similar(self, S_row, exclude_id=None, k=6):
        Z = self.c_scaler.transform(self._cluster_matrix(S_row))
        _, idx = self.nn.kneighbors(Z, n_neighbors=min(k + 1, len(self.nn_ids)))
        return [int(self.nn_ids[j]) for j in idx[0] if int(self.nn_ids[j]) != exclude_id][:k]


def _fmt(v):
    if isinstance(v, (float, np.floating)):
        if np.isnan(v): return "—"
        return int(v) if float(v).is_integer() else round(float(v), 1)
    if isinstance(v, np.integer): return int(v)
    return v
