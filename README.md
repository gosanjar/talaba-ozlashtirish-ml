# 🎓 Mashinali o'qitish asosida talabalar o'zlashtirish ko'rsatkichini tahlil qilish

Talabalar bazasini yurituvchi va har bir talaba uchun **yakuniy ball, toifa, qarz, chetlashtirish xavfi va grant imkoniyati**ni bashorat qiluvchi web tizim. Model qanday qurilgani Google Colab notebookida ko'rsatilgan.

| | Havola |
|---|---|
| 🌐 **Jonli demo (frontend)** | https://gosanjar.github.io/talaba-ozlashtirish-ml/ |
| 📓 **Google Colab notebook** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gosanjar/talaba-ozlashtirish-ml/blob/main/notebook/talaba_ozlashtirish_ML.ipynb) |
| 📊 **Dataset** | [`dataset/`](dataset) papkasi: 1500 talaba, 9000 baho |
| 📄 **Ilmiy reja** | [`docs/ilmiy_reja.md`](docs/ilmiy_reja.md) |

> Jonli demo — statik versiya: barcha sahifalar, grafiklar va 1500 talabaning profillari ochiladi. Talaba qo'shish, tahrirlash, "agar…" simulyatori va qayta o'qitish uchun to'liq versiyani pastdagi yo'riqnoma bo'yicha o'z kompyuteringizda ishga tushiring.

---

## Nima qiladi

**Ma'lumot:** 3 fakultet → 6 yo'nalish → 48 guruh (1–4 kurs) → 1500 talaba, 34 fan.

Har bir talaba haqida quyidagilar saqlanadi:
- fanlar bo'yicha joriy, oraliq va yakuniy ballar;
- sababli va sababsiz NB;
- grant yoki kontrakt, kontrakt qarzi;
- ishlaydimi va qayerda yashaydi;
- topshiriqlar, LMS faolligi va boshqalar.

| Vazifa | Natija | Eng yaxshi model | Aniqlik (5-fold CV) |
|---|---|---|---|
| **Regressiya** | Yakuniy o'rtacha ball (0–100) | Stacking (Ridge + XGBoost + HGB) | MAE **1.96** ball, R² **0.956** |
| **Klassifikatsiya** | A'lo / Yaxshi / Qoniqarli / Qoniqarsiz | Soft-voting ansambl | Accuracy 0.888, F1 **0.886** |
| **Klassifikatsiya** | O'tadi / o'tmaydi (kamida 1 qarz) | Soft-voting ansambl | ROC-AUC **0.983**, Recall 0.89 |
| **Klassifikatsiya** | Chetlashtirish xavfi (3+ qarz) | HistGradientBoosting | ROC-AUC **0.989** |
| **Fan darajasi** | Har bir fan bo'yicha ball va qarz ehtimoli | XGBoost (GroupKFold) | MAE 3.67, qarz AUC **0.977** |
| **Klasterlash** | Xavf guruhi, Faol lekin bahosi past va boshqalar (5 guruh) | K-Means + PCA | Silhouette 0.22 |
| **Anomaliyalar** | G'ayrioddiy profillar | Isolation Forest | 4% |
| **Izohlash** | Nima uchun shunday bashorat qilindi | SHAP | — |

Taqqoslash uchun "faqat oraliq o'rtachasi" bazaviy yechimi MAE = 3.10 beradi, ya'ni model **37% aniqroq**.

**Aniqlikni oshirish uchun qilingan ishlar:**
- giperparametrlarni sozlash (RandomizedSearchCV);
- ansambllar (Stacking, Soft-voting);
- hosila belgilar;
- belgilar guruhlari tahlili (ablation): faqat baholar bilan MAE 2.94 bo'lsa, to'liq belgilar bilan 2.01;
- ma'lumot hajmi ta'siri (learning curve): 100 talabada MAE 2.97, 1042 talabada 2.01.

### Web tizim sahifalari
- **Bosh panel:**
  - KPI'lar;
  - toifalar, ballar taqsimoti va bosqichlar bo'yicha grafiklar;
  - klasterlar xaritasi;
  - eng qiyin fanlar;
  - xavfli va a'lochi talabalar.
- **Tuzilma:** fakultet → yo'nalish → guruh → talaba bo'yicha drill-down va bosqich (kurs) filtri.
- **Talabalar:**
  - qidirish, saralash, sahifalash;
  - toifa, klaster, xavf va to'lov shakli bo'yicha filtrlar;
  - maxsus ro'yxatlar (grantga da'vogarlar, grantni yo'qotish xavfi, NB limiti, anomaliyalar).
- **Talaba profili:**
  - barcha bashoratlar va har bir fan bo'yicha bashorat;
  - SHAP sabablari va tavsiyalar;
  - guruh va kursdagi o'rni;
  - o'xshash talabalar;
  - **"Agar…" simulyatori**: NB yoki topshiriqni o'zgartirsangiz, natija qanday o'zgarishi ko'rinadi.
- **Yangi talaba:** guruh tanlanadi, o'quv rejadagi fanlar avtomatik chiqadi va bashorat jonli yangilanadi.
- **Analitika:**
  - omillar ta'siri;
  - korrelyatsiya matritsasi;
  - fanlar bo'yicha kutilayotgan qarzlar;
  - global SHAP.
- **Klasterlar** va **Modellar va aniqlik:** barcha metrikalar, bazaviy modellar bilan taqqoslash, tuning, ablation, learning curve.
- **Boshqaruv:**
  - fakultet, yo'nalish, guruh (kurs) va fan qo'shish yoki tahrirlash;
  - o'quv reja;
  - **baholash tizimlari** (5 baholik, ECTS, GPA 4.0 yoki o'zingizniki).

  Baholash tizimi o'zgarsa, model fonda avtomatik qayta o'qitiladi.

---

## 🚀 Ishga tushirish yo'riqnomasi (to'liq versiya)

**Kerak:** Python 3.10+ va Git.

### 1-usul: bitta buyruq

macOS / Linux:
```bash
git clone https://github.com/gosanjar/talaba-ozlashtirish-ml.git
```
```bash
cd talaba-ozlashtirish-ml && ./run.sh
```

Windows: papkani oching va `run.bat` faylini ikki marta bosing.

Skript quyidagilarni bajaradi:
1. virtual muhit yaratadi;
2. kutubxonalarni o'rnatadi;
3. serverni ishga tushiradi.

Brauzerda oching: **http://localhost:8000**

### 2-usul: qo'lda
```bash
python3 -m venv .venv
```
```bash
source .venv/bin/activate
```
Windows'da: `.venv\Scripts\activate`
```bash
pip install -r requirements.txt
```
```bash
uvicorn app.main:app --port 8000
```

**Birinchi ishga tushishda:**
- `data/talabalar.db` bazasi 1500 ta namunaviy talaba bilan avtomatik yaratiladi;
- barcha modellar o'qitiladi (~1 daqiqa, sahifada jarayon ko'rinib turadi).

Keyingi ishga tushishlarda saqlangan model (`data/engine.joblib`) darhol yuklanadi.

### Internetga ochish (ngrok)
```bash
ngrok http 8000
```
ngrok bergan `https://….ngrok-free.app` manzilini boshqalarga yuboring.

Tashqaridan kirish **login va parol** bilan himoyalangan. Ular birinchi ishga tushishda `data/auth.txt` fayliga yoziladi (`login:parol`). `localhost`dan kirishda parol so'ralmaydi.

### Google Colab
Yuqoridagi **Open in Colab** tugmasini bosing, so'ng `Runtime → Run all` ni tanlang. Dataset GitHub'dan avtomatik yuklanadi.

Notebook bosqichlari:
1. ma'lumotni yuklash va EDA;
2. feature engineering;
3. tuning;
4. 5 ta regressiya va 5 ta klassifikatsiya modelini bazaviy yechimlar bilan taqqoslash;
5. fan darajasidagi model;
6. ablation va learning curve;
7. klasterlash va anomaliyalar;
8. SHAP;
9. yangi talabalar uchun bashorat.

### O'z ma'lumotingiz bilan (HEMIS)
1. `dataset/` dagi CSV fayllarni namuna sifatida oling: xuddi shu ustunlar bilan o'z faylingizni tayyorlang.
2. Talaba, guruh va fanlarni web tizimning **Boshqaruv** va **+ Yangi talaba** sahifalari orqali kiriting.
3. **Modellar → To'liq qayta o'qitish** tugmasini bosing.

### Statik demoni qayta yig'ish
```bash
python tools/export_dataset.py
```
```bash
python tools/build_demo.py
```
```bash
python tools/build_notebook.py
```
- `export_dataset.py` bazani `dataset/` papkasiga CSV/JSON sifatida chiqaradi.
- `build_demo.py` API javoblarini `docs/api/*.json` ga yozadi, `docs/index.html` esa GitHub Pages'da ochiladi.
- `build_notebook.py` Colab notebookini qayta yaratadi.

---

## 📊 Dataset

| Fayl | Qatorlar | Mazmuni |
|---|---|---|
| [`talabalar.csv`](dataset/talabalar.csv) | 1500 | Tuzilma (fakultet, yo'nalish, guruh, kurs), shaxsiy, ijtimoiy-iqtisodiy, davomat va faollik ko'rsatkichlari |
| [`baholar.csv`](dataset/baholar.csv) | 9000 | Talaba × fan: joriy, oraliq, yakuniy (0–100). Yakuniy ~15% hollarda bo'sh: model aynan shuni bashorat qiladi |
| [`fanlar.csv`](dataset/fanlar.csv) | 34 | Fan, kredit, baholash tizimi |
| [`baholash_tizimlari.json`](dataset/baholash_tizimlari.json) | 3 | 5 baholik, ECTS (A–F), GPA 4.0 chegaralari |
| [`tuzilma.json`](dataset/tuzilma.json) | — | Fakultet → yo'nalish → guruh va o'quv reja |

**Ustunlar (`talabalar.csv`):**
- `talaba_id, fio, fakultet, yonalish_kod, yonalish, guruh, kurs, jinsi, hudud`;
- `tolov` (Grant/Kontrakt), `kontrakt_qarz, stipendiya, yashash, ishlaydi, ota_ona, togarak, olimpiada`;
- `yosh, ish_soati, yol_vaqti, kirish_bali, oldingi_ball, oldingi_qarzlar`;
- `nb_sababli, nb_sababsiz` (soat), `topshiriq` (%), `lms_kirish, kutubxona, mustaqil_soat`.

> ⚠️ Ma'lumotlar **sintetik**: real qonuniyatlar asosida generatsiya qilingan (yashirin omillar: qobiliyat, motivatsiya, ijtimoiy yuk). Ismlar tasodifiy, hech qaysi real shaxsga tegishli emas. Ilmiy ishda HEMIS ma'lumoti bilan almashtiring, shunda natijalar ham o'zgaradi.

---

## 🗂 Loyiha tuzilmasi
```
app/
  main.py          FastAPI server: API, boshqaruv (CRUD), fon rejimida o'qitish, parol himoyasi
  ml.py            ML dvigatel: tuning, regressiya, klassifikatsiya, fan modeli, klasterlash, SHAP
  db.py            SQLite tuzilmasi va namunaviy ma'lumot generatori
  static/index.html  Frontend (bitta fayl, Chart.js)
dataset/           Ochiq dataset (CSV/JSON)
notebook/          Google Colab notebook
docs/              GitHub Pages statik demo + ilmiy reja
tools/             Dataset eksporti, demo va notebook yig'uvchi skriptlar
```

**Texnologiyalar:**
- backend: Python, FastAPI, SQLite;
- ML: scikit-learn, XGBoost, SHAP;
- frontend: vanilla JS va Chart.js.
