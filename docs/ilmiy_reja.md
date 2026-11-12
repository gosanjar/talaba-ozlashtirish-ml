# Mashinali o'qitish asosida talabalar o'zlashtirish ko'rsatkichini tahlil qilish modelini ishlab chiqish

## Annotatsiya

**O'zbekcha.** Ushbu ishda talabalarning o'zlashtirish ko'rsatkichini tahlil qiluvchi va bashorat qiluvchi mashinali o'qitish tizimi ishlab chiqildi. Tizim fanlar bo'yicha oraliq nazorat natijalari, sababli va sababsiz qoldirilgan darslar (NB), topshiriqlarning bajarilishi, to'lov shakli (grant yoki kontrakt), kontrakt qarzdorligi va ijtimoiy omillardan foydalanib uchta vazifani hal qiladi. Birinchisi regressiya: semestr yakuniy o'rtacha balini bashorat qilish (Stacking ansambl, MAE ≈ 1.96 ball, R² ≈ 0.956). Ikkinchisi klassifikatsiya: "a'lo / yaxshi / qoniqarli / qoniqarsiz" toifalarini aniqlash (Soft-voting ansambl, F1 ≈ 0.89), akademik qarzdorlikni (ROC-AUC ≈ 0.98) va chetlashtirish xavfini (ROC-AUC ≈ 0.99) oldindan aniqlash. Har bir fan bo'yicha qarz ehtimolini beruvchi fan darajasidagi model ham qurildi (ROC-AUC ≈ 0.98). Uchinchisi klasterlash: K-Means yordamida 1500 talabani "xavf guruhi", "faol, lekin bahosi past" kabi guruhlarga ajratish. Model qarorlari SHAP usuli bilan izohlanadi. Har bir talaba uchun avtomatik pedagogik tavsiyalar beriladi. Tizim web ilova ko'rinishida amalga oshirildi va yangi ma'lumot qo'shilganda modellar avtomatik qayta o'qitiladi.

**Kalit so'zlar:** mashinali o'qitish, ta'lim ma'lumotlarini tahlil qilish (EDM), o'zlashtirishni bashorat qilish, XGBoost, SHAP, erta ogohlantirish tizimi.

**Аннотация (русский).** Разработана система машинного обучения для анализа и прогнозирования успеваемости студентов. На основе промежуточных оценок по предметам, пропусков занятий, выполнения заданий, формы обучения (грант или контракт) и социальных факторов решаются три задачи. Регрессия прогнозирует итоговый балл (MAE ≈ 1.96, R² ≈ 0.956). Классификация определяет категорию успеваемости (F1 ≈ 0.89), риск академической задолженности (ROC-AUC ≈ 0.98) и риск отчисления (ROC-AUC ≈ 0.99). Кластеризация K-Means выделяет группы студентов. Решения интерпретируются методом SHAP, система реализована в виде веб-приложения.

**Abstract (English).** We develop a machine learning system that analyses and predicts student performance from mid-term subject scores, absences, assignment completion, tuition type (state grant or contract) and social factors. It solves three tasks. Regression predicts the final score (MAE ≈ 1.96, R² ≈ 0.956). Classification predicts the grade category (F1 ≈ 0.89), the risk of academic debt (ROC-AUC ≈ 0.98) and the risk of expulsion (ROC-AUC ≈ 0.99). K-Means clustering groups students into interpretable profiles. Predictions are explained with SHAP, and the system is delivered as a web application that retrains automatically when new data is added.

---

## 1. Mavzuning dolzarbligi
- OTMlarda talabalarning o'qishdan chetlashishi va akademik qarzdorlik muhim muammo. Muammo ko'pincha semestr oxirida, kech aniqlanadi.
- HEMIS, LMS va elektron jurnallar sabab ta'lim ma'lumotlari hajmi ortdi, lekin ular asosan hisobot uchun ishlatiladi, bashorat uchun emas.
- Mashinali o'qitish xavf guruhini **erta** aniqlab, profilaktik choralar (tyutor, konsultatsiya, psixolog) ko'rishga imkon beradi.

## 2. Tadqiqot maqsadi
Talabalarning ijtimoiy-demografik va akademik ma'lumotlari asosida o'zlashtirish ko'rsatkichini bashorat qiluvchi va qarorlarini tushuntiruvchi mashinali o'qitish modelini ishlab chiqish hamda uni amaliy web ilova ko'rinishida joriy etish.

## 3. Tadqiqot vazifalari
1. Ta'lim ma'lumotlarini intellektual tahlil qilish (Educational Data Mining) bo'yicha adabiyotlarni o'rganish.
2. Ma'lumotlar to'plamini tanlash, razvedka tahlilini (EDA) o'tkazish va ma'lumotni qayta ishlash.
3. Klassifikatsiya va regressiya algoritmlarini qo'llash va ularni taqqoslash.
4. Eng yaxshi modelning giperparametrlarini optimallashtirish.
5. Model qarorlarini SHAP usuli bilan izohlash.
6. Modelni web ilovaga integratsiya qilish.

## 4. Tadqiqot obyekti va predmeti
- **Obyekt:** ta'lim muassasasi talabalarining o'zlashtirish jarayoni.
- **Predmet:** talabalar o'zlashtirishini bashorat qilishning mashinali o'qitish usullari va modellari.

## 5. Tadqiqot usullari
Statistik tahlil, korrelyatsion tahlil, nazoratli o'qitish (klassifikatsiya va regressiya), ansambl usullari (bagging, boosting), kross-validatsiya, giperparametrlarni tasodifiy qidiruv, tushuntiriladigan sun'iy intellekt (SHAP).

## 6. Ilmiy yangiligi
- Sakkiz xil algoritm yagona protokol asosida (bir xil preprocessing pipeline va 5-fold stratifikatsiyalangan CV) taqqoslandi.
- Regressiya, klassifikatsiya va klasterlash **yagona tizimda** birlashtirildi. To'lov shakli (grant yoki kontrakt) va kontrakt qarzdorligi kabi O'zbekiston OTMlariga xos omillar hisobga olindi.
- Yangi ma'lumot kiritilganda modellar avtomatik qayta o'qitiladi va eng yaxshi algoritm avtomatik tanlanadi.
- Global va individual (talaba darajasidagi) SHAP izohlari asosida avtomatik pedagogik tavsiyalar yaratish mexanizmi ishlab chiqildi.

## 7. Amaliy ahamiyati
O'qituvchi, kurator va dekanat web ilova orqali talaba yoki butun guruh ma'lumotini kiritib, xavf darajasini, taxminiy yakuniy bahoni, sabablarni va tavsiyalarni oladi. Model OTMning o'z ma'lumotlari (HEMIS, LMS) asosida qayta o'qitilishi mumkin.

## 8. Olingan natijalar (1500 talaba, 1302 tasi yakuniysi ma'lum; 5-fold cross-validation)

| Vazifa | Eng yaxshi algoritm | Natija | Bazaviy yechim / boshqa algoritmlar |
|---|---|---|---|
| Regressiya: yakuniy ball | Stacking (Ridge + XGBoost + HGB) | MAE = 1.96, R² = 0.956 | Faqat oraliq: MAE 3.10 · XGBoost 2.01 · Ridge 2.09 · RF 2.26 |
| Klassifikatsiya: 4 ta toifa | Soft-voting ansambl | Accuracy = 0.888, F1 = 0.886 | Oraliq bo'yicha qoida: F1 0.80 · HGB 0.884 · LR 0.863 |
| O'tadi / o'tmaydi | Soft-voting ansambl | ROC-AUC = 0.983, Recall = 0.89 | Qoida: AUC 0.87 · LR 0.982 · RF 0.980 |
| Chetlashtirish xavfi (3+ qarz) | HistGradientBoosting | ROC-AUC = 0.989 | Qoida: AUC 0.77 · XGBoost 0.989 |
| Fan darajasi (7812 qator) | XGBoost + GroupKFold | MAE = 3.67, qarz AUC = 0.977 | Oraliq + joriy: MAE 4.21 |
| Klasterlash | K-Means (k=5) | Silhouette = 0.22 | A'lochi va faol, Qobiliyatli lekin intizomi past, Faol lekin bahosi past, O'rtacha sust, Xavf guruhi |

**Aniqlikni oshirish:**
- Giperparametrlarni sozlash: XGBoost MAE 2.09 dan 2.05 ga tushdi.
- Belgilarni qo'shish (ablation): faqat baholar bilan MAE 2.94, davomat va faollik qo'shilganda 2.62, ijtimoiy-iqtisodiy omillar qo'shilganda 2.01.
- Ma'lumot hajmi (learning curve): 50 talabada MAE 3.16, 100 talabada 2.97, 400 talabada 2.30, 1042 talabada 2.01.

Eng muhim omillar (SHAP): oraliq va joriy nazorat o'rtachasi, sababsiz NB, kontrakt qarzdorligi, oldingi akademik qarzlar, oldingi semestr bali, faollik indeksi.

> Eslatma: natijalar namunaviy (sintetik) ma'lumotda olingan. Haqiqiy OTM ma'lumotida qayta o'qitilganda raqamlar o'zgaradi va dissertatsiyaga o'sha natijalar yoziladi.

## 9. Ishning taxminiy tuzilmasi (bitiruv ishi yoki magistrlik dissertatsiyasi)

**Kirish:** dolzarblik, maqsad, vazifalar, obyekt, predmet, yangilik, amaliy ahamiyat.

**I bob. Ta'lim ma'lumotlarini intellektual tahlil qilishning nazariy asoslari**
1.1. Educational Data Mining va Learning Analytics tushunchalari
1.2. Talabalar o'zlashtirishiga ta'sir etuvchi omillar
1.3. O'zlashtirishni bashorat qilish bo'yicha xorijiy va mahalliy tadqiqotlar tahlili

**II bob. Mashinali o'qitish usullari va modelni ishlab chiqish**
2.1. Ma'lumotlar tuzilmasi: fanlar bo'yicha baholar, NB, to'lov shakli va ijtimoiy omillar
2.2. Ma'lumotlarni qayta ishlash (preprocessing) va belgilarni tanlash
2.3. Regressiya, klassifikatsiya va klasterlash algoritmlari: nazariy asoslar
2.4. Baholash metrikalari va kross-validatsiya

**III bob. Eksperimental tadqiqot va dasturiy ta'minot**
3.1. Modellarni o'qitish va taqqoslash natijalari
3.2. Talabalarni klasterlash va guruhlar tavsifi
3.3. SHAP yordamida modelni izohlash
3.4. Pedagogik tavsiyalar mexanizmi
3.5. Web ilova arxitekturasi va interfeysi

**Xulosa. Foydalanilgan adabiyotlar. Ilovalar** (dastur kodi, notebook).

## 10. Asosiy adabiyotlar
1. Cortez P., Silva A. Using Data Mining to Predict Secondary School Student Performance // Proc. of 5th FUture BUsiness TEChnology Conference (FUBUTEC 2008). Porto, 2008. P. 5–12.
2. Romero C., Ventura S. Educational data mining and learning analytics: An updated survey // WIREs Data Mining and Knowledge Discovery. 2020. Vol. 10, No. 3. e1355.
3. Chen T., Guestrin C. XGBoost: A Scalable Tree Boosting System // Proc. 22nd ACM SIGKDD. 2016. P. 785–794.
4. Lundberg S. M., Lee S.-I. A Unified Approach to Interpreting Model Predictions // Advances in Neural Information Processing Systems 30 (NeurIPS). 2017.
5. Breiman L. Random Forests // Machine Learning. 2001. Vol. 45. P. 5–32.
6. Kuzilek J., Hlosta M., Zdrahal Z. Open University Learning Analytics dataset // Scientific Data. 2017. Vol. 4. 170171.
7. Pedregosa F. et al. Scikit-learn: Machine Learning in Python // JMLR. 2011. Vol. 12. P. 2825–2830.

> Eslatma: adabiyotlar ro'yxatini OTMingiz talab qiladigan formatga (GOST yoki APA) moslang. Kirish qismi uchun O'zbekistondagi ta'limni raqamlashtirish bo'yicha me'yoriy hujjatlarni ham qo'shing.
