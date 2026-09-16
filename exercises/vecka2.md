# Vecka 2 – Kapitel 2: Ett ML-projekt från början till slut

## Faktafrågor

### 1. Beskriv de sju stegen. Arbetar man rakt eller iterativt?

1. **Formulera problemet** – vad ska lösas, vilken nytta, vilket mått på framgång.
2. **Skaffa och utforska data (EDA)** – hämta data, förstå fördelningar och samband.
3. **Förbehandla data** – rensa saknade värden, encoda, skala, feature engineering.
4. **Träna modeller** – prova några olika på träningsdatan.
5. **Finjustera** – trimma hyperparametrar, välj variabler.
6. **Presentera lösningen** – kommunicera resultat och begränsningar.
7. **Produktionssätt och övervaka** – driftsätt och följ upp över tid.

Man arbetar **iterativt**, inte rakt – hoppar ofta tillbaka när nya problem dyker upp.

### 2. Vad menas med att en modell produktionssätts?

Den tas från experiment till skarp drift och gör prediktioner på riktig data för riktiga användare (t.ex. bakom ett API). Måste vara pålitlig, snabb nog, samt övervakas – data kan ändras över tid (*data drift*).

### 3. Vad är scikit-learn? Designprinciper? Estimators, predictors, transformers?

Pythons vanligaste bibliotek för klassisk ML, med enhetligt API.
Principer: **konsistens**, **inspektion**, **begränsad objekthierarki**, **komposition** (pipelines) och **förnuftiga standardvärden**.
- **Estimator** – lär sig från data via `.fit()`.
- **Predictor** – kan även prediktera via `.predict()` (t.ex. `LinearRegression`).
- **Transformer** – omvandlar data via `.transform()` (t.ex. `StandardScaler`).

### 4. Vad är TensorFlow och Keras?

- **TensorFlow** – Googles ramverk för djupinlärning, tränar neurala nät effektivt (även GPU/TPU).
- **Keras** – högnivå-API i TensorFlow som gör det enkelt att bygga nät lager för lager.

## Resonemangfrågor

### 5. Kalle justerar modellen tills den är bra på testdatan. Vad säger du?

Stina har rätt. Justerar man tills det blir bra på *testdatan* så överanpassar man mot just den – testdatan läcker in och slutar vara en ärlig mätning. Trimma mot **valideringsdata** (eller korsvalidering) och rör testdatan en enda gång, sist.

### 6. Varför når många ML-projekt inte målen?

Vanliga orsaker: dålig datakvalitet, oklar problemformulering, fel problem (ingen nytta), underskattad förbehandling/produktionssättning, brist på kompetens, för höga förväntningar. Förhållningssätt: börja litet med en mätbar fråga, validera nyttan tidigt (PoC), säkra datakvaliteten, och acceptera att många idéer inte bär – iterera.

## Koduppgifter

Körbar kod i **`vecka2.ipynb`**.

### 7. Avskrift av kapitlets kodexempel.

Görs i notebooken.

### 8. Vad gör koden? Varför spara en modell?

Skapar syntetisk data, tränar en linjär regression, sparar den med `joblib.dump`, laddar tillbaka med `load` och predikterar. **Varför spara:** träning kan ta lång tid. Sparar man slipper man träna om – tränar en gång och återanvänder modellen i produktion. Grunden för att produktionssätta.

### 9. data_01.csv – fullständigt flöde.

Läs in, dela X/y, dela train/validering/test (20 % test, 15 % av resten till validering). Träna linjär regression och beslutsträd, utvärdera på valideringsdatan, träna om bästa på träning+validering, utvärdera på testet, träna sist om på hela datan. Linjär regression vinner klart (val-RMSE ≈ 3,6 mot trädets ≈ 99) – datan är i princip linjär, trädet överanpassar.

### 10. salary_dataset.csv – korsvalidering, inget valideringsset.

`Salary` = `y` (det vi vill förutsäga), `YearsExperience` = `X`. Dela train/test, jämför två modeller med `cross_validate` (`neg_root_mean_squared_error`, `cv=5`), utvärdera bästa på testet. Linjär regression vinner – lön mot erfarenhet är ungefär linjärt.

### 11. mpg – kategorisk data.

Droppa rader med saknade värden och kolumnen `name` (305 unika värden → olämpligt att one-hot:a, och namnet påverkar inte bränsleåtgång). Dummy-encoda `origin` med `drop_first=True` (2 kolumner, undviker kollinearitet). `mpg` = `y`, dela train/test, träna linjär regression och utvärdera med RMSE.

### 12. Förbättra RandomForest på huspris-exemplet.

Idéer: trimma hyperparametrar (`n_estimators`, `max_depth`, `max_features`, `min_samples_leaf`) med `GridSearchCV`, skapa nya features (kvoter som `rooms_per_household`), variabelselektion via `feature_importances_`, logaritmera skeva variabler, eller prova gradient boosting (`HistGradientBoostingRegressor`).
