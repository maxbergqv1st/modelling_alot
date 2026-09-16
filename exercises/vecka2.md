# Vecka 2 – Kapitel 2: Ett ML projekt från början till slut

## Faktafrågor

### 1. I kapitlet beskrivs en checklista med sju steg. Beskriv de sju stegen översiktligt. I verkligheten, följs dessa steg i en rak progression eller arbetar man generellt sett mer iterativt?

1. **Formulera problemet** – vad ska lösas, vilken affärsnytta, vilket mått på framgång.
2. **Skaffa och utforska data** (EDA) – hämta data, titta på den, förstå fördelningar och samband.
3. **Förbered/förbehandla data** – rensa saknade värden, encoda kategorier, skala, feature engineering.
4. **Välj och träna modeller** – prova några olika modeller på träningsdatan.
5. **Finjustera** – trimma hyperparametrar (t.ex. grid search), variabelselektion.
6. **Presentera lösningen** – kommunicera resultat och begränsningar.
7. **Produktionssätt, övervaka och underhåll** – driftsätt modellen, följ upp prestanda över tid.

I verkligheten arbetar man **iterativt**, inte i en rak linje. Man hoppar ofta tillbaka – t.ex. upptäcker under modellträningen att mer feature engineering behövs, eller att problemformuleringen måste justeras.

### 2. Vad menas med att en modell produktionssätts?

Att modellen tas från utvecklings-/experimentstadiet och driftsätts i ett skarpt system där den gör prediktioner på riktig data för riktiga användare – t.ex. bakom ett API, i en app eller i en batch-process. Inkluderar att den måste vara pålitlig, snabb nog, samt övervakas och underhållas (data kan förändras över tid, *data drift*).

### 3. Vad är *scikit-learn* för något? Biblioteket följer några centrala designprinciper. Vilka är dessa? Vad är *estimators*, *predictors* och *transformers*?

scikit-learn är Pythons mest använda bibliotek för klassisk maskininlärning – med ett enhetligt API för modeller, förbehandling och utvärdering.
Centrala designprinciper: **konsistens** (samma gränssnitt överallt), **inspektion** (hyperparametrar och lärda parametrar är åtkomliga), **begränsad objekthierarki**, **komposition** (bygg pipelines av byggblock) och **förnuftiga standardvärden**.
- **Estimator** – vilket objekt som helst som lär sig från data via `.fit()` (t.ex. en modell eller en encoder).
- **Predictor** – en estimator som dessutom kan göra prediktioner via `.predict()` (t.ex. `LinearRegression`).
- **Transformer** – en estimator som omvandlar data via `.transform()` (t.ex. `StandardScaler`), ofta med bekvämligheten `.fit_transform()`.

### 4. Vad är *TensorFlow* och *Keras*?

- **TensorFlow** – Googles ramverk för numeriska beräkningar och djupinlärning, byggt för att träna och köra neurala nätverk effektivt (även på GPU/TPU).
- **Keras** – ett högnivå-API (numera integrerat i TensorFlow) som gör det enkelt och läsbart att bygga neurala nätverk lager för lager, utan att hantera lågnivådetaljerna.

## Resonemangfrågor

### 5. Kalle och Stina diskuterar maskininlärning över en lunch. Kalle säger "om jag tränat en modell och den inte presterar bra nog på testdatan så justerar jag den tills den gör det." Stina säger "det är ett stort fel att göra så, det enda du då åstadkommer är att du överanpassar testdatan. Hela syftet med testdatan försvinner då". Vad säger du om deras dialog?

Stina har rätt. Om man upprepat justerar modellen tills den presterar bra på *testdatan* så börjar man i praktiken anpassa (överanpassa) mot just den datan – testdatan läcker in i utvecklingsprocessen och slutar vara en oberoende, ärlig mätning. Testresultatet blir då för optimistiskt och säger inget om hur modellen klarar verkligt osedd data. Rätt tillvägagångssätt: trimma mot **valideringsdata** (eller korsvalidering), och rör testdatan en enda gång, allra sist.

### 6. Många AI/ML projekt uppnår inte de ursprungligen satta målen eller att ens passera någon form av prototyp-stadie. Vad tror du detta beror på och hur ska vi förhålla oss till det?

Vanliga orsaker: dålig eller otillräcklig datakvalitet; oklar eller orealistisk problemformulering; att man löser fel problem (ingen verklig affärsnytta); underskattad svårighet i förbehandling och produktionssättning; brist på infrastruktur/kompetens; överskattade förväntningar på vad ML kan. Förhållningssätt: börja litet med en tydlig, mätbar frågeställning, validera nyttan tidigt (prototyp/PoC), säkra datakvalitet först, och acceptera att många idéer inte bär – iterera och lär.

## Koduppgifter

### 7. Gå igenom samtliga kodexempel i kapitlet och skriv gärna av koden manuellt. Det är även bra att experimentera genom att ändra vissa delar av koden och läsa dokumentationen.

(Genomgång/avskrift av kapitlets kodexempel – görs i notebook.)

### 8. Förklara vad koden nedan gör. Varför är det viktigt att kunna spara en modell?
```python
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from joblib import dump, load

X, y = make_regression(n_samples=20000, n_features=3, noise=0.1)

model = LinearRegression().fit(X, y)
dump(model, "linear_model.joblib")
model_loaded = load("linear_model.joblib")

print(model_loaded.predict(X[:5]))
```

```python
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from joblib import dump, load

X, y = make_regression(n_samples=20000, n_features=3, noise=0.1)  # syntetisk regressionsdata

model = LinearRegression().fit(X, y)   # tränar modellen
dump(model, "linear_model.joblib")     # SPARAR den tränade modellen till fil
model_loaded = load("linear_model.joblib")  # LADDAR tillbaka den

print(model_loaded.predict(X[:5]))     # predikterar med den laddade modellen
```

Koden skapar syntetisk data, tränar en linjär regression, sparar modellen med `joblib`, laddar tillbaka den och predikterar de fem första raderna.
**Varför viktigt att spara:** träning kan ta lång tid/mycket resurser. Genom att spara slipper man träna om varje gång – man tränar en gång och återanvänder modellen i produktion, i andra script eller senare i tiden. Det är grunden för att produktionssätta.

### 9. Denna uppgift består av flera steg enligt nedan.
a) Läs in datasetet "data_01.csv" med funktionen `read_csv()` från *Pandas*. Funktionen returnerar en `DataFrame`.

b) Dela upp datasetet i `X` och `y`.

c) Dela upp datan ytterligare, i ett träning-, ett validering-, och ett testset med `train_test_split()`. Låt 20% av datan vara testdata och 15% av den återstående datan vara valideringsdata.

d) Träna två valfria regressionsmodeller (exempelvis `LinearRegression` och `DecisionTreeRegressor`) på träningsdatan. Notera, i kapitel 3 kommer vi lära oss mer om `DecisionTreeRegressor`.

e) Utvärdera modellerna på valideringsdatan.

f) Träna om den bäst presterande modellen på både tränings- och valideringsdatan.

g) Utvärdera modellen på testdatan.

h) Träna om modellen på hela datasetet.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# a) Läs in
df = pd.read_csv("../dataset/data_01.csv")   # kolumner: x1..x5, target

# b) Dela upp i X och y
X = df.drop(columns=["target"])
y = df["target"]

# c) Träning / validering / test  (20 % test, 15 % av resten till validering)
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.15, random_state=42)

# d) Träna två modeller
modeller = {"Linjär regression": LinearRegression(),
            "Beslutsträd":       DecisionTreeRegressor(random_state=42)}
for m in modeller.values():
    m.fit(X_train, y_train)

# e) Utvärdera på valideringsdatan
rmse = {namn: mean_squared_error(y_val, m.predict(X_val)) ** 0.5
        for namn, m in modeller.items()}
for namn, v in rmse.items():
    print(f"{namn:20} val-RMSE: {v:.3f}")

# f) Träna om bästa modellen på träning + validering
bäst = min(rmse, key=rmse.get)
modell = modeller[bäst].fit(
    pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))

# g) Utvärdera på testdatan
print(f"\nBäst: {bäst}  test-RMSE: "
      f"{mean_squared_error(y_test, modell.predict(X_test)) ** 0.5:.3f}")

# h) Träna om på hela datasetet (inför produktion)
modell.fit(X, y)
```

**Resultat (random_state=42):** linjär regression vinner klart (val-RMSE ≈ 3,6 mot trädets ≈ 99). Datan är alltså i princip ett linjärt samband → trädet överanpassar. Test-RMSE för linjär ≈ 3,4.

### 10. Datasetet "salary_dataset.csv" innehåller 29 observationer av personer. `YearsExperience` är hur många år de arbetat, och `Salary` är deras lön.

a) Läs in datasetet "salary_dataset.csv" med pandas `read_csv()`-funktion och dela upp i `X` och `y`. Avgör själv vilken variabel som ska vara den beroende variabeln `y`.

b) Dela upp datasetet i träning- och testset. (Inget validerings-set alltså!)

c) Träna två regressionsmodeller med k-delad korsvalidering med `cross_validate()`-funktionen från *scikit-learn*. Använd `neg_root_mean_squared_error` som `scoring`. Välj själv hur många iterationer den ska göra genom hyperparametern `cv`.

d) Utvärdera modellen som presterar bäst på testsetet.

```python
import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# a) Läs in. y = Salary (det vi vill prediktera), x = YearsExperience
df = pd.read_csv("../dataset/salary_dataset.csv")
X = df[["YearsExperience"]]
y = df["Salary"]

# b) Träning / test (inget valideringsset)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# c) Två modeller med k-delad korsvalidering (cv=5)
modeller = {"Linjär regression": LinearRegression(),
            "Beslutsträd":       DecisionTreeRegressor(random_state=42)}
cv_rmse = {}
for namn, m in modeller.items():
    cv = cross_validate(m, X_train, y_train,
                        scoring="neg_root_mean_squared_error", cv=5)
    cv_rmse[namn] = -cv["test_score"].mean()      # negera tillbaka
    print(f"{namn:20} CV-RMSE: {cv_rmse[namn]:.0f}")

# d) Utvärdera bästa modellen på testsetet
bäst = min(cv_rmse, key=cv_rmse.get)
modell = modeller[bäst].fit(X_train, y_train)
print(f"\nBäst: {bäst}  test-RMSE: "
      f"{mean_squared_error(y_test, modell.predict(X_test)) ** 0.5:.0f}")
```

`Salary` väljs som `y` eftersom lön är det vi rimligen vill förutsäga utifrån erfarenhet. Linjär regression vinner (CV-RMSE ≈ 5300 mot trädets ≈ 5600) – lön vs erfarenhet är ungefär linjärt.

### 11. I denna uppgift kommer vi arbeta med kategorisk data.
Nominaldata är kategorisk data där kategorierna inte har någon inbördes rangordning.

> Datasetet `mpg` som följer med biblioteket *seaborn* är ett dataset med 398 observationer av bilar. Den beroende variabeln `mpg` står för *miles per gallon* och beskriver bilarnas bränsleeffektivitet.

> Om vi vill träna en regressionsmodell på `mpg`-datasetet behöver vi hantera de två kategoriska variablerna `origin` och `name`. Variabeln `origin` är en kategorisk variabel med tre olika värden: `europe`, `japan` och `usa`. Den lämpar sig bra för *one hot encoding*. Variabeln `name` har 305 unika värden. Skulle vi utföra *one hot encoding* på den skulle vårt dataset få 305 nya dimensioner vilket inte är så lämpligt i detta fall. Dessutom - tror du att namnet på bilen har något att göra med hur långt den kör på en *gallon* bensin? Vi ska därför droppa `name`-kolumnen innan vi börjar träna en modell på datan.

```python
import seaborn as sns

df = sns.load_dataset("mpg")
print(df.head())
```

a) Läs in datasetet `mpg` med seaborns `load_dataset()`-funktion (det gjordes i koden ovan).

b) Droppa rader med saknade värden med `dropna()`-metoden.

c) Droppa kolumnen `name`.

d) Utför en *dummy-variable-encoding* på `origin`-kolumnen med pandas `get_dummies()`-funktion. Ange `drop_first=True` så att vi får 2 nya kolumner istället för 3.

e) Dela upp datasetet i `X` och `y`, med `mpg` som den beroende variabeln `y`.

f) Dela upp datasetet i träning- och testset.

g) Träna en linjär regressionsmodell på träningsdatan och utvärdera den på testdatan.

```python
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# a) Läs in (kräver internet första gången seaborn hämtar datan)
df = sns.load_dataset("mpg")

# b) Droppa rader med saknade värden
df = df.dropna()

# c) Droppa 'name' (305 unika värden -> olämpligt att one-hot-encoda)
df = df.drop(columns=["name"])

# d) Dummy-encoda 'origin' (drop_first=True -> 2 kolumner istället för 3)
df = pd.get_dummies(df, columns=["origin"], drop_first=True)

# e) X och y (mpg = beroende variabel)
X = df.drop(columns=["mpg"])
y = df["mpg"]

# f) Träning / test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# g) Träna linjär regression och utvärdera
modell = LinearRegression().fit(X_train, y_train)
rmse = mean_squared_error(y_test, modell.predict(X_test)) ** 0.5
print(f"test-RMSE: {rmse:.2f}")
```

`name` droppas för att one-hot av 305 unika värden skulle skapa 305 nya dimensioner (glest, överanpassningsrisk) – och bilens namn har ingen rimlig kausal koppling till bränsleeffektivitet. `origin` (europe/japan/usa) är nominal och passar för dummy-encoding; `drop_first=True` ger 2 kolumner och undviker perfekt kollinearitet.

> Obs: seaborn hämtar `mpg` från nätet första gången. Utan internet får du `HTTPError` – då kan datan läsas lokalt i stället.

### 12. I avsnitt 2.2 "Ett kodexempel från början till slut - Huspriser
i Kalifornien" så gås ett komplett kodexempel igenom. På valideringsdatan fick vi `RMSE Random Forest Regression: 52277.96578719621`. Försök få ett bättre resultat genom att exempelvis justera hyperparametrar eller genomföra variabelselektion.

Idéer att prova (kräver hela kodexemplet från avsnitt 2.2):
- **Hyperparametertrimning** av `RandomForestRegressor` via `GridSearchCV`/`RandomizedSearchCV` – t.ex. `n_estimators`, `max_depth`, `max_features`, `min_samples_leaf`.
- **Feature engineering** – skapa kvoter som `rooms_per_household`, `bedrooms_per_room`, `population_per_household`.
- **Variabelselektion** – ta bort svaga/brusiga features (kolla `feature_importances_`).
- **Hantera skevhet** – logaritmera skeva variabler (t.ex. inkomst, befolkning).
- **Prova andra modeller** – gradient boosting (`HistGradientBoostingRegressor`) slår ofta random forest.

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

param = {"n_estimators": [100, 300], "max_features": [4, 6, 8],
         "max_depth": [None, 10, 20]}
sok = GridSearchCV(RandomForestRegressor(random_state=42), param,
                   scoring="neg_root_mean_squared_error", cv=5, n_jobs=-1)
sok.fit(X_train, y_train)
print(sok.best_params_, -sok.best_score_)
```
