# Vecka 1 – Kapitel 1: Introduktion till maskininlärning

## Faktafrågor

### 1. Hur hänger AI, ML och DL ihop?

Tre nivåer inuti varandra: **DL ⊂ ML ⊂ AI**.
- **AI** – bredast: allt som får datorer att lösa uppgifter som kräver intelligens (även handkodade regler).
- **ML** – lär sig mönster ur data i stället för regler.
- **DL** – ML med djupa neurala nät. Starkt på bild, ljud och text.

### 2. De fyra problemkategorierna inom ML?

1. **Övervakad** – vi har facit (`y`). Ex: regression, klassificering.
2. **Oövervakad** – inget facit, hitta struktur själv. Ex: klustring.
3. **Semi-övervakad** – en liten del är etiketterad, resten inte.
4. **Förstärkning** – en agent lär sig via belöning/straff i en miljö.

### 3. Förklara följande:

**a) Tränings-, validerings- och testdata.**
- **Träning** – modellen lär sig här.
- **Validering** – väljer modell och trimmar hyperparametrar.
- **Test** – rörs inte förrän på slutet; ger en ärlig siffra på ny data. Skyddar mot att luras av överanpassning.

**b) K-delad korsvalidering.**
Dela träningsdatan i *k* delar. Träna *k* gånger, varje gång med en del som validering och resten som träning. Medelvärdera. Stabilare än ett enda split och utnyttjar all data.

**c) RMSE.**
Roten ur medelvärdet av de kvadrerade felen. Snittfel i samma enhet som `y`, straffar stora fel hårdare.

**d) Parameter vs hyperparameter.**
- **Parameter** – lärs av modellen ur datan (t.ex. vikterna i linjär regression).
- **Hyperparameter** – sätts av dig innan träning (t.ex. `max_depth`, `alpha`).

**e) Grid search + `refit=True`.**
Grid search provar alla kombinationer i ett rutnät ("grid") av hyperparametrar och söker fram den bästa (via korsvalidering). `refit=True` tränar om en slutmodell på hela träningsdatan med de bästa värdena, så den går att `.predict()` direkt.

**f) Kategorisk data.**
Variabler med kategorier i stället för tal.
- **Nominal** – ingen ordning (röd/grön/blå).
- **Ordinal** – naturlig ordning (låg/medel/hög).
- **One-hot** – en 0/1-kolumn per kategori. För nominal data.
- **Dummy** – som one-hot men släpp en kolumn (`drop_first=True`) → undviker kollinearitet.
- **Ordinal encoding** – kategori → heltal (0,1,2…). För ordinal data.

**g) Feature engineering.**
Skapa, omvandla eller välja variabler så modellen blir bättre – t.ex. kvoter, år ur datum, logaritmera skeva variabler, encoda kategorier.

**h) Principle of parsimony.**
Occams rakkniv: av två lika bra modeller, välj den enklare. Den generaliserar oftast bättre.

### 4. "En modell är en förenkling av verkligheten"?

Modellen fångar de viktigaste sambanden, inte varje detalj. Verkligheten är för komplex – modellen är en användbar approximation. "Alla modeller är fel, men vissa är användbara."

### 5. Vad betyder "överanpassad" (overfitted)?

Modellen har lärt sig träningsdatan för väl, inklusive bruset. Lågt fel på träning men högt på test → generaliserar dåligt.

### 6. "Högre är bättre" i scikit-learn scoring?

`scoring`-API:t är byggt så att högre värde = bättre modell. Då kan t.ex. `GridSearchCV` alltid maximera. För felmått (där lägre är bättre) negeras värdet, se fråga 9.

### 7. Tvärsnitts-, tidsserie- och paneldata?

- **Tvärsnitt** – många enheter vid en tidpunkt (500 diamanter idag).
- **Tidsserie** – en enhet över flera tidpunkter (en akties kurs under ett år).
- **Panel** – flera enheter över flera tidpunkter (BNP för 10 länder i 20 år).

## Resonemangfrågor

### 8. Exempel på verkliga ML-tillämpningar.

- Rekommendationer (Netflix, Spotify).
- Bildigenkänning (diagnostik, självkörande bilar).
- Bedrägeri- och kreditriskbedömning.
- Prisprediktion (bostäder, försäkring).
- Språkmodeller, översättning, chattbottar.
- Prediktivt underhåll och efterfrågeprognoser.

### 9. Varför "negative" mean squared error?

MSE är ett felmått där lägre är bättre, men scikit-learn vill att högre är bättre. Därför negeras det: `neg_mean_squared_error`. Att maximera −MSE är samma sak som att minimera MSE. Slipper specialfall i optimeringen.

## Koduppgifter

### 10. Avskrift av kapitlets kodexempel.

Görs i notebooken.

### 11. Varför ger `test_size=0.2` sedan `0.25` proportionerna 60-20-20?

- Steg 1: `0.2` → **test = 20 %**, kvar = 80 %.
- Steg 2: `0.25` räknas på de 80 % → `0.25 × 80 = 20 %` → **validering = 20 %**.
- Kvar: `80 − 20 = 60 %` → **träning = 60 %**.

Nyckeln: andra anropet räknar på de återstående 80 %, inte på hela datan.
