# Vecka 1 – Kapitel 1: Introduktion till maskininlärning (svar)

## Faktafrågor

**1. Hur hänger AI, ML och DL ihop?**
De är tre nästlade nivåer, som ringar inuti varandra.
- **AI (artificiell intelligens)** är det bredaste begreppet: allt som får datorer att lösa uppgifter som normalt kräver mänsklig intelligens (kan vara handkodade regler).
- **ML (maskininlärning)** är en delmängd av AI där systemet lär sig mönster ur data i stället för att programmeras med explicita regler.
- **DL (djupinlärning)** är en delmängd av ML som använder djupa neurala nätverk (många lager) och är särskilt starkt på ostrukturerad data som bild, ljud och text.

Alltså: **DL ⊂ ML ⊂ AI**.

**2. Vilka är de fyra problemkategorierna inom ML?**
1. **Övervakad inlärning** (*supervised*) – vi har facit (etiketter `y`). Ex. regression och klassificering.
2. **Oövervakad inlärning** (*unsupervised*) – inget facit, hitta struktur själv. Ex. klustring, dimensionsreducering.
3. **Semi-övervakad inlärning** – en liten del av datan är etiketterad, resten inte.
4. **Förstärkningsinlärning** (*reinforcement learning*) – en agent lär sig genom att agera i en miljö och få belöning/straff.

**3. Förklara följande:**

**a) Syftet med att dela upp i tränings-, validerings- och testdata.**
- **Träningsdata** – modellen lär sig (anpassar sina parametrar) på denna.
- **Valideringsdata** – används för att välja modell och trimma hyperparametrar utan att titta på testdatan.
- **Testdata** – rörs *inte* under utvecklingen; används en enda gång på slutet för en ärlig uppskattning av hur modellen presterar på ny, osedd data. Delningen skyddar mot att man luras av överanpassning.

**b) Vad är k-delad korsvalidering?**
Man delar träningsdatan i *k* lika stora delar (*folds*). Modellen tränas *k* gånger; varje gång används en fold som valideringsset och de övriga *k*−1 till träning. Resultatet medelvärdesbildas. Ger en stabilare prestandauppskattning än ett enda train/val-split och utnyttjar all data.

**c) Vad är RMSE?**
*Root Mean Squared Error* = roten ur medelvärdet av de kvadrerade felen:
$$\text{RMSE} = \sqrt{\tfrac{1}{n}\sum (y_i - \hat{y}_i)^2}$$
Det är genomsnittsfelet i samma enhet som `y` (t.ex. kronor). Kvadreringen gör att stora fel straffas hårdare.

**d) Hyperparameter vs parameter.**
- **Parameter** – lärs av modellen under träning ur datan (t.ex. koefficienterna/vikterna i linjär regression).
- **Hyperparameter** – sätts av dig *innan* träning och styr hur inlärningen går till (t.ex. `max_depth` för ett träd, `alpha` för Ridge). Modellen lär sig inte dessa själv.

**e) Grid search + `refit=True`.**
*Grid search* provar systematiskt alla kombinationer i ett rutnät ("**grid**") av hyperparametervärden och "**söker**" igenom dem för att hitta den bästa kombinationen (mätt med korsvalidering). Namnet: varje axel i rutnätet är en hyperparameter, varje skärningspunkt en kombination som testas.
`refit=True` (standard) innebär att `GridSearchCV`, efter att bästa kombinationen hittats, automatiskt **tränar om** en slutmodell på *hela* träningsdatan med de bästa hyperparametrarna. Då kan man direkt använda det anpassade objektet för `.predict()`.

**f) Kategorisk data och hantering.**
Kategorisk data är variabler med diskreta kategorier snarare än tal.
- **Nominal data** – kategorier *utan* rangordning (t.ex. `färg`: röd/grön/blå).
- **Ordinal data** – kategorier *med* naturlig ordning (t.ex. `betyg`: låg/medel/hög).
- **One-hot-encoding** – skapar en 0/1-kolumn per kategori. Lämpligt för nominal data (ingen falsk ordning införs).
- **Dummy-variable-encoding** – som one-hot men man släpper en kategori (`drop_first=True`) → *k*−1 kolumner, undviker perfekt kollinearitet.
- **Ordinal encoding** – mappar kategorier till heltal (0,1,2…). Lämpligt för ordinal data där ordningen är meningsfull.

**g) Vad är feature engineering?**
Att skapa, omvandla eller välja ut variabler (*features*) för att modellen ska prestera bättre – t.ex. kombinera kolumner, skapa kvoter, extrahera år ur ett datum, logaritmera skeva variabler, encoda kategorier. Ofta det som avgör hur bra en modell blir.

**h) Vad menas med *principle of parsimony*?**
Sparsamhetsprincipen (Occams rakkniv): av två modeller som förklarar data lika bra, föredra den enklare. Enklare modeller generaliserar oftast bättre och är lättare att tolka.

**4. Vad menas med att "en modell är en förenkling av verkligheten"?**
En modell fångar bara de viktigaste sambanden, inte varje detalj. Verkligheten är för komplex och brusig för att beskrivas exakt; modellen är en användbar approximation. "Alla modeller är fel, men vissa är användbara."

**5. Vad menas med att en modell är överanpassad (*overfitted*)?**
Modellen har lärt sig träningsdatan för väl – inklusive brus och slumpmässiga särdrag – i stället för de generella mönstren. Kännetecken: mycket lågt fel på träningsdatan men högt fel på testdatan. Den generaliserar dåligt till ny data.

**6. Högre är bättre i scikit-learn scoring – vad innebär det?**
scikit-learns `scoring`-API är byggt så att ett *högre* värde alltid betyder en *bättre* modell. Det gör att samma maskineri (t.ex. `GridSearchCV`) kan optimera på alla mått genom att maximera. För felmått (där lägre är bättre) negeras därför värdet, se fråga 9.

**7. Tvärsnittsdata, tidsseriedata och paneldata.**
- **Tvärsnittsdata** – många enheter vid *en* tidpunkt (t.ex. 500 diamanter mätta idag).
- **Tidsseriedata** – *en* enhet över *flera* tidpunkter (t.ex. en akties dagliga slutkurs under ett år).
- **Paneldata** – *flera* enheter över *flera* tidpunkter (t.ex. BNP för 10 länder under 20 år).

## Resonemangfrågor

**8. Verkliga tillämpningsområden inom ML.**
- Rekommendationssystem (Netflix, Spotify, e-handel).
- Bildigenkänning (medicinsk diagnostik, ansiktsigenkänning, självkörande bilar).
- Bedrägeridetektering och kreditriskbedömning inom bank/finans.
- Prisprediktion (bostäder, försäkring, dynamisk prissättning).
- Språkmodeller och chattbottar, maskinöversättning.
- Prediktivt underhåll i industrin, efterfrågeprognoser i logistik.

**9. Logiken bakom "negative" mean squared error.**
MSE är ett *felmått* där lägre är bättre. Men scikit-learns konvention är att högre score alltid är bättre. För att båda ska stämma negeras MSE: `neg_mean_squared_error`. Då blir t.ex. −10 bättre än −25, dvs. att *maximera* den negativa MSE:n är samma sak som att *minimera* MSE. Så slipper man specialfall i optimeringsrutinerna.

## Koduppgifter

**10.** (Genomgång/avskrift av kapitlets kodexempel – görs i notebook, inget enskilt svar.)

**11. Varför ger `test_size=0.2` följt av `test_size=0.25` proportionerna 60-20-20?**

```python
# Steg 1: 20 % blir test, 80 % blir "full träning"
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Steg 2: 25 % av de återstående 80 % blir validering
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.25, random_state=42)
```

- Efter steg 1: **test = 20 %**, kvar = 80 % av allt.
- I steg 2 tas 25 % av dessa 80 %: `0.25 × 80 % = 20 %` → **validering = 20 %**.
- Kvar blir `80 % − 20 % = 60 %` → **träning = 60 %**.

Nyckeln: `test_size=0.25` i andra anropet räknas på den *redan minskade* mängden (80 %), inte på hela datan. Därför 0,25 och inte 0,20 för att få jämna 20 %.
