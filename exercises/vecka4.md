# Vecka 4 – Kapitel 4: Klassificering

## Faktafrågor

### 1. Vad kännetecknar klassificeringsproblem? Ge exempel.

Klassificering = man förutsäger en **klass/etikett** (kategoriskt `y`), inte ett tal. Exempel: spam eller inte, sjuk/frisk, katt/hund, kreditrisk, churn, siffror 0–9.

### 2. Förklara OvR och OvO.

Två sätt att lösa **flerklassproblem** med binära klassificerare:
- **OvR (One-vs-Rest)** – en modell per klass ("denna mot alla andra"). N klasser → N modeller. Väljer klassen med högst score.
- **OvO (One-vs-One)** – en modell per klasspar. N klasser → N·(N−1)/2 modeller. Klassen som vinner flest dueller väljs. Bra för modeller som skalar dåligt med datamängd (t.ex. SVM). OvR är default för de flesta.

### 3. Förklara utvärderingsmåtten.

TP/FP/FN/TN = sant/falskt positiv/negativ.
- **a) Confusion matrix** – tabell: verklig klass (rader) mot predikterad (kolumner). Diagonalen = rätt, resten visar vilka fel som görs.
- **b) Accuracy** = (TP+TN)/alla. Enkel, men **missvisande vid obalans** (99 % "frisk" ger 99 % accuracy).
- **c) Precision** = TP/(TP+FP). Hur pålitliga de positiva prediktionerna är.
- **d) Recall** = TP/(TP+FN). Hur många av de verkligt positiva vi hittade.
- **e) F1** = harmoniskt medel av precision och recall. Högt bara när båda är höga.
- **f) ROC-kurvan** – True Positive Rate mot False Positive Rate vid olika trösklar. **AUC**: 1.0 = perfekt, 0.5 = gissning.

### 4. Vad är precision–recall tradeoff?

Måtten drar åt olika håll via beslutströskeln. Högre tröskel → högre precision, lägre recall (missar fler). Lägre tröskel → tvärtom. Man får välja balans efter vad som är dyrast: falska larm eller missade fall.

### 5. Förklara modellerna översiktligt.

- **a) Logistisk regression** – linjär modell som ger en sannolikhet (sigmoid/softmax); en tröskel avgör klassen. Enkel, tolkbar.
- **b) SVC** – hittar hyperplanet med **störst marginal** mellan klasserna. Med kernel (t.ex. RBF) även icke-linjära gränser.
- **c) Beslutsträd** – ja/nej-frågor som delar datan, väljer splits som minskar orenheten (gini/entropi). Tolkbart men överanpassar lätt.
- **d) Ensemble** – kombinerar modeller. `VotingClassifier` låter modeller rösta; `BaggingClassifier` tränar samma modell på många bootstrap-urval → minskar varians.
- **e) Random forest** – bagging av beslutsträd där varje split bara ser slumpade features. Avkorrelerar träden → lägre varians, robust.
- **f) Extra trees** – som random forest men slumpar **även trösklarna**. Snabbare, ännu lägre varians, lite mer bias.

### 6. Vad innebär feature importance i trädmodeller?

Träd mäter hur mycket varje feature minskar orenheten över alla splits. Ger en **rangordning av vilka variabler som betyder mest** – bra för att förstå datan och välja variabler. (Obs: partisk mot features med många nivåer; permutation importance är pålitligare.)

## Resonemangfrågor

### 7. Stina vill ha högsta möjliga precision. Vad händer med recall?

Hög precision fås genom att bara säga "positiv" när modellen är mycket säker (hög tröskel) – priset är **lägre recall**, man missar verkliga fall. Hög precision är bra när en falsk positiv är dyr (spamfilter får inte slänga riktig e-post). Dåligt när man inte får missa fall (cancerscreening, bedrägeri → prioritera recall).
Rättsväsendet: "hellre fria tio skyldiga än fälla en oskyldig" → **hög precision på 'skyldig'** (döm bara vid stark bevisning) och därmed lägre recall. Ett medvetet val: en falsk fällning värderas mycket dyrare än en missad.

### 8. Tolka figur 4.8 (s. 175).

Visar **precision och recall som funktion av tröskeln**. Högre tröskel → precisionen stiger, recall faller; lägre tröskel → tvärtom. Används för att **välja tröskel** efter vad verksamheten prioriterar, i stället för standard 0.5.

### 9. Varför `fit_transform` på träning men bara `transform` på validering/test?

Förbehandlaren (t.ex. `StandardScaler`) lär sig medelvärde och standardavvikelse i `fit` – och det ska ske **bara på träningsdatan**. Kör man `fit` även på test läcker information in (**data leakage**) och utvärderingen blir för optimistisk. På validering/test tillämpar man bara den redan inlärda transformen. Speglar verkligheten: ny data skalas med träningens statistik.

## Koduppgifter

Körbar kod i **`vecka4.ipynb`** (uppgift 15 i `vecka4_mnist.ipynb`).

### 10. Avskrift av kapitlets kodexempel.

Görs i notebooken.

### 11. Vad gör `classification_report`-koden och hur tolkas resultatet?

Jämför facit (`y_true`) mot prediktioner (`y_pred`) och skriver ut precision, recall och F1 per klass. Se notebooken för utskrift och tolkning.

### 12. Visualisera ett beslutsträd med `plot_tree`.

Görs i notebooken.

### 13. hr_employee_data.xlsx – komplett ML-flöde med `left` som `y`.

`left` = 1 (slutat) / 0 (kvar). Klassificeringsflöde, se notebooken.

### 14. Iris-datasetet – komplett ML-flöde.

Klassificera blomsorten (setosa/versicolor/virginica). Grundflöde med två features, se notebooken.

### 15. MNIST – prediktera egna handskrivna siffror.

Huvudarbetet är **preprocessing** av mobilbilderna så modellen kan läsa dem. Se `vecka4_mnist.ipynb`.
