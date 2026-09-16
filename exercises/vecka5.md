# Vecka 5 – Kapitel 5: Dimensionsreducering

## Faktafrågor

### 1. Vad menas med curse of dimensionality?

Fler dimensioner (features) → datan blir glesare, punkterna hamnar långt ifrån varandra och avståndsmått tappar mening. Man behöver mycket mer data, träningen blir långsammare och risken för överanpassning ökar.

### 2. Vad är dimensionsreducering och varför?

Att minska antalet features men behålla så mycket information (varians) som möjligt. Skäl:
- snabbare träning, mindre minne,
- motverkar curse of dimensionality och överanpassning,
- gör 2D/3D-visualisering möjlig,
- tar bort redundanta variabler.

### 3. Hur fungerar PCA (översiktligt)?

PCA hittar riktningarna med störst varians – *principalkomponenterna*. Första komponenten är linjen som bevarar mest varians när datan projiceras på den (fig 5.4). Andra är ortogonal mot den och fångar näst mest, osv. Behåller man bara de *k* första projiceras datan ner till *k* dimensioner med minimal informationsförlust.

### 4. Hur kan kernel PCA utvärderas?

Det saknar "rätt svar", så indirekt:
- **Som steg i en pipeline** – mät hur bra nedströmsmodellen presterar (t.ex. `GridSearchCV` över kärna och `n_components`).
- Via **reconstruction error** – hur väl datan kan återskapas från den reducerade formen.

## Resonemangfrågor

### 5. Är bästa möjliga prediktion alltid målet?

Kalle har rätt – tid, minne och kostnad väger tungt. En modell som är marginellt sämre men tränar och predikterar mycket snabbare är ofta bättre i produktion (t.ex. realtid). Man optimerar mot helheten: träffsäkerhet *och* resurser.

### 6. Vad händer med tolkningen efter PCA?

Man tappar tolkbarheten. Komponenterna är linjärkombinationer av alla originalvariabler, så en axel motsvarar ingen namngiven variabel längre. Man vinner komprimering men förlorar "den här featuren betyder X".

## Koduppgifter

Körbar kod i **`vecka5.ipynb`**.

### 7. Avskrift av kapitlets kodexempel.

Görs i notebooken.

### 8. Vad gör PCA-koden?

Skapar data med 3 features, reducerar till 2 med `PCA.fit_transform`, och återskapar sedan 3 dimensioner med `inverse_transform`. `np.allclose` blir `False` – återskapningen är inte exakt eftersom information gick förlorad i reduceringen.

### 9. PCA på car_price_dataset.csv före modellering – hur påverkas resultatet?

Det blir **sämre**. RandomForest hanterar alla 52 features bra; PCA slänger bort signal (R² faller från ≈0.99 till ≈0.76 vid k=10). PCA lönar sig när features är många och starkt korrelerade, inte som standardsteg.
