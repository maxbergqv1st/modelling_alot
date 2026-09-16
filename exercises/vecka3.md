# Vecka 3 – Kapitel 3: Regression

## Faktafrågor

### 1. Vad kännetecknar regressionsproblem? Ge exempel.

Regression = man förutsäger ett **tal** (kontinuerligt `y`), inte en kategori. Exempel: huspriser, lön, temperatur, diamantpriser.

### 2. Förklara RMSE, MSE och MAE.

Fel = `y − ŷ`.
- **MAE** – snittet av felens absolutbelopp. Samma enhet som `y`, tål extremvärden.
- **MSE** – snittet av felen i kvadrat. Straffar stora fel hårt, enhet i kvadrat.
- **RMSE** – roten ur MSE. Tillbaka i `y`:s enhet, men fortfarande känslig för stora fel.

### 3. Spelar det roll om vi rangordnar modeller med RMSE eller MSE?

Nej, samma ordning. RMSE är bara roten ur MSE, så lägst MSE = lägst RMSE. Skillnaden är tolkbarheten: RMSE är i `y`:s enhet.

### 4. Vad är gradient descent (översiktligt)?

En metod som stegvis minskar felet. Man räknar ut lutningen (gradienten) och tar små steg nedför tills man når ett minimum. Steglängden = *learning rate*. Som att gå ned i en dal i den brantaste riktningen.

### 5. Vad är bias–variance trade-off? Varför är komplexa modeller inte alltid bättre?

Felet består av **bias** (för enkel modell → underanpassar) och **varians** (för känslig modell → överanpassar).
- Enkel modell: hög bias, låg varians.
- Komplex modell: låg bias, hög varians.

Minskar man den ena ökar ofta den andra. En för komplex modell lär sig bruset och blir sämre på ny data. Målet är balansen med minst totalt fel.

### 6. Förklara modellerna översiktligt.

- **a) Linjär regression** – drar en rät linje som minimerar kvadrerade fel. Enkel, tolkbar.
- **b) Ridge** – linjär + L2-straff på stora koefficienter (`alpha`). Krymper vikter, minskar överanpassning.
- **c) Lasso** – linjär + L1-straff. Kan **nolla** koefficienter → väljer variabler automatiskt.
- **d) Elastic net** – blandning av Ridge och Lasso (`alpha`, `l1_ratio`).
- **e) SVR** – lägger en marginal ("slang") runt datan; kan fånga icke-linjära samband via kernels.
- **f) Beslutsträd** – ställer ja/nej-frågor och gissar medelvärdet i varje löv. Fångar icke-linjärt, men överanpassar lätt.
- **g) Ensemble** – kombinerar flera modeller. `VotingRegressor` medelvärdar olika modeller; `BaggingRegressor` tränar många på slumpade delmängder.
- **h) Random forest** – många beslutsträd (bagging + slumpade features). Medelvärdar ihop → lägre varians, stark standardmodell.

### 7. White box vs black box?

- **White box** – lätt att förstå *varför*, t.ex. linjär regression och beslutsträd.
- **Black box** – svår att tolka trots bra resultat, t.ex. random forest och neurala nät.

### 8. Skillnaden mellan bagging och pasting?

Båda tränar modeller på slumpade delmängder. Skillnaden:
- **Bagging** – urval **med** återläggning (samma rad kan komma flera gånger).
- **Pasting** – urval **utan** återläggning.

## Resonemangfrågor

### 9. Tolka figur 3.1 (s. 113).

Visar grundidén: en linje dras genom ett punktmoln. Avståndet mellan punkt och linje är felet. Linjen väljs så att de kvadrerade felen blir så små som möjligt.

### 10. Tolka figur 3.13 (s. 140) och kopplingen till 3.14 (s. 141).

De visar under- respektive överanpassning. När modellen blir mer komplex sjunker träningsfelet, men valideringsfelet vänder uppåt vid överanpassning. Samma sak från två håll: anpassningen i datan och fel-kurvorna.

### 11. Vad är R² (determinationskoefficienten)?

Andelen av variationen i `y` som modellen förklarar.
- `R² = 1` – perfekt.
- `R² = 0` – lika bra som att alltid gissa medelvärdet.
- Negativt – sämre än att gissa medelvärdet.

Formel: $R^2 = 1 - \frac{\sum(y_i-\hat{y}_i)^2}{\sum(y_i-\bar{y})^2}$. Högre är bättre.

## Koduppgifter

Körbar kod i **`vecka3.ipynb`**.

### 12. Avskrift av kapitlets kodexempel.

Görs i notebooken.

### 13. EDA på hr_employee_data.xlsx (ingen ML).

Läs in med `read_excel`. Titta på storlek, datatyper, saknade värden och statistik. Nyckelfrågor för ledningen: hur många har slutat (`left`), och skiljer sig nöjdhet/timmar/projekt mellan de som stannat och slutat, per avdelning och lönenivå. Nöjdheten är klart lägre bland de som slutat; låg lön och vissa avdelningar har högre personalomsättning.

### 14. Diabetes-datasetet – komplett ML-flöde.

Läs in som DataFrame, dela train/test, träna linjär regression och random forest, utvärdera med MAE/RMSE/R². Datan är svår: `R² ≈ 0.45` på testet, men flödet fungerar.

### 15. car_price_dataset.csv.

**a)** Komplett flöde med `Price` som `y`. Obs: filen är **semikolon-separerad** (`sep=";"`). One-hot på kategorierna, random forest ger `R² ≈ 0.99`.
**b)** Streamlit-app där man matar in bildata och får ett pris (kodmall i notebooken, kör med `streamlit run app.py`).
**c)** Ja, det används i verkligheten: värdering av begagnade bilar, försäkringsbolag och bilhandlare.

### 16. diamonds.csv – komplett ML-flöde.

`price` som `y`. Droppa indexkolumnen `Unnamed: 0`, one-hot på `cut`/`color`/`clarity`, random forest ger `R² ≈ 0.98`. Viktigaste variabeln är `carat`.
