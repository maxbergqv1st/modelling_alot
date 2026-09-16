# Lathund: hyperparametrar

Hyperparameter = inställning du sätter **innan** träning (`.fit`). Modellen lär
sig inte dessa själv — du väljer dem. Rätt värden hittas genom att prova och
mäta MAE på testdatan (eller korsvalidering).

Grundregeln bakom nästan alla: **överanpassning** (modellen pluggar facit
utantill: lågt träningsfel, högt testfel) vs **underanpassning** (för trubbig:
högt fel på båda). Du trimmar hyperparametrar för att landa mittemellan.

---

## Beslutsträd — `DecisionTreeRegressor` / `Classifier`

Det är här det mesta trimmandet sker. Ett obegränsat träd överanpassar nästan
alltid.

| Parameter | Vad den gör | När du drar i den |
|---|---|---|
| `max_depth` | Max antal frågor på djupet (trädets höjd). | **Första du testar.** Sänk (t.ex. 5–10) om test-MAE >> tränings-MAE (överanpassning). Höj om båda är dåliga. |
| `min_samples_leaf` | Minsta antal rader i ett löv. | Höj (t.ex. 5, 20, 50) för att jämna ut. Stoppar löv som bara memorerar 1 diamant. Bra alternativ/komplement till `max_depth`. |
| `min_samples_split` | Minsta antal rader för att alls dela en nod. | Höj för att bromsa djupa, spretiga träd. Liknar `min_samples_leaf`. |
| `max_features` | Hur många features trädet får titta på per delning. | Sänk för att minska överanpassning och göra träd olika (viktigt i Random Forest). |
| `max_leaf_nodes` | Tak på totalt antal löv. | Alternativ hävstång mot `max_depth` — begränsar bredd i stället för djup. |
| `random_state` | Frö för slumpen. | Sätt ett fast tal (t.ex. 42) så resultatet blir **reproducerbart**. Inte en trimknapp — bara för att få samma svar varje körning. |
| `criterion` | Hur "bra delning" mäts (`squared_error`, `absolute_error`...). | Rör sällan. Byt till `absolute_error` om du bryr dig mer om medianfel / robusthet mot extremvärden. |

**Typiskt arbetsflöde:** börja med bara `max_depth`, loopa värden 3–15, plotta
tränings- vs test-MAE, välj där testfelet är lägst innan det börjar öka igen.

---

## Linjär regression — `LinearRegression`

Har i praktiken **inga** hyperparametrar att trimma (`fit_intercept` sätter du
nästan aldrig om). Om vanlig linjär regression överanpassar (många features,
`get_dummies` gav massa kolumner) → byt modell till en **regulariserad** variant:

| Modell | Parameter | När |
|---|---|---|
| `Ridge` | `alpha` | Många features, vill dämpa alla vikter jämnt. Höj `alpha` = mer dämpning. |
| `Lasso` | `alpha` | Vill att modellen **nollar** oviktiga features (automatiskt urval). |
| `ElasticNet` | `alpha`, `l1_ratio` | Blandning av Ridge + Lasso. |

`alpha=0` ≈ vanlig regression. Öka `alpha` om test-MAE >> tränings-MAE.

---

## Random Forest — `RandomForestRegressor` (nästa steg efter enskilt träd)

Många träd som röstar. Ärver alla trädets parametrar ovan, plus:

| Parameter | Vad den gör | När |
|---|---|---|
| `n_estimators` | Antal träd. | Höj (100 → 300 → 500) tills MAE slutar förbättras. Mer = bättre men långsammare, överanpassar inte av fler träd. |
| `max_features` | Features per delning. | Standard brukar duga. Sänk för mer olika (mindre korrelerade) träd. |
| `n_jobs` | Antal CPU-kärnor. | Sätt `-1` för att använda alla → tränar snabbare. Påverkar inte resultatet. |

---

## Hur man hittar bra värden utan att gissa

Loopa för hand (som din cell 14), eller låt sklearn söka:

```python
from sklearn.model_selection import GridSearchCV

rutnat = {"max_depth": [4, 6, 8, 12], "min_samples_leaf": [1, 10, 50]}
sok = GridSearchCV(DecisionTreeRegressor(random_state=1337),
                   rutnat, scoring="neg_mean_absolute_error", cv=5)
sok.fit(X_train, y_train)
print(sok.best_params_)   # bästa kombinationen
```

`cv=5` = korsvalidering: delar träningsdatan i 5 bitar och mäter stabilt,
i stället för att lita på ett enda test-split.

---

**Tumregel:** börja alltid enkelt (default), mät, och trimma bara den knapp
som din tränings-vs-test-MAE säger är problemet. Ändra en sak i taget.
