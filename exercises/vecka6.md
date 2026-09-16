# Vecka 6 – Kapitel 6: Klustring

## Faktafrågor

### 1. Vad är klustring? Ge exempel.

*Oövervakad* inlärning där man grupperar punkter så att lika hamnar ihop och olika i skilda grupper – utan facit (`y`). Exempel: kundsegmentering, bildkomprimering, anomalidetektion, gruppering av dokument, eller som förbehandling.

### 2. Hur fungerar K-means (översiktligt)?

Bestäm antalet kluster *k* i förväg. Sedan:
1. Placera *k* centroider (slumpvis).
2. **Tilldela** varje punkt till närmaste centroid (fig 6.3).
3. **Flytta** centroiden till medelvärdet av sina punkter (fig 6.4).
4. Upprepa tills centroiderna slutar röra sig.

Resultatet beror på startläget, så man kör flera gånger (`n_init`) och behåller bästa (lägst inertia).

## Resonemangfrågor

### 3. Hur väljer man antal kluster?

- **Inertia** – summan av kvadrerade avstånd till egen centroid. Faller alltid med *k*; leta "armbågen" (elbow) där kurvan planar ut.
- **Silhouette score** ∈ [−1, 1] – hur väl en punkt passar sitt kluster mot närmaste grannkluster. Högre är bättre; välj det *k* som maximerar.
- **Silhouette-diagram** – silhuettvärde per punkt, grupperat per kluster (knivformer). Breda, jämna knivar över medelstrecket = bra kluster.

### 4. Figur 6.10 (s. 247) – hur många kluster, och är det exakt vetenskap?

Nej, ingen exakt vetenskap. Måtten pekar ofta åt olika håll och rätt antal beror på syftet. Välj det *k* som ger tolkbara, användbara grupper – lika mycket affärsbeslut som statistik.

### 5. Tolka figur 6.13 (s. 251).

Ett silhouette-diagram: varje kluster är en "kniv" där bredden = antal punkter och längden = hur väl de sitter. Streckade linjen = medel-silhouetten. Knivar mestadels under strecket = dåligt separerade; knivar förbi strecket = bra.

## Koduppgifter

Körbar kod i **`vecka6.ipynb`**.

### 6. Avskrift av kapitlets kodexempel.

Görs i notebooken.

### 7. K-means på housing.csv.

**a)** Koden väljer tre kolumner (`median_income`, `latitude`, `longitude`), kör K-means med 6 kluster och färgar en karta efter kluster – geografiska inkomstgrupper.
**b)** I en marknadsundersökning skulle man vilja veta klustrens preferenser: kontaktväg, hustyp, budget, demografi – med enkäter och köpdata per segment.
**c)** Användbart för riktad marknadsföring: anpassa budskap och erbjudanden per segment i stället för till alla lika.
**d)** Välj antal kluster med inertia (elbow) och silhouette. K-means är känslig för skala → standardisera först.

### 8. Klusteranalys på hr_employee_data.xlsx.

Alla variabler betraktas som `x` (inget `y`). Kluster 2 sticker ut: högst arbetsbelastning (~237 tim/mån) och betyg, och **35 % har slutat** mot 6 % i kluster 0. Klustringen hittar "överbelastade högpresterare" som riskgrupp – helt utan `left`. Användbart för HR att rikta insatser mot ett segment.

### 9. Klustring på 80-tals Spotify-datasetet.

Samma recept: välj numeriska ljud-features (tempo, energy, danceability …), `StandardScaler` → `KMeans`, välj *k* med silhouette. Ej körd här – datasetet finns inte lokalt.
