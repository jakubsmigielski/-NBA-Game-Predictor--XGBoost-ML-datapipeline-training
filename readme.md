### 🏀 NBA Game Predictor 🏀 : Predicting Match Outcomes (2012-2024) 

This Data Science training project focuses on building a robust Machine Learning model (XGBoost Classifier) to predict the outcome of NBA games based on historical performance data from the **2012-2024** seasons. The core innovation lies in engineering time-series features to prevent data leakage and simulate real-world prediction.

###  Project Goal

The main objective was to perform a binary classification (Home Win: 1 / Away Win: 0) using only data known *before* the game started.

---

### 🛠️ Methodology and Feature Engineering

RAW data downloaded from Kaagle

The project utilized a robust feature engineering pipeline to handle the temporal nature of sports data.

#### 1. Feature Creation: Cumulative Statistics

To ensure the model only learns from past events and prevent **data leakage**, features were engineered as the **cumulative average** of a team's performance up to the game being predicted.

The **Average Points Before Game $k$** was calculated as:

$$
\text{AVG PTS}_{k} = \frac{\sum_{i=1}^{k-1} \text{PTS}_{i}}{k-1}
$$

This was applied to all key metrics (Points, Rebounds, Assists, etc.) and the **Win Percentage** (`WINS`).

#### 2. Predictive Features (DIFF Features)

The final predictive variables ($X$) were generated as the **difference** between the Home Team's cumulative average and the Away Team's cumulative average. This forces the model to learn the relative strength of the matchup.

$$
\text{WINS DIFF} = \text{AVG WINS}_{\text{Home}} - \text{AVG WINS}_{\text{Away}}
$$
  
#### 3. Model & Validation

* **Model:** **XGBoost Classifier** optimized via **Grid Search**.
* **Data Split:** **Chronological Split** (last 20% of games used as the Test Set).

---

##  Final Weighted Model Results

The model was optimized using Grid Search, resulting in the following performance metrics on the independent test set:

| Metric | Result | Interpretation |
| :--- | :--- | :--- |
| **Accuracy** | **0.6394** | The model correctly predicted the outcome of $\approx 63.9\%$ of games. |
| **AUC-ROC** | **0.6937** | The model has a strong ability to distinguish between winning and losing outcomes (close to the $0.70$ threshold). |

### Feature Importance & Key Findings

The **WINS\_DIFF** (difference in cumulative winning percentage) was the most decisive factor, confirming that **current team form and strength** are the best predictors.



---

##  Visualizations

The model's performance is further illustrated by the ROC curve:

![c](plots/plot1.png)

![b](plots/plot2.png)





