import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, roc_curve
from sklearn.model_selection import GridSearchCV

try:
    df = pd.read_csv('data/nbadata.csv')
except FileNotFoundError:
    print("Error: File 'nbadata.csv' not found. Check the path.")
    exit()


df_cleaned = df.dropna(how='any').copy()
df_cleaned = df_cleaned.drop(columns=['MIN'])

df_cleaned['IS_HOME'] = (df_cleaned['TEAM_NAME'] == df_cleaned['HOME_TEAM']).astype(int)

df_cleaned['WINS'] = (df_cleaned['PLUS_MINUS'] > 0).astype(int)

df_cleaned = df_cleaned.sort_values(by=['SEASON', 'GAME_ID']).reset_index(drop=True)


STATS_TO_TRACK = ['WINS', 'PTS', 'REB', 'AST', 'STL', 'BLK', 'FGM', 'FGA']

for stat in STATS_TO_TRACK:

    df_cleaned[f'CUM_{stat}_SUM'] = df_cleaned.groupby(['SEASON', 'TEAM_ID'])[stat].cumsum()


df_cleaned['GAMES_PLAYED_CUM'] = df_cleaned.groupby(['SEASON', 'TEAM_ID']).cumcount() + 1


for stat in STATS_TO_TRACK:
    numerator = df_cleaned[f'CUM_{stat}_SUM'] - df_cleaned[stat]
    denominator = df_cleaned['GAMES_PLAYED_CUM'] - 1
    df_cleaned[f'PREV_AVG_{stat}'] = numerator / denominator
    df_cleaned.loc[df_cleaned['GAMES_PLAYED_CUM'] == 1, f'PREV_AVG_{stat}'] = 0

df_cleaned['PREV_AVG_FG_PCT'] = df_cleaned['PREV_AVG_FGM'] / df_cleaned['PREV_AVG_FGA']
df_cleaned.loc[df_cleaned['PREV_AVG_FGA'] == 0, 'PREV_AVG_FG_PCT'] = 0


df = df_cleaned.copy()


df_home = df[df['IS_HOME'] == 1].copy()
df_away = df[df['IS_HOME'] == 0].copy()

AVG_COLUMNS = [col for col in df.columns if col.startswith('PREV_AVG_')]
COLS_TO_MERGE_HOME = ['GAME_ID', 'SEASON', 'WINS'] + AVG_COLUMNS
COLS_TO_MERGE_AWAY = ['GAME_ID', 'SEASON'] + AVG_COLUMNS


df_home = df_home[COLS_TO_MERGE_HOME].rename(columns={'WINS': 'HOME_WINS'})
home_cols_mapping = {col: f'{col}_HOME' for col in AVG_COLUMNS}
df_home = df_home.rename(columns=home_cols_mapping)

away_cols_mapping = {col: f'{col}_AWAY' for col in AVG_COLUMNS}
df_away = df_away[COLS_TO_MERGE_AWAY].rename(columns=away_cols_mapping)

df_matchup = pd.merge(df_home, df_away, on=['GAME_ID', 'SEASON'], how='inner')


AVG_BASE_COLUMNS = [col.replace('_HOME', '') for col in df_home.columns if col.startswith('PREV_AVG_')]
DIFF_FEATURES = []

for base_col in AVG_BASE_COLUMNS:
    home_col = f'{base_col}_HOME'
    away_col = f'{base_col}_AWAY'
    diff_col = f'{base_col.replace("PREV_AVG_", "")}_DIFF'

    df_matchup[diff_col] = df_matchup[home_col] - df_matchup[away_col]
    DIFF_FEATURES.append(diff_col)


df_matchup['HOME_ADVANTAGE'] = 1
DIFF_FEATURES.append('HOME_ADVANTAGE')

columns_to_drop = [col for col in df_matchup.columns if col.startswith('PREV_AVG_')]
df_matchup = df_matchup.drop(columns=columns_to_drop)


df_matchup = df_matchup.sort_values(by=['SEASON', 'GAME_ID']).reset_index(drop=True)

X = df_matchup[DIFF_FEATURES]
y = df_matchup['HOME_WINS']


TEST_SIZE_FRACTION = 0.20
SPLIT_INDEX = int(len(df_matchup) * (1 - TEST_SIZE_FRACTION))

X_train = X.iloc[:SPLIT_INDEX]
X_test = X.iloc[SPLIT_INDEX:]
y_train = y.iloc[:SPLIT_INDEX]
y_test = y.iloc[SPLIT_INDEX:]


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

xgb_model = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=5, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train_scaled, y_train)

param_grid = {
    'n_estimators': [250],
    'max_depth': [3],
    'learning_rate': [0.05],
    'min_child_weight': [1]
}

grid_search = GridSearchCV(
    estimator=XGBClassifier(random_state=42, eval_metric='logloss'),
    param_grid=param_grid,
    scoring='roc_auc',
    cv=3,
    verbose=0,
    n_jobs=-1
)

grid_search.fit(X_train_scaled, y_train)
best_xgb_model = grid_search.best_estimator_


neg_count = len(y_train[y_train == 0])
pos_count = len(y_train[y_train == 1])
pos_weight = neg_count / pos_count


xgb_weighted_model = XGBClassifier(
    n_estimators=best_xgb_model.get_params()['n_estimators'],
    learning_rate=best_xgb_model.get_params()['learning_rate'],
    max_depth=best_xgb_model.get_params()['max_depth'],
    min_child_weight=best_xgb_model.get_params()['min_child_weight'],
    scale_pos_weight=pos_weight,
    random_state=42
)

xgb_weighted_model.fit(X_train_scaled, y_train)

y_pred_weighted = xgb_weighted_model.predict(X_test_scaled)
y_proba_weighted = xgb_weighted_model.predict_proba(X_test_scaled)[:, 1]

print("\n--- FINAL WEIGHTED MODEL RESULTS ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_weighted):.4f}")
print(f"AUC-ROC: {roc_auc_score(y_test, y_proba_weighted):.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred_weighted))

feature_importances = pd.Series(xgb_weighted_model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances.values, y=feature_importances.index)
plt.title('Feature Importance (XGBoost)')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

fpr, tpr, thresholds = roc_curve(y_test, y_proba_weighted)
plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc_score(y_test, y_proba_weighted):.4f}')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve (XGBoost)')
plt.legend(loc='lower right')
plt.show()