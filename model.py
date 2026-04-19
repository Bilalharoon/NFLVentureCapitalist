import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np

rookie_roi_df = pd.read_csv('./merged.csv')
# data cleainng
rookie_roi_df['draft_grade'] = rookie_roi_df['draft_grade'].fillna(rookie_roi_df['draft_grade'].median())

rookie_roi_df['BMI'] = (rookie_roi_df['weight'] * 703) / (rookie_roi_df['height']**2)
ppa_cols = ['career_avgPPA', 'career_totalPPA', 'countablePlays', 'draft_grade', 'BMI']
for col in ppa_cols:
    rookie_roi_df[col] = rookie_roi_df.groupby('position_x')[col].transform(
        lambda x: x.fillna(x.median())
    )


# feaature engineering
# Create a 'Power Score' for RBs and Linemen
rookie_roi_df['BMI_PPA_Interaction'] = rookie_roi_df['BMI'] * rookie_roi_df['career_totalPPA']
# Create a 'Physical Age' factor
rookie_roi_df['BMI_Age_Ratio'] = rookie_roi_df['BMI'] / rookie_roi_df['age']


for col in ['BMI', 'career_avgPPA', 'draft_grade']:
    rookie_roi_df[f'{col}_rel'] = rookie_roi_df.groupby('position_x')[col].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
    )
# Fill the NaNs created by players who are the only ones at their position
rookie_roi_df = rookie_roi_df.fillna(0)

# 1. Select your 'Predictors' (X) and your 'Target' (y)
# features = ['pick', 'round', 'age', 'BMI', 'allpro', 'probowls']
# features = rookie_roi_df.select_dtypes(include='number')

# position_dummies = pd.get_dummies(rookie_roi_df['position_x'], prefix='pos')

features = [
    'pick_x', 'age', 'draft_grade', 
    'BMI_rel', 'career_avgPPA_rel', 'draft_grade_rel',
    'countablePlays'
]
# print(features.isna().sum())

# data cleaining

X =  rookie_roi_df[features] # Simple cleaning
X[features].fillna(0)
y = rookie_roi_df['roi_ratio']
# 2. Split into Training and Testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_train_log = np.log1p(y_train)

# 3. Initialize and Train
model = RandomForestRegressor(
    n_estimators=100,      # More trees = more stability
    max_depth=6,
    max_features='sqrt',    # Forces trees to look at different features, reducing bias
    min_samples_leaf=5,
    random_state=42
)
model.fit(X_train, y_train_log)

# 4. GET THE TRUTH: Which features matter?
importances = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)


# 3. Sort by ROI
print(importances)

# Get predictions
predictions = np.expm1(model.predict(X_test))

# Calculate metrics
print(f"MAE: {mean_absolute_error(y_test, predictions)}")
print(f"R2 Score: {r2_score(y_test, predictions)}")

top_10_actual = y_test.quantile(0.9)
top_10_pred = pd.Series(predictions).quantile(0.9)

hits = ((y_test > top_10_actual) & (pd.Series(predictions, index=y_test.index) > top_10_pred)).sum()
print(f"The model correctly identified {hits} high-value players out of {len(y_test[y_test > top_10_actual])}.")


# 1. THE HEURISTIC MODEL (Baseline)
X_baseline = X_train[['pick_x']]
baseline_model = LinearRegression().fit(X_baseline, y_train)
baseline_preds = baseline_model.predict(X_test[['pick_x']])

# 2. YOUR 'VC' MODEL (BMI + Age + Pick)
# It knows the 'pick' AND the physical traits
vc_preds = np.expm1(model.predict(X_test)) # Your existing model predictions

# 3. COMPARE THE HIT RATES
def calculate_hits(preds, actuals):
    top_10_actual = actuals.quantile(0.9)
    top_10_pred = pd.Series(preds).quantile(0.9)
    return ((actuals > top_10_actual) & (pd.Series(preds, index=actuals.index) > top_10_pred)).sum()

baseline_hits = calculate_hits(baseline_preds, y_test)
vc_hits = calculate_hits(vc_preds, y_test)

print(f"Heuristic Linear Regresion Hits: {baseline_hits}")
r2 = r2_score(y_test, baseline_preds)
print(f'lr r2 score:{r2}')
print(f"VC Model (BMI/Age/Pick) Hits: {vc_hits}")


results_df = pd.DataFrame({
    'Player': rookie_roi_df.loc[X_test.index, 'pfr_player_name'],
    'Actual_ROI': y_test,
    'Predicted_ROI': predictions,
    'Pick': X_test['pick_x'],
})

# 3. Sort by Predicted ROI to find the "Model Favorites"
top_10_picks = results_df.sort_values(by='Predicted_ROI', ascending=False).head(10)

print("--- TOP 10 VENTURE CAPITALIST PICKS (TEST DATA) ---")
print(top_10_picks[['Player', 'Pick', 'Predicted_ROI', 'Actual_ROI']])