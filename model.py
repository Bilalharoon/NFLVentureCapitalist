import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np

rookie_roi_df = pd.read_csv('./outputs/rookie_roi.csv')
# data cleainng
# rookie_roi_df['draft_grade'] = rookie_roi_df['draft_grade'].fillna(rookie_roi_df['draft_grade'].median())
rookie_roi_df = rookie_roi_df.drop_duplicates(subset=['pfr_player_name', 'pick'])

rookie_roi_df['BMI'] = (rookie_roi_df['weight'] * 703) / (rookie_roi_df['height']**2)

# feaature engineering
rookie_roi_df['BMI_Age_Ratio'] = rookie_roi_df['BMI'] / rookie_roi_df['age']

features = [
    'pick',
    'age',
    'weight',
    'height',
    'BMI_Age_Ratio'
]
for col in features + ['BMI']:
    rookie_roi_df[f'{col}_rel'] = rookie_roi_df.groupby('position')[col].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
    )

features += [f'{col}_rel' for col in features]
features.append('BMI_rel')
# Fill the NaNs created by players who are the only ones at their position
rookie_roi_df = rookie_roi_df.fillna(0)

# 1. Select your 'Predictors' (X) and your 'Target' (y)
# features = ['pick', 'round', 'age', 'BMI', 'allpro', 'probowls']
# features = rookie_roi_df.select_dtypes(include='number')

# position_dummies = pd.get_dummies(rookie_roi_df['position_x'], prefix='pos')


# data cleaining

X =  rookie_roi_df[features] # Simple cleaning
X[features].fillna(0)
y = rookie_roi_df['roi_ratio']
# 2. Split into Training and Testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Initialize and Train
model = RandomForestRegressor(
    n_estimators=100,      # More trees = more stability
    max_depth=6,
    max_features='sqrt',    # Forces trees to look at different features, reducing bias
    min_samples_leaf=5,
    random_state=42
)
model.fit(X_train, y_train)

# 4. GET THE TRUTH: Which features matter?
importances = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)


# 3. Sort by ROI
print(importances)

# Get predictions
predictions = model.predict(X_test)

# Calculate metrics
print(f"MAE: {mean_absolute_error(y_test, predictions)}")
print(f"R2 Score: {r2_score(y_test, predictions)}")

top_25_actual = y_test.quantile(0.75)
top_25_pred = pd.Series(predictions).quantile(0.75)

hits = ((y_test > top_25_actual) & (pd.Series(predictions, index=y_test.index) > top_25_pred)).sum()
print(f"The model correctly identified {hits} high-value players out of {len(y_test[y_test > top_25_actual])}.")


# 1. THE HEURISTIC MODEL (Baseline)
X_baseline = X_train[['pick']]
baseline_model = LinearRegression().fit(X_baseline, y_train)
baseline_preds = baseline_model.predict(X_test[['pick']])

# 2. YOUR 'VC' MODEL (BMI + Age + Pick)
# It knows the 'pick' AND the physical traits
vc_preds = model.predict(X_test) # Your existing model predictions

# 3. COMPARE THE HIT RATES
def calculate_hits(preds, actuals):
    top_25_actual = actuals.quantile(0.75)
    top_25_pred = pd.Series(preds).quantile(0.75)
    return ((actuals > top_25_actual) & (pd.Series(preds, index=actuals.index) > top_25_pred)).sum()

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
    'Pick': X_test['pick'],
    'Position': rookie_roi_df.loc[X_test.index, 'position'],
    'BMI_rel': X_test['BMI_rel'],
    'Age': X_test['age'],
    'Gem_detected':predictions <= y_test
})

# 3. Sort by Predicted ROI to find the "Model Favorites"
top_10_picks = results_df.sort_values(by='Predicted_ROI', ascending=False).head(10)

print("--- TOP 10 VENTURE CAPITALIST PICKS (TEST DATA) ---")
print(top_10_picks[['Player', 'Pick', 'Position', 'BMI_rel', 'Age', 'Predicted_ROI', 'Actual_ROI', 'Gem_detected']])