import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
import numpy as np

rookie_roi_df = pd.read_csv('./outputs/rookie_roi_cfb_merge.csv')
# data cleainng
# rookie_roi_df['draft_grade'] = rookie_roi_df['draft_grade'].fillna(rookie_roi_df['draft_grade'].median())
rookie_roi_df = rookie_roi_df.drop_duplicates(subset=['pfr_player_name', 'pick'])

# position_dummies = pd.get_dummies(rookie_roi_df['position'])
rookie_roi_df['BMI'] = (rookie_roi_df['weight'] * 703) / (rookie_roi_df['height']**2)

# feaature engineering
rookie_roi_df['BMI_Age_Ratio'] = rookie_roi_df['BMI'] / rookie_roi_df['age']

# 1. Size-Adjusted Speed (Speed Score)
rookie_roi_df['speed_score'] = (rookie_roi_df['weight'] * 200) / (rookie_roi_df['forty']**4)

# 2. Explosive Burst (Size-Adjusted)
rookie_roi_df['burst_score'] = (rookie_roi_df['vertical'] + rookie_roi_df['broad_jump']) * rookie_roi_df['weight']

# 3. Density-Adjusted Power (Your Vertical/BMI interaction)
rookie_roi_df['power_density'] = rookie_roi_df['vertical'] / rookie_roi_df['BMI']

# 4. Agility Premium
rookie_roi_df['agility_score'] = rookie_roi_df['weight'] / rookie_roi_df['cone']
features = [
    'pick',
    'BMI_Age_Ratio',
    'forty',
    'vertical',
    'broad_jump',
    'cone',
    'shuttle',
    'bench',
    'speed_score',
    'burst_score',
    'power_density',
    'agility_score',
    'BMI',
    'kicking_xpm',
    'kicking_xpa', 'kicking_xp_pct', 'kicking_fgm', 'kicking_fga',
    'passing_pass_cmp', 'passing_pass_att', 'passing_pass_cmp_pct', 'passing_pass_yds',
    'passing_pass_td', 'passing_pass_td_pct', 'passing_pass_int',
    'passing_pass_int_pct', 'passing_pass_yds_per_att',
    'passing_pass_adj_yds_per_att', 'passing_pass_yds_per_cmp',
    'passing_pass_yds_per_g', 'passing_pass_rating', 'punting_punt',
    'punting_punt_yds', 'punting_punt_yds_per_punt', 'receiving_rec',
    'receiving_rec_yds', 'receiving_rec_yds_per_rec', 'receiving_rec_td',
    'receiving_rec_yds_per_g', 'receiving_rush_att', 'receiving_rush_yds',
    'receiving_rush_yds_per_att', 'receiving_rush_td',
    'receiving_rush_yds_per_g', 'receiving_scrim_att',
    'receiving_yds_from_scrimmage', 'receiving_scrim_yds_per_att',
    'receiving_scrim_td', 'rushing_rush_att', 'rushing_rush_yds',
    'rushing_rush_yds_per_att', 'rushing_rush_td', 'rushing_rush_yds_per_g',
    'rushing_rec', 'rushing_rec_yds', 'rushing_rec_yds_per_rec',
    'rushing_rec_td', 'rushing_rec_yds_per_g', 'rushing_scrim_att',
    'rushing_yds_from_scrimmage', 'rushing_scrim_yds_per_att',
    'rushing_scrim_td', 'scoring_rush_td', 'scoring_rec_td',
    'scoring_punt_ret_td', 'scoring_kick_ret_td', 'scoring_fumbles_rec_td',
    'scoring_def_int_td', 'scoring_all_td', 'scoring_xpm',
    'scoring_fga', 'scoring_two_pt_md', 'scoring_safety_md',
    'scoring_total_points', 'scoring_points_per_game', 'scoring_other_td',
    'career_games_total']



# Fill the NaNs created by players who are the only ones at their position
rookie_roi_df = rookie_roi_df.fillna(0)



final_features = ['pick'] 

for col in features:
    rookie_roi_df[f'{col}_rel'] = rookie_roi_df.groupby('position')[col].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
    )

for col in features:
    rookie_roi_df[f'{col}_imputed'] = rookie_roi_df[col].isna()
    
    means = rookie_roi_df.groupby('position')[col].transform('mean')
    rookie_roi_df[col] = rookie_roi_df[col].fillna(means)

final_features.extend([f'{col}_rel' for col in features])
final_features.extend([f'{col}_imputed' for col in features])
position_dummies = pd.get_dummies(rookie_roi_df['position'])

# Calculate how good the player was compared to OTHER players at their position
rookie_roi_df['roi_position_zscore'] = rookie_roi_df.groupby('position')['roi_ratio'].transform(
    lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
)




X =  pd.concat([rookie_roi_df[final_features], position_dummies], axis=1) # Simple cleaning
print(X.shape)
final_features += list(position_dummies.columns)
X[final_features].fillna(0)

# Set this new relative metric as your target (y)
y = rookie_roi_df['roi_position_zscore']
# 2. Split into Training and Testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Initialize and Train
# model = RandomForestRegressor(
#     n_estimators=100,      # More trees = more stability
#     max_depth=6,
#     max_features='sqrt',    # Forces trees to look at different features, reducing bias
#     min_samples_leaf=5,
#     random_state=42
# )

model = GradientBoostingRegressor(
    loss='quantile',   # This activates the asymmetric loss!
    alpha=0.67,        # 85th percentile (focuses on the ceiling)
    learning_rate=0.1,
    n_estimators=500,
    max_depth=4,
    min_samples_leaf=12,
    random_state=42
)
# model = xgb.XGBRegressor(
#     objective='reg:quantileerror', # Continues your Quantile Loss strategy
#     quantile_alpha=0.67,           # Your target ceiling
#     reg_alpha=10,                # THE FILTER: Higher = more aggressive deletion
#     reg_lambda=0.01,                # L2 penalty (smooths out the remaining features)
#     n_estimators=300,
#     learning_rate=0.1,
#     max_depth=4,
# )
model.fit(X_train, y_train)


# 4. GET THE TRUTH: Which features matter?
importances = pd.DataFrame({
    'feature': final_features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

# 3. Sort by ROI
print(importances)
importances.to_csv('./outputs/feature_importances.csv')

# Get predictions
predictions = model.predict(X_test)

# Calculate metrics
top_25_actual = y_test.quantile(0.75)
top_25_pred = pd.Series(predictions).quantile(0.75)

top_ten_actual = y_test.quantile(0.9)
top_ten_pred = pd.Series(predictions).quantile(0.9)

hits25 = ((y_test > top_25_actual) & (pd.Series(predictions, index=y_test.index) > top_25_pred)).sum()
hits10 = ((y_test > top_ten_actual) & (pd.Series(predictions, index=y_test.index) > top_ten_pred)).sum()

print(f"The model correctly identified {hits25} high-value players out of {len(y_test[y_test > top_25_actual])} (Top 25%).")
print(f"The model correctly identified {hits10} high-value players out of {len(y_test[y_test > top_ten_actual])} (Top 10%).")


# 1. THE HEURISTIC MODEL (Baseline)
X_baseline = X_train[['pick']]
baseline_model = LinearRegression().fit(X_baseline, y_train)
baseline_preds = baseline_model.predict(X_test[['pick']])

# 2. YOUR 'VC' MODEL (BMI + Age + Pick)
# It knows the 'pick' AND the physical traits
vc_preds = model.predict(X_test) # Your existing model predictions

# 3. COMPARE THE HIT RATES
def calculate_hits(preds, actuals, quants):
    top_percentile_actual = actuals.quantile(quants)
    top_percentile_pred = pd.Series(preds).quantile(quants)
    return ((actuals > top_percentile_actual) & (pd.Series(preds, index=actuals.index) > top_percentile_pred)).sum()

baseline_25_hits = calculate_hits(baseline_preds, y_test, 0.75)
vc_25_hits = calculate_hits(vc_preds, y_test, 0.75)

print('== TOP 25% HITS ==')
print(f"Heuristic Linear Regresion Hits: {baseline_25_hits}")
print(f"VC Model (BMI/Age/Pick) Hits: {vc_25_hits}")

baseline_10_hits = calculate_hits(baseline_preds, y_test, 0.9)
vc_10_hits = calculate_hits(vc_preds, y_test, 0.9)
print('== TOP 10% HITS ==')
print(f"Heuristic Linear Regresion Hits: {baseline_10_hits}")
print(f"VC Model (BMI/Age/Pick) Hits: {vc_10_hits}")



results_df = pd.DataFrame({
    'Player': rookie_roi_df.loc[X_test.index, 'pfr_player_name'],
    'Actual_ROI': y_test,
    'Predicted_ROI': predictions,
    'Pick': X_test['pick'],
    'Position': rookie_roi_df.loc[X_test.index, 'position'],
    'BMI_rel': X_test['BMI_rel'],
    # 'Age': X_test['age'],
    'Gem_detected':predictions <= y_test
})

# 3. Sort by Predicted ROI to find the "Model Favorites"
top_10_picks = results_df.sort_values(by='Predicted_ROI', ascending=False).head(10)

print("--- TOP 10 VENTURE CAPITALIST PICKS (TEST DATA) ---")
print(top_10_picks[['Player', 'Pick', 'Position', 'BMI_rel', 'Predicted_ROI', 'Actual_ROI', 'Gem_detected']])
results_df.to_csv('outputs/results.csv')