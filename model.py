import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from xgboost import XGBClassifier
import numpy as np

rookie_roi_df = pd.read_csv('./outputs/rookie_roi_cfb_merge.csv')
# rookie_roi_df = rookie_roi_df[rookie_roi_df['draft_year_x'] >= 2000]
# data cleainng
# rookie_roi_df['draft_grade'] = rookie_roi_df['draft_grade'].fillna(rookie_roi_df['draft_grade'].median())
# rookie_roi_df = rookie_roi_df[rookie_roi_df['pick'] > 50]
rookie_roi_df = rookie_roi_df.drop_duplicates(subset=['pfr_id'])

QUANTILE_THRESHOLD = 0.9
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

# rookie_roi_df['production_x_pick'] = rookie_roi_df['yards_per_game'] / rookie_roi_df['pick']

rookie_roi_df['surplus_av'] = rookie_roi_df['expected_av'] - rookie_roi_df['dr_av']
threshold = rookie_roi_df['surplus_av'].quantile(QUANTILE_THRESHOLD)
rookie_roi_df['vc_hit'] = (rookie_roi_df['surplus_av'] >= threshold).astype(int)

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
    'career_games_total'
    ]



# Fill the NaNs created by players who are the only ones at their position
rookie_roi_df = rookie_roi_df.fillna(0)

# final_features = ['pick'] 
final_features = []

for col in features:
    rookie_roi_df[f'{col}_rel'] = rookie_roi_df.groupby('position')[col].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
    )

    rookie_roi_df[f'{col}_resid'] = rookie_roi_df.groupby('pick')[col].transform(
        lambda x: rookie_roi_df[col] - rookie_roi_df.groupby('pick')[col].transform('mean')
    )
    rookie_roi_df[f'{col}_imputed'] = rookie_roi_df[col].isna()
    
    means = rookie_roi_df.groupby('position')[col].transform('mean')
    rookie_roi_df[col] = rookie_roi_df[col].fillna(means)

final_features.extend([f'{col}_rel' for col in features])
final_features.extend([f'{col}_imputed' for col in features])
final_features.extend([f'{col}_resid' for col in features])
position_dummies = pd.get_dummies(rookie_roi_df['position'], prefix='pos')

rookie_roi_df['speed_x_wr'] = rookie_roi_df['speed_score_resid'] * (rookie_roi_df['position'] == 'WR')
rookie_roi_df['speed_x_cb'] = rookie_roi_df['speed_score_resid'] * (rookie_roi_df['position'] == 'CB')
final_features += ['speed_x_wr', 'speed_x_cb']
# Calculate how good the player was compared to OTHER players at their position
# rookie_roi_df['roi_ratio'] = rookie_roi_df['roi_ratio'].clip(-20, 20)
rookie_roi_df['roi_position_zscore'] = rookie_roi_df.groupby('position')['roi_ratio'].transform(
    lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
)

pos_stats = rookie_roi_df.groupby('position')['roi_ratio'].agg(['mean', 'std']).reset_index()
pos_stats.columns = ['Position', 'pos_mean_roi', 'pos_std_roi']

X =   pd.concat([rookie_roi_df[final_features], position_dummies], axis=1)
# X = rookie_roi_df[final_features]
print(X.shape)
final_features += list(position_dummies.columns)
X[final_features].fillna(0)

y = rookie_roi_df['vc_hit']
# 2. Split into Training and Testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
pick_cutoff = 0
year_cutoff = 2024
X_train = X[rookie_roi_df['pick'] > pick_cutoff][rookie_roi_df['draft_year_x'] < year_cutoff]
y_train = y[rookie_roi_df['pick'] > pick_cutoff][rookie_roi_df['draft_year_x'] < year_cutoff]
# X_train = X[rookie_roi_df['draft_year_x'] < 2018]
# y_train = y[rookie_roi_df['draft_year_x'] < 2018]
X_test = X[rookie_roi_df['draft_year_x'] >= year_cutoff]
y_test = y[rookie_roi_df['draft_year_x'] >= year_cutoff]

print(f'train size: {len(y_train)}')
print(f'test size; {len(y_test)}')

# cat_features = [X.columns.get_loc(col) for col in X.columns if col.startswith('pos_')]
# model = CatBoostRegressor(
#     loss_function='Quantile:alpha=0.67',
#     iterations=300,
#     learning_rate=0.1,
#     depth=4,
#     l2_leaf_reg=0.1,        # L2 Regularization for stability
#     random_seed=42,
#     logging_level='Silent' # Keeps your console clean
# )
# 3. Initialize and Train
# model = RandomForestRegressor(
#     n_estimators=500,      # More trees = more stability
#     max_depth=6,
#     max_features='sqrt',    # Forces trees to look at different features, reducing bias
#     min_samples_leaf=5,
#     random_state=42
# )

# model = GradientBoostingRegressor(
#     loss='quantile',   # This activates the asymmetric loss!
#     alpha=0.67,        # 85th percentile (focuses on the ceiling)
#     n_estimators=500,
#     learning_rate=0.5,
#     # max_depth=4,
#     max_features='sqrt',
#     min_samples_leaf=4
# )
# model = xgb.XGBRegressor(
#     objective='reg:quantileerror', 
#     quantile_alpha=0.67,           
#     reg_alpha=15,                
#     reg_lambda=1,            
#     n_estimators=300,
#     max_depth=4,
# )

# model = RandomForestClassifier(
#     n_estimators=300,
#     # max_depth=6,
#     class_weight='balanced',
#     # max_features='sqrt',
#     random_state=42,
#     # max_depth=8,
#     # min_samples_leaf=5
# )

model = XGBClassifier(
    n_estimators=200
)

cal_model = CalibratedClassifierCV(model, method='isotonic', cv=3)
cal_model.fit(X_train, y_train)


model.fit(X_train, y_train)

# 4. GET THE TRUTH: Which features matter?
importances = pd.DataFrame({
    'feature': final_features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

# 3. Sort by ROI
print(importances)
importances.to_csv('./outputs/feature_importances.csv')

# # Get predictions
# predictions = model.predict(X_test)

# # Calculate metrics
# top_25_actual = y_test.quantile(0.75)
# top_25_pred = pd.Series(predictions).quantile(0.75)

# top_ten_actual = y_test.quantile(0.9)
# top_ten_pred = pd.Series(predictions).quantile(0.9)

# hits25 = ((y_test > top_25_actual) & (pd.Series(predictions, index=y_test.index) > top_25_pred)).sum()
# hits10 = ((y_test > top_ten_actual) & (pd.Series(predictions, index=y_test.index) > top_ten_pred)).sum()

# print(f"The model correctly identified {hits25} high-value players out of {len(y_test[y_test > top_25_actual])} (Top 25%).")
# print(f"The model correctly identified {hits10} high-value players out of {len(y_test[y_test > top_ten_actual])} (Top 10%).")



# # 3. COMPARE THE HIT RATES
def calculate_hits(preds, actuals, quants):
    top_percentile_actual = actuals.quantile(quants)
    top_percentile_pred = pd.Series(preds).quantile(quants)
    return ((actuals > top_percentile_actual) & (pd.Series(preds, index=actuals.index) > top_percentile_pred)).sum()

# baseline_25_hits = calculate_hits(baseline_preds, y_test, 0.75)
# vc_25_hits = calculate_hits(vc_preds, y_test, 0.75)

# preds = pd.Series(model.predict_proba(X_test)[:, 1])
# top_k = int(0.1 * len(preds))  # top 10%
# top_preds = preds.nlargest(top_k, keep='first')

# print(rookie_roi_df.loc[top_preds.index, 'vc_hit'])

# print(model.predict(X_test))

# print('== TOP 25% HITS ==')
# print(f"Heuristic Linear Regresion Hits: {baseline_25_hits}")
# print(f"VC Model (BMI/Age/Pick) Hits: {vc_25_hits}")

# vc_10_hits = calculate_hits(vc_preds, y_test, 0.9)
# print('== TOP 10% HITS ==')
# print(f"VC Model (BMI/Age/Pick) Hits: {vc_10_hits}")



results_df = pd.DataFrame({
    'Player': rookie_roi_df.loc[X_test.index, 'pfr_player_name'],
    'Actual_ROI': rookie_roi_df.loc[X_test.index, 'roi_ratio'],
    'Pick': rookie_roi_df.loc[X_test.index, 'pick'],
    'Position': rookie_roi_df.loc[X_test.index, 'position'],
    'Predicted_Probability' : cal_model.predict_proba(X_test)[:, 1],
    'Surplus_AV': rookie_roi_df.loc[X_test.index, 'surplus_av'],
    'Expected_AV': rookie_roi_df.loc[X_test.index, 'expected_av'],
    'Actual_hit': y_test,
})


# results_df = results_df.merge(pos_stats, on='Position', how='left')
# results_df['Predicted_ROI'] = (results_df['Predicted_ROI_Zscore'] * results_df['pos_std_roi']) + results_df['pos_mean_roi']
# results_df = results_df.drop(['pos_std_roi', 'pos_mean_roi'], axis=1)

# 3. Sort by Predicted ROI to find the "Model Favorites"
# top_10_picks = results_df.sort_values(by='Predicted_Probability', ascending=False).head(10)
k = int((1-QUANTILE_THRESHOLD) * len(results_df))  # top 10%
top_preds = results_df.nlargest(k, 'Predicted_Probability')

hit_rate = top_preds['Actual_hit'].mean()
hits = top_preds['Actual_hit'].sum()

model_hits = top_preds['Actual_hit'].sum()

print("--- TOP 10 VENTURE CAPITALIST PICKS (TEST DATA) ---")
print(top_preds[['Player', 'Position', 'Pick', 'Actual_hit', 'Actual_ROI', 'Expected_AV', 'Surplus_AV']])
results_df.to_csv('outputs/results.csv')

# top_k = results_df['Predicted_Probability'].quantile(QUANTILE_THRESHOLD)
# total_hits = len(results_df[results_df['Predicted_Probability'] > top_k])

# top_pct = results_df[results_df['Predicted_Probability'] > top_k]
# accuracy = (top_pct['Predicted_hit'] == top_pct['Actual_hit']).sum()

print(f'Predicted {model_hits} out of {k}. Model hit rate at {(model_hits/k):.2%}')


baseline_x_train = rookie_roi_df.loc[X_train.index][['pick']]
baseline_y_train = y_train

baseline_model = LinearRegression().fit(baseline_x_train, baseline_y_train)
baseline_input = results_df.rename(columns={'Pick':'pick'})[['pick']]
baseline_preds = baseline_model.predict(baseline_input)

results_df['baseline_pred'] = baseline_preds

baseline_top = results_df.nlargest(k, 'baseline_pred')

baseline_hits = baseline_top['Actual_hit'].sum()
baseline_hit_rate = baseline_hits / k
print(f'Heuristic predicted {baseline_hits} out of {k}. Hit rate at {(baseline_hit_rate):.2%}')
# print(results_df['Actual_hit'].mean())