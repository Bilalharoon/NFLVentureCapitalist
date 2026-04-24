import nflreadpy as nfl
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# 1. Load and immediately convert to Pandas
print('loading data...')
draft = nfl.load_draft_picks().to_pandas()
players = nfl.load_players().to_pandas()
contracts = nfl.load_contracts().to_pandas()
combine = nfl.load_combine().to_pandas()

# 2. Fix the DataType Mismatch (Pandas style)
# We make sure otc_id is a string in both places for a clean merge
players['otc_id'] = players['otc_id'].astype(str)
contracts['otc_id'] = contracts['otc_id'].astype(str)

# 3. Join Draft to Players
# left_on/right_on handles the different column names (pfr_player_id vs pfr_id)
print('merging data')
master_df = pd.merge(
    draft, 
    players, 
    left_on="pfr_player_id", 
    right_on="pfr_id", 
    how="inner"
)
# contracts['year_signed'] = contracts['year_signed'].fillna(0)
# players['draft_year'] = players['draft_year'].fillna(0)

def total_cap_pct(cols, draft_year):
    if cols is None:
        return
    total = 0
    for d in cols:
        if not isinstance(d, dict):
            continue
            
        year = d.get('year')
        cap_pct = d.get('cap_percent')
        
        # skip bad rows
        if year in [None, 'Total'] or cap_pct is None:
            continue
        
        year = int(year)
        
        # rookie window (first 4 years)
        if draft_year <= year < draft_year + 4:
            total += cap_pct
            
    return total

# 4. Join the result to Contracts
final_df = pd.merge(
    master_df, 
    contracts, 
    on="otc_id", 
    how="inner"
)


# rookie_roi_df = final_df[
#     (final_df['year_signed'] == final_df['draft_year_x']) 
# ].copy()

rookie_roi_df = final_df[[
    'pfr_id',
    'pick',
    'age',
    'season',
    'round',
    'team_x',
    'pfr_player_name',
    'position',
    'round',
    'w_av',
    'dr_av',
    'years',
    'draft_year_x',
    'cfb_player_id',
    'apy_cap_pct',
    'weight_x',
    'height_x',
]]
# 3. Keep only the athletic traits (and IDs for merging)
safe_combine_cols = [
    'pfr_id', 'forty', 'vertical', 'broad_jump', 'cone', 'shuttle', 'bench'
]
clean_combine = combine[safe_combine_cols].drop_duplicates(subset=['pfr_id'])

rookie_roi_df = pd.merge(rookie_roi_df, clean_combine, on='pfr_id', how='left')

print('engineering features...')
rookie_roi_df['total_cap_pct'] = final_df.apply(
    lambda row: total_cap_pct(row['cols'], row['draft_year_x']),
    axis=1
)

rookie_roi_df = rookie_roi_df.drop_duplicates(subset=['pfr_id'])
# ROI Target Variable

clean_df = rookie_roi_df[['pick', 'dr_av']].dropna()
coeffs = np.polyfit(clean_df['pick'], np.log1p(clean_df['dr_av'].clip(lower=0.02)), 3)
print(coeffs)
rookie_roi_df['expected_av'] = np.expm1(np.polyval(coeffs, rookie_roi_df['pick']))
rookie_roi_df['roi_ratio'] = (rookie_roi_df['dr_av'] - rookie_roi_df['expected_av']) / (rookie_roi_df['total_cap_pct'].clip(lower=0.03) * 100)
rookie_roi_df['roi_ratio'] = rookie_roi_df['roi_ratio'].replace([np.inf, -np.inf], np.nan)
print(len(rookie_roi_df))
rookie_roi_df = rookie_roi_df.dropna(subset=['roi_ratio'])
print(len(rookie_roi_df))
final_modeling_df = rookie_roi_df[[
    'season', 'round', 'pick', 'team_x', 'pfr_player_name', 
    'position', 'w_av', 'total_cap_pct', 'roi_ratio'
]].dropna()

rookie_roi_df = rookie_roi_df.rename(columns={
    'height_x': 'height',
    'weight_x': 'weight',
    'position_x': 'position'
})


rookie_roi_df.to_csv('./outputs/rookie_roi.csv')

print(rookie_roi_df.head(20))

print(f"Successfully created a Pandas DataFrame with {len(rookie_roi_df)} rows.")


# Filter out extreme outliers if necessary
plot_df = final_modeling_df[final_modeling_df['roi_ratio'] < 20000]

plt.figure(figsize=(12, 6))
sns.regplot(data=plot_df, x='pick', y='roi_ratio', lowess=True, 
            line_kws={'color': 'red'}, scatter_kws={'alpha': 0.3})

plt.title('The ROI Curve: Where is the Profit in the NFL Draft?')
plt.xlabel('Draft Pick Number')
plt.ylabel('ROI (Performance / total of Cap)')
plt.grid(True, alpha=0.3)
plt.savefig('./outputs/ROIvsPick.png')
plt.show()