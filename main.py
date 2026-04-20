import nflreadpy as nfl
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. Load and immediately convert to Pandas
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
master_df = pd.merge(
    draft, 
    players, 
    left_on="pfr_player_id", 
    right_on="pfr_id", 
    how="inner"
)

# 4. Join the result to Contracts
final_df = pd.merge(
    master_df, 
    contracts, 
    on="otc_id", 
    how="inner"
)

rookie_roi_df = final_df[
    (final_df['year_signed'] == final_df['season']) & 
    (final_df['years'] == 4)
].copy()

rookie_roi_df = rookie_roi_df[[
    'pfr_id',
    'pick',
    'age',
    'season',
    'round',
    'team_x',
    'pfr_player_name',
    'position',
    'w_av',
    'dr_av',
    'cfb_id',
    'apy_cap_pct',
    'weight_x',
    'height_x',
]]
# 3. Keep only the athletic traits (and IDs for merging)
safe_combine_cols = [
    'pfr_id', 'forty', 'vertical', 'broad_jump', 'cone', 'shuttle', 'bench'
]
clean_combine = combine[safe_combine_cols].drop_duplicates(subset=['pfr_id'])

# 4. Merge them together using the Pro Football Reference ID
rookie_roi_df = pd.merge(rookie_roi_df, clean_combine, on='pfr_id', how='left')
print(len(rookie_roi_df))

# 6. Create your 'Profitability' Target Variable
rookie_roi_df['roi_ratio'] = rookie_roi_df['dr_av'] / rookie_roi_df['apy_cap_pct']
rookie_roi_df['roi_ratio'] = rookie_roi_df['roi_ratio'].replace([np.inf, -np.inf], np.nan)
rookie_roi_df = rookie_roi_df.dropna(subset=['roi_ratio'])

final_modeling_df = rookie_roi_df[[
    'season', 'round', 'pick', 'team_x', 'pfr_player_name', 
    'position', 'w_av', 'apy_cap_pct', 'roi_ratio'
]].dropna()

rookie_roi_df = rookie_roi_df.rename(columns={
    'height_x': 'height',
    'weight_x': 'weight',
    'position_x': 'position'
})


rookie_roi_df.to_csv('outputs/rookie_roi.csv')

print(rookie_roi_df.head(20))

print(f"Successfully created a Pandas DataFrame with {len(rookie_roi_df)} rows.")


# Filter out extreme outliers if necessary
plot_df = final_modeling_df[final_modeling_df['roi_ratio'] < 20000]

plt.figure(figsize=(12, 6))
sns.regplot(data=plot_df, x='pick', y='roi_ratio', lowess=True, 
            line_kws={'color': 'red'}, scatter_kws={'alpha': 0.3})

plt.title('The ROI Curve: Where is the Profit in the NFL Draft?')
plt.xlabel('Draft Pick Number')
plt.ylabel('ROI (Performance / % of Cap)')
plt.grid(True, alpha=0.3)
plt.savefig('./outputs/ROIvsPick.png')
plt.show()