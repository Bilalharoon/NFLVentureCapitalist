import pandas as pd
import numpy as np
from pathlib import Path
from functools import reduce

# Configuration
DATA_DIR = Path('./data/')
FILES = {
    'kicking': DATA_DIR / 'kicking.csv',
    'passing': DATA_DIR / 'passing.csv',
    'punting': DATA_DIR / 'punting.csv',
    'receiving': DATA_DIR / 'receiving.csv',
    'rushing': DATA_DIR / 'rushing.csv',
    'scoring': DATA_DIR / 'scoring.csv'
}

rookie_roi_df = pd.read_csv(Path('./outputs/') / 'rookie_roi.csv')
# 1. Load DataFrames into a dictionary or list
dfs = {name: pd.read_csv(path) for name, path in FILES.items()}

# 2. Define columns to drop
to_drop = ['Season', 'ranker', 'name_display', 'team_name_abbr', 'conf_abbr', 'awards']

def transform_df(df, name):
    # Drop columns that exist in the dataframe to avoid errors
    cols_to_remove = [c for c in to_drop if c in df.columns]
    transformed = df.drop(columns=cols_to_remove).groupby(['player_id']).sum()
    transformed = transformed.add_prefix(f'{name}_')
    
    # Transform: Drop -> GroupBy -> Sum
    # This returns a new DataFrame where 'name_display' is the index
    return transformed


# 3. Apply transformation to all dataframes
transformed_stats = [transform_df(df, name) for name, df in dfs.items()]

# 4. Merge DataFrames
# Since we grouped by 'name_display', it is now the index. 
# We merge on the index.
merged_df = reduce(lambda left, right: pd.merge(left, right, on='player_id', how='outer'), transformed_stats)

game_cols = [col for col in merged_df.columns if '_games' in col]

merged_df['career_games_total'] = merged_df[game_cols].max(axis=1)
merged_df = merged_df.drop(columns=game_cols)

merged_df = merged_df.fillna(0).astype(int)
print(merged_df.reset_index().head())
print(merged_df.shape)

rookie_merge = pd.merge(left=rookie_roi_df, right=merged_df, left_on='cfb_player_id', right_on='player_id', how='left', indicator=True)

print(rookie_merge.head())
print(rookie_merge.shape)

print(rookie_merge['_merge'].value_counts())

rookie_merge.to_csv(Path('./outputs/') / 'rookie_roi_cfb_merge.csv')
print(f'final df length: {len(rookie_merge)}')
