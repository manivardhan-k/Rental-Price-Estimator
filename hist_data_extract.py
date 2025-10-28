import pandas as pd

# Load and clean
df = pd.read_csv("State_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv")

# Drop irrelevant columns
df = df.drop(columns=["RegionID", "SizeRank", "RegionType", "StateName"])

# Remove DC
df = df[df['RegionName'] != 'District of Columbia']

# Keep only date columns after 2015-01-30 + RegionName
date_cols = pd.to_datetime(df.columns, errors='coerce')  # Non-dates become NaT
df = df.loc[:, (date_cols > pd.to_datetime('2015-01-30')) | (df.columns == 'RegionName')]

# --- Compute yearly averages ---
df_dates = df[['RegionName']]
df_values = df.drop(columns=['RegionName'])

# Convert column headers to datetime
df_values.columns = pd.to_datetime(df_values.columns)

# Group columns by year and average
yearly_avg = df_values.groupby(df_values.columns.year, axis=1).mean()

# Combine with RegionName
df_yearly = pd.concat([df_dates.reset_index(drop=True), yearly_avg.reset_index(drop=True)], axis=1)

# --- Filter to keep only years 2020–2025 ---
# Get only the year columns (integers)
year_cols = [col for col in df_yearly.columns if isinstance(col, (int, float)) and 2020 <= col <= 2025]

# Rebuild dataframe with RegionName + those years
df_yearly = df_yearly[['RegionName'] + year_cols]

# Optional: make columns strings for clean CSV export
df_yearly.columns = ['RegionName'] + [str(c) for c in year_cols]

df_yearly[df_yearly.columns[1:]] = df_yearly[df_yearly.columns[1:]].round(2)

# Show result
print(df_yearly.info())
print(df_yearly.head())
