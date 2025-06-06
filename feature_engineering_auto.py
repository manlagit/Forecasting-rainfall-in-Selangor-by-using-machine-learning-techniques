"""
Automated Feature Engineering Script for Rainfall Forecasting
This script automates the feature engineering process from the notebook.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Create processed directory if not exists
os.makedirs('data/processed', exist_ok=True)
os.makedirs('reports/figures', exist_ok=True)

print("=== AUTOMATED FEATURE ENGINEERING STARTED ===")

# 1. Load Raw Data
print("\n1. Loading raw data...")
df = pd.read_csv("data/raw/230731665812CCD_weekly1.csv")
df['Date'] = pd.to_datetime(df['Date'])

print(f"Dataset shape: {df.shape}")
print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")

# 2. Create Lag Features
print("\n2. Creating lag features...")
def create_lag_features(df, columns, lags=[1, 2, 3]):
    """
    Create lag features for specified columns.
    """
    df_lagged = df.copy()
    
    for col in columns:
        for lag in lags:
            df_lagged[f'{col}_lag_{lag}'] = df_lagged[col].shift(lag)
    
    return df_lagged

lag_columns = ['Precipitation_mm', 'Temp_avg', 'Relative_Humidity', 'Wind_kmh']
df_with_lags = create_lag_features(df, lag_columns, lags=[1, 2])

print(f"Added {len(df_with_lags.columns) - len(df.columns)} lag features")

# 3. Create Moving Averages
print("\n3. Creating moving averages...")
def create_moving_averages(df, columns, windows=[3, 4, 6]):
    """
    Create moving average features.
    """
    df_ma = df.copy()
    
    for col in columns:
        for window in windows:
            df_ma[f'{col}_ma_{window}'] = df_ma[col].rolling(window=window).mean()
    
    return df_ma

ma_columns = ['Precipitation_mm', 'Temp_avg', 'Relative_Humidity']
df_with_ma = create_moving_averages(df_with_lags, ma_columns, windows=[3, 4])

print(f"Added {len(df_with_ma.columns) - len(df_with_lags.columns)} moving average features")

# 4. Create Seasonal Features
print("\n4. Creating seasonal features...")
def create_seasonal_features(df):
    """
    Create seasonal features based on Malaysian climate.
    """
    df_seasonal = df.copy()
    
    # Extract month
    df_seasonal['Month'] = df_seasonal['Date'].dt.month
    
    # Monsoon season (heavy rainfall)
    df_seasonal['is_monsoon'] = df_seasonal['Month'].isin([10, 11, 12, 4]).astype(int)
    
    # Dry season (low rainfall)
    df_seasonal['is_dry_season'] = df_seasonal['Month'].isin([6, 7, 8]).astype(int)
    
    # Cyclical encoding for month
    df_seasonal['month_sin'] = np.sin(2 * np.pi * df_seasonal['Month'] / 12)
    df_seasonal['month_cos'] = np.cos(2 * np.pi * df_seasonal['Month'] / 12)
    
    # Week of year cyclical encoding
    df_seasonal['week_sin'] = np.sin(2 * np.pi * df_seasonal['Week_Number'] / 52)
    df_seasonal['week_cos'] = np.cos(2 * np.pi * df_seasonal['Week_Number'] / 52)
    
    return df_seasonal

df_with_seasonal = create_seasonal_features(df_with_ma)

print(f"Added {len(df_with_seasonal.columns) - len(df_with_ma.columns)} seasonal features")

# 5. Create Interaction Features
print("\n5. Creating interaction features...")
def create_interaction_features(df):
    """
    Create interaction features between weather variables.
    """
    df_interaction = df.copy()
    
    # Temperature-Humidity interaction (heat index proxy)
    df_interaction['temp_humidity_interaction'] = (
        df_interaction['Temp_avg'] * df_interaction['Relative_Humidity']
    )
    
    # Wind-Precipitation ratio
    df_interaction['wind_precip_ratio'] = (
        df_interaction['Wind_kmh'] / (df_interaction['Precipitation_mm'] + 1)
    )
    
    # Temperature difference from mean
    temp_mean = df_interaction['Temp_avg'].mean()
    df_interaction['temp_deviation'] = df_interaction['Temp_avg'] - temp_mean
    
    # Humidity categories
    df_interaction['humidity_category'] = pd.cut(
        df_interaction['Relative_Humidity'], 
        bins=[0, 60, 80, 100], 
        labels=['Low', 'Medium', 'High']
    )
    
    # One-hot encode humidity categories
    humidity_dummies = pd.get_dummies(df_interaction['humidity_category'], prefix='humidity')
    df_interaction = pd.concat([df_interaction, humidity_dummies], axis=1)
    
    return df_interaction

df_engineered = create_interaction_features(df_with_seasonal)

print(f"Added {len(df_engineered.columns) - len(df_with_seasonal.columns)} interaction features")

# 6. Feature Analysis
print("\n6. Analyzing features...")
numeric_cols = df_engineered.select_dtypes(include=[np.number]).columns
correlations = df_engineered[numeric_cols].corr()['Precipitation_mm'].abs().sort_values(ascending=False)

print("Top 10 features correlated with Precipitation:")
print(correlations.head(10))

# Save correlation analysis to figure
plt.figure(figsize=(12, 8))
top_features = correlations.head(15).index[1:]  # Exclude self-correlation
plt.barh(range(len(top_features)), correlations[top_features].values)
plt.yticks(range(len(top_features)), top_features)
plt.xlabel('Absolute Correlation with Precipitation')
plt.title('Top Features Correlated with Precipitation')
plt.tight_layout()
plt.savefig('reports/figures/feature_correlation.png', dpi=300, bbox_inches='tight')
print("Correlation plot saved to reports/figures/feature_correlation.png")

# 7. Feature Selection
print("\n7. Selecting top features...")
def select_features(df, target_col='Precipitation_mm', top_n=20):
    """
    Select top features based on correlation with target.
    """
    # Calculate correlations
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlations = df[numeric_cols].corr()[target_col].abs().sort_values(ascending=False)
    
    # Select top features (excluding target itself)
    selected_features = correlations.head(top_n + 1).index[1:].tolist()
    
    # Always include basic weather variables
    essential_features = ['Temp_avg', 'Relative_Humidity', 'Wind_kmh']
    for feature in essential_features:
        if feature not in selected_features:
            selected_features.append(feature)
    
    return selected_features

# Select features
selected_features = select_features(df_engineered, top_n=15)
print(f"Selected {len(selected_features)} features:")
for i, feature in enumerate(selected_features, 1):
    print(f"{i:2d}. {feature}")

# 8. Create final dataset
print("\n8. Creating final dataset...")
final_features = ['Date', 'Year', 'Week_Number'] + selected_features + ['Precipitation_mm']
df_final = df_engineered[final_features].copy()

# Remove rows with NaN values (due to lag features)
df_final_clean = df_final.dropna()

print(f"Original dataset: {len(df_engineered)} rows")
print(f"After removing NaN: {len(df_final_clean)} rows")

# Create directory if it doesn't exist
os.makedirs('data/processed', exist_ok=True)

# Save the engineered dataset
df_final_clean.to_csv('data/processed/engineered_features.csv', index=False)
print("\nEngineered dataset saved to data/processed/engineered_features.csv")

# Also save a smaller feature dataset for experimentation
df_minimal = df_final_clean[['Date', 'Year', 'Temp_avg', 'Relative_Humidity', 
                             'Wind_kmh', 'Precipitation_mm', 'Precipitation_mm_lag_1', 
                             'is_monsoon', 'is_dry_season']].copy()
df_minimal.to_csv('data/processed/minimal_features.csv', index=False)
print("Minimal feature set saved to data/processed/minimal_features.csv")

# 9. Summary
print("\n=== FEATURE ENGINEERING SUMMARY ===")
print(f"Total features created: {len(df_engineered.columns)}")
print(f"Selected features: {len(selected_features)}")
print(f"Final dataset shape: {df_final_clean.shape}")
print(f"Date range: {df_final_clean['Date'].min()} to {df_final_clean['Date'].max()}")
print("\nFeature engineering completed successfully!")
