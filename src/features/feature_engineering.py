import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Define base directory for relative paths
BASE_DIR = os.path.join(os.path.dirname(__file__), '..', '..') # Goes up two levels to the project root

# Define file paths
RAW_DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', '230731665812CCD_weekly1.csv')
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'engineered_features.csv')
FIGURES_PATH = os.path.join(BASE_DIR, 'reports', 'figures')

# Ensure output directories exist
os.makedirs(os.path.join(BASE_DIR, 'data', 'processed'), exist_ok=True)
os.makedirs(FIGURES_PATH, exist_ok=True)

def create_lag_features(df, columns, lags=[1, 2]):
    """Creates lag features for specified columns."""
    df_lagged = df.copy()
    for col in columns:
        for lag in lags:
            df_lagged[f'{col}_lag_{lag}'] = df_lagged[col].shift(lag)
    return df_lagged

def create_moving_averages(df, columns, windows=[3, 4]):
    """Creates moving average features for specified columns."""
    df_ma = df.copy()
    for col in columns:
        for window in windows:
            df_ma[f'{col}_ma_{window}'] = df_ma[col].rolling(window=window, min_periods=1).mean() # Added min_periods=1
    return df_ma

def create_seasonal_features(df):
    """Creates seasonal features from the Date column."""
    df_seasonal = df.copy()
    if 'Date' not in df_seasonal.columns:
        raise ValueError("DataFrame must contain a 'Date' column for seasonal features.")
    
    # Ensure 'Date' is datetime
    if not pd.api.types.is_datetime64_any_dtype(df_seasonal['Date']):
        df_seasonal['Date'] = pd.to_datetime(df_seasonal['Date'])
        
    df_seasonal['Month'] = df_seasonal['Date'].dt.month
    df_seasonal['is_monsoon'] = df_seasonal['Month'].isin([4, 10, 11, 12]).astype(int) # April, Oct, Nov, Dec
    df_seasonal['is_dry_season'] = df_seasonal['Month'].isin([6, 7, 8]).astype(int) # June, July, August
    
    # Cyclical features for month
    df_seasonal['month_sin'] = np.sin(2 * np.pi * df_seasonal['Month'] / 12)
    df_seasonal['month_cos'] = np.cos(2 * np.pi * df_seasonal['Month'] / 12)
    
    # Cyclical features for week number (assuming 'Week_Number' exists)
    if 'Week_Number' in df_seasonal.columns:
        df_seasonal['week_sin'] = np.sin(2 * np.pi * df_seasonal['Week_Number'] / 52)
        df_seasonal['week_cos'] = np.cos(2 * np.pi * df_seasonal['Week_Number'] / 52)
    else:
        # Create Week_Number if it doesn't exist
        df_seasonal['Week_Number'] = df_seasonal['Date'].dt.isocalendar().week.astype(int)
        df_seasonal['week_sin'] = np.sin(2 * np.pi * df_seasonal['Week_Number'] / 52)
        df_seasonal['week_cos'] = np.cos(2 * np.pi * df_seasonal['Week_Number'] / 52)
        
    return df_seasonal

def create_interaction_features(df):
    """Creates interaction features."""
    df_interaction = df.copy()
    if 'Temp_avg' in df_interaction.columns and 'Relative_Humidity' in df_interaction.columns:
        df_interaction['temp_humidity_interaction'] = df_interaction['Temp_avg'] * df_interaction['Relative_Humidity']
    
    if 'Wind_kmh' in df_interaction.columns and 'Precipitation_mm' in df_interaction.columns:
        df_interaction['wind_precip_ratio'] = df_interaction['Wind_kmh'] / (df_interaction['Precipitation_mm'] + 1) # Add 1 to avoid division by zero
    
    if 'Temp_avg' in df_interaction.columns:
        df_interaction['temp_deviation'] = df_interaction['Temp_avg'] - df_interaction['Temp_avg'].mean()
        
    if 'Relative_Humidity' in df_interaction.columns:
        df_interaction['humidity_category'] = pd.cut(df_interaction['Relative_Humidity'], bins=[0, 60, 80, 101], labels=['Low', 'Medium', 'High'], right=False) # Adjusted bins
        humidity_dummies = pd.get_dummies(df_interaction['humidity_category'], prefix='humidity', dummy_na=False) # dummy_na=False
        df_interaction = pd.concat([df_interaction, humidity_dummies], axis=1)
        df_interaction.drop('humidity_category', axis=1, inplace=True) # Drop original category column
        
    return df_interaction

def select_features(df, target_col='Precipitation_mm', top_n=15):
    """Selects features based on correlation with the target column."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if target_col not in numeric_cols:
        print(f"Target column '{target_col}' not found or not numeric. Skipping feature selection by correlation.")
        # Return all numeric columns except the target if it exists, or just all numeric if target is not numeric
        return [col for col in numeric_cols if col != target_col]

    correlations = df[numeric_cols].corr()[target_col].abs().sort_values(ascending=False)
    # Exclude the target column itself from the selection
    selected_features = correlations.drop(target_col, errors='ignore').head(top_n).index.tolist()
    
    # Ensure essential meteorological features are included
    essential_features = ['Temp_avg', 'Relative_Humidity', 'Wind_kmh']
    for feature in essential_features:
        if feature in df.columns and feature not in selected_features:
            selected_features.append(feature)
            
    # Remove target_col if accidentally included
    if target_col in selected_features:
        selected_features.remove(target_col)
        
    return selected_features

def plot_top_correlated_features(df, target_col='Precipitation_mm', top_n=15, save_path=FIGURES_PATH):
    """Plots and saves the top N features correlated with the target column."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if target_col not in numeric_cols:
        print(f"Target column '{target_col}' not found or not numeric for correlation plot.")
        return
        
    correlations = df[numeric_cols].corr()[target_col].abs().sort_values(ascending=False)
    top_features = correlations.drop(target_col, errors='ignore').head(top_n)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x=top_features.values, y=top_features.index)
    plt.title(f'Top {top_n} Features Correlated with {target_col}')
    plt.xlabel('Absolute Correlation')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'correlation_plot.png'), dpi=300)
    plt.close()
    print(f"Correlation plot saved to {os.path.join(save_path, 'correlation_plot.png')}")

def plot_seasonal_patterns(df, target_col='Precipitation_mm', save_path=FIGURES_PATH):
    """Plots and saves seasonal precipitation patterns."""
    if 'Month' not in df.columns or target_col not in df.columns:
        print("Required columns ('Month', '{target_col}') not found for seasonal plot.")
        return

    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    
    # Precipitation by Month
    sns.boxplot(x='Month', y=target_col, data=df, ax=ax[0])
    ax[0].set_title('Precipitation by Month')
    ax[0].set_xlabel('Month')
    ax[0].set_ylabel('Precipitation (mm)')
    
    # Precipitation by Monsoon Status
    if 'is_monsoon' in df.columns:
        sns.boxplot(x='is_monsoon', y=target_col, data=df, ax=ax[1])
        ax[1].set_title('Precipitation by Monsoon Status')
        ax[1].set_xlabel('Is Monsoon (1=Yes, 0=No)')
        ax[1].set_ylabel('Precipitation (mm)')
    else:
        ax[1].set_visible(False) # Hide if 'is_monsoon' is not present
        
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'seasonal_patterns_plot.png'), dpi=300)
    plt.close()
    print(f"Seasonal patterns plot saved to {os.path.join(save_path, 'seasonal_patterns_plot.png')}")

def plot_temp_humidity_scatter(df, target_col='Precipitation_mm', save_path=FIGURES_PATH):
    """Plots and saves scatter plot of Temperature vs. Humidity, colored by Precipitation."""
    if 'Temp_avg' not in df.columns or 'Relative_Humidity' not in df.columns or target_col not in df.columns:
        print("Required columns ('Temp_avg', 'Relative_Humidity', '{target_col}') not found for scatter plot.")
        return

    plt.figure(figsize=(10, 6))
    scatter = sns.scatterplot(x='Temp_avg', y='Relative_Humidity', hue=target_col, size=target_col, data=df, palette='viridis', sizes=(20, 200))
    plt.title('Temperature vs. Humidity (Colored by Precipitation)')
    plt.xlabel('Average Temperature (°C)')
    plt.ylabel('Relative Humidity (%)')
    plt.legend(title='Precipitation (mm)')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'temp_humidity_precipitation_scatter.png'), dpi=300)
    plt.close()
    print(f"Temp vs. Humidity scatter plot saved to {os.path.join(save_path, 'temp_humidity_precipitation_scatter.png')}")

def plot_lag_effectiveness(df, target_col='Precipitation_mm', lag_cols_prefixes=['Precipitation_mm_lag_'], save_path=FIGURES_PATH):
    """Plots and saves correlation of lag features with the target."""
    lag_columns_to_plot = [col for col in df.columns if any(col.startswith(prefix) for prefix in lag_cols_prefixes)]
    
    if not lag_columns_to_plot or target_col not in df.columns:
        print(f"No lag columns found with prefixes {lag_cols_prefixes} or target '{target_col}' not in DataFrame for lag effectiveness plot.")
        return

    correlations = df[lag_columns_to_plot + [target_col]].corr()[target_col].drop(target_col, errors='ignore').sort_values(ascending=False)
    
    if correlations.empty:
        print("No lag feature correlations to plot.")
        return

    plt.figure(figsize=(10, 6))
    sns.barplot(x=correlations.values, y=correlations.index)
    plt.title('Effectiveness of Lag Features (Correlation with Precipitation)')
    plt.xlabel('Absolute Correlation')
    plt.ylabel('Lag Feature')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'lag_effectiveness_plot.png'), dpi=300)
    plt.close()
    print(f"Lag effectiveness plot saved to {os.path.join(save_path, 'lag_effectiveness_plot.png')}")


def main():
    """Main function to run the feature engineering pipeline."""
    print("Starting feature engineering...")

    # Load raw data
    try:
        df = pd.read_csv(RAW_DATA_PATH)
        print(f"Raw data loaded successfully from {RAW_DATA_PATH}")
    except FileNotFoundError:
        print(f"Error: Raw data file not found at {RAW_DATA_PATH}")
        return
    
    # Convert 'Date' column to datetime
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    else:
        print("Error: 'Date' column not found in the raw data.")
        return

    print("Initial data info:")
    df.info()
    print("\nInitial data description:")
    print(df.describe())

    # Create lag features
    lag_columns = ['Precipitation_mm', 'Temp_avg', 'Relative_Humidity', 'Wind_kmh']
    df_with_lags = create_lag_features(df, lag_columns, lags=[1, 2])
    print("\nLag features created.")

    # Create moving averages
    ma_columns = ['Precipitation_mm', 'Temp_avg', 'Relative_Humidity']
    df_with_ma = create_moving_averages(df_with_lags, ma_columns, windows=[3, 4])
    print("Moving average features created.")

    # Create seasonal features
    df_with_seasonal = create_seasonal_features(df_with_ma)
    print("Seasonal features created.")

    # Create interaction features
    df_engineered = create_interaction_features(df_with_seasonal)
    print("Interaction features created.")

    # Feature Analysis and Visualization
    print("\nPerforming feature analysis and visualization...")
    
    # Print top correlated features
    numeric_cols_engineered = df_engineered.select_dtypes(include=[np.number]).columns
    if 'Precipitation_mm' in numeric_cols_engineered:
        correlations_engineered = df_engineered[numeric_cols_engineered].corr()['Precipitation_mm'].abs().sort_values(ascending=False)
        print("\nTop 15 features correlated with Precipitation_mm:")
        print(correlations_engineered.head(16)) # Print 16 to include target itself if present
    else:
        print("'Precipitation_mm' not found or not numeric in engineered data for correlation printing.")

    # Generate and save plots
    plot_top_correlated_features(df_engineered, target_col='Precipitation_mm', top_n=15)
    plot_seasonal_patterns(df_engineered, target_col='Precipitation_mm')
    plot_temp_humidity_scatter(df_engineered, target_col='Precipitation_mm')
    plot_lag_effectiveness(df_engineered, target_col='Precipitation_mm', lag_cols_prefixes=['Precipitation_mm_lag_', 'Temp_avg_lag_', 'Relative_Humidity_lag_', 'Wind_kmh_lag_'])


    # Feature Selection
    print("\nSelecting final features...")
    selected_features = select_features(df_engineered, target_col='Precipitation_mm', top_n=15)
    print(f"Selected {len(selected_features)} features: {selected_features}")

    # Prepare final dataset
    # Ensure 'Date', 'Year', 'Week_Number', and target 'Precipitation_mm' are in the final list if they exist
    final_feature_columns = []
    for col in ['Date', 'Year', 'Week_Number']:
        if col in df_engineered.columns:
            final_feature_columns.append(col)
    
    final_feature_columns.extend(selected_features)
    
    if 'Precipitation_mm' in df_engineered.columns:
         final_feature_columns.append('Precipitation_mm')
    
    # Remove duplicates just in case
    final_feature_columns = sorted(list(set(final_feature_columns)), key=final_feature_columns.index)


    df_final = df_engineered[final_feature_columns].copy()
    
    # Handle missing values created by lags/MA by dropping rows
    df_final_clean = df_final.dropna()
    print(f"\nShape of data before dropna: {df_final.shape}")
    print(f"Shape of data after dropna: {df_final_clean.shape}")


    # Save processed data
    try:
        df_final_clean.to_csv(PROCESSED_DATA_PATH, index=False)
        print(f"\nEngineered features saved to {PROCESSED_DATA_PATH}")
    except Exception as e:
        print(f"Error saving processed data: {e}")

    print("\nFeature engineering script completed.")

if __name__ == '__main__':
    main()
