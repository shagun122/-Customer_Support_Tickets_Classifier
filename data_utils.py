import pandas as pd
import numpy as np

def load_data(file_path: str) -> pd.DataFrame:
    """
    Loads data from an Excel file.
    """
    try:
        df = pd.read_excel(file_path)
        print(f"Dataset loaded successfully from {file_path}.")
        return df
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return pd.DataFrame()
    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        return pd.DataFrame()

def initial_data_inspection(df: pd.DataFrame):
    """
    Performs and prints initial data inspection.
    """
    if df.empty:
        print("DataFrame is empty. No inspection to perform.")
        return

    print("\nFirst 5 rows of the dataset:")
    print(df.head())

    print("\nDataset Info:")
    df.info()

    print("\nDescriptive Statistics:")
    print(df.describe(include='all'))

    print("\nMissing values per column:")
    print(df.isnull().sum())

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handles missing data in critical columns.
    """
    if df.empty:
        print("DataFrame is empty. No cleaning to perform.")
        return df

    print(f"\nOriginal dataset shape: {df.shape}")
    
    # Drop rows where critical information for training is missing
    critical_cols = ['ticket_text', 'issue_type', 'urgency_level']
    df.dropna(subset=critical_cols, inplace=True)
    print(f"Shape after dropping NA from {critical_cols}: {df.shape}")
    
    # Handle missing 'product' values - fill with 'Unknown'
    # This column is ground truth for entity extraction.
    if 'product' in df.columns:
        missing_products_before_fill = df['product'].isnull().sum()
        if missing_products_before_fill > 0:
            print(f"Missing values in 'product' before fill: {missing_products_before_fill}")
            df['product'].fillna('Unknown', inplace=True)
            print(f"Missing values in 'product' after fill: {df['product'].isnull().sum()}")
    else:
        print("Warning: 'product' column not found.")
        
    print("\nMissing values after cleaning:")
    print(df.isnull().sum())
    
    return df

if __name__ == '__main__':
    # Example usage:
    FILE_PATH = 'ai_dev_assignment_tickets_complex_1000.xls'
    
    data_df = load_data(FILE_PATH)
    
    if not data_df.empty:
        initial_data_inspection(data_df)
        cleaned_df = clean_data(data_df.copy()) # Use .copy() to avoid modifying the original df in this example
        
        print("\nCleaned DataFrame head:")
        print(cleaned_df.head())
        print(f"\nShape of cleaned DataFrame: {cleaned_df.shape}")
