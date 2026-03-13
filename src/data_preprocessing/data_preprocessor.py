"""
Data Preprocessing Module
Handles cleaning and preprocessing data
"""

import pandas as pd
import numpy as np


class DataPreprocessor:
    """
    Cleans and preprocesses data:
    - Fill missing numeric values with median
    - Fill missing categorical values with mode
    - Handle corrupt/invalid values
    - Cap outliers using IQR method
    """
    
    def __init__(self, df):
        self.df = df.copy() if df is not None else None
    
    def fill_missing_numeric(self):
        """Fill missing numeric values with median."""
        print("\n2A: Filling missing numeric values...")
        if self.df is None:
            return
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if self.df[col].isnull().any():
                median_value = self.df[col].median()
                self.df[col] = self.df[col].fillna(median_value)
                print(f"   → '{col}': filled with median {median_value:.2f}")
    
    def fill_missing_categorical(self):
        """Fill missing categorical values with mode."""
        print("\n2B: Filling missing categorical values...")
        cat_cols = ['genre', 'mood', 'track_name', 'artists', 'album_name']
        
        for cat_col in cat_cols:
            if cat_col in self.df.columns and self.df[cat_col].isnull().any():
                mode_result = self.df[cat_col].mode()
                if len(mode_result) > 0:
                    mode_value = mode_result[0]
                    self.df[cat_col].fillna(mode_value, inplace=True)
                    print(f"   → '{cat_col}': filled with mode '{mode_value}'")
    
    def handle_corrupt_values(self):
        """Fix corrupt/invalid categorical values."""
        print("\n2C: Handling corrupt/invalid values...")
        invalid_values = ["UNKNOWN", "???", "##corrupt##", "N/A", "", "null", "None"]
        
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                self.df[col] = self.df[col].replace(invalid_values, np.nan)
                if self.df[col].isnull().any():
                    mode_result = self.df[col].mode()
                    if len(mode_result) > 0:
                        mode_value = mode_result[0]
                        self.df[col].fillna(mode_value, inplace=True)
                        print(f"   → '{col}': corrupt values replaced")
    
    def cap_outliers(self):
        """Cap outliers using IQR method."""
        print("\n2D: Capping outliers (IQR method)...")
        outlier_cols = ['popularity', 'danceability', 'energy', 'loudness', 'speechiness',
                        'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
        
        for col in outlier_cols:
            if col in self.df.columns:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                self.df[col] = self.df[col].clip(lower_bound, upper_bound)
                print(f"   → '{col}' capped")
    
    def preprocess(self):
        """Execute all preprocessing steps."""
        print("\n" + "="*60)
        print("STEP 2: DATA PREPROCESSING")
        print("="*60)
        
        if self.df is None:
            print("✗ Error: No data to preprocess")
            return None
        
        self.fill_missing_numeric()
        self.fill_missing_categorical()
        self.handle_corrupt_values()
        self.cap_outliers()
        
        print("\n✓ Data preprocessing completed!")
        print(f"  Final shape: {self.df.shape}")
        
        return self.df
    
    def get_data(self):
        """Get preprocessed dataframe."""
        return self.df
