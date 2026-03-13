"""
Data Loading Module
Handles loading data from CSV files
"""

import pandas as pd


class DataLoader:
    """Loads data from CSV files."""
    
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None
    
    def load(self):
        """Load data from CSV file."""
        print("\n" + "="*60)
        print("STEP 1: DATA LOADING")
        print("="*60)
        
        try:
            self.df = pd.read_csv(self.filepath)
            print(f"✓ Data loaded successfully!")
            print(f"  Shape: {self.df.shape}")
            print(f"  Columns: {list(self.df.columns)}")
            return self.df
        except FileNotFoundError:
            print(f"✗ Error: File '{self.filepath}' not found")
            return None
    
    def get_data(self):
        """Get loaded dataframe."""
        return self.df
