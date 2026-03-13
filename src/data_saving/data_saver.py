"""
Data Saving Module
Handles saving cleaned data to CSV files
"""


class DataSaver:
    """Saves cleaned data to CSV files."""
    
    def __init__(self, df, output_path):
        self.df = df
        self.output_path = output_path
    
    def save(self):
        """Save dataframe to CSV."""
        print("\n" + "="*60)
        print("STEP 3: DATA SAVING")
        print("="*60)
        
        if self.df is None:
            print("✗ Error: No data to save")
            return False
        
        try:
            self.df.to_csv(self.output_path, index=False)
            print(f"✓ Cleaned data saved successfully!")
            print(f"  Location: {self.output_path}")
            print(f"  Shape: {self.df.shape}")
            return True
        except Exception as e:
            print(f"✗ Error saving data: {e}")
            return False
