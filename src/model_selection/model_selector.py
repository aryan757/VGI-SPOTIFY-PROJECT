"""
Model Selection Module
Prepares data for different ML tasks
"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


class ModelSelector:
    """
    Prepares data for different ML tasks:
    - Regression: Predict song popularity
    - Classification: Predict mood
    - Clustering: Find song vibe groups
    """
    
    def __init__(self, df):
        self.df = df
        self.model_data = {}
    
    def prepare_regression_data(self):
        """Prepare data for regression (predict popularity)."""
        print("\nA. REGRESSION - Predict Song Popularity")
        
        features_reg = ['danceability', 'energy', 'loudness', 'valence', 'tempo']
        X_reg = self.df[features_reg]
        y_reg = self.df['popularity']
        
        X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
            X_reg, y_reg, test_size=0.2, random_state=42
        )
        
        scaler_reg = StandardScaler()
        X_train_reg_scaled = scaler_reg.fit_transform(X_train_reg)
        X_test_reg_scaled = scaler_reg.transform(X_test_reg)
        
        self.model_data['regression'] = {
            'X_train': X_train_reg,
            'X_test': X_test_reg,
            'y_train': y_train_reg,
            'y_test': y_test_reg,
            'X_train_scaled': X_train_reg_scaled,
            'X_test_scaled': X_test_reg_scaled,
            'scaler': scaler_reg,
            'features': features_reg
        }
        print(f"   Training set: {X_train_reg.shape[0]} samples")
        print(f"   Test set: {X_test_reg.shape[0]} samples")
    
    def prepare_classification_data(self):
        """Prepare data for classification (predict mood)."""
        print("\nB. CLASSIFICATION - Predict Mood")
        
        features_clf = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
                        'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
        X_clf = self.df[features_clf]
        
        le_mood = LabelEncoder()
        y_clf = le_mood.fit_transform(self.df['mood'])
        
        X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
            X_clf, y_clf, test_size=0.2, random_state=42
        )
        
        self.model_data['classification'] = {
            'X_train': X_train_clf,
            'X_test': X_test_clf,
            'y_train': y_train_clf,
            'y_test': y_test_clf,
            'label_encoder': le_mood,
            'features': features_clf
        }
        print(f"   Training set: {X_train_clf.shape[0]} samples")
        print(f"   Test set: {X_test_clf.shape[0]} samples")
        print(f"   Classes: {list(le_mood.classes_)}")
    
    def prepare_clustering_data(self):
        """Prepare data for clustering (find song vibe groups)."""
        print("\nC. CLUSTERING - Find Song Vibe Groups")
        
        features_cluster = ['danceability', 'energy', 'key', 'loudness', 'mode',
                            'speechiness', 'acousticness', 'instrumentalness',
                            'liveness', 'valence', 'tempo']
        X_cluster = self.df[features_cluster]
        
        scaler_cluster = StandardScaler()
        X_cluster_scaled = scaler_cluster.fit_transform(X_cluster)
        
        self.model_data['clustering'] = {
            'X': X_cluster,
            'X_scaled': X_cluster_scaled,
            'scaler': scaler_cluster,
            'features': features_cluster
        }
        print(f"   Number of samples: {X_cluster.shape[0]}")
    
    def select(self):
        """Execute all data preparation steps."""
        print("\n" + "="*60)
        print("STEP 4: MODEL SELECTION (DATA PREPARATION)")
        print("="*60)
        
        if self.df is None:
            print("✗ Error: No data to prepare")
            return None
        
        self.prepare_regression_data()
        self.prepare_classification_data()
        self.prepare_clustering_data()
        
        print("\n✓ Data preparation completed!")
        return self.model_data
    
    def get_model_data(self):
        """Get prepared model data."""
        return self.model_data
