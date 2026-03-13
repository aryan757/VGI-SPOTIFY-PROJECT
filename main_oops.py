
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import mean_absolute_error, r2_score, classification_report, silhouette_score, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from xgboost import XGBRegressor, XGBClassifier
import os

warnings.filterwarnings('ignore')

# =====================================================================
# STEP 1: DATA LOADING
# =====================================================================

class DataLoader:

    def __init__(self, filepath='data/cleaned_data_final.csv'):

        self.filepath = filepath
        self.df = None


    def load_data(self):

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
        """ get loaded dataframe"""
        return self.df



# =====================================================================
# STEP 2: DATA PREPROCESSING (CLEANING)
# =====================================================================

class DataPreprocessor:
    """
    Handles data cleaning and preprocessing:
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


# =====================================================================
# STEP 3: DATA SAVING (SAVE CLEANED DATA)
# =====================================================================

class DataSaver:


    """
    Handles saving cleaned data to CSV files.
    """
    def __init__(self, df, output_path='data/cleaned_data_final.csv'):
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




# =====================================================================
# STEP 4: MODEL SELECTION (PREPARE DATA FOR MODELS)
# =====================================================================


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

#====================================================================

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
    
#====================================================================

    
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
    


#====================================================================



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


# =====================================================================
# STEP 5: MODEL TRAINING : 
# =====================================================================



class RegressionModelTrainer:
    """
    Trains regression models for popularity prediction.
    """
    def __init__(self, reg_data):
        self.reg_data = reg_data
        self.models = {}
        self.predictions = {}
    
    def train_xgboost(self):
        """Train XGBoost Regressor."""
        print("\n   1. XGBoost Regressor")
        
        X_train = self.reg_data['X_train_scaled']
        y_train = self.reg_data['y_train']
        X_test = self.reg_data['X_test_scaled']
        
        xgb_reg = XGBRegressor(
            n_estimators=30,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
            n_jobs=-1
        )
        xgb_reg.fit(X_train, y_train)
        y_pred = xgb_reg.predict(X_test)
        
        self.models['XGBoost'] = xgb_reg
        self.predictions['XGBoost'] = y_pred
        print("      ✓ Trained")
    
    def train_random_forest(self):
        """Train Random Forest Regressor."""
        print("\n   2. Random Forest Regressor")
        
        X_train = self.reg_data['X_train']
        y_train = self.reg_data['y_train']
        X_test = self.reg_data['X_test']
        
        rf_reg = RandomForestRegressor(n_estimators=10, random_state=42)
        rf_reg.fit(X_train, y_train)
        y_pred = rf_reg.predict(X_test)
        
        self.models['RandomForest'] = rf_reg
        self.predictions['RandomForest'] = y_pred
        print("      ✓ Trained")
    
    def train_neural_network(self):
        """Train Neural Network Regressor."""
        print("\n   3. Neural Network Regressor")
        
        X_train = self.reg_data['X_train_scaled']
        y_train = self.reg_data['y_train']
        X_test = self.reg_data['X_test_scaled']
        
        nn_reg = Sequential([
            Dense(32, input_dim=X_train.shape[1], activation='relu'),
            Dense(16, activation='relu'),
            Dense(1, activation='linear')
        ])
        nn_reg.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
        nn_reg.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )
        y_pred = nn_reg.predict(X_test, verbose=0).flatten()
        
        self.models['NeuralNetwork'] = nn_reg
        self.predictions['NeuralNetwork'] = y_pred
        print("      ✓ Trained")
    
    def train(self):
        """Train all regression models."""
        print("\nA. REGRESSION MODELS - Predict Song Popularity")
        print("-" * 50)
        
        self.train_xgboost()
        self.train_random_forest()
        self.train_neural_network()
    
    def get_models(self):
        """Get trained models."""
        return self.models
    
    def get_predictions(self):
        """Get predictions."""
        return self.predictions



# =====================================================================
# CLASS : CLASSIFICATION MODEL TRAINER
# =====================================================================


class ClassificationModelTrainer:
    """
    Trains classification models for mood prediction.
    """
    def __init__(self, clf_data):
        self.clf_data = clf_data
        self.model = None
        self.predictions = None
    
    def train_xgboost(self):
        """Train XGBoost Classifier."""
        print("\n   1. XGBoost Classifier")
        
        X_train = self.clf_data['X_train']
        y_train = self.clf_data['y_train']
        X_test = self.clf_data['X_test']
        
        xgb_clf = XGBClassifier(
            n_estimators=13,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            n_jobs=1,
            eval_metric='mlogloss'
        )
        xgb_clf.fit(X_train, y_train)
        y_pred = xgb_clf.predict(X_test)
        
        self.model = xgb_clf
        self.predictions = y_pred
        print("      ✓ Trained")
    
    def train(self):
        """Train classification model."""
        print("\n\nB. CLASSIFICATION MODEL - Predict Mood")
        print("-" * 50)
        
        self.train_xgboost()
    
    def get_model(self):
        """Get trained model."""
        return self.model
    
    def get_predictions(self):
        """Get predictions."""
        return self.predictions



# =====================================================================
# CLASS : CLUSTERING MODEL TRAINER
# =====================================================================



class ClusteringModelTrainer:
    """
    Trains clustering models for song vibe grouping.
    """
    def __init__(self, cluster_data):
        self.cluster_data = cluster_data
        self.model = None
        self.labels = None
    
    def train_dbscan(self):
        """Train DBSCAN Clustering."""
        print("\n   1. DBSCAN Clustering")
        
        X_scaled = self.cluster_data['X_scaled']
        
        dbscan = DBSCAN(eps=2.3, min_samples=10)
        labels = dbscan.fit_predict(X_scaled)
        
        self.model = dbscan
        self.labels = labels
        print("      ✓ Trained")
    
    def train(self):
        """Train clustering model."""
        print("\n\nC. CLUSTERING MODEL - Find Song Vibe Groups")
        print("-" * 50)
        
        self.train_dbscan()
    
    def get_model(self):
        """Get trained model."""
        return self.model
    
    def get_labels(self):
        """Get cluster labels."""
        return self.labels



# =====================================================================
# STEP 6: MODEL SAVING
# =====================================================================


class ModelSaver:
    """
    Saves trained models to disk using joblib.
    """
    def __init__(self, trained_models, model_dir='model/'):
        self.trained_models = trained_models
        self.model_dir = model_dir
    
    def save_regression_models(self):
        """Save regression models."""
        print("\nA. Saving Regression Models...")
        
        reg_models = self.trained_models['regression_models']
        
        joblib.dump(reg_models['XGBoost'], f'{self.model_dir}xgb_regressor.pkl')
        print(f"   ✓ XGBoost Regressor → {self.model_dir}xgb_regressor.pkl")
        
        joblib.dump(reg_models['RandomForest'], f'{self.model_dir}rf_regressor.pkl')
        print(f"   ✓ Random Forest Regressor → {self.model_dir}rf_regressor.pkl")
        
        joblib.dump(reg_models['NeuralNetwork'], f'{self.model_dir}nn_regressor.h5')
        print(f"   ✓ Neural Network Regressor → {self.model_dir}nn_regressor.h5")
        
        joblib.dump(self.trained_models['regression_data']['scaler'], f'{self.model_dir}regression_scaler.pkl')
        print(f"   ✓ Regression Scaler → {self.model_dir}regression_scaler.pkl")
    
    def save_classification_model(self):
        """Save classification model."""
        print("\nB. Saving Classification Model...")
        
        joblib.dump(self.trained_models['classification_model'], f'{self.model_dir}xgb_classifier.pkl')
        print(f"   ✓ XGBoost Classifier → {self.model_dir}xgb_classifier.pkl")
        
        joblib.dump(self.trained_models['classification_data']['label_encoder'], 
                   f'{self.model_dir}mood_label_encoder.pkl')
        print(f"   ✓ Mood Label Encoder → {self.model_dir}mood_label_encoder.pkl")
    
    def save_clustering_model(self):
        """Save clustering model."""
        print("\nC. Saving Clustering Model...")
        
        joblib.dump(self.trained_models['clustering_model'], f'{self.model_dir}dbscan_clustering.pkl')
        print(f"   ✓ DBSCAN Clustering → {self.model_dir}dbscan_clustering.pkl")
        
        joblib.dump(self.trained_models['clustering_data']['scaler'], f'{self.model_dir}clustering_scaler.pkl')
        print(f"   ✓ Clustering Scaler → {self.model_dir}clustering_scaler.pkl")
    
    def save(self):
        """Save all models."""
        print("\n" + "="*60)
        print("STEP 6: MODEL SAVING")
        print("="*60)
        
        if self.trained_models is None:
            print("✗ Error: No models to save")
            return False
        
        try:
            self.save_regression_models()
            self.save_classification_model()
            self.save_clustering_model()
            
            print("\n✓ All models saved successfully!")
            return True
        except Exception as e:
            print(f"✗ Error saving models: {e}")
            return False




# =====================================================================
# STEP 7: MODEL TESTING & EVALUATION
# =====================================================================



class ModelEvaluator:
    """
    Evaluates and tests all trained models.
    """
    def __init__(self, trained_models):
        self.trained_models = trained_models
        self.metrics = {}
    
    def evaluate_regression(self):
        """Evaluate regression models."""
        print("\n\nA. REGRESSION MODELS - Predict Song Popularity")
        print("-" * 50)
        
        y_test_reg = self.trained_models['regression_data']['y_test']
        predictions_reg = self.trained_models['regression_predictions']
        
        reg_metrics = {}
        
        for model_name, y_pred in predictions_reg.items():
            mae = mean_absolute_error(y_test_reg, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred))
            r2 = r2_score(y_test_reg, y_pred)
            
            reg_metrics[model_name] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}
            
            print(f"\n   {model_name}:")
            print(f"      MAE:  {mae:.4f}")
            print(f"      RMSE: {rmse:.4f}")
            print(f"      R²:   {r2:.4f}")
        
        self.metrics['regression'] = reg_metrics
    
    def evaluate_classification(self):
        """Evaluate classification model."""
        print("\n\nB. CLASSIFICATION MODEL - Predict Mood")
        print("-" * 50)
        
        y_test_clf = self.trained_models['classification_data']['y_test']
        y_pred_clf = self.trained_models['classification_predictions']
        
        print("\n   XGBoost Classifier:")
        print("\n   Classification Report:")
        print(classification_report(y_test_clf, y_pred_clf))
        
        self.metrics['classification'] = {
            'y_test': y_test_clf,
            'y_pred': y_pred_clf
        }
    
    def evaluate_clustering(self):
        """Evaluate clustering model."""
        print("\n\nC. CLUSTERING MODEL - Song Vibe Groups")
        print("-" * 50)
        
        cluster_labels = self.trained_models['clustering_data']['labels']
        
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        
        print(f"\n   DBSCAN Results:")
        print(f"      Number of clusters: {n_clusters}")
        print(f"      Number of noise points: {n_noise}")
        print(f"      Unique cluster labels: {sorted(set(cluster_labels))}")
        
        self.metrics['clustering'] = {
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'labels': cluster_labels
        }
    
    def evaluate(self):
        """Evaluate all models."""
        print("\n" + "="*60)
        print("STEP 7: MODEL TESTING & EVALUATION")
        print("="*60)
        
        self.evaluate_regression()
        self.evaluate_classification()
        self.evaluate_clustering()
        
        print("\n✓ Model testing completed!")
        return self.metrics
    
    def get_metrics(self):
        """Get evaluation metrics."""
        return self.metrics


# =====================================================================
# CLASS FOR ML PIPELINE ORCHESTRATOR
# =====================================================================


class MLPipeline:
    """
    Orchestrates the complete ML pipeline.
    Coordinates all steps from data loading to model evaluation.
    """
    def __init__(self, data_filepath='data/cleaned_data_final.csv', model_dir='model/'):
        self.data_filepath = data_filepath
        self.model_dir = model_dir
        self.data_loader = None
        self.data_preprocessor = None
        self.data_saver = None
        self.model_selector = None
        self.regression_trainer = None
        self.classification_trainer = None
        self.clustering_trainer = None
        self.model_saver = None
        self.model_evaluator = None
        self.trained_models = {}
    
    def execute(self):
        """Execute the complete pipeline."""
        print("\n" + "="*60)
        print("SPOTIFY ML PIPELINE - COMPLETE WORKFLOW")
        print("="*60)
        
        # Step 1: Load Data
        self.data_loader = DataLoader(self.data_filepath)
        df = self.data_loader.load_data()
        if df is None:
            print("✗ Pipeline failed at data loading")
            return False
        
        # Step 2: Preprocess Data
        self.data_preprocessor = DataPreprocessor(df)
        df_clean = self.data_preprocessor.preprocess()
        if df_clean is None:
            print("✗ Pipeline failed at data preprocessing")
            return False
        
        # Step 3: Save Cleaned Data
        self.data_saver = DataSaver(df_clean, self.data_filepath)
        self.data_saver.save()
        
        # Step 4: Prepare Data for Models
        self.model_selector = ModelSelector(df_clean)
        model_data = self.model_selector.select()
        if model_data is None:
            print("✗ Pipeline failed at model selection")
            return False
        
        # Step 5: Train Models
        print("\n" + "="*60)
        print("STEP 5: MODEL TRAINING")
        print("="*60)
        
        # Regression Models
        self.regression_trainer = RegressionModelTrainer(model_data['regression'])
        self.regression_trainer.train()
        
        # Classification Models
        self.classification_trainer = ClassificationModelTrainer(model_data['classification'])
        self.classification_trainer.train()
        
        # Clustering Models
        self.clustering_trainer = ClusteringModelTrainer(model_data['clustering'])
        self.clustering_trainer.train()
        
        print("\n✓ All models trained successfully!")
        
        # Prepare trained models dictionary
        self._prepare_trained_models_dict(model_data)
        
        # Step 6: Save Models
        self.model_saver = ModelSaver(self.trained_models, self.model_dir)
        self.model_saver.save()
        
        # Step 7: Evaluate Models
        self.model_evaluator = ModelEvaluator(self.trained_models)
        self.model_evaluator.evaluate()
        
        print("\n" + "="*60)
        print("✓ PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        return True
    
    def _prepare_trained_models_dict(self, model_data):
        """Prepare trained models dictionary."""
        self.trained_models = {
            'regression_models': self.regression_trainer.get_models(),
            'regression_predictions': self.regression_trainer.get_predictions(),
            'regression_data': {
                'y_test': model_data['regression']['y_test'],
                'scaler': model_data['regression']['scaler']
            },
            'classification_model': self.classification_trainer.get_model(),
            'classification_predictions': self.classification_trainer.get_predictions(),
            'classification_data': {
                'y_test': model_data['classification']['y_test'],
                'label_encoder': model_data['classification']['label_encoder']
            },
            'clustering_model': self.clustering_trainer.get_model(),
            'clustering_data': {
                'labels': self.clustering_trainer.get_labels(),
                'scaler': model_data['clustering']['scaler']
            }
        }


# =====================================================================
# MAIN EXECUTION
# =====================================================================



def main():
    """
    Main entry point for the ML pipeline.
    """
    pipeline = MLPipeline(
        data_filepath='/Users/aryan/Desktop/my_ml_project/data/spotify_400.csv',
        model_dir='model/'
    )
    pipeline.execute()


if __name__ == "__main__":
    main()
