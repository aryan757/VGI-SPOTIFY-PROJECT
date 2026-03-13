"""
Model Training Module
Trains all ML models for regression, classification, and clustering
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import DBSCAN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from xgboost import XGBRegressor, XGBClassifier


class RegressionModelTrainer:
    """Trains regression models for popularity prediction."""
    
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
        self.train_xgboost()
        self.train_random_forest()
        self.train_neural_network()
    
    def get_models(self):
        """Get trained models."""
        return self.models
    
    def get_predictions(self):
        """Get predictions."""
        return self.predictions


class ClassificationModelTrainer:
    """Trains classification models for mood prediction."""
    
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
        self.train_xgboost()
    
    def get_model(self):
        """Get trained model."""
        return self.model
    
    def get_predictions(self):
        """Get predictions."""
        return self.predictions


class ClusteringModelTrainer:
    """Trains clustering models for song vibe grouping."""
    
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
        self.train_dbscan()
    
    def get_model(self):
        """Get trained model."""
        return self.model
    
    def get_labels(self):
        """Get cluster labels."""
        return self.labels
