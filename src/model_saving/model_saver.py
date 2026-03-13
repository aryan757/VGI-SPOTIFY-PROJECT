"""
Model Saving Module
Saves trained models to disk using joblib
"""

import joblib


class ModelSaver:
    """Saves trained models to disk."""
    
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
