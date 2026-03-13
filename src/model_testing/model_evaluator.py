"""
Model Testing Module
Evaluates and tests all trained models
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score, classification_report, mean_squared_error


class ModelEvaluator:
    """Evaluates and tests all trained models."""
    
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
