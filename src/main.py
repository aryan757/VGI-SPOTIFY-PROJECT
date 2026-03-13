"""
ML Pipeline Orchestrator
Main entry point that coordinates all pipeline steps
"""

import warnings
warnings.filterwarnings('ignore')

from data_loading import DataLoader
from data_preprocessing import DataPreprocessor
from data_saving import DataSaver
from model_selection import ModelSelector
from model_training import (
    RegressionModelTrainer,
    ClassificationModelTrainer,
    ClusteringModelTrainer
)
from model_saving import ModelSaver
from model_testing import ModelEvaluator


class MLPipeline:
    """
    Orchestrates the complete ML pipeline.
    Coordinates all steps from data loading to model evaluation.
    """
    
    def __init__(self, data_filepath, model_dir='model/'):
        self.data_filepath = data_filepath
        self.model_dir = model_dir
        self.trained_models = {}
    
    def execute(self):
        """Execute the complete pipeline."""
        print("\n" + "="*60)
        print("SPOTIFY ML PIPELINE - COMPLETE WORKFLOW")
        print("="*60)
        
        # Step 1: Load Data
        print("\n[Step 1/7] Loading data...")
        data_loader = DataLoader(self.data_filepath)
        df = data_loader.load()
        if df is None:
            print("✗ Pipeline failed at data loading")
            return False
        
        # Step 2: Preprocess Data
        print("\n[Step 2/7] Preprocessing data...")
        data_preprocessor = DataPreprocessor(df)
        df_clean = data_preprocessor.preprocess()
        if df_clean is None:
            print("Pipeline failed at data preprocessing")
            return False
        
        # Step 3: Save Cleaned Data
        print("\n[Step 3/7] Saving cleaned data...")
        data_saver = DataSaver(df_clean, self.data_filepath)
        data_saver.save()
        
        # Step 4: Prepare Data for Models
        print("\n[Step 4/7] Preparing data for models...")
        model_selector = ModelSelector(df_clean)
        model_data = model_selector.select()
        if model_data is None:
            print("Pipeline failed at model selection")
            return False
        
        # Step 5: Train Models
        print("\n[Step 5/7] Training models...")
        print("\n" + "="*60)
        print("STEP 5: MODEL TRAINING")
        print("="*60)
        
        # Regression Models
        print("\nA. REGRESSION MODELS - Predict Song Popularity")
        print("-" * 50)
        regression_trainer = RegressionModelTrainer(model_data['regression'])
        regression_trainer.train()
        
        # Classification Models
        print("\n\nB. CLASSIFICATION MODEL - Predict Mood")
        print("-" * 50)
        classification_trainer = ClassificationModelTrainer(model_data['classification'])
        classification_trainer.train()
        
        # Clustering Models
        print("\n\nC. CLUSTERING MODEL - Find Song Vibe Groups")
        print("-" * 50)
        clustering_trainer = ClusteringModelTrainer(model_data['clustering'])
        clustering_trainer.train()
        
        print("\nAll models trained successfully!")
        
        # Prepare trained models dictionary
        self._prepare_trained_models_dict(
            regression_trainer,
            classification_trainer,
            clustering_trainer,
            model_data
        )
        
        # Step 6: Save Models
        print("\n[Step 6/7] Saving models...")
        model_saver = ModelSaver(self.trained_models, self.model_dir)
        model_saver.save()
        
        # Step 7: Evaluate Models
        print("\n[Step 7/7] Evaluating models...")
        model_evaluator = ModelEvaluator(self.trained_models)
        model_evaluator.evaluate()
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        return True
    
    def _prepare_trained_models_dict(self, reg_trainer, clf_trainer, 
                                     cluster_trainer, model_data):
        """Prepare trained models dictionary."""
        self.trained_models = {
            'regression_models': reg_trainer.get_models(),
            'regression_predictions': reg_trainer.get_predictions(),
            'regression_data': {
                'y_test': model_data['regression']['y_test'],
                'scaler': model_data['regression']['scaler']
            },
            'classification_model': clf_trainer.get_model(),
            'classification_predictions': clf_trainer.get_predictions(),
            'classification_data': {
                'y_test': model_data['classification']['y_test'],
                'label_encoder': model_data['classification']['label_encoder']
            },
            'clustering_model': cluster_trainer.get_model(),
            'clustering_data': {
                'labels': cluster_trainer.get_labels(),
                'scaler': model_data['clustering']['scaler']
            }
        }


def main():
    """Main entry point for the ML pipeline."""
    pipeline = MLPipeline(
        data_filepath='/Users/aryan/Desktop/my_ml_project_Helper/data/spotify_400.csv',
        model_dir='model/'
    )
    pipeline.execute()


if __name__ == "__main__":
    main()
