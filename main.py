### we will first do the Necessary imports ##

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


def data_loading(filepath='data/cleaned_data_final.csv'):

    """
    Load data from CSV file.
    
    Args:
        filepath (str): Path to the data file
        
    Returns:
        pd.DataFrame: Loaded dataframe
    """

    print("\n" + "="*60)
    print("STEP 1: DATA LOADING")
    print("="*60)

    try:
        df = pd.read_csv(filepath)
        print(f"✓ Data loaded successfully!")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        return df
    except FileNotFoundError:
        print(f"✗ Error: File '{filepath}' not found")
        return None



# =====================================================================
# STEP 2: DATA PREPROCESSING (CLEANING)
# =====================================================================


def data_preprocessing(df):

    """
    Clean and preprocess the data.
    
    Handles:
    - Missing numeric values (fill with median)
    - Missing categorical values (fill with mode)
    - Invalid/corrupt values
    - Outlier capping using IQR method
    
    Args:
        df (pd.DataFrame): Raw dataframe
        
    Returns:
        pd.DataFrame: Cleaned dataframe
    """

    print("\n" + "="*60)
    print("STEP 2: DATA PREPROCESSING")
    print("="*60)

    if df is None:
        return None
    
    df_clean = df.copy()

    # 2A: Fill missing numeric values with median
    print("\n2A: Filling missing numeric values...")
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df_clean[col].isnull().any():
            median_value = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(median_value)
            print(f"   → '{col}': filled with median {median_value:.2f}")

    # 2B: Fill missing categorical values with mode
    print("\n2B: Filling missing categorical values...")
    cat_cols = ['genre', 'mood', 'track_name', 'artists', 'album_name']
    for cat_col in cat_cols:
        if cat_col in df_clean.columns and df_clean[cat_col].isnull().any():
            mode_result = df_clean[cat_col].mode()
            if len(mode_result) > 0:
                mode_value = mode_result[0]
                df_clean[cat_col].fillna(mode_value, inplace=True)
                print(f"   → '{cat_col}': filled with mode '{mode_value}'")

    # 2C: Fix corrupt/invalid categorical values
    print("\n2C: Handling corrupt/invalid values...")
    invalid_values = ["UNKNOWN", "???", "##corrupt##", "N/A", "", "null", "None"]

    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':  # String columns
            df_clean[col] = df_clean[col].replace(invalid_values, np.nan)
            if df_clean[col].isnull().any():
                mode_result = df_clean[col].mode()
                if len(mode_result) > 0:
                    mode_value = mode_result[0]
                    df_clean[col].fillna(mode_value, inplace=True)
                    print(f"   → '{col}': corrupt values replaced")

    # 2D: Cap outliers using IQR method
    print("\n2D: Capping outliers (IQR method)...")
    outlier_cols = ['popularity', 'danceability', 'energy', 'loudness', 'speechiness',
                    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
    


    for col in outlier_cols:
        if col in df_clean.columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)
            print(f"   → '{col}' capped")
    
     


    print("\n✓ Data preprocessing completed!")
    print(f"  Final shape: {df_clean.shape}")

    return df_clean


# =====================================================================
# STEP 3: DATA SAVING (SAVE CLEANED DATA)
# =====================================================================

def data_saving(df, output_path = 'data/cleaned_data_final.csv'):

    """
    Save cleaned data to CSV file.
    
    Args:
        df (pd.DataFrame): Cleaned dataframe
        output_path (str): Path to save the cleaned data
        
    Returns:
        None
    """

    print("\n" + "="*60)
    print("STEP 3: DATA SAVING")
    print("="*60)
    
    if df is None:
        print("✗ Error: No data to save")
        return False
    
    try:
        df.to_csv(output_path, index=False)
        print(f"✓ Cleaned data saved successfully!")
        print(f"  Location: {output_path}")
        print(f"  Shape: {df.shape}")
        return True
    except Exception as e:
        print(f"✗ Error saving data: {e}")
        return False


# =====================================================================
# STEP 4: MODEL SELECTION (PREPARE DATA FOR MODELS)
# =====================================================================


def model_selection(df):
    """
    Prepare data for different machine learning tasks.
    - Regression: Predict song popularity
    - Classification: Predict mood
    - Clustering: Find song vibe groups
    
    Args:
        df (pd.DataFrame): Cleaned dataframe
        
    Returns:
        dict: Dictionary containing prepared data for each task
    """


    if df is None:
        return None
    
    model_data = {}

     # ---- REGRESSION: Predict Song Popularity ----
    print("\nA. REGRESSION - Predict Song Popularity")
    features_reg = ['danceability', 'energy', 'loudness', 'valence', 'tempo']
    X_reg = df[features_reg]
    y_reg = df['popularity']
    
    # Train-test split for regression
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )
    
    # Scaling for regression models
    scaler_reg = StandardScaler()
    X_train_reg_scaled = scaler_reg.fit_transform(X_train_reg)
    X_test_reg_scaled = scaler_reg.transform(X_test_reg)
    
    model_data['regression'] = {
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
       

    # ---- CLASSIFICATION: Predict Mood ----
    print("\nB. CLASSIFICATION - Predict Mood")
    features_clf = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
                    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
    X_clf = df[features_clf]
    
    # Label encode the mood
    le_mood = LabelEncoder()
    y_clf = le_mood.fit_transform(df['mood'])
    
    # Train-test split for classification
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
        X_clf, y_clf, test_size=0.2, random_state=42
    )
    
    model_data['classification'] = {
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
    
    # ---- CLUSTERING: Find Song Vibe Groups ----
    print("\nC. CLUSTERING - Find Song Vibe Groups")
    features_cluster = ['danceability', 'energy', 'key', 'loudness', 'mode',
                        'speechiness', 'acousticness', 'instrumentalness',
                        'liveness', 'valence', 'tempo']
    X_cluster = df[features_cluster]
    
    # Scale for clustering (distance-based algorithms benefit from scaling)
    scaler_cluster = StandardScaler()
    X_cluster_scaled = scaler_cluster.fit_transform(X_cluster)
    
    model_data['clustering'] = {
        'X': X_cluster,
        'X_scaled': X_cluster_scaled,
        'scaler': scaler_cluster,
        'features': features_cluster
    }
    print(f"   Number of samples: {X_cluster.shape[0]}")
    
    print("\n✓ Data preparation completed!")
    return model_data


# =====================================================================
# STEP 5: MODEL TRAINING
# =====================================================================



def model_training(model_data):
    """
    Train all models:
    - XGBoost Regressor (Popularity prediction)
    - Random Forest Regressor
    - Neural Network Regressor
    - XGBoost Classifier (Mood prediction)
    - DBSCAN Clustering
    
    Args:
        model_data (dict): Prepared data from model_selection()
        
    Returns:
        dict: Dictionary containing trained models
    """

    print("\n" + "="*60)
    print("STEP 5: MODEL TRAINING")
    print("="*60)
    
    if model_data is None:
        return None
    
    trained_models = {}

    
    # ---- 1. REGRESSION MODELS ----
    print("\nA. REGRESSION MODELS - Predict Song Popularity")
    print("-" * 50)
    
    reg_data = model_data['regression']
    X_train_reg = reg_data['X_train']
    X_test_reg = reg_data['X_test']
    y_train_reg = reg_data['y_train']
    y_test_reg = reg_data['y_test']
    X_train_reg_scaled = reg_data['X_train_scaled']
    X_test_reg_scaled = reg_data['X_test_scaled']
    
    reg_models = {}
    
    # 1.1: XGBoost Regressor
    print("\n   1. XGBoost Regressor")
    xgb_reg = XGBRegressor(
        n_estimators=30,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        n_jobs=-1
    )
    xgb_reg.fit(X_train_reg_scaled, y_train_reg)
    y_pred_xgb_reg = xgb_reg.predict(X_test_reg_scaled)
    reg_models['XGBoost'] = xgb_reg
    print("      ✓ Trained")
    
    # 1.2: Random Forest Regressor
    print("\n   2. Random Forest Regressor")
    rf_reg = RandomForestRegressor(n_estimators=10, random_state=42)
    rf_reg.fit(X_train_reg, y_train_reg)
    y_pred_rf_reg = rf_reg.predict(X_test_reg)
    reg_models['RandomForest'] = rf_reg
    print("      ✓ Trained")
    
    # 1.3: Neural Network Regressor
    print("\n   3. Neural Network Regressor")
    nn_reg = Sequential([
        Dense(32, input_dim=X_train_reg_scaled.shape[1], activation='relu'),
        Dense(16, activation='relu'),
        Dense(1, activation='linear')
    ])
    nn_reg.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    nn_reg.fit(
        X_train_reg_scaled, y_train_reg,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        verbose=0
    )
    y_pred_nn_reg = nn_reg.predict(X_test_reg_scaled, verbose=0).flatten()
    reg_models['NeuralNetwork'] = nn_reg
    print("      ✓ Trained")
    
    trained_models['regression_models'] = reg_models
    trained_models['regression_predictions'] = {
        'XGBoost': y_pred_xgb_reg,
        'RandomForest': y_pred_rf_reg,
        'NeuralNetwork': y_pred_nn_reg
    }
    trained_models['regression_data'] = {
        'y_test': y_test_reg,
        'scaler': reg_data['scaler']
    }
    


    # ---- 2. CLASSIFICATION MODELS ----
    print("\n\nB. CLASSIFICATION MODEL - Predict Mood")
    print("-" * 50)
    
    clf_data = model_data['classification']
    X_train_clf = clf_data['X_train']
    X_test_clf = clf_data['X_test']
    y_train_clf = clf_data['y_train']
    y_test_clf = clf_data['y_test']
    le_mood = clf_data['label_encoder']
    
    print("\n   1. XGBoost Classifier")
    xgb_clf = XGBClassifier(
        n_estimators=13,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        n_jobs=1,
        eval_metric='mlogloss'
    )
    xgb_clf.fit(X_train_clf, y_train_clf)
    y_pred_clf = xgb_clf.predict(X_test_clf)
    print("      ✓ Trained")
    
    trained_models['classification_model'] = xgb_clf
    trained_models['classification_predictions'] = y_pred_clf
    trained_models['classification_data'] = {
        'y_test': y_test_clf,
        'label_encoder': le_mood
    }
    

    # ---- 3. CLUSTERING MODEL ----
    print("\n\nC. CLUSTERING MODEL - Find Song Vibe Groups")
    print("-" * 50)
    
    cluster_data = model_data['clustering']
    X_cluster_scaled = cluster_data['X_scaled']
    
    print("\n   1. DBSCAN Clustering")
    dbscan = DBSCAN(eps=2.3, min_samples=10)
    cluster_labels = dbscan.fit_predict(X_cluster_scaled)
    print("      ✓ Trained")
    
    trained_models['clustering_model'] = dbscan
    trained_models['clustering_data'] = {
        'labels': cluster_labels,
        'scaler': cluster_data['scaler']
    }
    
    print("\n✓ All models trained successfully!")
    return trained_models




# =====================================================================
# STEP 6: MODEL SAVING
# =====================================================================

def model_saving(trained_models, model_dir='model/'):
    """
    Save all trained models to disk using joblib.
    
    Args:
        trained_models (dict): Trained models from model_training()
        model_dir (str): Directory to save models
        
    Returns:
        bool: True if successful, False otherwise
    """
    print("\n" + "="*60)
    print("STEP 6: MODEL SAVING")
    print("="*60)
    
    if trained_models is None:
        print("✗ Error: No models to save")
        return False
    
    try:
        # Regression models
        print("\nA. Saving Regression Models...")
        reg_models = trained_models['regression_models']
        
        joblib.dump(reg_models['XGBoost'], f'{model_dir}xgb_regressor.pkl')
        print(f"   ✓ XGBoost Regressor → {model_dir}xgb_regressor.pkl")
        
        joblib.dump(reg_models['RandomForest'], f'{model_dir}rf_regressor.pkl')
        print(f"   ✓ Random Forest Regressor → {model_dir}rf_regressor.pkl")
        
        joblib.dump(reg_models['NeuralNetwork'], f'{model_dir}nn_regressor.h5')
        print(f"   ✓ Neural Network Regressor → {model_dir}nn_regressor.h5")
        
        joblib.dump(trained_models['regression_data']['scaler'], f'{model_dir}regression_scaler.pkl')
        print(f"   ✓ Regression Scaler → {model_dir}regression_scaler.pkl")
        
        # Classification model
        print("\nB. Saving Classification Model...")
        joblib.dump(trained_models['classification_model'], f'{model_dir}xgb_classifier.pkl')
        print(f"   ✓ XGBoost Classifier → {model_dir}xgb_classifier.pkl")
        
        joblib.dump(trained_models['classification_data']['label_encoder'], 
                   f'{model_dir}mood_label_encoder.pkl')
        print(f"   ✓ Mood Label Encoder → {model_dir}mood_label_encoder.pkl")
        
        # Clustering model
        print("\nC. Saving Clustering Model...")
        joblib.dump(trained_models['clustering_model'], f'{model_dir}dbscan_clustering.pkl')
        print(f"   ✓ DBSCAN Clustering → {model_dir}dbscan_clustering.pkl")
        
        joblib.dump(trained_models['clustering_data']['scaler'], f'{model_dir}clustering_scaler.pkl')
        print(f"   ✓ Clustering Scaler → {model_dir}clustering_scaler.pkl")
        
        print("\n✓ All models saved successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Error saving models: {e}")
        return False


# =====================================================================
# STEP 7: MODEL TESTING & EVALUATION
# =====================================================================

def model_testing(trained_models):
    """
    Test all trained models and print evaluation metrics.
    
    Args:
        trained_models (dict): Trained models from model_training()
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    print("\n" + "="*60)
    print("STEP 7: MODEL TESTING & EVALUATION")
    print("="*60)
    
    if trained_models is None:
        return None
    
    metrics = {}

    
    # ---- REGRESSION EVALUATION ----
    print("\n\nA. REGRESSION MODELS - Predict Song Popularity")
    print("-" * 50)
    
    y_test_reg = trained_models['regression_data']['y_test']
    predictions_reg = trained_models['regression_predictions']
    
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
    
    metrics['regression'] = reg_metrics

    
    # ---- CLASSIFICATION EVALUATION ----
    print("\n\nB. CLASSIFICATION MODEL - Predict Mood")
    print("-" * 50)
    
    y_test_clf = trained_models['classification_data']['y_test']
    y_pred_clf = trained_models['classification_predictions']
    le_mood = trained_models['classification_data']['label_encoder']
    
    print("\n   XGBoost Classifier:")
    print("\n   Classification Report:")
    print(classification_report(y_test_clf, y_pred_clf))
    
    metrics['classification'] = {
        'y_test': y_test_clf,
        'y_pred': y_pred_clf,
        'classes': le_mood.classes_
    }


    # ---- CLUSTERING EVALUATION ----
    print("\n\nC. CLUSTERING MODEL - Song Vibe Groups")
    print("-" * 50)
    
    cluster_labels = trained_models['clustering_data']['labels']
    
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = list(cluster_labels).count(-1)
    
    print(f"\n   DBSCAN Results:")
    print(f"      Number of clusters: {n_clusters}")
    print(f"      Number of noise points: {n_noise}")
    print(f"      Unique cluster labels: {sorted(set(cluster_labels))}")
    
    metrics['clustering'] = {
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'labels': cluster_labels
    }
    
    print("\n✓ Model testing completed!")
    return metrics




# =====================================================================
# MAIN PIPELINE EXECUTION
# =====================================================================

def main():
    """
    Execute the complete ML pipeline sequentially.
    """
    print("\n" + "="*60)
    print("SPOTIFY ML PIPELINE - COMPLETE WORKFLOW")
    print("="*60)

    # Step 1: Load Data
    df = data_loading('/Users/aryan/Desktop/my_ml_project_Helper/data/spotify_400.csv')
    if df is None:
        print("✗ Pipeline failed at data loading")
        return    
    
    # Step 2: Preprocess Data
    df_clean = data_preprocessing(df)
    if df_clean is None:
        print("✗ Pipeline failed at data preprocessing")
        return
    
     
    # Step 3: Save Cleaned Data
    data_saving(df_clean, '/Users/aryan/Desktop/my_ml_project_Helper/cleaned_data_final.csv')

       

    # Step 4: Prepare Data for Models
    model_data = model_selection(df_clean)
    if model_data is None:
        print("✗ Pipeline failed at model selection")
        return
    
    # Step 5: Train Models
    trained_models = model_training(model_data)
    if trained_models is None:
        print("✗ Pipeline failed at model training")
        return
    
    # Step 6: Save Models
    model_saving(trained_models, 'model/')
    
    # Step 7: Test and Evaluate Models
    metrics = model_testing(trained_models)
    
    print("\n" + "="*60)
    print("✓ PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)



if __name__ == "__main__":
    main()

