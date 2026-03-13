# Spotify ML Pipeline (my_ml_project)

This repository contains an end-to-end machine learning pipeline for analyzing Spotify track metadata and audio features.
It includes data loading, preprocessing, model training (regression, classification, clustering), model saving/loading, and an API for inference.

---

## ✅ What This Project Does

The project is built around a dataset of Spotify tracks (`data/spotify_400.csv` / `data/spotify_400_cleaned.csv`) and provides:

- **Data loading + cleaning** (filling missing values, fixing corrupt rows, outlier handling)
- **Model preparation** (train/test splitting, scaling, encoding)
- **Model training**:
  - Regression: Predict song popularity
  - Classification: Predict song mood label
  - Clustering: Group songs into "vibe" clusters
- **Model saving** (to `model/` folder) and inference via a **FastAPI** web server
- Optional notebooks for exploration + training insights

---

## 📁 Repository Structure

```
my_ml_project/
├── api/                  # FastAPI inference server
│   └── app.py
├── data/                 # Input data and cleaned output
│   ├── spotify_400.csv
│   ├── spotify_400_cleaned.csv
│   └── cleaned_data.csv
├── model/                # Saved model artifacts (output after training)
├── notebook/             # Jupyter notebooks for exploration + training
│   ├── 01_data_exploration_and_preprocessing.ipynb
│   └── 02_model_training_and_evaluation.ipynb
├── src/                  # Modular pipeline implementation
│   ├── data_loading/
│   ├── data_preprocessing/
│   ├── data_saving/
│   ├── model_selection/
│   ├── model_training/
│   ├── model_saving/
│   └── model_testing/
├── main.py               # Script with functions for a full pipeline run (non-OOP)
├── main_oops.py          # (Likely older/test code) - keep or remove as desired
└── requirements.txt      # Python dependencies
```

> 💡 The **`src/`** package contains a more structured and modular pipeline, while `main.py` is a single-script version of a similar workflow.

---

## 🧰 Setup

### 1) Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

> ✅ If you use conda, you can also install the same packages into a conda environment.

---

## ▶️ Running the Pipeline

### Option A: Run the structured pipeline (recommended)

This uses `src/main.py` (the orchestrator class) and saves trained models in `model/`.

```bash
python src/main.py
```

> 🔎 By default, `src/main.py` is configured to use:
> - `data_filepath='/Users/aryan/Desktop/my_ml_project_Helper/data/spotify_400.csv'`
> - `model_dir='model/'`

If you want to use a different dataset path, edit the `main()` function in `src/main.py`.


### Option B: Run the single-script pipeline (quick demo)

```bash
python main.py
```

This script runs the same logical steps (loading, cleaning, training, evaluation) in a single file.

---

## 🧠 What Models Are Trained

### Regression (Popularity prediction)
- XGBoost Regressor
- Random Forest Regressor
- Neural Network (Keras)

### Classification (Mood prediction)
- XGBoost Classifier

### Clustering (Song vibe groups)
- DBSCAN clustering

Model artifacts are saved into `model/` as `.pkl` files using `joblib`.

---

## 🚀 Running the FastAPI Inference Server

The API is located at `api/app.py` and exposes endpoints for predictions. It expects trained models to exist at:

- `model/best_reg_model.pkl`
- `model/best_clf_model.pkl`
- `model/dbscan_clustering.pkl`

### Start the server

```bash
uvicorn api.app:app --reload
```

### Available endpoints

#### 1) Predict popularity score

`POST /song_viral_score`

Body (JSON):
```json
{
  "danceability": 0.7,
  "energy": 0.8,
  "loudness": -5.2,
  "valence": 0.6,
  "tempo": 120
}
```

#### 2) Predict mood

`POST /song_mood_classification`

Body (JSON):
```json
{
  "danceability": 0.7,
  "energy": 0.8,
  "key": 5,
  "loudness": -5.2,
  "mode": 1,
  "speechiness": 0.05,
  "acousticness": 0.2,
  "instrumentalness": 0.0,
  "liveness": 0.12,
  "valence": 0.6,
  "tempo": 120
}
```

#### 3) Predict cluster label

`POST /song_cluster`

Body (JSON):
```json
{
  "danceability": 0.7,
  "energy": 0.8,
  "key": 5,
  "loudness": -5.2,
  "mode": 1,
  "speechiness": 0.05,
  "acousticness": 0.2,
  "instrumentalness": 0.0,
  "liveness": 0.12,
  "valence": 0.6,
  "tempo": 120,
  "time_signature": 4
}
```

---

## 🧪 Notebooks

There are two Jupyter notebooks for exploration and evaluation:

- `notebook/01_data_exploration_and_preprocessing.ipynb`
- `notebook/02_model_training_and_evaluation.ipynb`

Run them with:

```bash
jupyter notebook
```

---

## 🔧 Notes / Extensions

- The preprocessor in `src/data_preprocessing/data_preprocessor.py` and `main.py` fill missing values and cap outliers using IQR.
- The pipeline assumes the dataset contains `mood` labels and `popularity` scores.
- If you want to add new models, update the trainers under `src/model_training/`.

---

## 📌 Troubleshooting

- **`FileNotFoundError`**: Ensure the CSV path exists and the correct file is specified.
- **Model loading errors**: Confirm the model files exist in `model/` after training.

---

If you want, I can also help you add a Makefile, Dockerfile, or a `requirements-dev.txt` for testing.
