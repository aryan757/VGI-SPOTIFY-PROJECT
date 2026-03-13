from fastapi import FastAPI
import joblib
import numpy as np
from fastapi import Body
from sklearn.cluster import DBSCAN


app = FastAPI()


# Load models
reg_model = joblib.load('/Users/aryan/Desktop/my_ml_project/model/best_reg_model.pkl')
clf_model = joblib.load('/Users/aryan/Desktop/my_ml_project/model/best_clf_model.pkl')
clust_model = joblib.load('/Users/aryan/Desktop/my_ml_project/model/dbscan_clustering.pkl')



@app.post("/song_viral_score")
def regression_model_song_viral_score(data: dict = Body(...)):
    # Assuming data contains the features: danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, liveness, valence, tempo, time_signature
    features = np.array([[
        data['danceability'],
        data['energy'],
        data['loudness'],
        data['valence'],
        data['tempo']
    ]])
    prediction = reg_model.predict(features)
    return {"predicted_mood_score": float(prediction[0])}


@app.post("/song_mood_classification")
def classification_model_song_mood(data: dict = Body(...)):
    # Assuming data contains the features: danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, liveness, valence, tempo, time_signature
    features = np.array([[
        data['danceability'],
        data['energy'],
        data['key'],
        data['loudness'],
        data['mode'],
        data['speechiness'],
        data['acousticness'],
        data['instrumentalness'],
        data['liveness'],
        data['valence'],
        data['tempo']
    ]])
    prediction = clf_model.predict(features)
    return {"predicted_mood": str(prediction[0])}


@app.post("/song_cluster")
def clustering_model_song_cluster(data: dict = Body(...)):
    # Assuming data contains the features: danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, liveness, valence, tempo, time_signature
    features = np.array([[
        data['danceability'],
        data['energy'],
        data['key'],
        data['loudness'],
        data['mode'],
        data['speechiness'],
        data['acousticness'],
        data['instrumentalness'],
        data['liveness'],
        data['valence'],
        data['tempo'],
        data['time_signature']
    ]])


    new_dbscan = DBSCAN(eps=clust_model.eps, min_samples=clust_model.min_samples)
    prediction = new_dbscan.fit_predict(features)
    return {"predicted_cluster": int(prediction[0])}