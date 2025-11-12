from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import os

app = Flask(__name__, template_folder="templates", static_folder="static")

# Load dataset
file_path = "Data_recommendationafter.csv"

if not os.path.exists(file_path):
    print("ERROR: Dataset file not found!")
    exit()  # Stop execution if file is missing

df = pd.read_csv(file_path)

# Ensure necessary columns exist
if "id" not in df.columns or "song_name" not in df.columns:
    raise ValueError("ERROR: Missing 'id' or 'song_name' in dataset. Check your CSV file!")

# Compute cosine similarity
drop_cols = [col for col in ["id", "song_name"] if col in df.columns]
df_cs = df.drop(columns=drop_cols, errors="ignore")  # Drop only if present
cs = cosine_similarity(df_cs)

# Convert duration from milliseconds to MM:SS format
if "duration_ms" in df.columns:
    df["duration"] = (df["duration_ms"] // 60000).astype(str) + ":" + ((df["duration_ms"] % 60000) // 1000).astype(str).str.zfill(2)
    df.drop(columns=["duration_ms"], inplace=True)

df["duration"] = df["duration"].fillna("Unknown")  # Ensure duration is always available

def find_song(song_input):
    """Find song by ID or Name."""
    df["id"] = df["id"].astype(str)  # Ensure ID is a string
    df["song_name"] = df["song_name"].astype(str).fillna("")  # Handle NaN values

    song_row = df[(df["id"] == song_input) | (df["song_name"].str.lower().str.strip() == song_input.lower().strip())]
    
    if song_row.empty:
        return None, None
    
    return song_row.index[0], song_row.iloc[0]["song_name"]

def recommend(song_idx, top_n=10):
    """Get top song recommendations."""
    if song_idx is None or song_idx < 0 or song_idx >= len(cs):
        return []

    most_similar_idx = np.argsort(-cs[song_idx])[1:top_n+1]
    recommended_songs = df.iloc[most_similar_idx][["id", "song_name", "duration"]].to_dict(orient="records")

    return recommended_songs if recommended_songs else []

@app.route("/")
def home():
    """Serve the HTML frontend."""
    return render_template("index.html")

@app.route("/recommend", methods=["POST"])
def recommend_songs():
    """API endpoint to get song recommendations."""
    data = request.json
    song_input = data.get("song")

    if not song_input:
        return jsonify({"error": "Please provide a song name or ID"}), 400

    song_idx, song_title = find_song(song_input)

    if song_idx is None:
        return jsonify({"error": f"Song '{song_input}' not found! Try another title or ID."}), 404

    recommendations = recommend(song_idx)

    if not recommendations:
        return jsonify({"error": f"No recommendations found for '{song_title}'"}), 404

    return jsonify({"input_song": song_title, "recommendations": recommendations})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5002, debug=True)

