import numpy as np
import pandas as pd
import streamlit as st
import gym
from gym import spaces
from stable_baselines3 import DQN

# Load Dataset
music_data = pd.read_csv("cleaned_dataset_35k.csv")
music_data.dropna(subset=['Genre'], inplace=True)

# Process Genre Column
music_data['Genre'] = music_data['Genre'].str.lower().str.replace(' ', '')
music_data = music_data.assign(Genre=music_data['Genre'].str.split(',')).explode('Genre')
music_data.reset_index(drop=True, inplace=True)

# Define RL Environment
class MusicRecommenderEnv(gym.Env):
    def __init__(self, music_data):
        super(MusicRecommenderEnv, self).__init__()
        self.music_data = music_data
        self.n_songs = len(music_data)
        self.observation_space = spaces.Discrete(self.n_songs)  # Song index as state
        self.action_space = spaces.Discrete(self.n_songs)  # Recommend a song
        self.user_history = set()  # Store previously listened songs

    def step(self, action):
        song = self.music_data.iloc[action]
        reward = 2 if song['Genre'] in self.user_history else -1  # Increase reward strength
        self.user_history.add(song['Genre'])
        done = len(self.user_history) >= 10  # Stop after 10 recommendations
        return action, reward, done, {}

    def reset(self):
        self.user_history.clear()
        return np.random.randint(0, self.n_songs)

# Cache the trained model
@st.cache_resource
def train_rl_model():
    env = MusicRecommenderEnv(music_data)
    model = DQN("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=50000)  # Increase training steps for better learning
    return model

# Train model once and store in session state
if "model" not in st.session_state:
    st.session_state.model = train_rl_model()

# Streamlit UI
st.title("🎵 RL-Based Music Recommendation System")
selected_song = st.selectbox("🎶 Select a song:", music_data["Track Name"].unique())

# **🔹 Function to Recommend Songs Using RL Model**
def recommend_songs_rl(selected_song, num_recommendations=10):
    song_index = music_data[music_data["Track Name"] == selected_song].index[0]
    selected_genres = music_data[music_data["Track Name"] == selected_song]['Genre'].unique()

    recommended_indices = set()
    attempts = 0

    # **Use RL model predictions to recommend songs**
    while len(recommended_indices) < num_recommendations and attempts < 50:
        action, _ = st.session_state.model.predict(song_index)
        
        # Validate recommendation
        if action != song_index and action not in recommended_indices:
            recommended_indices.add(action)
        attempts += 1

    # **Convert indices to song names**
    recommended_songs = [(music_data.iloc[idx]["Track Name"], music_data.iloc[idx]["Genre"]) for idx in recommended_indices]
    
    return recommended_songs[:num_recommendations]  # Return only required number

if st.button("🎧 Get RL Recommendations"):
    recommended_songs = recommend_songs_rl(selected_song, num_recommendations=10)

    st.success("✅ Recommended Songs:")
    for song, genre in recommended_songs:
        st.write(f"- 🎵 {song} (Genre: {genre})")

    # **🔹 Accuracy Calculation**
    def calculate_accuracy(selected_song, recommended_songs):
        selected_genres = music_data[music_data["Track Name"] == selected_song]['Genre'].tolist()
        correct_recommendations = sum(any(genre in selected_genres for genre in song_genre.split(',')) for _, song_genre in recommended_songs)
        accuracy = (correct_recommendations / len(recommended_songs)) * 100
        return accuracy

    accuracy = calculate_accuracy(selected_song, recommended_songs)
    st.write(f"🎯 Recommendation Accuracy: **{accuracy:.2f}%**")
