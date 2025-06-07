import numpy as np
import pandas as pd
import random
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
        self.observation_space = spaces.Discrete(self.n_songs)
        self.action_space = spaces.Discrete(self.n_songs)
        self.user_history = set()

    def step(self, action):
        song = self.music_data.iloc[action]
        reward = 2 if song['Genre'] in self.user_history else -1
        self.user_history.add(song['Genre'])
        done = len(self.user_history) >= 10
        return action, reward, done, {}

    def reset(self):
        self.user_history.clear()
        return np.random.randint(0, self.n_songs)

# Train RL Model
env = MusicRecommenderEnv(music_data)
model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=50000)

# Function to Recommend Songs

def recommend_songs_rl(selected_song, num_recommendations=10):
    song_index = music_data[music_data["Track Name"] == selected_song].index[0]
    recommended_indices = set()
    attempts = 0
    while len(recommended_indices) < num_recommendations and attempts < 50:
        action, _ = model.predict(song_index)
        if action != song_index and action not in recommended_indices:
            recommended_indices.add(action)
        attempts += 1
    recommended_songs = [(music_data.iloc[idx]["Track Name"], music_data.iloc[idx]["Genre"]) for idx in recommended_indices]
    return recommended_songs[:num_recommendations]

# Function to Evaluate Recommendations
def evaluate_recommendations(selected_song, recommended_songs):
    selected_song_data = music_data[music_data["Track Name"] == selected_song]
    if selected_song_data.empty:
        return 0, 0, 0, 0
    selected_genres = set(selected_song_data["Genre"].tolist())
    recommended_genres = set()
    for _, song_genre in recommended_songs:
        recommended_genres.add(song_genre)
    
    true_positives = len(selected_genres & recommended_genres)
    false_positives = len(recommended_genres - selected_genres)
    false_negatives = len(selected_genres - recommended_genres)
    true_negatives = len(set(music_data['Genre'].unique()) - (selected_genres | recommended_genres))
    
    precision = (true_positives / (true_positives + false_positives)) * 100 if (true_positives + false_positives) > 0 else 0
    recall = (true_positives / (true_positives + false_negatives)) * 100 if (true_positives + false_negatives) > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = ((true_positives + true_negatives) / len(set(music_data['Genre'].unique()))) * 100 if len(set(music_data['Genre'].unique())) > 0 else 0
    
    return precision, recall, f1_score, accuracy

# Evaluate Model
num_tests = 30
precision_scores = []
recall_scores = []
f1_scores = []
accuracy_scores = []

random_songs = random.sample(list(music_data["Track Name"].unique()), num_tests)
for song in random_songs:
    recommended_songs = recommend_songs_rl(song, num_recommendations=10)
    precision, recall, f1_score, accuracy = evaluate_recommendations(song, recommended_songs)
    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1_score)
    accuracy_scores.append(accuracy)

# Compute Average Metrics
avg_precision = np.mean(precision_scores)
avg_recall = np.mean(recall_scores)
avg_f1 = np.mean(f1_scores)
avg_accuracy = np.mean(accuracy_scores)

# Print Results
print("\n📊 Model Evaluation:")
print(f"🎯 Precision: {avg_precision:.2f}%")
print(f"📌 Recall: {avg_recall:.2f}%")
print(f"⚡ F1 Score: {avg_f1:.2f}%")
print(f"✅ Accuracy: {avg_accuracy:.2f}%\n")
