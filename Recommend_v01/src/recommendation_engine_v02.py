# Iteration History (one of the most important things in software development)
# Iteration 1: Basic Data Processing and simple User listing (csv)
# Iteration 2: Enhanced User Profiling and Data Processing
# Iteration 3: Baseline Recommendation System Implementation (popularity + recency + personalization)

import pandas as pd
import numpy as np
from collections import defaultdict
import json
import math
import os

class ProcessData:
    def __init__(self, interactions_path, user_features_path=None, video_features_path=None):
        self.interactions_path = interactions_path
        self.user_features_path = user_features_path
        self.video_features_path = video_features_path

        # Load datasets
        self.df = self.load_and_process_interactions()
        self.user_features = pd.read_csv(self.user_features_path) if self.user_features_path else None
        self.video_features = pd.read_csv(self.video_features_path) if self.video_features_path else None

        self.user_profiles = None
        self.flattened_user_data = None
        self.video_summary = None
        self.video_hashtags = None

    def load_and_process_interactions(self):
        df = pd.read_csv(self.interactions_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        df['weight'] = (
            1
            + 4 * df['watch_fraction']
            + 2 * df['quiz_correct'].clip(lower=0)
            + 1 * df['like']
            + 0.5 * df['save']
        )

        df['positive'] = (
            (df['watch_fraction'] >= 0.6)
            | (df['quiz_correct'] == 1)
            | (df['like'] == 1)
        ).astype(int)

        return df

    def extract_video_hashtags(self):
        video_hashtags = {}
        for video_id, video_data in self.df.groupby('video_id'):
            all_hashtags = set()
            for _, row in video_data.iterrows():
                if pd.notna(row['hashtag']):
                    hashtags = [tag.strip() for tag in row['hashtag'].split('#') if tag.strip()]
                    all_hashtags.update(hashtags)
            video_hashtags[video_id] = list(all_hashtags)
        return video_hashtags

    def create_user_profiles(self):
        user_profiles = {}
        for user_id, user_data in self.df.groupby('user_id'):
            total_interactions = int(len(user_data))
            avg_watch_fraction = float(user_data['watch_fraction'].mean())
            positive_interactions = int(user_data['positive'].sum())
            engagement_score = float(user_data['weight'].mean())

            hashtag_engagement = defaultdict(float)
            hashtag_count = defaultdict(int)
            for _, row in user_data.iterrows():
                if pd.notna(row['hashtag']):
                    hashtags = [tag.strip() for tag in row['hashtag'].split('#') if tag.strip()]
                    for tag in hashtags:
                        hashtag_engagement[tag] += float(row['weight'])
                        hashtag_count[tag] += 1

            hashtag_preferences = {}
            for tag, total_weight in hashtag_engagement.items():
                hashtag_preferences[tag] = {
                    'total_engagement': float(total_weight),
                    'avg_engagement': float(total_weight / hashtag_count[tag]),
                    'interaction_count': int(hashtag_count[tag])
                }

            recent_videos = user_data.sort_values('timestamp', ascending=False)['video_id'].head(10).tolist()

            user_profiles[user_id] = {
                'general_metrics': {
                    'total_interactions': total_interactions,
                    'avg_watch_fraction': avg_watch_fraction,
                    'positive_interactions': positive_interactions,
                    'engagement_score': engagement_score
                },
                'hashtag_preferences': hashtag_preferences,
                'recent_engagements': recent_videos,
                'last_updated': pd.Timestamp.now().isoformat()
            }
        return user_profiles

    def create_flattened_user_data(self, user_profiles):
        flattened_data = []
        for user_id, profile in user_profiles.items():
            base_record = {'user_id': user_id, **profile['general_metrics']}
            sorted_hashtags = sorted(profile['hashtag_preferences'].items(),
                                    key=lambda x: x[1]['total_engagement'],
                                    reverse=True)
            for i, (hashtag, stats) in enumerate(sorted_hashtags[:3]):
                base_record[f'top_hashtag_{i+1}'] = hashtag
                base_record[f'top_hashtag_{i+1}_engagement'] = float(stats['total_engagement'])
            flattened_data.append(base_record)
        return pd.DataFrame(flattened_data)

    def create_video_summary(self):
        popularity = self.df.groupby('video_id').agg({
            'weight': 'sum',
            'user_id': 'count'
        }).rename(columns={'weight': 'popularity_score', 'user_id': 'total_interactions'})
        last_ts = self.df.groupby('video_id')['timestamp'].max().reset_index()
        last_ts.rename(columns={'timestamp': 'last_watch'}, inplace=True)
        video_summary = popularity.merge(last_ts, on='video_id')
        video_summary['popularity_score'] = video_summary['popularity_score'].astype(float)
        video_summary['total_interactions'] = video_summary['total_interactions'].astype(int)
        return video_summary

    def process_all_data(self, save_dir="Recommend_v01/data/processed"):
        os.makedirs(save_dir, exist_ok=True)

        print("Extracting video hashtags...")
        self.video_hashtags = self.extract_video_hashtags()

        print("Processing user profiles...")
        self.user_profiles = self.create_user_profiles()

        print("Creating flattened user data...")
        self.flattened_user_data = self.create_flattened_user_data(self.user_profiles)

        print("Creating video summary...")
        self.video_summary = self.create_video_summary()

        print("Saving processed data...")

        # 1. Save user profiles (JSON)
        with open(os.path.join(save_dir, "user_profiles.json"), "w") as f:
            json.dump(self.user_profiles, f, indent=2)

        # 2. Save flattened user profiles (CSV)
        self.flattened_user_data.to_csv(os.path.join(save_dir, "user_profiles_flattened.csv"), index=False)

        # 3. Save video hashtags (JSON)
        with open(os.path.join(save_dir, "video_hashtags.json"), "w") as f:
            json.dump(self.video_hashtags, f, indent=2)

        # 4. Save video summary (CSV)
        self.video_summary.to_csv(os.path.join(save_dir, "video_summary.csv"), index=False)

        # 5. Save interactions with weights (CSV)
        self.df.to_csv(os.path.join(save_dir, "user_interactions_with_weights.csv"), index=False)

        print(f"Processed {len(self.user_profiles)} user profiles")
        print(f"Processed {len(self.video_summary)} videos")

    def generate_baseline_recommendations(self, user_id, k=10):
        """Generate recommendations (popularity + recency + hashtag overlap)"""
        now = pd.Timestamp.now()
        tau = 6  # hours
        self.video_summary['recency_score'] = self.video_summary['last_watch'].apply(
            lambda x: math.exp(-(now - x).total_seconds() / (tau * 3600))
        )
        self.video_summary['pop_norm'] = (
            (self.video_summary['popularity_score'] - self.video_summary['popularity_score'].min()) /
            (self.video_summary['popularity_score'].max() - self.video_summary['popularity_score'].min())
        )
        self.video_summary['rec_norm'] = (
            (self.video_summary['recency_score'] - self.video_summary['recency_score'].min()) /
            (self.video_summary['recency_score'].max() - self.video_summary['recency_score'].min())
        )

        user_profile = self.user_profiles.get(user_id)
        if user_profile:
            user_top_hashtags = sorted(
                user_profile['hashtag_preferences'].items(),
                key=lambda x: x[1]['total_engagement'],
                reverse=True
            )[:5]
            user_hashtags = [tag for tag, _ in user_top_hashtags]

            tag_overlap_scores = []
            for video_id in self.video_summary.index:
                video_hashtags = self.video_hashtags.get(video_id, [])
                overlap = len(set(user_hashtags) & set(video_hashtags)) / max(len(set(user_hashtags)), 1)
                tag_overlap_scores.append(overlap)
            self.video_summary['tag_overlap'] = tag_overlap_scores
        else:
            self.video_summary['tag_overlap'] = 0

        self.video_summary['final_score'] = (
            0.6 * self.video_summary['pop_norm'] +
            0.3 * self.video_summary['rec_norm'] +
            0.1 * self.video_summary['tag_overlap']
        )
        recommendations = self.video_summary.nlargest(k, 'final_score').index.tolist()
        return recommendations
    
# === Backend için kullanılacak fonksiyon ===
# Burda 2 ana şey istenecek:
# 1. kişi id'si ve kaç video istendiği girilecek
# 2. proceessor içine csv'ler yüklenecek
# Output direkt video id listesi olacak
def get_recommendations(user_id, k=20):
    """
    Backend burada sadece bu fonksiyonu çağıracak.
    Input: user_id, k (kaç video istendiği)
    Output: video_id listesi
    """
    processor = ProcessData(
        "Recommend_v01/data/raw/user_interactions.csv",
        "Recommend_v01/data/raw/user_features.csv",
        "Recommend_v01/data/raw/video_features.csv"
    )
    processor.process_all_data()
    recs = processor.generate_baseline_recommendations(user_id, k=k)
    return recs

# Test (sen çalıştırabilirsin, backend gerekmez)
if __name__ == "__main__":
    sample = get_recommendations("user_97", k=5)
    print("Sample recs:", sample)

