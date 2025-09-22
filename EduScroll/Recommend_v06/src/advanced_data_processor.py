"""
Advanced Data Processing Module for Recommendation Systems
Designed for ALS and FAISS algorithms

Author: Enhanced for EduScroll v05
Date: September 2025
"""

import pandas as pd
import numpy as np
from collections import defaultdict
import json
import os
import sys
import pickle
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

# Required libraries for advanced processing
from scipy import sparse
from sklearn.preprocessing import StandardScaler, LabelEncoder, MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')


class AdvancedProcessData:
    """
    Enhanced data processor for advanced recommendation algorithms.
    Supports ALS (Collaborative Filtering) and FAISS (Vector Search).
    """
    
    def __init__(self, interactions_path: str, user_features_path: str = None, 
                 video_features_path: str = None):
        """
        Initialize the advanced data processor.
        
        Args:
            interactions_path: Path to user interactions CSV
            user_features_path: Path to user features CSV
            video_features_path: Path to video features CSV
        """
        self.interactions_path = interactions_path
        self.user_features_path = user_features_path
        self.video_features_path = video_features_path
        
        # Core datasets
        self.df = None
        self.user_features = None
        self.video_features = None
        
        # Processed data containers
        self.user_profiles = None
        self.video_summary = None
        self.video_hashtags = None
        
        # Algorithm-specific data structures
        self.user_item_matrix = None
        self.user_mapping = {}
        self.video_mapping = {}
        self.reverse_user_mapping = {}
        self.reverse_video_mapping = {}
        
        # Feature matrices for different algorithms
        self.user_feature_matrix = None
        self.video_feature_matrix = None
        self.user_embeddings = None
        self.video_embeddings = None
        
        # Encoders and scalers
        self.label_encoders = {}
        self.scalers = {}
        
        # Load and initialize data
        self._load_datasets()
        
    def _load_datasets(self):
        """Load all datasets and perform initial processing."""
        print("Loading datasets...")
        
        # Load interactions
        self.df = self._load_and_process_interactions()
        
        # Load features
        if self.user_features_path:
            self.user_features = pd.read_csv(self.user_features_path)
            
        if self.video_features_path:
            self.video_features = pd.read_csv(self.video_features_path)
            
        print(f"Loaded {len(self.df)} interactions, {len(self.user_features) if self.user_features is not None else 0} users, {len(self.video_features) if self.video_features is not None else 0} videos")
        
    def _load_and_process_interactions(self) -> pd.DataFrame:
        """Load and process interaction data with enhanced features."""
        df = pd.read_csv(self.interactions_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Enhanced weight calculation for better implicit feedback
        df['weight'] = (
            1.0  # Base weight
            + 5.0 * df['watch_fraction']  # Increased importance of watch time
            + 3.0 * df['quiz_correct'].clip(lower=0)  # Quiz performance
            + 2.0 * df['like']  # Likes
            + 1.5 * df['save']  # Saves
            + 1.0 * df['share']  # Shares
        )
        
        # Binary positive feedback indicator
        df['positive'] = (
            (df['watch_fraction'] >= 0.6) |
            (df['quiz_correct'] == 1) |
            (df['like'] == 1) |
            (df['save'] == 1)
        ).astype(int)
        
        # Time-based features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['days_since_epoch'] = (df['timestamp'] - pd.Timestamp('2024-01-01')).dt.days
        
        # Engagement intensity (for ALS confidence)
        df['confidence'] = 1 + df['weight'] * 0.1
        
        return df
        
    def create_user_item_mappings(self):
        """Create bidirectional mappings between user/video IDs and matrix indices."""
        unique_users = sorted(self.df['user_id'].unique())
        unique_videos = sorted(self.df['video_id'].unique())
        
        self.user_mapping = {user_id: idx for idx, user_id in enumerate(unique_users)}
        self.video_mapping = {video_id: idx for idx, video_id in enumerate(unique_videos)}
        
        self.reverse_user_mapping = {idx: user_id for user_id, idx in self.user_mapping.items()}
        self.reverse_video_mapping = {idx: video_id for video_id, idx in self.video_mapping.items()}
        
        print(f"Created mappings: {len(unique_users)} users, {len(unique_videos)} videos")
        
    def create_user_item_matrix(self, use_confidence: bool = False) -> sparse.csr_matrix:
        """
        Create user-item interaction matrix for ALS.
        
        Args:
            use_confidence: Whether to use confidence values or binary interactions
            
        Returns:
            Sparse CSR matrix of user-item interactions
        """
        if not self.user_mapping:
            self.create_user_item_mappings()
            
        n_users = len(self.user_mapping)
        n_items = len(self.video_mapping)
        
        # Map IDs to indices
        user_indices = self.df['user_id'].map(self.user_mapping)
        item_indices = self.df['video_id'].map(self.video_mapping)
        
        if use_confidence:
            values = self.df['confidence'].values
        else:
            values = self.df['positive'].values
            
        # Create sparse matrix
        self.user_item_matrix = sparse.csr_matrix(
            (values, (user_indices, item_indices)),
            shape=(n_users, n_items)
        )
        
        print(f"Created user-item matrix: {self.user_item_matrix.shape}, "
              f"sparsity: {1 - self.user_item_matrix.nnz / (n_users * n_items):.4f}")
        
        return self.user_item_matrix
        
    def extract_hashtag_features(self) -> Tuple[Dict, np.ndarray, np.ndarray]:
        """Extract hashtag features for users and videos."""
        # Extract all unique hashtags
        all_hashtags = set()
        
        for _, row in self.df.iterrows():
            if pd.notna(row['hashtag']):
                hashtags = [tag.strip() for tag in row['hashtag'].split('#') if tag.strip()]
                all_hashtags.update(hashtags)
                
        if self.video_features is not None:
            for _, row in self.video_features.iterrows():
                if pd.notna(row['tags']):
                    hashtags = [tag.strip() for tag in row['tags'].split('#') if tag.strip()]
                    all_hashtags.update(hashtags)
                    
        hashtag_list = sorted(list(all_hashtags))
        hashtag_to_idx = {tag: idx for idx, tag in enumerate(hashtag_list)}
        
        print(f"Found {len(hashtag_list)} unique hashtags")
        
        # Create user hashtag preference matrix
        n_users = len(self.user_mapping)
        n_hashtags = len(hashtag_list)
        user_hashtag_matrix = np.zeros((n_users, n_hashtags))
        
        for user_id, user_data in self.df.groupby('user_id'):
            user_idx = self.user_mapping[user_id]
            hashtag_weights = defaultdict(float)
            
            for _, row in user_data.iterrows():
                if pd.notna(row['hashtag']):
                    hashtags = [tag.strip() for tag in row['hashtag'].split('#') if tag.strip()]
                    for tag in hashtags:
                        if tag in hashtag_to_idx:
                            hashtag_weights[tag] += row['weight']
                            
            for tag, weight in hashtag_weights.items():
                hashtag_idx = hashtag_to_idx[tag]
                user_hashtag_matrix[user_idx, hashtag_idx] = weight
                
        # Create video hashtag matrix
        n_videos = len(self.video_mapping)
        video_hashtag_matrix = np.zeros((n_videos, n_hashtags))
        
        # From interactions
        for video_id, video_data in self.df.groupby('video_id'):
            video_idx = self.video_mapping[video_id]
            video_hashtags = set()
            
            for _, row in video_data.iterrows():
                if pd.notna(row['hashtag']):
                    hashtags = [tag.strip() for tag in row['hashtag'].split('#') if tag.strip()]
                    video_hashtags.update(hashtags)
                    
            for tag in video_hashtags:
                if tag in hashtag_to_idx:
                    hashtag_idx = hashtag_to_idx[tag]
                    video_hashtag_matrix[video_idx, hashtag_idx] = 1.0
                    
        # From video features
        if self.video_features is not None:
            for _, row in self.video_features.iterrows():
                if row['video_id'] in self.video_mapping and pd.notna(row['tags']):
                    video_idx = self.video_mapping[row['video_id']]
                    hashtags = [tag.strip() for tag in row['tags'].split('#') if tag.strip()]
                    
                    for tag in hashtags:
                        if tag in hashtag_to_idx:
                            hashtag_idx = hashtag_to_idx[tag]
                            video_hashtag_matrix[video_idx, hashtag_idx] = 1.0
                            
        return hashtag_to_idx, user_hashtag_matrix, video_hashtag_matrix
        
    def create_user_feature_matrix(self) -> np.ndarray:
        """Create comprehensive user feature matrix for LightFM and FAISS."""
        if not self.user_mapping:
            self.create_user_item_mappings()
            
        n_users = len(self.user_mapping)
        features = []
        
        # Get hashtag features
        hashtag_to_idx, user_hashtag_matrix, _ = self.extract_hashtag_features()
        
        # Normalize hashtag features
        user_hashtag_normalized = StandardScaler().fit_transform(user_hashtag_matrix)
        features.append(user_hashtag_normalized)
        
        # User behavior features from interactions
        user_behavior = np.zeros((n_users, 8))  # 8 behavioral features
        
        for user_id, user_data in self.df.groupby('user_id'):
            user_idx = self.user_mapping[user_id]
            
            # Compute behavioral metrics
            user_behavior[user_idx, 0] = len(user_data)  # Total interactions
            user_behavior[user_idx, 1] = user_data['watch_fraction'].mean()  # Avg watch fraction
            user_behavior[user_idx, 2] = user_data['positive'].mean()  # Positive rate
            user_behavior[user_idx, 3] = user_data['weight'].mean()  # Avg engagement
            user_behavior[user_idx, 4] = user_data['quiz_correct'].clip(lower=0).mean()  # Quiz performance
            user_behavior[user_idx, 5] = user_data['like'].mean()  # Like rate
            user_behavior[user_idx, 6] = user_data['save'].mean()  # Save rate
            user_behavior[user_idx, 7] = user_data['share'].mean()  # Share rate
            
        # Normalize behavioral features
        user_behavior_normalized = StandardScaler().fit_transform(user_behavior)
        features.append(user_behavior_normalized)
        
        # User profile features if available
        if self.user_features is not None:
            # Class encoding
            class_features = np.zeros((n_users, 5))  # 5 possible classes: 9,10,11,12,mezun
            subject_features = np.zeros((n_users, 10))  # Top 10 subjects
            
            # Get all unique subjects
            all_subjects = set()
            for _, row in self.user_features.iterrows():
                if pd.notna(row['top_subjects']):
                    subjects = [s.strip() for s in row['top_subjects'].split(',')]
                    all_subjects.update(subjects)
                    
            subject_list = sorted(list(all_subjects))[:10]  # Top 10 subjects
            subject_to_idx = {subj: idx for idx, subj in enumerate(subject_list)}
            
            class_mapping = {'9': 0, '10': 1, '11': 2, '12': 3, 'mezun': 4}
            
            for _, row in self.user_features.iterrows():
                if row['user_id'] in self.user_mapping:
                    user_idx = self.user_mapping[row['user_id']]
                    
                    # Encode class
                    if str(row['class']) in class_mapping:
                        class_idx = class_mapping[str(row['class'])]
                        class_features[user_idx, class_idx] = 1.0
                        
                    # Encode subjects
                    if pd.notna(row['top_subjects']):
                        subjects = [s.strip() for s in row['top_subjects'].split(',')]
                        for subj in subjects:
                            if subj in subject_to_idx:
                                subj_idx = subject_to_idx[subj]
                                subject_features[user_idx, subj_idx] = 1.0
                                
            features.extend([class_features, subject_features])
            
        # Combine all features
        self.user_feature_matrix = np.hstack(features)
        
        print(f"Created user feature matrix: {self.user_feature_matrix.shape}")
        return self.user_feature_matrix
        
    def create_video_feature_matrix(self) -> np.ndarray:
        """Create comprehensive video feature matrix for FAISS."""
        if not self.video_mapping:
            self.create_user_item_mappings()
            
        n_videos = len(self.video_mapping)
        features = []
        
        # Get hashtag features
        hashtag_to_idx, _, video_hashtag_matrix = self.extract_hashtag_features()
        features.append(video_hashtag_matrix)
        
        # Video content features from interactions
        video_content = np.zeros((n_videos, 6))  # 6 content features
        
        for video_id, video_data in self.df.groupby('video_id'):
            video_idx = self.video_mapping[video_id]
            
            # Compute content metrics
            video_content[video_idx, 0] = len(video_data)  # Total interactions
            video_content[video_idx, 1] = video_data['watch_fraction'].mean()  # Avg watch fraction
            video_content[video_idx, 2] = video_data['positive'].mean()  # Positive rate
            video_content[video_idx, 3] = video_data['weight'].mean()  # Avg engagement
            video_content[video_idx, 4] = video_data['quiz_correct'].clip(lower=0).mean()  # Quiz performance
            video_content[video_idx, 5] = video_data['like'].mean()  # Like rate
            
        # Normalize content features
        video_content_normalized = StandardScaler().fit_transform(video_content)
        features.append(video_content_normalized)
        
        # Video metadata features if available
        if self.video_features is not None:
            duration_features = np.zeros((n_videos, 1))
            
            for _, row in self.video_features.iterrows():
                if row['video_id'] in self.video_mapping:
                    video_idx = self.video_mapping[row['video_id']]
                    duration_features[video_idx, 0] = row['video_duration_seconds']
                    
            # Normalize duration
            duration_normalized = StandardScaler().fit_transform(duration_features)
            features.append(duration_normalized)
            
        # Combine all features
        self.video_feature_matrix = np.hstack(features)
        
        print(f"Created video feature matrix: {self.video_feature_matrix.shape}")
        return self.video_feature_matrix
        
    def prepare_als_data(self) -> Dict[str, Any]:
        """Prepare data specifically for ALS (Alternating Least Squares) algorithm."""
        print("Preparing ALS data...")
        
        # Create user-item matrix with confidence values
        user_item_matrix = self.create_user_item_matrix(use_confidence=True)
        
        # Create binary preference matrix
        binary_matrix = self.create_user_item_matrix(use_confidence=False)
        
        als_data = {
            'user_item_matrix': user_item_matrix,
            'binary_matrix': binary_matrix,
            'user_mapping': self.user_mapping,
            'video_mapping': self.video_mapping,
            'reverse_user_mapping': self.reverse_user_mapping,
            'reverse_video_mapping': self.reverse_video_mapping,
            'n_users': len(self.user_mapping),
            'n_items': len(self.video_mapping)
        }
        
        print(f"ALS data prepared: {user_item_matrix.shape} matrix, {user_item_matrix.nnz} non-zero entries")
        return als_data
        
    def prepare_faiss_data(self) -> Dict[str, Any]:
        """Prepare data specifically for FAISS (vector similarity search)."""
        print("Preparing FAISS data...")
        
        # Create feature matrices
        user_features = self.create_user_feature_matrix()
        video_features = self.create_video_feature_matrix()
        
        # Normalize for cosine similarity
        user_features_norm = user_features / (np.linalg.norm(user_features, axis=1, keepdims=True) + 1e-8)
        video_features_norm = video_features / (np.linalg.norm(video_features, axis=1, keepdims=True) + 1e-8)
        
        faiss_data = {
            'user_features': user_features_norm.astype(np.float32),
            'video_features': video_features_norm.astype(np.float32),
            'user_mapping': self.user_mapping,
            'video_mapping': self.video_mapping,
            'reverse_user_mapping': self.reverse_user_mapping,
            'reverse_video_mapping': self.reverse_video_mapping,
        }
        
        print(f"FAISS data prepared: {user_features_norm.shape} user features, {video_features_norm.shape} video features")
        return faiss_data
        
    def save_processed_data(self, save_dir: str = "data/processed_train"):
        """Save all processed data for different algorithms."""
        os.makedirs(save_dir, exist_ok=True)
        
        print("Saving processed data...")
        
        # Prepare data for algorithms
        als_data = self.prepare_als_data()
        faiss_data = self.prepare_faiss_data()
        
        # Save ALS data
        als_dir = os.path.join(save_dir, "als")
        os.makedirs(als_dir, exist_ok=True)
        
        sparse.save_npz(os.path.join(als_dir, "user_item_matrix.npz"), als_data['user_item_matrix'])
        sparse.save_npz(os.path.join(als_dir, "binary_matrix.npz"), als_data['binary_matrix'])
        
        with open(os.path.join(als_dir, "mappings.json"), "w") as f:
            json.dump({
                'user_mapping': als_data['user_mapping'],
                'video_mapping': als_data['video_mapping'],
                'reverse_user_mapping': als_data['reverse_user_mapping'],
                'reverse_video_mapping': als_data['reverse_video_mapping']
            }, f, indent=2)
            
        # Save FAISS data
        faiss_dir = os.path.join(save_dir, "faiss")
        os.makedirs(faiss_dir, exist_ok=True)
        
        np.save(os.path.join(faiss_dir, "user_features.npy"), faiss_data['user_features'])
        np.save(os.path.join(faiss_dir, "video_features.npy"), faiss_data['video_features'])
        
        with open(os.path.join(faiss_dir, "mappings.json"), "w") as f:
            json.dump({
                'user_mapping': faiss_data['user_mapping'],
                'video_mapping': faiss_data['video_mapping'],
                'reverse_user_mapping': faiss_data['reverse_user_mapping'],
                'reverse_video_mapping': faiss_data['reverse_video_mapping']
            }, f, indent=2)
            
        # Save original processed data for backward compatibility
        original_dir = os.path.join(save_dir, "original")
        os.makedirs(original_dir, exist_ok=True)
        
        # Create user profiles (keeping original format)
        self.user_profiles = self._create_original_user_profiles()
        self.video_hashtags = self._extract_original_video_hashtags()
        self.video_summary = self._create_original_video_summary()
        
        with open(os.path.join(original_dir, "user_profiles.json"), "w") as f:
            json.dump(self.user_profiles, f, indent=2)
            
        with open(os.path.join(original_dir, "video_hashtags.json"), "w") as f:
            json.dump(self.video_hashtags, f, indent=2)
            
        # Save enhanced interactions
        self.df.to_csv(os.path.join(original_dir, "user_interactions_enhanced.csv"), index=False)
        
        print(f"All processed data saved to {save_dir}")
        print(f"- ALS: user-item matrices and mappings")
        print(f"- FAISS: normalized feature vectors")
        print(f"- Original: backward-compatible formats")
        
    def _create_original_user_profiles(self) -> Dict:
        """Create user profiles in original format for backward compatibility."""
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
        
    def _extract_original_video_hashtags(self) -> Dict:
        """Extract video hashtags in original format."""
        video_hashtags = {}
        
        for video_id, video_data in self.df.groupby('video_id'):
            all_hashtags = set()
            for _, row in video_data.iterrows():
                if pd.notna(row['hashtag']):
                    hashtags = [tag.strip() for tag in row['hashtag'].split('#') if tag.strip()]
                    all_hashtags.update(hashtags)
            video_hashtags[video_id] = list(all_hashtags)
            
        return video_hashtags
        
    def _create_original_video_summary(self) -> pd.DataFrame:
        """Create video summary in original format."""
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


def process_data_for_algorithms(data_dir: str = "data"):
    """
    Main function to process data for all recommendation algorithms.
    
    Args:
        data_dir: Base directory containing raw data
    """
    print("=== Advanced Data Processing for Recommendation Algorithms ===")
    
    # Initialize processor
    processor = AdvancedProcessData(
        interactions_path=os.path.join(data_dir, "raw", "user_interactions.csv"),
        user_features_path=os.path.join(data_dir, "raw", "user_features.csv"),
        video_features_path=os.path.join(data_dir, "raw", "video_features.csv")
    )
    
    # Process and save all data
    processor.save_processed_data(os.path.join(data_dir, "processed_train"))
    
    print("=== Processing Complete ===")
    return processor


if __name__ == "__main__":
    # Get the directory of this script and find the data directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)  # Go up one level from src/
    data_dir = os.path.join(project_dir, "data")
    
    print(f"Script directory: {script_dir}")
    print(f"Project directory: {project_dir}")
    print(f"Data directory: {data_dir}")
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        print(f"❌ Data directory not found: {data_dir}")
        print("Please run this script from a directory that contains a 'data' folder")
        sys.exit(1)
    
    # Run the processing pipeline
    processor = process_data_for_algorithms(data_dir)
    
    # Display some statistics
    print("\n=== Data Statistics ===")
    print(f"Total interactions: {len(processor.df)}")
    print(f"Unique users: {len(processor.user_mapping)}")
    print(f"Unique videos: {len(processor.video_mapping)}")
    print(f"Average interactions per user: {len(processor.df) / len(processor.user_mapping):.2f}")
    print(f"Average interactions per video: {len(processor.df) / len(processor.video_mapping):.2f}")
    print(f"Sparsity: {1 - len(processor.df) / (len(processor.user_mapping) * len(processor.video_mapping)):.4f}")