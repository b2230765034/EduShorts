"""
FAISS-specific Data Processing Extensions
Specialized methods for vector similarity search and content-based recommendations

Author: Enhanced for EduScroll v05
Date: September 2025
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import json
import os
from collections import defaultdict
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')


class FAISSDataProcessor:
    """
    Specialized processor for FAISS (Facebook AI Similarity Search).
    Optimized for content-based and hybrid recommendations using dense vector representations.
    """
    
    def __init__(self, base_processor):
        """
        Initialize FAISS processor with base data processor.
        
        Args:
            base_processor: AdvancedProcessData instance
        """
        self.base_processor = base_processor
        self.df = base_processor.df
        self.user_features = base_processor.user_features
        self.video_features = base_processor.video_features
        self.user_mapping = base_processor.user_mapping
        self.video_mapping = base_processor.video_mapping
        
        # FAISS-specific vectors
        self.user_vectors = None
        self.video_vectors = None
        self.user_content_vectors = None
        self.video_content_vectors = None
        self.user_behavior_vectors = None
        
        # Dimensionality reduction models
        self.user_pca = None
        self.video_pca = None
        self.svd_model = None
        
    def create_hashtag_embeddings(self, embedding_dim: int = 64) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Create hashtag-based embeddings using co-occurrence and TF-IDF.
        
        Args:
            embedding_dim: Dimension of hashtag embeddings
            
        Returns:
            Tuple of (user_hashtag_vectors, video_hashtag_vectors, hashtag_info)
        """
        print("Creating hashtag embeddings...")
        
        # Extract all hashtags and create vocabulary
        all_hashtags = set()
        hashtag_docs = {'users': {}, 'videos': {}}
        
        # Collect hashtags from interactions
        for user_id, user_data in self.df.groupby('user_id'):
            hashtag_list = []
            for _, row in user_data.iterrows():
                if pd.notna(row['hashtag']):
                    tags = [tag.strip() for tag in row['hashtag'].split('#') if tag.strip()]
                    hashtag_list.extend(tags)
                    all_hashtags.update(tags)
            hashtag_docs['users'][user_id] = ' '.join(hashtag_list)
            
        for video_id, video_data in self.df.groupby('video_id'):
            hashtag_list = []
            for _, row in video_data.iterrows():
                if pd.notna(row['hashtag']):
                    tags = [tag.strip() for tag in row['hashtag'].split('#') if tag.strip()]
                    hashtag_list.extend(tags)
                    all_hashtags.update(tags)
            hashtag_docs['videos'][video_id] = ' '.join(hashtag_list)
            
        # Add hashtags from video features
        if self.video_features is not None:
            for _, row in self.video_features.iterrows():
                if pd.notna(row['tags']):
                    tags = [tag.strip() for tag in row['tags'].split('#') if tag.strip()]
                    all_hashtags.update(tags)
                    if row['video_id'] in hashtag_docs['videos']:
                        hashtag_docs['videos'][row['video_id']] += ' ' + ' '.join(tags)
                    else:
                        hashtag_docs['videos'][row['video_id']] = ' '.join(tags)
                        
        print(f"Found {len(all_hashtags)} unique hashtags")
        
        # Create TF-IDF vectors for hashtags
        max_features = min(len(all_hashtags), 200)  # Limit for computational efficiency
        tfidf = TfidfVectorizer(max_features=max_features, stop_words=None)
        
        # Fit on all documents
        all_docs = list(hashtag_docs['users'].values()) + list(hashtag_docs['videos'].values())
        all_docs = [doc for doc in all_docs if doc.strip()]  # Remove empty docs
        
        if len(all_docs) == 0:
            print("Warning: No hashtag documents found")
            n_users = len(self.user_mapping)
            n_videos = len(self.video_mapping)
            return (np.zeros((n_users, embedding_dim)), 
                   np.zeros((n_videos, embedding_dim)), 
                   {'n_hashtags': 0})
        
        tfidf.fit(all_docs)
        
        # Create user hashtag vectors
        n_users = len(self.user_mapping)
        user_hashtag_vectors = np.zeros((n_users, max_features))
        
        for user_id in self.user_mapping:
            user_idx = self.user_mapping[user_id]
            doc = hashtag_docs['users'].get(user_id, '')
            if doc.strip():
                try:
                    vector = tfidf.transform([doc]).toarray()[0]
                    user_hashtag_vectors[user_idx] = vector
                except:
                    pass  # Skip if transformation fails
                    
        # Create video hashtag vectors
        n_videos = len(self.video_mapping)
        video_hashtag_vectors = np.zeros((n_videos, max_features))
        
        for video_id in self.video_mapping:
            video_idx = self.video_mapping[video_id]
            doc = hashtag_docs['videos'].get(video_id, '')
            if doc.strip():
                try:
                    vector = tfidf.transform([doc]).toarray()[0]
                    video_hashtag_vectors[video_idx] = vector
                except:
                    pass  # Skip if transformation fails
                    
        # Dimensionality reduction if needed
        if max_features > embedding_dim:
            print(f"Reducing hashtag embeddings from {max_features} to {embedding_dim} dimensions")
            svd = TruncatedSVD(n_components=embedding_dim, random_state=42)
            user_hashtag_vectors = svd.fit_transform(user_hashtag_vectors)
            video_hashtag_vectors = svd.transform(video_hashtag_vectors)
        else:
            # Pad with zeros if needed
            if max_features < embedding_dim:
                padding_users = np.zeros((n_users, embedding_dim - max_features))
                padding_videos = np.zeros((n_videos, embedding_dim - max_features))
                user_hashtag_vectors = np.hstack([user_hashtag_vectors, padding_users])
                video_hashtag_vectors = np.hstack([video_hashtag_vectors, padding_videos])
                
        hashtag_info = {
            'n_hashtags': len(all_hashtags),
            'max_features': max_features,
            'embedding_dim': embedding_dim,
            'vocabulary': list(tfidf.vocabulary_.keys()) if hasattr(tfidf, 'vocabulary_') else []
        }
        
        print(f"Created hashtag embeddings: users {user_hashtag_vectors.shape}, videos {video_hashtag_vectors.shape}")
        return user_hashtag_vectors, video_hashtag_vectors, hashtag_info
        
    def create_user_behavior_vectors(self, normalize: bool = True) -> np.ndarray:
        """
        Create dense behavior vectors for users based on interaction patterns.
        
        Args:
            normalize: Whether to normalize the vectors
            
        Returns:
            User behavior vectors
        """
        print("Creating user behavior vectors...")
        
        n_users = len(self.user_mapping)
        behavior_features = []
        
        # Basic interaction statistics
        user_stats = np.zeros((n_users, 12))  # 12 behavioral features
        
        for user_id, user_data in self.df.groupby('user_id'):
            user_idx = self.user_mapping[user_id]
            
            # Engagement metrics
            user_stats[user_idx, 0] = len(user_data)  # Total interactions
            user_stats[user_idx, 1] = user_data['watch_fraction'].mean()  # Avg watch fraction
            user_stats[user_idx, 2] = user_data['watch_fraction'].std()  # Watch fraction variability
            user_stats[user_idx, 3] = user_data['positive'].mean()  # Positive rate
            user_stats[user_idx, 4] = user_data['weight'].mean()  # Avg engagement score
            user_stats[user_idx, 5] = user_data['weight'].std()  # Engagement variability
            
            # Interaction type preferences
            user_stats[user_idx, 6] = user_data['like'].mean()  # Like rate
            user_stats[user_idx, 7] = user_data['save'].mean()  # Save rate
            user_stats[user_idx, 8] = user_data['share'].mean()  # Share rate
            
            # Quiz performance
            quiz_data = user_data[user_data['quiz_correct'] >= 0]
            if len(quiz_data) > 0:
                user_stats[user_idx, 9] = quiz_data['quiz_correct'].mean()  # Quiz success rate
            
            # Temporal patterns
            user_stats[user_idx, 10] = user_data['hour'].std()  # Time diversity
            user_stats[user_idx, 11] = user_data['day_of_week'].nunique()  # Day diversity
            
        behavior_features.append(user_stats)
        
        # Time-based patterns
        temporal_features = np.zeros((n_users, 6))
        
        for user_id, user_data in self.df.groupby('user_id'):
            user_idx = self.user_mapping[user_id]
            
            # Activity patterns
            hourly_activity = user_data.groupby('hour').size()
            temporal_features[user_idx, 0] = hourly_activity.std()  # Activity spread across hours
            
            # Peak activity hours
            peak_hours = hourly_activity.nlargest(3).index.tolist()
            if len(peak_hours) >= 3:
                temporal_features[user_idx, 1] = np.mean(peak_hours)  # Average peak hour
                
            # Session patterns (approximate)
            timestamps = user_data['timestamp'].sort_values()
            if len(timestamps) > 1:
                time_diffs = timestamps.diff().dt.total_seconds() / 3600  # Hours between interactions
                temporal_features[user_idx, 2] = time_diffs.mean()  # Avg time between interactions
                temporal_features[user_idx, 3] = time_diffs.std()  # Time pattern variability
                
            # Recency
            latest_interaction = user_data['timestamp'].max()
            days_since_last = (pd.Timestamp.now() - latest_interaction).days
            temporal_features[user_idx, 4] = days_since_last  # Days since last interaction
            
            # Consistency
            temporal_features[user_idx, 5] = len(user_data) / max(days_since_last, 1)  # Interaction rate
            
        behavior_features.append(temporal_features)
        
        # Content preferences (video duration, subject diversity)
        content_features = np.zeros((n_users, 4))
        
        for user_id, user_data in self.df.groupby('user_id'):
            user_idx = self.user_mapping[user_id]
            
            # Video duration preferences
            if self.video_features is not None:
                video_durations = []
                for video_id in user_data['video_id'].unique():
                    video_info = self.video_features[self.video_features['video_id'] == video_id]
                    if len(video_info) > 0:
                        video_durations.append(video_info.iloc[0]['video_duration_seconds'])
                        
                if video_durations:
                    content_features[user_idx, 0] = np.mean(video_durations)  # Avg preferred duration
                    content_features[user_idx, 1] = np.std(video_durations)  # Duration preference variance
                    
            # Subject diversity
            all_hashtags = []
            for _, row in user_data.iterrows():
                if pd.notna(row['hashtag']):
                    tags = [tag.strip() for tag in row['hashtag'].split('#') if tag.strip()]
                    all_hashtags.extend(tags)
                    
            if all_hashtags:
                unique_hashtags = set(all_hashtags)
                content_features[user_idx, 2] = len(unique_hashtags)  # Hashtag diversity
                content_features[user_idx, 3] = len(all_hashtags) / len(unique_hashtags)  # Hashtag repetition
                
        behavior_features.append(content_features)
        
        # Combine all behavioral features
        self.user_behavior_vectors = np.hstack(behavior_features)
        
        # Handle NaN values
        self.user_behavior_vectors = np.nan_to_num(self.user_behavior_vectors, nan=0.0)
        
        # Normalize if requested
        if normalize:
            scaler = StandardScaler()
            self.user_behavior_vectors = scaler.fit_transform(self.user_behavior_vectors)
            
        print(f"Created user behavior vectors: {self.user_behavior_vectors.shape}")
        return self.user_behavior_vectors
        
    def create_video_content_vectors(self, normalize: bool = True) -> np.ndarray:
        """
        Create dense content vectors for videos based on metadata and engagement.
        
        Args:
            normalize: Whether to normalize the vectors
            
        Returns:
            Video content vectors
        """
        print("Creating video content vectors...")
        
        n_videos = len(self.video_mapping)
        content_features = []
        
        # Basic video statistics from interactions
        video_stats = np.zeros((n_videos, 10))
        
        for video_id, video_data in self.df.groupby('video_id'):
            video_idx = self.video_mapping[video_id]
            
            # Engagement metrics
            video_stats[video_idx, 0] = len(video_data)  # Total views
            video_stats[video_idx, 1] = video_data['watch_fraction'].mean()  # Avg completion rate
            video_stats[video_idx, 2] = video_data['watch_fraction'].std()  # Completion variance
            video_stats[video_idx, 3] = video_data['positive'].mean()  # Positive rate
            video_stats[video_idx, 4] = video_data['weight'].mean()  # Avg engagement
            video_stats[video_idx, 5] = video_data['like'].mean()  # Like rate
            video_stats[video_idx, 6] = video_data['save'].mean()  # Save rate
            video_stats[video_idx, 7] = video_data['share'].mean()  # Share rate
            
            # Quiz performance (if applicable)
            quiz_data = video_data[video_data['quiz_correct'] >= 0]
            if len(quiz_data) > 0:
                video_stats[video_idx, 8] = quiz_data['quiz_correct'].mean()  # Quiz success rate
                
            # Viewer diversity
            video_stats[video_idx, 9] = len(video_data['user_id'].unique())  # Unique viewers
            
        content_features.append(video_stats)
        
        # Video metadata features
        metadata_features = np.zeros((n_videos, 6))
        
        if self.video_features is not None:
            for _, row in self.video_features.iterrows():
                if row['video_id'] in self.video_mapping:
                    video_idx = self.video_mapping[row['video_id']]
                    
                    # Duration features
                    duration = row['video_duration_seconds']
                    metadata_features[video_idx, 0] = duration  # Raw duration
                    metadata_features[video_idx, 1] = np.log1p(duration)  # Log duration
                    
                    # Duration category (short/medium/long)
                    if duration < 120:  # < 2 minutes
                        metadata_features[video_idx, 2] = 1
                    elif duration < 300:  # 2-5 minutes
                        metadata_features[video_idx, 3] = 1
                    else:  # > 5 minutes
                        metadata_features[video_idx, 4] = 1
                        
                    # Upload recency
                    upload_date = pd.to_datetime(row['upload_date'])
                    days_old = (pd.Timestamp.now() - upload_date).days
                    metadata_features[video_idx, 5] = np.log1p(days_old)  # Log age
                    
        content_features.append(metadata_features)
        
        # Temporal engagement patterns
        temporal_features = np.zeros((n_videos, 4))
        
        for video_id, video_data in self.df.groupby('video_id'):
            video_idx = self.video_mapping[video_id]
            
            # When is this video typically watched?
            temporal_features[video_idx, 0] = video_data['hour'].mean()  # Avg watch hour
            temporal_features[video_idx, 1] = video_data['hour'].std()  # Hour variability
            temporal_features[video_idx, 2] = video_data['day_of_week'].mean()  # Avg watch day
            
            # Viewing pattern consistency
            timestamps = video_data['timestamp'].sort_values()
            if len(timestamps) > 1:
                time_span = (timestamps.max() - timestamps.min()).days
                temporal_features[video_idx, 3] = len(video_data) / max(time_span, 1)  # Views per day
                
        content_features.append(temporal_features)
        
        # Combine all content features
        self.video_content_vectors = np.hstack(content_features)
        
        # Handle NaN values
        self.video_content_vectors = np.nan_to_num(self.video_content_vectors, nan=0.0)
        
        # Normalize if requested
        if normalize:
            scaler = StandardScaler()
            self.video_content_vectors = scaler.fit_transform(self.video_content_vectors)
            
        print(f"Created video content vectors: {self.video_content_vectors.shape}")
        return self.video_content_vectors
        
    def create_hybrid_vectors(self, include_hashtags: bool = True, 
                             embedding_dim: int = 128) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create comprehensive hybrid vectors combining all features.
        
        Args:
            include_hashtags: Whether to include hashtag embeddings
            embedding_dim: Final embedding dimension
            
        Returns:
            Tuple of (user_vectors, video_vectors)
        """
        print("Creating hybrid vectors...")
        
        user_features = []
        video_features = []
        
        # Add behavior and content vectors
        user_behavior = self.create_user_behavior_vectors(normalize=True)
        video_content = self.create_video_content_vectors(normalize=True)
        
        user_features.append(user_behavior)
        video_features.append(video_content)
        
        # Add hashtag embeddings if requested
        if include_hashtags:
            hashtag_dim = min(64, embedding_dim // 2)  # Use half the embedding dim for hashtags
            user_hashtag, video_hashtag, _ = self.create_hashtag_embeddings(hashtag_dim)
            user_features.append(user_hashtag)
            video_features.append(video_hashtag)
            
        # Add user profile features if available
        if self.user_features is not None:
            user_profile_features = self._create_user_profile_vectors()
            user_features.append(user_profile_features)
            
        # Combine features
        user_combined = np.hstack(user_features)
        video_combined = np.hstack(video_features)
        
        # Dimensionality reduction to target size
        if user_combined.shape[1] > embedding_dim:
            print(f"Reducing user vectors from {user_combined.shape[1]} to {embedding_dim} dimensions")
            self.user_pca = PCA(n_components=embedding_dim, random_state=42)
            user_combined = self.user_pca.fit_transform(user_combined)
            
        if video_combined.shape[1] > embedding_dim:
            print(f"Reducing video vectors from {video_combined.shape[1]} to {embedding_dim} dimensions")
            self.video_pca = PCA(n_components=embedding_dim, random_state=42)
            video_combined = self.video_pca.fit_transform(video_combined)
            
        # Pad with zeros if needed
        if user_combined.shape[1] < embedding_dim:
            padding = np.zeros((user_combined.shape[0], embedding_dim - user_combined.shape[1]))
            user_combined = np.hstack([user_combined, padding])
            
        if video_combined.shape[1] < embedding_dim:
            padding = np.zeros((video_combined.shape[0], embedding_dim - video_combined.shape[1]))
            video_combined = np.hstack([video_combined, padding])
            
        # Final normalization for cosine similarity
        user_norms = np.linalg.norm(user_combined, axis=1, keepdims=True)
        user_norms[user_norms == 0] = 1  # Avoid division by zero
        self.user_vectors = user_combined / user_norms
        
        video_norms = np.linalg.norm(video_combined, axis=1, keepdims=True)
        video_norms[video_norms == 0] = 1  # Avoid division by zero
        self.video_vectors = video_combined / video_norms
        
        print(f"Created hybrid vectors: users {self.user_vectors.shape}, videos {self.video_vectors.shape}")
        return self.user_vectors, self.video_vectors
        
    def _create_user_profile_vectors(self) -> np.ndarray:
        """Create vectors from user profile features."""
        n_users = len(self.user_mapping)
        profile_features = np.zeros((n_users, 8))  # 8 profile features
        
        # Class encoding
        class_mapping = {'9': 0, '10': 1, '11': 2, '12': 3, 'mezun': 4}
        
        # Subject encoding (one-hot for top subjects)
        all_subjects = set()
        for _, row in self.user_features.iterrows():
            if pd.notna(row['top_subjects']):
                subjects = [s.strip() for s in row['top_subjects'].split(',')]
                all_subjects.update(subjects)
                
        subject_list = sorted(list(all_subjects))[:3]  # Top 3 subjects
        
        for _, row in self.user_features.iterrows():
            if row['user_id'] in self.user_mapping:
                user_idx = self.user_mapping[row['user_id']]
                
                # Class (5 categories)
                if str(row['class']) in class_mapping:
                    class_val = class_mapping[str(row['class'])]
                    profile_features[user_idx, class_val] = 1.0
                    
                # Top subjects (3 subjects)
                if pd.notna(row['top_subjects']):
                    subjects = [s.strip() for s in row['top_subjects'].split(',')]
                    for i, subject in enumerate(subject_list):
                        if subject in subjects and i < 3:
                            profile_features[user_idx, 5 + i] = 1.0
                            
        return profile_features
        
    def save_faiss_data(self, save_dir: str, embedding_dim: int = 128):
        """Save all FAISS-specific processed data."""
        faiss_dir = os.path.join(save_dir, "faiss_enhanced")
        os.makedirs(faiss_dir, exist_ok=True)
        
        print("Saving enhanced FAISS data...")
        
        # Create hybrid vectors
        user_vectors, video_vectors = self.create_hybrid_vectors(
            include_hashtags=True, 
            embedding_dim=embedding_dim
        )
        
        # Save vectors as float32 (FAISS requirement)
        np.save(os.path.join(faiss_dir, "user_vectors.npy"), user_vectors.astype(np.float32))
        np.save(os.path.join(faiss_dir, "video_vectors.npy"), video_vectors.astype(np.float32))
        
        # Save individual component vectors
        components_dir = os.path.join(faiss_dir, "components")
        os.makedirs(components_dir, exist_ok=True)
        
        if self.user_behavior_vectors is not None:
            np.save(os.path.join(components_dir, "user_behavior.npy"), 
                   self.user_behavior_vectors.astype(np.float32))
                   
        if self.video_content_vectors is not None:
            np.save(os.path.join(components_dir, "video_content.npy"), 
                   self.video_content_vectors.astype(np.float32))
                   
        # Save hashtag embeddings separately
        user_hashtag, video_hashtag, hashtag_info = self.create_hashtag_embeddings()
        np.save(os.path.join(components_dir, "user_hashtag.npy"), user_hashtag.astype(np.float32))
        np.save(os.path.join(components_dir, "video_hashtag.npy"), video_hashtag.astype(np.float32))
        
        # Save mappings and metadata
        mappings = {
            'user_mapping': self.user_mapping,
            'video_mapping': self.video_mapping,
            'reverse_user_mapping': {v: k for k, v in self.user_mapping.items()},
            'reverse_video_mapping': {v: k for k, v in self.video_mapping.items()},
            'embedding_dim': embedding_dim,
            'n_users': len(self.user_mapping),
            'n_videos': len(self.video_mapping)
        }
        
        with open(os.path.join(faiss_dir, "mappings.json"), "w") as f:
            json.dump(mappings, f, indent=2)
            
        # Save hashtag info
        with open(os.path.join(faiss_dir, "hashtag_info.json"), "w") as f:
            json.dump(hashtag_info, f, indent=2)
            
        # Save PCA models if they exist
        if self.user_pca is not None:
            import pickle
            with open(os.path.join(faiss_dir, "user_pca.pkl"), "wb") as f:
                pickle.dump(self.user_pca, f)
                
        if self.video_pca is not None:
            import pickle
            with open(os.path.join(faiss_dir, "video_pca.pkl"), "wb") as f:
                pickle.dump(self.video_pca, f)
                
        print(f"Enhanced FAISS data saved to {faiss_dir}")
        
        return {
            'user_vectors': user_vectors,
            'video_vectors': video_vectors,
            'mappings': mappings,
            'hashtag_info': hashtag_info
        }


def load_faiss_data(data_dir: str) -> Dict[str, Any]:
    """
    Load FAISS-specific processed data.
    
    Args:
        data_dir: Directory containing FAISS data
        
    Returns:
        Dictionary with loaded FAISS data
    """
    faiss_dir = os.path.join(data_dir, "faiss_enhanced")
    
    if not os.path.exists(faiss_dir):
        raise FileNotFoundError(f"FAISS data directory not found: {faiss_dir}")
        
    print("Loading FAISS data...")
    
    # Load vectors
    user_vectors = np.load(os.path.join(faiss_dir, "user_vectors.npy"))
    video_vectors = np.load(os.path.join(faiss_dir, "video_vectors.npy"))
    
    # Load mappings
    with open(os.path.join(faiss_dir, "mappings.json"), "r") as f:
        mappings = json.load(f)
        
    # Load hashtag info
    with open(os.path.join(faiss_dir, "hashtag_info.json"), "r") as f:
        hashtag_info = json.load(f)
        
    # Load component vectors if available
    components = {}
    components_dir = os.path.join(faiss_dir, "components")
    if os.path.exists(components_dir):
        for file_name in os.listdir(components_dir):
            if file_name.endswith('.npy'):
                component_name = file_name[:-4]  # Remove .npy extension
                components[component_name] = np.load(os.path.join(components_dir, file_name))
                
    faiss_data = {
        'user_vectors': user_vectors,
        'video_vectors': video_vectors,
        'mappings': mappings,
        'hashtag_info': hashtag_info,
        'components': components
    }
    
    print(f"Loaded FAISS data with {user_vectors.shape} user vectors and {video_vectors.shape} video vectors")
    return faiss_data