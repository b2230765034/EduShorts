"""
ALS-specific Data Processing Extensions
Specialized methods for Alternating Least Squares collaborative filtering

Author: Enhanced for EduScroll v05
Date: September 2025
"""

import pandas as pd
import numpy as np
from scipy import sparse
from typing import Dict, List, Tuple, Optional, Any
import json
import os


class ALSDataProcessor:
    """
    Specialized processor for ALS (Alternating Least Squares) algorithm.
    Optimized for implicit feedback collaborative filtering.
    """
    
    def __init__(self, base_processor):
        """
        Initialize ALS processor with base data processor.
        
        Args:
            base_processor: AdvancedProcessData instance
        """
        self.base_processor = base_processor
        self.df = base_processor.df
        
        # Ensure mappings are created
        if not base_processor.user_mapping:
            base_processor.create_user_item_mappings()
            
        self.user_mapping = base_processor.user_mapping
        self.video_mapping = base_processor.video_mapping
        
        # ALS-specific matrices
        self.confidence_matrix = None
        self.preference_matrix = None
        self.weighted_matrix = None
        
    def create_confidence_matrix(self, alpha: float = 40.0) -> sparse.csr_matrix:
        """
        Create confidence matrix for ALS implicit feedback.
        
        Args:
            alpha: Confidence scaling factor
            
        Returns:
            Sparse confidence matrix
        """
        if not self.user_mapping:
            self.base_processor.create_user_item_mappings()
            
        n_users = len(self.user_mapping)
        n_items = len(self.video_mapping)
        
        # Filter data to only include valid user/video IDs
        valid_data = self.df[
            (self.df['user_id'].isin(self.user_mapping.keys())) &
            (self.df['video_id'].isin(self.video_mapping.keys()))
        ].copy()
        
        if len(valid_data) == 0:
            raise ValueError("No valid user-video interactions found after filtering")
            
        print(f"Filtered data: {len(valid_data)}/{len(self.df)} valid interactions")
        
        # Map IDs to indices
        user_indices = valid_data['user_id'].map(self.user_mapping)
        item_indices = valid_data['video_id'].map(self.video_mapping)
        
        # Verify no NaN values
        if user_indices.isna().any() or item_indices.isna().any():
            raise ValueError("Found NaN values in user or item indices after mapping")
            
        # Create confidence values: 1 + alpha * rating
        # Using enhanced weight as implicit rating
        confidence_values = 1 + alpha * valid_data['weight'].values
        
        self.confidence_matrix = sparse.csr_matrix(
            (confidence_values, (user_indices, item_indices)),
            shape=(n_users, n_items)
        )
        
        print(f"Created confidence matrix: {self.confidence_matrix.shape}, "
              f"avg confidence: {confidence_values.mean():.2f}")
        
        return self.confidence_matrix
        
    def create_preference_matrix(self, threshold: float = 0.5) -> sparse.csr_matrix:
        """
        Create binary preference matrix for ALS.
        
        Args:
            threshold: Threshold for positive preference
            
        Returns:
            Binary preference matrix
        """
        if not self.user_mapping:
            self.base_processor.create_user_item_mappings()
            
        n_users = len(self.user_mapping)
        n_items = len(self.video_mapping)
        
        # Filter data to only include valid user/video IDs
        valid_data = self.df[
            (self.df['user_id'].isin(self.user_mapping.keys())) &
            (self.df['video_id'].isin(self.video_mapping.keys()))
        ].copy()
        
        if len(valid_data) == 0:
            raise ValueError("No valid user-video interactions found after filtering")
        
        # Map IDs to indices
        user_indices = valid_data['user_id'].map(self.user_mapping)
        item_indices = valid_data['video_id'].map(self.video_mapping)
        
        # Verify no NaN values
        if user_indices.isna().any() or item_indices.isna().any():
            raise ValueError("Found NaN values in user or item indices after mapping")
        
        # Create binary preferences based on multiple signals
        preferences = (
            (valid_data['watch_fraction'] >= 0.6) |
            (valid_data['quiz_correct'] == 1) |
            (valid_data['like'] == 1) |
            (valid_data['save'] == 1) |
            (valid_data['weight'] >= threshold * valid_data['weight'].max())
        ).astype(float)
        
        self.preference_matrix = sparse.csr_matrix(
            (preferences, (user_indices, item_indices)),
            shape=(n_users, n_items)
        )
        
        print(f"Created preference matrix: {self.preference_matrix.shape}, "
              f"positive rate: {preferences.mean():.3f}")
        
        return self.preference_matrix
        
    def create_weighted_interaction_matrix(self) -> sparse.csr_matrix:
        """
        Create weighted interaction matrix with time decay and engagement boost.
        
        Returns:
            Weighted interaction matrix
        """
        if not self.user_mapping:
            self.base_processor.create_user_item_mappings()
            
        n_users = len(self.user_mapping)
        n_items = len(self.video_mapping)
        
        # Filter data to only include valid user/video IDs
        valid_data = self.df[
            (self.df['user_id'].isin(self.user_mapping.keys())) &
            (self.df['video_id'].isin(self.video_mapping.keys()))
        ].copy()
        
        if len(valid_data) == 0:
            raise ValueError("No valid user-video interactions found after filtering")
        
        # Calculate time decay (more recent interactions get higher weight)
        latest_timestamp = valid_data['timestamp'].max()
        time_diff_days = (latest_timestamp - valid_data['timestamp']).dt.days
        time_decay = np.exp(-time_diff_days / 30.0)  # 30-day half-life
        
        # Enhanced weight with time decay
        weighted_values = valid_data['weight'] * time_decay
        
        # Map IDs to indices
        user_indices = valid_data['user_id'].map(self.user_mapping)
        item_indices = valid_data['video_id'].map(self.video_mapping)
        
        # Verify no NaN values
        if user_indices.isna().any() or item_indices.isna().any():
            raise ValueError("Found NaN values in user or item indices after mapping")
        
        self.weighted_matrix = sparse.csr_matrix(
            (weighted_values, (user_indices, item_indices)),
            shape=(n_users, n_items)
        )
        
        print(f"Created weighted matrix: {self.weighted_matrix.shape}, "
              f"avg weight: {weighted_values.mean():.2f}")
        
        return self.weighted_matrix
        
    def get_user_item_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics for ALS model tuning."""
        stats = {}
        
        # User statistics
        user_interactions = self.df.groupby('user_id').size()
        stats['user_stats'] = {
            'min_interactions': int(user_interactions.min()),
            'max_interactions': int(user_interactions.max()),
            'mean_interactions': float(user_interactions.mean()),
            'median_interactions': float(user_interactions.median()),
            'std_interactions': float(user_interactions.std())
        }
        
        # Item statistics
        item_interactions = self.df.groupby('video_id').size()
        stats['item_stats'] = {
            'min_interactions': int(item_interactions.min()),
            'max_interactions': int(item_interactions.max()),
            'mean_interactions': float(item_interactions.mean()),
            'median_interactions': float(item_interactions.median()),
            'std_interactions': float(item_interactions.std())
        }
        
        # Sparsity statistics
        total_possible = len(self.user_mapping) * len(self.video_mapping)
        total_observed = len(self.df)
        
        stats['sparsity_stats'] = {
            'total_users': len(self.user_mapping),
            'total_items': len(self.video_mapping),
            'total_interactions': total_observed,
            'sparsity': 1 - (total_observed / total_possible),
            'density': total_observed / total_possible
        }
        
        # Engagement statistics
        stats['engagement_stats'] = {
            'avg_watch_fraction': float(self.df['watch_fraction'].mean()),
            'avg_weight': float(self.df['weight'].mean()),
            'positive_rate': float(self.df['positive'].mean()),
            'like_rate': float(self.df['like'].mean()),
            'save_rate': float(self.df['save'].mean()),
            'quiz_success_rate': float(self.df['quiz_correct'].clip(lower=0).mean())
        }
        
        return stats
        
    def create_cold_start_matrices(self) -> Tuple[sparse.csr_matrix, sparse.csr_matrix]:
        """
        Create matrices for handling cold start users and items.
        
        Returns:
            Tuple of (warm_users_matrix, warm_items_matrix)
        """
        # Define warm users (users with >= 5 interactions)
        user_interaction_counts = self.df.groupby('user_id').size()
        warm_users = user_interaction_counts[user_interaction_counts >= 5].index
        
        # Define warm items (items with >= 3 interactions)
        item_interaction_counts = self.df.groupby('video_id').size()
        warm_items = item_interaction_counts[item_interaction_counts >= 3].index
        
        # Filter data for warm users and items
        warm_data = self.df[
            (self.df['user_id'].isin(warm_users)) & 
            (self.df['video_id'].isin(warm_items))
        ].copy()
        
        if len(warm_data) == 0:
            print("Warning: No warm users/items found with current thresholds")
            return self.create_confidence_matrix(), self.create_preference_matrix()
        
        # Create new mappings for warm users/items only
        warm_user_mapping = {user_id: idx for idx, user_id in enumerate(sorted(warm_users))}
        warm_item_mapping = {item_id: idx for idx, item_id in enumerate(sorted(warm_items))}
        
        n_warm_users = len(warm_user_mapping)
        n_warm_items = len(warm_item_mapping)
        
        # Map to new indices
        user_indices = warm_data['user_id'].map(warm_user_mapping)
        item_indices = warm_data['video_id'].map(warm_item_mapping)
        
        # Create warm confidence matrix
        confidence_values = 1 + 40.0 * warm_data['weight'].values
        warm_confidence_matrix = sparse.csr_matrix(
            (confidence_values, (user_indices, item_indices)),
            shape=(n_warm_users, n_warm_items)
        )
        
        # Create warm preference matrix
        preferences = warm_data['positive'].values.astype(float)
        warm_preference_matrix = sparse.csr_matrix(
            (preferences, (user_indices, item_indices)),
            shape=(n_warm_users, n_warm_items)
        )
        
        print(f"Created warm matrices: {n_warm_users} users, {n_warm_items} items")
        print(f"Warm data coverage: {len(warm_data) / len(self.df):.3f}")
        
        # Save warm mappings for later use
        self.warm_user_mapping = warm_user_mapping
        self.warm_item_mapping = warm_item_mapping
        
        return warm_confidence_matrix, warm_preference_matrix
        
    def prepare_train_test_split(self, test_ratio: float = 0.2, 
                               temporal_split: bool = True) -> Dict[str, Any]:
        """
        Prepare train/test split for ALS evaluation.
        
        Args:
            test_ratio: Ratio of data for testing
            temporal_split: Whether to use temporal splitting
            
        Returns:
            Dictionary with train/test matrices and metadata
        """
        if temporal_split:
            # Split by time (latest interactions for testing)
            split_time = self.df['timestamp'].quantile(1 - test_ratio)
            train_data = self.df[self.df['timestamp'] <= split_time].copy()
            test_data = self.df[self.df['timestamp'] > split_time].copy()
        else:
            # Random split
            test_data = self.df.sample(frac=test_ratio, random_state=42)
            train_data = self.df.drop(test_data.index)
            
        # Ensure we have consistent user/item mappings
        all_users = set(train_data['user_id']) | set(test_data['user_id'])
        all_items = set(train_data['video_id']) | set(test_data['video_id'])
        
        user_mapping = {user_id: idx for idx, user_id in enumerate(sorted(all_users))}
        item_mapping = {item_id: idx for idx, item_id in enumerate(sorted(all_items))}
        
        n_users = len(user_mapping)
        n_items = len(item_mapping)
        
        # Create train matrices
        train_user_indices = train_data['user_id'].map(user_mapping)
        train_item_indices = train_data['video_id'].map(item_mapping)
        train_confidence = 1 + 40.0 * train_data['weight'].values
        train_preferences = train_data['positive'].values.astype(float)
        
        train_confidence_matrix = sparse.csr_matrix(
            (train_confidence, (train_user_indices, train_item_indices)),
            shape=(n_users, n_items)
        )
        
        train_preference_matrix = sparse.csr_matrix(
            (train_preferences, (train_user_indices, train_item_indices)),
            shape=(n_users, n_items)
        )
        
        # Create test matrices
        test_user_indices = test_data['user_id'].map(user_mapping)
        test_item_indices = test_data['video_id'].map(item_mapping)
        test_preferences = test_data['positive'].values.astype(float)
        
        test_matrix = sparse.csr_matrix(
            (test_preferences, (test_user_indices, test_item_indices)),
            shape=(n_users, n_items)
        )
        
        split_data = {
            'train_confidence': train_confidence_matrix,
            'train_preference': train_preference_matrix,
            'test_matrix': test_matrix,
            'user_mapping': user_mapping,
            'item_mapping': item_mapping,
            'train_size': len(train_data),
            'test_size': len(test_data),
            'split_method': 'temporal' if temporal_split else 'random'
        }
        
        print(f"Created train/test split:")
        print(f"- Train: {len(train_data)} interactions")
        print(f"- Test: {len(test_data)} interactions")
        print(f"- Split method: {split_data['split_method']}")
        
        return split_data
        
    def save_als_data(self, save_dir: str):
        """Save all ALS-specific processed data."""
        als_dir = os.path.join(save_dir, "als_enhanced")
        os.makedirs(als_dir, exist_ok=True)
        
        print("Saving enhanced ALS data...")
        
        # Create all matrices
        confidence_matrix = self.create_confidence_matrix()
        preference_matrix = self.create_preference_matrix()
        weighted_matrix = self.create_weighted_interaction_matrix()
        
        # Save main matrices
        sparse.save_npz(os.path.join(als_dir, "confidence_matrix.npz"), confidence_matrix)
        sparse.save_npz(os.path.join(als_dir, "preference_matrix.npz"), preference_matrix)
        sparse.save_npz(os.path.join(als_dir, "weighted_matrix.npz"), weighted_matrix)
        
        # Create and save cold start matrices
        warm_conf, warm_pref = self.create_cold_start_matrices()
        sparse.save_npz(os.path.join(als_dir, "warm_confidence_matrix.npz"), warm_conf)
        sparse.save_npz(os.path.join(als_dir, "warm_preference_matrix.npz"), warm_pref)
        
        # Save mappings
        mappings = {
            'user_mapping': self.user_mapping,
            'video_mapping': self.video_mapping,
            'reverse_user_mapping': {v: k for k, v in self.user_mapping.items()},
            'reverse_video_mapping': {v: k for k, v in self.video_mapping.items()}
        }
        
        if hasattr(self, 'warm_user_mapping'):
            mappings['warm_user_mapping'] = self.warm_user_mapping
            mappings['warm_item_mapping'] = self.warm_item_mapping
            
        with open(os.path.join(als_dir, "mappings.json"), "w") as f:
            json.dump(mappings, f, indent=2)
            
        # Save statistics
        stats = self.get_user_item_statistics()
        with open(os.path.join(als_dir, "statistics.json"), "w") as f:
            json.dump(stats, f, indent=2)
            
        # Create and save train/test split
        split_data = self.prepare_train_test_split()
        
        sparse.save_npz(os.path.join(als_dir, "train_confidence.npz"), split_data['train_confidence'])
        sparse.save_npz(os.path.join(als_dir, "train_preference.npz"), split_data['train_preference'])
        sparse.save_npz(os.path.join(als_dir, "test_matrix.npz"), split_data['test_matrix'])
        
        split_metadata = {k: v for k, v in split_data.items() 
                         if k not in ['train_confidence', 'train_preference', 'test_matrix']}
        
        with open(os.path.join(als_dir, "split_metadata.json"), "w") as f:
            json.dump(split_metadata, f, indent=2)
            
        print(f"Enhanced ALS data saved to {als_dir}")
        
        return {
            'confidence_matrix': confidence_matrix,
            'preference_matrix': preference_matrix,
            'weighted_matrix': weighted_matrix,
            'statistics': stats,
            'split_data': split_data
        }


def load_als_data(data_dir: str) -> Dict[str, Any]:
    """
    Load ALS-specific processed data.
    
    Args:
        data_dir: Directory containing ALS data
        
    Returns:
        Dictionary with loaded ALS data
    """
    als_dir = os.path.join(data_dir, "als_enhanced")
    
    if not os.path.exists(als_dir):
        raise FileNotFoundError(f"ALS data directory not found: {als_dir}")
        
    print("Loading ALS data...")
    
    # Load matrices
    confidence_matrix = sparse.load_npz(os.path.join(als_dir, "confidence_matrix.npz"))
    preference_matrix = sparse.load_npz(os.path.join(als_dir, "preference_matrix.npz"))
    weighted_matrix = sparse.load_npz(os.path.join(als_dir, "weighted_matrix.npz"))
    
    # Load mappings
    with open(os.path.join(als_dir, "mappings.json"), "r") as f:
        mappings = json.load(f)
        
    # Load statistics
    with open(os.path.join(als_dir, "statistics.json"), "r") as f:
        statistics = json.load(f)
        
    # Load train/test data if available
    train_test_data = {}
    if os.path.exists(os.path.join(als_dir, "train_confidence.npz")):
        train_test_data['train_confidence'] = sparse.load_npz(os.path.join(als_dir, "train_confidence.npz"))
        train_test_data['train_preference'] = sparse.load_npz(os.path.join(als_dir, "train_preference.npz"))
        train_test_data['test_matrix'] = sparse.load_npz(os.path.join(als_dir, "test_matrix.npz"))
        
        with open(os.path.join(als_dir, "split_metadata.json"), "r") as f:
            train_test_data.update(json.load(f))
            
    als_data = {
        'confidence_matrix': confidence_matrix,
        'preference_matrix': preference_matrix,
        'weighted_matrix': weighted_matrix,
        'mappings': mappings,
        'statistics': statistics,
        'train_test_data': train_test_data
    }
    
    print(f"Loaded ALS data with {confidence_matrix.shape} matrix")
    return als_data