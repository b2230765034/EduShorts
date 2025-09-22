"""
Enhanced Recommendation Engine V2 with FAISS Integration

Bu versiyon mevcut enhanced_recommendation_engine.py'a FAISS content-based
similarity search entegre eder. Hybrid approach ile cold-start problemini çözer.

Yeni Özellikler:
1. FAISS Content Engine entegrasyonu
2. Cold-start için instant recommendations
3. Hybrid recommendation (FAISS + Content + ALS)
4. Açıklanabilir öneriler
5. User preference bazlı similarity
"""

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'  # OpenBLAS thread uyarısı için

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from faiss_content_engine import FAISSContentEngine
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging
from collections import defaultdict
from scipy.sparse import csr_matrix, coo_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ALS için import
try:
    from implicit.als import AlternatingLeastSquares
    from implicit.nearest_neighbours import ItemItemRecommender
    ALS_AVAILABLE = True
except ImportError:
    print("⚠️ Warning: implicit library not found. ALS will not be available.")
    ALS_AVAILABLE = False

class EnhancedRecommendationEngineV2:
    """
    FAISS entegrasyonlu enhanced recommendation engine
    Hybrid approach: FAISS + Content-based + Collaborative filtering
    """
    
    def __init__(self, user_interactions_path: str, user_features_path: str, video_features_path: str):
        # File paths
        self.user_interactions_path = user_interactions_path
        self.user_features_path = user_features_path
        self.video_features_path = video_features_path
        
        # Data storage
        self.df = None
        self.video_features = None
        self.user_features = None
        
        # FAISS engine
        self.faiss_engine = FAISSContentEngine(embedding_dim=128)
        self.faiss_ready = False
        
        # ALS engine
        self.als_model = None
        self.als_ready = False
        self.user_item_matrix = None
        self.user_id_map = {}
        self.video_id_map = {}
        self.reverse_user_map = {}
        self.reverse_video_map = {}
        
        # Content-based components
        self.video_profiles = None
        self.tfidf_vectorizer = None
        self.content_similarity_matrix = None
        
        # User features for FAISS
        self.user_features_df = None
        
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        
        print("Enhanced Recommendation Engine V2 with FAISS + ALS initialized!")
    
    def load_and_enhance_data(self):
        """
        Load data, apply feature engineering ve FAISS + ALS index build et
        """
        print("Loading and enhancing data with FAISS + ALS integration...")
        
        try:
            # Load raw data
            self.df = pd.read_csv(self.user_interactions_path)
            self.video_features = pd.read_csv(self.video_features_path)
            self.user_features = pd.read_csv(self.user_features_path)
            
            print(f"✅ Loaded {len(self.df)} interactions, {len(self.user_features)} users, {len(self.video_features)} videos")
            
            # Apply feature engineering - add weight and positive columns
            self.df['weight'] = (
                1
                + 4 * self.df['watch_fraction']
                + 2 * self.df['quiz_correct'].clip(lower=0)
                + 1 * self.df['like']
                + 0.5 * self.df['save']
            )

            self.df['positive'] = (
                (self.df['watch_fraction'] >= 0.6)
                | (self.df['quiz_correct'] == 1)
                | (self.df['like'] == 1)
            ).astype(int)
            
            # Convert timestamp
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
            
            # Load user features for FAISS
            self.user_features_df = self.user_features.copy()
            
            # Build FAISS index
            self.build_faiss_index()
            
            # Build ALS model
            if ALS_AVAILABLE:
                self.build_als_model()
            else:
                print("⚠️ ALS not available, skipping ALS model building")
            
            # Build content-based components
            self.build_content_similarity()
            
            print("✅ Data enhancement with FAISS + ALS + Content completed!")
            
        except Exception as e:
            print(f"❌ Data loading failed: {e}")
            self.faiss_ready = False
            self.als_ready = False
    
    def build_faiss_index(self):
        """
        FAISS index'i build et
        """
        try:
            self.logger.info("Building FAISS index...")
            
            # Video dataframe hazırla
            video_df = self.video_features.copy()
            if 'tags' not in video_df.columns and 'video_hashtag' in video_df.columns:
                video_df['tags'] = video_df['video_hashtag']
            elif 'tags' not in video_df.columns:
                # Varsayılan tags oluştur
                video_df['tags'] = '#genel#10#tyt'
            
            # User dataframe
            user_df = self.user_features_df
            
            # FAISS index build
            self.faiss_engine.build_index(video_df, user_df)
            self.faiss_ready = True
            
            self.logger.info("✅ FAISS index built successfully")
            
        except Exception as e:
            self.logger.error(f"❌ FAISS index building failed: {e}")
            self.faiss_ready = False
    
    def get_faiss_recommendations(self, user_id: str, k: int = 10, exclude_videos: List[str] = None) -> List[str]:
        """
        FAISS tabanlı öneri
        
        Args:
            user_id: Kullanıcı ID'si
            k: Öneri sayısı
            exclude_videos: Hariç tutulacak video ID'leri
            
        Returns:
            List[str]: FAISS önerileri
        """
        if not self.faiss_ready:
            self.logger.warning("FAISS not ready, trying to build index...")
            self.build_faiss_index()
        
        if not self.faiss_ready:
            return []
        
        return self.faiss_engine.recommend(user_id, top_k=k, exclude_videos=exclude_videos)
    
    def build_als_model(self):
        """
        ALS collaborative filtering model build et
        """
        try:
            if not ALS_AVAILABLE:
                self.logger.warning("ALS not available, skipping ALS model building")
                return
                
            self.logger.info("Building ALS model...")
            
            # User-item matrix oluştur - String sorting yerine numeric sorting
            user_ids = sorted(self.df['user_id'].unique(), key=lambda x: int(x.split('_')[1]))
            video_ids = sorted(self.df['video_id'].unique(), key=lambda x: int(x.split('_')[1]))
            
            # ID mapping oluştur
            self.user_id_map = {user: idx for idx, user in enumerate(user_ids)}
            self.video_id_map = {video: idx for idx, video in enumerate(video_ids)}
            self.reverse_user_map = {idx: user for user, idx in self.user_id_map.items()}
            self.reverse_video_map = {idx: video for video, idx in self.video_id_map.items()}
            
            # Sparse matrix oluştur - user x item format (implicit library için)
            rows = [self.user_id_map[u] for u in self.df['user_id']]
            cols = [self.video_id_map[v] for v in self.df['video_id']]
            data = self.df['weight'].astype(float).tolist()
            
            # COO matrix oluştur ve CSR'ye çevir
            self.user_item_matrix = coo_matrix(
                (data, (rows, cols)), 
                shape=(len(user_ids), len(video_ids))
            ).tocsr()
            
            print(f"✅ User-item matrix created: {self.user_item_matrix.shape}")
            print(f"Matrix density: {self.user_item_matrix.nnz / (self.user_item_matrix.shape[0] * self.user_item_matrix.shape[1]) * 100:.4f}%")
            
            # ALS model eğit
            self.als_model = AlternatingLeastSquares(
                factors=64,
                regularization=0.01,
                iterations=20,
                random_state=42
            )
            
            self.als_model.fit(self.user_item_matrix)
            self.als_ready = True
            
            self.logger.info("✅ ALS model built successfully")
            
        except Exception as e:
            self.logger.error(f"❌ ALS model building failed: {e}")
            self.als_ready = False
    
    def get_als_recommendations(self, user_id: str, k: int = 10, exclude_videos: List[str] = None) -> List[str]:
        """
        ALS tabanlı öneri - verdiğiniz kod yapısına uygun
        """
        if not self.als_ready or user_id not in self.user_id_map:
            return []
        
        try:
            user_idx = self.user_id_map[user_id]
            
            # user_items vektörünü CSR sparse matrix olarak al
            # Matrix is (user x item), so we can directly index by user
            user_items = self.user_item_matrix[user_idx].tocsr()
            
            # ALS recommend method - returns (item_indices_array, scores_array)
            recommended = self.als_model.recommend(user_idx, user_items, N=k + 20)
            
            # Tuple unpacking - ALS returns (indices, scores)
            item_indices, scores = recommended
            
            # Video ID'lerine çevir
            recommended_videos = []
            for item_idx in item_indices:
                item_idx = int(item_idx)
                if item_idx in self.reverse_video_map:
                    video_id = self.reverse_video_map[item_idx]
                    
                    # Exclusion kontrolü
                    if exclude_videos and video_id in exclude_videos:
                        continue
                        
                    recommended_videos.append(video_id)
                    
                    if len(recommended_videos) >= k:
                        break
                else:
                    self.logger.warning(f"item_idx {item_idx} not found in reverse_video_map")
            
            return recommended_videos
            
        except Exception as e:
            self.logger.error(f"ALS recommendation failed for {user_id}: {e}")
            return []
    
    def build_content_similarity(self):
        """
        Content-based similarity matrix build et
        """
        try:
            self.logger.info("Building content similarity matrix...")
            
            # Video özelliklerini text olarak birleştir
            video_texts = []
            for _, video in self.video_features.iterrows():
                # Video özellikleri var mı kontrol et
                text_features = []
                
                if 'video_hashtag' in video and pd.notna(video['video_hashtag']):
                    text_features.append(str(video['video_hashtag']))
                elif 'tags' in video and pd.notna(video['tags']):
                    text_features.append(str(video['tags']))
                
                # Diğer metin özelliklerini ekle
                for col in ['title', 'description', 'category']:
                    if col in video and pd.notna(video[col]):
                        text_features.append(str(video[col]))
                
                video_text = ' '.join(text_features) if text_features else 'unknown'
                video_texts.append(video_text)
            
            # TF-IDF vectorizer
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            # TF-IDF matrix
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(video_texts)
            
            # Cosine similarity matrix
            self.content_similarity_matrix = cosine_similarity(tfidf_matrix)
            
            self.logger.info(f"✅ Content similarity matrix built: {self.content_similarity_matrix.shape}")
            
        except Exception as e:
            self.logger.error(f"❌ Content similarity building failed: {e}")
            self.content_similarity_matrix = None
    
    def get_content_based_recommendations(self, user_id: str, k: int = 10, exclude_videos: List[str] = None) -> List[str]:
        """
        Content-based öneriler
        """
        if self.content_similarity_matrix is None:
            return []
        
        try:
            # Kullanıcının geçmiş etkileşimlerini al
            user_interactions = self.df[self.df['user_id'] == user_id]
            if user_interactions.empty:
                return []
            
            # Video ID'lerini index'e çevir
            video_id_to_idx = {video_id: idx for idx, video_id in enumerate(self.video_features['video_id'])}
            
            # Kullanıcının sevdiği videoları bul (pozitif etkileşimler)
            if 'positive' in user_interactions.columns:
                liked_videos = user_interactions[user_interactions['positive'] == 1]['video_id'].tolist()
            else:
                # Eğer positive column yoksa tüm etkileşimleri pozitif kabul et
                liked_videos = user_interactions['video_id'].tolist()
            
            # Content-based scoring
            video_scores = defaultdict(float)
            
            for liked_video in liked_videos:
                if liked_video in video_id_to_idx:
                    liked_idx = video_id_to_idx[liked_video]
                    
                    # Bu videonun benzer videolarını bul
                    similarities = self.content_similarity_matrix[liked_idx]
                    
                    for video_idx, similarity in enumerate(similarities):
                        candidate_video = self.video_features.iloc[video_idx]['video_id']
                        
                        # Kendisi ve zaten izlenenler hariç
                        watched_videos = set(user_interactions['video_id'].tolist())
                        if candidate_video != liked_video and candidate_video not in watched_videos:
                            video_scores[candidate_video] += similarity
            
            # En iyi skorları sırala
            sorted_videos = sorted(video_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Exclusion uygula
            recommendations = []
            for video_id, score in sorted_videos:
                if exclude_videos and video_id in exclude_videos:
                    continue
                recommendations.append(video_id)
                if len(recommendations) >= k:
                    break
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Content-based recommendation failed for {user_id}: {e}")
            return []
    
    def get_hybrid_recommendations(self, user_id: str, k: int = 10, weights: Dict[str, float] = None) -> List[str]:
        """
        Hybrid öneriler (FAISS + ALS + Content-based)
        """
        if weights is None:
            # Varsayılan ağırlıklar
            weights = {
                'faiss': 0.4,
                'als': 0.35,
                'content': 0.25
            }
        
        # Her yöntemden öneri al
        faiss_recs = self.get_faiss_recommendations(user_id, k=k*2)
        als_recs = self.get_als_recommendations(user_id, k=k*2) if self.als_ready else []
        content_recs = self.get_content_based_recommendations(user_id, k=k*2)
        
        # Scoring combine et
        video_scores = defaultdict(float)
        
        # FAISS skorları
        for idx, video_id in enumerate(faiss_recs):
            score = (len(faiss_recs) - idx) / len(faiss_recs)  # Rank-based scoring
            video_scores[video_id] += weights['faiss'] * score
        
        # ALS skorları
        for idx, video_id in enumerate(als_recs):
            score = (len(als_recs) - idx) / len(als_recs)
            video_scores[video_id] += weights['als'] * score
        
        # Content skorları
        for idx, video_id in enumerate(content_recs):
            score = (len(content_recs) - idx) / len(content_recs)
            video_scores[video_id] += weights['content'] * score
        
        # En iyi skorları sırala
        sorted_videos = sorted(video_scores.items(), key=lambda x: x[1], reverse=True)
        
        return [video_id for video_id, score in sorted_videos[:k]]
    
    def get_cold_start_recommendations(self, user_id: str, k: int = 10) -> List[str]:
        """
        Cold start users için recommendation
        Öncelik: FAISS (user_features.csv'deki top_subjects kullanarak)
        
        Args:
            user_id: Kullanıcı ID'si
            k: Öneri sayısı
            
        Returns:
            List[str]: Cold start önerileri
        """
        # Kullanıcının etkileşim sayısını kontrol et
        user_interaction_count = 0
        if hasattr(self, 'df') and self.df is not None:
            user_interaction_count = self.df[self.df['user_id'] == user_id].shape[0]
        
        is_cold_start = user_interaction_count <= 2  # Cold start threshold
        
        if is_cold_start:
            self.logger.info(f"Cold start recommendation for {user_id} (interactions: {user_interaction_count})")
            
            # FAISS'i öncelikle kullan (user_features.csv'den profile building)
            if self.faiss_ready:
                faiss_recs = self.get_faiss_recommendations(user_id, k=k)
                if faiss_recs:
                    self.logger.info(f"FAISS provided {len(faiss_recs)} cold start recommendations")
                    return faiss_recs
            
            # FAISS yoksa fallback: content-based
            content_recs = self.get_content_based_recommendations(user_id, k=k)
            self.logger.info(f"Content-based fallback provided {len(content_recs)} recommendations")
            return content_recs
        
        # Normal kullanıcı için hybrid approach
        return self.get_hybrid_advanced_recommendations(user_id, k=k)
    
    def get_hybrid_advanced_recommendations(self, user_id: str, k: int = 10) -> List[str]:
        """
        Hybrid recommendation: FAISS + Content-based + ALS
        
        Args:
            user_id: Kullanıcı ID'si
            k: Öneri sayısı
            
        Returns:
            List[str]: Hybrid öneriler
        """
        recommendations = {}
        
        # Kullanıcının izlediği videolar (exclude için)
        user_videos = set()
        if hasattr(self, 'df') and self.df is not None:
            user_videos = set(self.df[self.df['user_id'] == user_id]['video_id'].tolist())
        
        exclude_list = list(user_videos)
        
        # 1. FAISS recommendations (40% weight)
        try:
            faiss_recs = self.get_faiss_recommendations(user_id, k=k*2, exclude_videos=exclude_list)
            for i, video_id in enumerate(faiss_recs[:k]):
                score = (k - i) / k * 0.4  # Decreasing score with rank
                recommendations[video_id] = recommendations.get(video_id, 0) + score
                
            self.logger.info(f"FAISS contributed {len(faiss_recs)} recommendations")
        except Exception as e:
            self.logger.warning(f"FAISS recommendation failed: {e}")
        
        # 2. Content-based recommendations (35% weight)
        try:
            content_recs = self.get_content_based_recommendations(user_id, k=k*2)
            # Exclude watched videos
            content_recs = [vid for vid in content_recs if vid not in exclude_list]
            
            for i, video_id in enumerate(content_recs[:k]):
                score = (k - i) / k * 0.35
                recommendations[video_id] = recommendations.get(video_id, 0) + score
                
            self.logger.info(f"Content-based contributed {len(content_recs)} recommendations")
        except Exception as e:
            self.logger.warning(f"Content-based recommendation failed: {e}")
        
        # 3. ALS collaborative filtering (25% weight) - eğer mevcut ise
        try:
            if hasattr(self, 'als_model') and self.als_model is not None:
                als_recs = self.get_als_recommendations(user_id, k=k*2)
                for i, video_id in enumerate(als_recs[:k]):
                    if video_id not in exclude_list:
                        score = (k - i) / k * 0.25
                        recommendations[video_id] = recommendations.get(video_id, 0) + score
                        
                self.logger.info(f"ALS contributed {len(als_recs)} recommendations")
        except Exception as e:
            self.logger.warning(f"ALS recommendation failed: {e}")
        
        # Sonuçları score'a göre sırala
        sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        final_recs = [video_id for video_id, _ in sorted_recommendations[:k]]
        
        self.logger.info(f"Hybrid approach generated {len(final_recs)} final recommendations")
        return final_recs
    
    def explain_recommendation(self, user_id: str, video_id: str) -> Dict:
        """
        Öneri açıklaması (multiple sources)
        
        Args:
            user_id: Kullanıcı ID'si
            video_id: Video ID'si
            
        Returns:
            Dict: Açıklama bilgileri
        """
        explanation = {
            'user_id': user_id,
            'video_id': video_id,
            'sources': {},
            'overall_score': 0.0
        }
        
        # FAISS explanation
        if self.faiss_ready:
            try:
                faiss_explanation = self.faiss_engine.explain_recommendation(user_id, video_id)
                explanation['sources']['faiss'] = faiss_explanation
                
                if 'similarity_score' in faiss_explanation:
                    explanation['overall_score'] += faiss_explanation['similarity_score'] * 0.4
                    
            except Exception as e:
                self.logger.warning(f"FAISS explanation failed: {e}")
        
        # Content-based explanation
        try:
            # Kullanıcı profili
            if hasattr(self, 'enhanced_user_features') and self.enhanced_user_features is not None:
                user_profile = self.enhanced_user_features[
                    self.enhanced_user_features['user_id'] == user_id
                ]
                
                if not user_profile.empty:
                    profile = user_profile.iloc[0]
                    preferred_subjects = profile.get('preferred_subjects', [])
                    
                    # Video hashtags
                    video_hashtags = []
                    if hasattr(self, 'enhanced_video_features') and self.enhanced_video_features is not None:
                        video_row = self.enhanced_video_features[
                            self.enhanced_video_features['video_id'] == video_id
                        ]
                        if not video_row.empty:
                            hashtag = video_row.iloc[0].get('hashtag', '')
                            video_hashtags = [tag.strip('#') for tag in str(hashtag).split('#') if tag.strip('#')]
                    
                    # Subject match
                    subject_match = len(set(preferred_subjects) & set(video_hashtags))
                    
                    explanation['sources']['content_based'] = {
                        'subject_matches': subject_match,
                        'preferred_subjects': preferred_subjects,
                        'video_hashtags': video_hashtags,
                        'grade_level': profile.get('grade_level'),
                        'target_exam': profile.get('target_exam')
                    }
                    
                    # Content-based score contribution
                    explanation['overall_score'] += subject_match * 0.35 / 3  # Normalize by max possible matches
                    
        except Exception as e:
            self.logger.warning(f"Content-based explanation failed: {e}")
        
        return explanation
    
    def get_recommendation_metrics(self, user_id: str, recommended_videos: List[str]) -> Dict:
        """
        Öneri kalitesi metrikleri
        
        Args:
            user_id: Kullanıcı ID'si
            recommended_videos: Önerilen video listesi
            
        Returns:
            Dict: Kalite metrikleri
        """
        metrics = {
            'total_recommendations': len(recommended_videos),
            'faiss_coverage': 0,
            'content_coverage': 0,
            'diversity_score': 0,
            'avg_similarity': 0,
            'cold_start_user': False
        }
        
        if not recommended_videos:
            return metrics
        
        # Cold start check
        user_interaction_count = 0
        if hasattr(self, 'df') and self.df is not None:
            user_interaction_count = self.df[self.df['user_id'] == user_id].shape[0]
        metrics['cold_start_user'] = user_interaction_count <= 2
        
        # FAISS coverage
        if self.faiss_ready:
            try:
                faiss_recs = self.get_faiss_recommendations(user_id, k=20)
                faiss_overlap = len(set(recommended_videos) & set(faiss_recs))
                metrics['faiss_coverage'] = faiss_overlap / len(recommended_videos)
            except:
                pass
        
        # Content coverage
        try:
            content_recs = self.get_content_based_recommendations(user_id, k=20)
            content_overlap = len(set(recommended_videos) & set(content_recs))
            metrics['content_coverage'] = content_overlap / len(recommended_videos)
        except:
            pass
        
        # Diversity score (hashtag diversity)
        all_hashtags = set()
        if hasattr(self, 'enhanced_video_features') and self.enhanced_video_features is not None:
            for video_id in recommended_videos:
                video_row = self.enhanced_video_features[
                    self.enhanced_video_features['video_id'] == video_id
                ]
                if not video_row.empty:
                    hashtag = video_row.iloc[0].get('hashtag', '')
                    video_hashtags = [tag.strip('#') for tag in str(hashtag).split('#') if tag.strip('#')]
                    all_hashtags.update(video_hashtags)
        
        metrics['diversity_score'] = len(all_hashtags) / len(recommended_videos) if recommended_videos else 0
        
        # Average similarity (if FAISS available)
        if self.faiss_ready and len(recommended_videos) > 1:
            similarities = []
            for video_id in recommended_videos:
                try:
                    similarity = self.faiss_engine.get_user_video_similarity(user_id, video_id)
                    if similarity > 0:
                        similarities.append(similarity)
                except:
                    pass
            
            metrics['avg_similarity'] = np.mean(similarities) if similarities else 0
        
        return metrics
    
    def evaluate_faiss_model(self, test_df: pd.DataFrame, k: int = 10) -> Dict:
        """
        FAISS model evaluation
        
        Args:
            test_df: Test dataframe
            k: Evaluation k value
            
        Returns:
            Dict: Evaluation metrics
        """
        print(f"Evaluating FAISS recommendations with k={k}...")
        
        if not self.faiss_ready:
            return {'error': 'FAISS not ready'}
        
        # Get test users with positive interactions
        test_users = test_df[test_df['positive'] == 1]['user_id'].unique()[:50]  # Limit for speed
        
        precisions = []
        recalls = []
        cold_start_performance = []
        
        for user_id in test_users:
            # Get actual positive videos from test set
            actual_videos = set(test_df[(test_df['user_id'] == user_id) & (test_df['positive'] == 1)]['video_id'].tolist())
            
            if not actual_videos:
                continue
            
            # Check if cold start user
            train_interactions = self.df[self.df['user_id'] == user_id].shape[0] if hasattr(self, 'df') else 0
            is_cold_start = train_interactions <= 2
            
            # Get FAISS recommendations
            try:
                recommended_videos = self.get_faiss_recommendations(user_id, k=k)
            except:
                continue
            
            if not recommended_videos:
                continue
            
            # Calculate metrics
            recommended_set = set(recommended_videos)
            hits = len(recommended_set & actual_videos)
            
            precision = hits / len(recommended_videos) if recommended_videos else 0
            recall = hits / len(actual_videos) if actual_videos else 0
            
            precisions.append(precision)
            recalls.append(recall)
            
            if is_cold_start:
                cold_start_performance.append(precision)
        
        results = {
            'method': 'FAISS',
            'precision@{}'.format(k): np.mean(precisions) if precisions else 0,
            'recall@{}'.format(k): np.mean(recalls) if recalls else 0,
            'evaluated_users': len(precisions),
            'cold_start_precision': np.mean(cold_start_performance) if cold_start_performance else 0,
            'cold_start_users': len(cold_start_performance)
        }
        
        print(f"FAISS Results:")
        print(f"  Precision@{k}: {results['precision@{}'.format(k)]:.4f}")
        print(f"  Recall@{k}: {results['recall@{}'.format(k)]:.4f}")
        print(f"  Cold Start Precision@{k}: {results['cold_start_precision']:.4f}")
        print(f"  Evaluated on {results['evaluated_users']} users ({results['cold_start_users']} cold start)")
        
        return results
    
    def get_comprehensive_recommendations(self, user_id: str, k: int = 10, strategy: str = 'auto') -> Dict:
        """
        Comprehensive recommendation with multiple strategies
        
        Args:
            user_id: Kullanıcı ID'si
            k: Öneri sayısı
            strategy: 'auto', 'faiss', 'als', 'content', 'hybrid'
            
        Returns:
            Dict: Recommendations with metadata
        """
        # Determine strategy
        if strategy == 'auto':
            # Auto-detect based on user interaction history
            user_interaction_count = 0
            if hasattr(self, 'df') and self.df is not None:
                user_interaction_count = self.df[self.df['user_id'] == user_id].shape[0]
            
            if user_interaction_count <= 2:
                strategy_used = 'faiss_cold_start'
                recommendations = self.get_faiss_recommendations(user_id, k=k)
            elif user_interaction_count <= 10:
                strategy_used = 'hybrid_light'
                recommendations = self.get_hybrid_recommendations(user_id, k=k, weights={'faiss': 0.5, 'als': 0.3, 'content': 0.2})
            else:
                strategy_used = 'hybrid_full'
                recommendations = self.get_hybrid_recommendations(user_id, k=k)
        
        elif strategy == 'faiss':
            strategy_used = 'faiss'
            recommendations = self.get_faiss_recommendations(user_id, k=k)
        
        elif strategy == 'als':
            strategy_used = 'als'
            recommendations = self.get_als_recommendations(user_id, k=k) if self.als_ready else []
        
        elif strategy == 'content':
            strategy_used = 'content'
            recommendations = self.get_content_based_recommendations(user_id, k=k)
        
        elif strategy == 'hybrid':
            strategy_used = 'hybrid'
            recommendations = self.get_hybrid_recommendations(user_id, k=k)
        
        else:
            strategy_used = 'faiss_fallback'
            recommendations = self.get_faiss_recommendations(user_id, k=k)
        
        # Enhanced metrics
        metrics = {
            'cold_start_user': len(recommendations) == 0 or strategy_used.endswith('cold_start'),
            'recommendation_count': len(recommendations),
            'diversity_score': self._calculate_diversity_score(recommendations),
            'confidence_score': self._calculate_confidence_score(user_id, recommendations, strategy_used),
            'coverage_score': len(recommendations) / k if k > 0 else 0,
            'engines_used': self._get_engines_used(strategy_used)
        }
        
        # Enhanced explanations for top recommendations
        explanations = []
        for video_id in recommendations[:3]:  # Top 3 explanations
            try:
                explanation = self.explain_recommendation(user_id, video_id)
                explanations.append(explanation)
            except:
                explanations.append({
                    'video_id': video_id, 
                    'similarity_score': 0.5, 
                    'reason': f'{strategy_used} recommendation'
                })
        
        return {
            'user_id': user_id,
            'strategy_used': strategy_used,
            'recommendations': recommendations,
            'metrics': metrics,
            'explanations': explanations,
            'total_count': len(recommendations)
        }
    
    def _calculate_diversity_score(self, recommendations: List[str]) -> float:
        """
        Önerilerin çeşitlilik skorunu hesapla
        """
        if not recommendations or len(recommendations) < 2:
            return 0.0
        
        # Video hashtag çeşitliliği
        unique_tags = set()
        for video_id in recommendations:
            if hasattr(self, 'video_features'):
                video_row = self.video_features[self.video_features['video_id'] == video_id]
                if not video_row.empty:
                    hashtag = video_row.iloc[0].get('video_hashtag', '') or video_row.iloc[0].get('tags', '')
                    if hashtag:
                        tags = [tag.strip() for tag in str(hashtag).split('#') if tag.strip()]
                        unique_tags.update(tags)
        
        # Çeşitlilik = unique tags / total recommendations
        return min(len(unique_tags) / len(recommendations), 1.0)
    
    def _calculate_confidence_score(self, user_id: str, recommendations: List[str], strategy: str) -> float:
        """
        Önerilerin güven skorunu hesapla
        """
        base_score = 0.5
        
        # Kullanıcı etkileşim geçmişine göre
        if hasattr(self, 'df') and self.df is not None:
            user_history = len(self.df[self.df['user_id'] == user_id])
            history_boost = min(user_history / 20, 0.3)  # Max 0.3 boost
            base_score += history_boost
        
        # Strateji tipine göre
        strategy_boosts = {
            'hybrid_full': 0.2,
            'hybrid_light': 0.15,
            'faiss': 0.1,
            'als': 0.1,
            'content': 0.05
        }
        
        base_score += strategy_boosts.get(strategy, 0)
        
        # Öneri sayısına göre
        if len(recommendations) >= 5:
            base_score += 0.1
        
        return min(base_score, 1.0)
    
    def _get_engines_used(self, strategy: str) -> List[str]:
        """
        Kullanılan motorları döndür
        """
        if 'hybrid' in strategy:
            engines = ['faiss']
            if self.als_ready:
                engines.append('als')
            if self.content_similarity_matrix is not None:
                engines.append('content')
            return engines
        elif strategy == 'faiss' or 'faiss' in strategy:
            return ['faiss']
        elif strategy == 'als':
            return ['als'] if self.als_ready else []
        elif strategy == 'content':
            return ['content'] if self.content_similarity_matrix is not None else []
        else:
            return ['unknown']


def test_enhanced_v2_engine():
    """
    Enhanced V2 engine test with real data
    """
    print("🧪 Testing Enhanced Recommendation Engine V2 with FAISS...")
    
    try:
        # Get script directory and build absolute paths
        script_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.dirname(script_dir)  # Go up one level from src/
        
        # Build absolute paths to data files
        interactions_path = os.path.join(base_dir, "data", "raw", "user_interactions.csv")
        user_features_path = os.path.join(base_dir, "data", "raw", "user_features.csv")
        video_features_path = os.path.join(base_dir, "data", "raw", "video_features.csv")
        
        print(f"Looking for data files in: {base_dir}/data/raw/")
        print(f"Interactions file exists: {os.path.exists(interactions_path)}")
        print(f"User features file exists: {os.path.exists(user_features_path)}")
        print(f"Video features file exists: {os.path.exists(video_features_path)}")
        
        # Engine oluştur
        engine = EnhancedRecommendationEngineV2(
            interactions_path,
            user_features_path, 
            video_features_path
        )
        
        print("✅ Engine initialized")
        
        # Data process (includes FAISS index building)
        engine.load_and_enhance_data()
        print("✅ Data processing and FAISS index building successful")
        
        # FAISS stats
        faiss_stats = engine.faiss_engine.get_stats()
        print(f"✅ FAISS Stats: {faiss_stats}")
        
        # Test user
        test_user = "user_3"  # From user_features.csv: "coğrafya,matematik,kimya",11
        
        print(f"\n🎯 Testing recommendations for {test_user}...")
        
        # Comprehensive recommendation test
        result = engine.get_comprehensive_recommendations(test_user, k=5, strategy='auto')
        print(f"✅ Comprehensive recommendations:")
        print(f"  Strategy used: {result['strategy_used']}")
        print(f"  Recommendations: {result['recommendations']}")
        print(f"  Metrics: {result['metrics']}")
        
        # Test different strategies
        strategies = ['faiss', 'hybrid', 'content', 'cold_start']
        for strategy in strategies:
            try:
                recs = engine.get_comprehensive_recommendations(test_user, k=3, strategy=strategy)
                print(f"✅ {strategy.upper()}: {recs['recommendations']}")
            except Exception as e:
                print(f"⚠️ {strategy.upper()} failed: {e}")
        
        # Explanation test
        if result['recommendations']:
            explanation = engine.explain_recommendation(test_user, result['recommendations'][0])
            print(f"✅ Explanation for {result['recommendations'][0]}: {explanation}")
        
        print("🎉 Enhanced V2 Engine with FAISS test successful!")
        return True
        
    except Exception as e:
        print(f"❌ Enhanced V2 Engine test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_enhanced_v2_engine()