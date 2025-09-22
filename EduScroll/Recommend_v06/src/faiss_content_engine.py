"""
FAISS Content-Based Recommendation Engine

Bu modül FAISS (Facebook AI Similarity Search) kullanarak content-based 
recommendation sistemi sağlar. Video embeddings ve user profile embeddings
oluşturarak hızlı similarity search yapar.

Özellikler:
- Video özelliklerinden embedding oluşturma
- User tercihlerinden profile embedding oluşturma  
- FAISS index ile hızlı similarity search
- Cold-start problemine çözüm
- Açıklanabilir öneriler
"""

import numpy as np
import pandas as pd
import json
try:
    import faiss
except ImportError:
    import sys
    print("faiss module not found. Please ensure faiss-cpu is installed for your Python version.", file=sys.stderr)
    raise
from typing import List, Dict, Tuple, Optional
from sklearn.preprocessing import OneHotEncoder
from collections import defaultdict
import logging

class FAISSContentEngine:
    """
    FAISS tabanlı content-based recommendation engine
    Video embeddings ve user profile embeddings kullanarak similarity search yapar
    """
    
    def __init__(self, embedding_dim: int = 128):
        self.embedding_dim = embedding_dim
        self.faiss_index = None
        self.video_embeddings = {}
        self.user_embeddings = {}
        self.video_ids = []
        self.video_hashtags = {}
        
        # Subject mapping (Turkish subjects)
        self.subjects = [
            'matematik', 'fizik', 'kimya', 'biyoloji', 'tarih', 
            'coğrafya', 'edebiyat', 'geometri', 'paragraf'
        ]
        
        # Grade mapping
        self.grades = ['9', '10', '11', '12', 'mezun']
        
        # Exam type mapping
        self.exam_types = ['tyt', 'ayt', 'lgs']
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _extract_video_features(self, hashtag: str) -> Dict:
        """
        Hashtag'den video özelliklerini çıkar
        
        Args:
            hashtag: Video hashtag string (örn: "#matematik#10#tyt")
            
        Returns:
            Dict: Video özellikleri
        """
        features = {
            'subjects': [],
            'grade': 'unknown',
            'exam_type': 'tyt',
            'difficulty': 1
        }
        
        if pd.isna(hashtag) or not hashtag:
            return features
            
        hashtag_lower = str(hashtag).lower()
        
        # Subject detection
        for subject in self.subjects:
            if subject in hashtag_lower:
                features['subjects'].append(subject)
        
        # Grade detection
        for grade in self.grades:
            if grade in hashtag_lower:
                features['grade'] = grade
                break
        
        # Exam type detection
        if 'ayt' in hashtag_lower:
            features['exam_type'] = 'ayt'
        elif 'lgs' in hashtag_lower:
            features['exam_type'] = 'lgs'
        else:
            features['exam_type'] = 'tyt'
        
        # Difficulty estimation (based on grade and exam type)
        if features['grade'] in ['9', '10']:
            features['difficulty'] = 1
        elif features['grade'] == '11':
            features['difficulty'] = 2
        elif features['grade'] == '12':
            features['difficulty'] = 3
        elif features['grade'] == 'mezun':
            features['difficulty'] = 4
        else:
            features['difficulty'] = 2  # Default
            
        if features['exam_type'] == 'ayt':
            features['difficulty'] += 1
        elif features['exam_type'] == 'lgs':
            features['difficulty'] = max(1, features['difficulty'] - 1)
            
        return features
    
    # İleri de görüntü işleme veya NLP tabanlı embedding eklenmesi planlanıyor
    def _create_video_embedding(self, video_features: Dict) -> np.ndarray:
        """
        Video özelliklerinden embedding oluştur
        
        Args:
            video_features: _extract_video_features'dan gelen özellikler
            
        Returns:
            np.ndarray: Normalized video embedding
        """
        embedding = np.zeros(self.embedding_dim, dtype=np.float32)
        
        # Subject one-hot encoding (0-8 indices for 9 subjects)
        subject_dim = len(self.subjects)
        for i, subject in enumerate(self.subjects):
            if subject in video_features['subjects']:
                embedding[i] = 1.0
        
        # Grade encoding (9-13 indices for 5 grades)
        grade_dim = len(self.grades)
        if video_features['grade'] in self.grades:
            grade_idx = self.grades.index(video_features['grade'])
            embedding[subject_dim + grade_idx] = 1.0
        
        # Exam type encoding (14-16 indices)
        exam_dim = len(self.exam_types)
        exam_idx = self.exam_types.index(video_features['exam_type'])
        embedding[subject_dim + grade_dim + exam_idx] = 1.0
        
        # Difficulty encoding (17 index)
        embedding[subject_dim + grade_dim + exam_dim] = video_features['difficulty'] / 5.0
        
        # Random component for diversity (18-127 indices)
        # Bu kısım videoların benzersizliğini sağlar
        random_start = subject_dim + grade_dim + exam_dim + 1
        random_dim = self.embedding_dim - random_start
        
        if random_dim > 0:
            # Deterministic random based on video content for reproducibility
            content_hash = hash(str(video_features)) % 2**32
            np.random.seed(content_hash)
            embedding[random_start:] = np.random.normal(0, 0.1, random_dim)
        
        # L2 normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
            
        return embedding
    
    def _create_user_embedding(self, user_preferences: Dict) -> np.ndarray:
        """
        Kullanıcı tercihlerinden embedding oluştur
        
        Args:
            user_preferences: User tercihleri (top_subjects, class)
            
        Returns:
            np.ndarray: Normalized user embedding
        """
        embedding = np.zeros(self.embedding_dim, dtype=np.float32)
        
        # Subject preferences (weighted by preference order)
        subject_dim = len(self.subjects)
        if 'top_subjects' in user_preferences and user_preferences['top_subjects']:
            subjects = user_preferences['top_subjects']
            # İlk tercih %50, ikinci %30, üçüncü %20 ağırlık
            weights = [0.5, 0.3, 0.2]
            for i, subject in enumerate(subjects[:3]):
                if subject in self.subjects:
                    subject_idx = self.subjects.index(subject)
                    weight = weights[i] if i < len(weights) else 0.1
                    embedding[subject_idx] = weight
        
        # Grade preference
        grade_dim = len(self.grades)
        if 'class' in user_preferences:
            grade = str(user_preferences['class'])
            if grade in self.grades:
                grade_idx = self.grades.index(grade)
                embedding[subject_dim + grade_idx] = 1.0
        
        # Exam type preference (infer from grade)
        exam_dim = len(self.exam_types)
        user_grade = str(user_preferences.get('class', '10'))
        if user_grade in ['12', 'mezun']:
            embedding[subject_dim + grade_dim + 1] = 1.0  # AYT
        elif user_grade == '9':
            embedding[subject_dim + grade_dim + 2] = 1.0  # LGS
        else:
            embedding[subject_dim + grade_dim] = 1.0  # TYT
        
        # Difficulty preference (based on grade)
        if user_grade == '9':
            embedding[subject_dim + grade_dim + exam_dim] = 0.2
        elif user_grade == '10':
            embedding[subject_dim + grade_dim + exam_dim] = 0.4
        elif user_grade == '11':
            embedding[subject_dim + grade_dim + exam_dim] = 0.6
        elif user_grade == '12':
            embedding[subject_dim + grade_dim + exam_dim] = 0.8
        elif user_grade == 'mezun':
            embedding[subject_dim + grade_dim + exam_dim] = 1.0
        else:
            embedding[subject_dim + grade_dim + exam_dim] = 0.5
        
        # L2 normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
            
        return embedding
    
    def build_index(self, video_df: pd.DataFrame, user_df: pd.DataFrame):
        """
        Video ve user embeddings oluşturup FAISS index'i build et
        
        Args:
            video_df: Video dataframe (video_id, hashtag/tags columns)
            user_df: User dataframe (user_id, top_subjects, class columns)
        """
        
        # Video embeddings oluştur
        video_embeddings_list = []
        self.video_ids = []
        self.video_hashtags = {}
        
        # Video dataframe'den hashtag column'ı bul
        hashtag_col = None
        for col in ['hashtag', 'tags', 'video_hashtag']:
            if col in video_df.columns:
                hashtag_col = col
                break
        
        if hashtag_col is None:
            self.logger.warning("No hashtag column found in video_df")
            return
        
        for _, row in video_df.iterrows():
            video_id = row['video_id']
            hashtag = row.get(hashtag_col, '')
            
            video_features = self._extract_video_features(hashtag)
            embedding = self._create_video_embedding(video_features)
            
            self.video_embeddings[video_id] = embedding
            video_embeddings_list.append(embedding)
            self.video_ids.append(video_id)
            self.video_hashtags[video_id] = hashtag
        
        # User embeddings oluştur
        for _, row in user_df.iterrows():
            user_id = row['user_id']
            
            # top_subjects parse et
            top_subjects = []
            if 'top_subjects' in row and pd.notna(row['top_subjects']):
                subjects_str = str(row['top_subjects'])
                top_subjects = [s.strip() for s in subjects_str.split(',')]
            
            user_preferences = {
                'top_subjects': top_subjects,
                'class': row.get('class', '10')
            }
            
            embedding = self._create_user_embedding(user_preferences)
            self.user_embeddings[user_id] = embedding
        
        # FAISS index oluştur
        if video_embeddings_list:
            embeddings_matrix = np.vstack(video_embeddings_list)
            
            # L2 distance için IndexFlatL2 kullan
            self.faiss_index = faiss.IndexFlatL2(self.embedding_dim)
            self.faiss_index.add(embeddings_matrix)
        else:
            self.logger.warning("No videos found for FAISS index")
    
    # MAİN ÖNERİ FONKSİYONU
    def recommend(self, user_id: str, top_k: int = 5, exclude_videos: List[str] = None) -> List[str]:
        """
        FAISS kullanarak kullanıcıya video önerisi yap
        
        Args:
            user_id: Kullanıcı ID'si
            top_k: Öneri sayısı
            exclude_videos: Hariç tutulacak video ID'leri
            
        Returns:
            List[str]: Önerilen video ID'leri
        """
        if self.faiss_index is None:
            self.logger.warning("FAISS index not built yet")
            return []
            
        if user_id not in self.user_embeddings:
            self.logger.warning(f"User {user_id} not found in embeddings")
            return self._fallback_recommendations(top_k, exclude_videos)
        
        # User embedding al
        user_embedding = self.user_embeddings[user_id].reshape(1, -1)
        
        # FAISS search
        # top_k * 3 al ki exclude işleminden sonra yeterli öneri kalsın
        search_k = min(top_k * 3, len(self.video_ids))
        distances, indices = self.faiss_index.search(user_embedding, search_k)
        
        # Sonuçları video ID'lere çevir
        recommendations = []
        exclude_set = set(exclude_videos) if exclude_videos else set()
        
        for idx in indices[0]:
            if 0 <= idx < len(self.video_ids):
                video_id = self.video_ids[idx]
                if video_id not in exclude_set:
                    recommendations.append(video_id)
                    
                if len(recommendations) >= top_k:
                    break
        
        return recommendations
    
    def _fallback_recommendations(self, top_k: int, exclude_videos: List[str] = None) -> List[str]:
        """
        FAISS recommendation başarısız olduğunda fallback öneriler
        
        Args:
            top_k: Öneri sayısı
            exclude_videos: Hariç tutulacak videolar
            
        Returns:
            List[str]: Fallback öneriler
        """
        exclude_set = set(exclude_videos) if exclude_videos else set()
        
        # Random sampling with popular bias
        available_videos = [vid for vid in self.video_ids if vid not in exclude_set]
        
        if not available_videos:
            return []
        
        # Return first top_k videos as fallback
        return available_videos[:top_k]
    
    def get_video_similarity(self, video_id1: str, video_id2: str) -> float:
        """
        İki video arasındaki similarity'yi hesapla
        
        Args:
            video_id1, video_id2: Video ID'leri
            
        Returns:
            float: Cosine similarity (0-1)
        """
        if video_id1 not in self.video_embeddings or video_id2 not in self.video_embeddings:
            return 0.0
        
        emb1 = self.video_embeddings[video_id1]
        emb2 = self.video_embeddings[video_id2]
        
        # Cosine similarity
        similarity = np.dot(emb1, emb2)
        return max(0.0, float(similarity))  # Clamp to positive
    
    def get_user_video_similarity(self, user_id: str, video_id: str) -> float:
        """
        User ile video arasındaki similarity'yi hesapla
        
        Args:
            user_id: Kullanıcı ID'si
            video_id: Video ID'si
            
        Returns:
            float: Cosine similarity (0-1)
        """
        if user_id not in self.user_embeddings or video_id not in self.video_embeddings:
            return 0.0
        
        user_emb = self.user_embeddings[user_id]
        video_emb = self.video_embeddings[video_id]
        
        # Cosine similarity
        similarity = np.dot(user_emb, video_emb)
        return max(0.0, float(similarity))  # Clamp to positive
    
    # ÖNERİ AÇIKLAMASI
    def explain_recommendation(self, user_id: str, video_id: str) -> Dict:
        """
        Öneri açıklaması (hangi özelliklerin eşleştiğini göster)
        
        Args:
            user_id: Kullanıcı ID'si
            video_id: Video ID'si
            
        Returns:
            Dict: Açıklama bilgileri
        """
        if user_id not in self.user_embeddings or video_id not in self.video_embeddings:
            return {'error': 'User or video not found'}
        
        user_emb = self.user_embeddings[user_id]
        video_emb = self.video_embeddings[video_id]
        
        explanation = {
            'similarity_score': self.get_user_video_similarity(user_id, video_id),
            'matched_subjects': [],
            'grade_match': False,
            'exam_type_match': False,
            'video_hashtag': self.video_hashtags.get(video_id, '')
        }
        
        # Subject matches (0-8 indices)
        subject_dim = len(self.subjects)
        for i, subject in enumerate(self.subjects):
            if user_emb[i] > 0.1 and video_emb[i] > 0.1:
                explanation['matched_subjects'].append(subject)
        
        # Grade match (9-13 indices)
        grade_start = subject_dim
        user_grade_scores = user_emb[grade_start:grade_start + len(self.grades)]
        video_grade_scores = video_emb[grade_start:grade_start + len(self.grades)]
        
        user_grade_idx = np.argmax(user_grade_scores) if user_grade_scores.max() > 0 else -1
        video_grade_idx = np.argmax(video_grade_scores) if video_grade_scores.max() > 0 else -1
        
        explanation['grade_match'] = (user_grade_idx == video_grade_idx and user_grade_idx != -1)
        
        if user_grade_idx >= 0:
            explanation['user_grade'] = self.grades[user_grade_idx]
        if video_grade_idx >= 0:
            explanation['video_grade'] = self.grades[video_grade_idx]
        
        # Exam type match (14-16 indices)
        exam_start = subject_dim + len(self.grades)
        user_exam_scores = user_emb[exam_start:exam_start + len(self.exam_types)]
        video_exam_scores = video_emb[exam_start:exam_start + len(self.exam_types)]
        
        user_exam_idx = np.argmax(user_exam_scores) if user_exam_scores.max() > 0 else -1
        video_exam_idx = np.argmax(video_exam_scores) if video_exam_scores.max() > 0 else -1
        
        explanation['exam_type_match'] = (user_exam_idx == video_exam_idx and user_exam_idx != -1)
        
        if user_exam_idx >= 0:
            explanation['user_exam'] = self.exam_types[user_exam_idx]
        if video_exam_idx >= 0:
            explanation['video_exam'] = self.exam_types[video_exam_idx]
        
        return explanation
    
    def get_stats(self) -> Dict:
        """
        FAISS engine istatistikleri
        
        Returns:
            Dict: İstatistik bilgileri
        """
        return {
            'total_videos': len(self.video_ids),
            'total_users': len(self.user_embeddings),
            'embedding_dim': self.embedding_dim,
            'index_ready': self.faiss_index is not None,
            'subjects': self.subjects,
            'grades': self.grades,
            'exam_types': self.exam_types
        }

# Bu sadece modülün temel işlevselliğini test etmek için basit bir test fonksiyonudur. Gerçek ML modellerini demo da test ediniz.
def test_faiss_engine():
    """
    FAISS engine'in temel functionality'sini test et
    """
    print("🧪 Testing FAISS Content Engine...")
    
    # Mock data oluştur
    video_data = {
        'video_id': ['video_1', 'video_2', 'video_3', 'video_4', 'video_5'],
        'tags': [
            '#matematik#10#tyt',
            '#fizik#11#ayt', 
            '#biyoloji#12#ayt',
            '#edebiyat#9#lgs',
            '#kimya#mezun#ayt'
        ]
    }
    video_df = pd.DataFrame(video_data)
    
    user_data = {
        'user_id': ['user_1', 'user_2', 'user_3'],
        'top_subjects': [
            'matematik,fizik,kimya', 
            'edebiyat,tarih,biyoloji',
            'geometri,coğrafya,paragraf'
        ],
        'class': ['10', '11', 'mezun']
    }
    user_df = pd.DataFrame(user_data)
    
    # Engine'i test et
    engine = FAISSContentEngine(embedding_dim=64)
    
    try:
        # Index build
        engine.build_index(video_df, user_df)
        print("✅ Index building successful")
        
        # Statistics
        stats = engine.get_stats()
        print(f"✅ Stats: {stats}")
        
        # Recommendation test
        for user_id in ['user_1', 'user_2', 'user_3']:
            recs = engine.recommend(user_id, top_k=3)
            print(f"✅ Recommendations for {user_id}: {recs}")
            
            # Explanation for first recommendation
            if recs:
                explanation = engine.explain_recommendation(user_id, recs[0])
                print(f"  💡 Explanation for {recs[0]}: {explanation}")
        
        # Similarity test
        sim = engine.get_user_video_similarity('user_1', 'video_1')
        print(f"✅ User-video similarity: {sim:.3f}")
        
        # Video-video similarity test
        vid_sim = engine.get_video_similarity('video_1', 'video_2')
        print(f"✅ Video-video similarity: {vid_sim:.3f}")
        
        print("🎉 All FAISS tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ FAISS test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_faiss_engine()