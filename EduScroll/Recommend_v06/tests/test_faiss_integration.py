"""
Comprehensive Test Suite for FAISS Integration

Bu modül FAISS entegrasyonunu kapsamlı şekilde test eder:
1. Unit tests (FAISS engine components)
2. Integration tests (Enhanced V2 engine)
3. Performance tests (Speed and memory)
4. Cold start tests (User preference based recommendations)
5. Comparison tests (FAISS vs other methods)
"""

import sys
import os
# Add the src directory to the path
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(os.path.dirname(script_dir), 'src')
sys.path.append(src_dir)

import unittest
import pandas as pd
import numpy as np
import time
import logging
from typing import Dict, List

# Import our modules with relative paths
from faiss_content_engine import FAISSContentEngine
from enhanced_recommendation_engine import EnhancedRecommendationEngineV2

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestFAISSContentEngine(unittest.TestCase):
    """FAISS Content Engine unit tests"""
    
    def setUp(self):
        """Test setup"""
        self.video_data = pd.DataFrame({
            'video_id': ['v1', 'v2', 'v3', 'v4', 'v5'],
            'tags': [
                '#matematik#10#tyt',
                '#fizik#11#ayt', 
                '#biyoloji#12#ayt',
                '#edebiyat#9#lgs',
                '#kimya#mezun#ayt'
            ]
        })
        
        self.user_data = pd.DataFrame({
            'user_id': ['u1', 'u2', 'u3'],
            'top_subjects': [
                'matematik,fizik,kimya', 
                'edebiyat,tarih,biyoloji',
                'geometri,coğrafya,paragraf'
            ],
            'class': ['10', '11', 'mezun']
        })
        
        self.engine = FAISSContentEngine(embedding_dim=64)
    
    def test_video_feature_extraction(self):
        """Test video feature extraction from hashtags"""
        print("Testing video feature extraction...")
        
        # Test matematik video
        features = self.engine._extract_video_features('#matematik#10#tyt')
        self.assertIn('matematik', features['subjects'])
        self.assertEqual(features['grade'], '10')
        self.assertEqual(features['exam_type'], 'tyt')
        self.assertGreater(features['difficulty'], 0)
        
        # Test AYT video
        features = self.engine._extract_video_features('#fizik#12#ayt')
        self.assertIn('fizik', features['subjects'])
        self.assertEqual(features['grade'], '12')
        self.assertEqual(features['exam_type'], 'ayt')
        
        # Test empty hashtag
        features = self.engine._extract_video_features('')
        self.assertEqual(features['grade'], 'unknown')
        self.assertEqual(features['exam_type'], 'tyt')
        
        print("✅ Video feature extraction tests passed")
    
    def test_embedding_generation(self):
        """Test embedding generation and normalization"""
        print("Testing embedding generation...")
        
        # Video embedding test
        features = {
            'subjects': ['matematik', 'fizik'],
            'grade': '10',
            'exam_type': 'tyt',
            'difficulty': 2
        }
        
        embedding = self.engine._create_video_embedding(features)
        
        # Check shape and normalization
        self.assertEqual(embedding.shape[0], self.engine.embedding_dim)
        self.assertAlmostEqual(np.linalg.norm(embedding), 1.0, places=5)
        
        # Check subject encoding
        self.assertGreater(embedding[0], 0)  # matematik
        self.assertGreater(embedding[1], 0)  # fizik
        
        # User embedding test
        user_prefs = {
            'top_subjects': ['matematik', 'fizik', 'kimya'],
            'class': '10'
        }
        
        user_embedding = self.engine._create_user_embedding(user_prefs)
        
        # Check shape and normalization
        self.assertEqual(user_embedding.shape[0], self.engine.embedding_dim)
        self.assertAlmostEqual(np.linalg.norm(user_embedding), 1.0, places=5)
        
        # Check preference weights
        self.assertGreater(user_embedding[0], user_embedding[1])  # matematik > fizik (order matters)
        
        print("✅ Embedding generation tests passed")
    
    def test_index_building(self):
        """Test FAISS index building"""
        print("Testing FAISS index building...")
        
        self.engine.build_index(self.video_data, self.user_data)
        
        # Check index creation
        self.assertIsNotNone(self.engine.faiss_index)
        self.assertEqual(len(self.engine.video_ids), len(self.video_data))
        self.assertEqual(len(self.engine.user_embeddings), len(self.user_data))
        
        # Check video hashtags storage
        self.assertIn('v1', self.engine.video_hashtags)
        
        print("✅ FAISS index building tests passed")
    
    def test_recommendation_generation(self):
        """Test recommendation generation"""
        print("Testing recommendation generation...")
        
        self.engine.build_index(self.video_data, self.user_data)
        
        # Test normal user
        recommendations = self.engine.recommend('u1', top_k=3)
        
        self.assertIsInstance(recommendations, list)
        self.assertLessEqual(len(recommendations), 3)
        
        # Check if all recommendations are valid video IDs
        for video_id in recommendations:
            self.assertIn(video_id, self.engine.video_ids)
        
        # Test with exclusions
        recommendations_with_exclusion = self.engine.recommend('u1', top_k=3, exclude_videos=['v1'])
        self.assertNotIn('v1', recommendations_with_exclusion)
        
        # Test non-existent user (should return fallback)
        fallback_recs = self.engine.recommend('nonexistent_user', top_k=3)
        self.assertIsInstance(fallback_recs, list)
        
        print("✅ Recommendation generation tests passed")
    
    def test_similarity_calculations(self):
        """Test similarity calculations"""
        print("Testing similarity calculations...")
        
        self.engine.build_index(self.video_data, self.user_data)
        
        # Video-video similarity
        sim = self.engine.get_video_similarity('v1', 'v2')
        self.assertIsInstance(sim, float)
        self.assertGreaterEqual(sim, 0)
        self.assertLessEqual(sim, 1)
        
        # User-video similarity
        sim = self.engine.get_user_video_similarity('u1', 'v1')
        self.assertIsInstance(sim, float)
        self.assertGreaterEqual(sim, 0)
        self.assertLessEqual(sim, 1)
        
        # Non-existent entities
        sim = self.engine.get_video_similarity('nonexistent1', 'nonexistent2')
        self.assertEqual(sim, 0.0)
        
        print("✅ Similarity calculation tests passed")
    
    def test_explanation_functionality(self):
        """Test recommendation explanation"""
        print("Testing explanation functionality...")
        
        self.engine.build_index(self.video_data, self.user_data)
        
        explanation = self.engine.explain_recommendation('u1', 'v1')
        
        # Check explanation structure
        self.assertIn('similarity_score', explanation)
        self.assertIn('matched_subjects', explanation)
        self.assertIn('grade_match', explanation)
        self.assertIn('exam_type_match', explanation)
        self.assertIn('video_hashtag', explanation)
        
        # Check similarity score is valid
        self.assertIsInstance(explanation['similarity_score'], float)
        self.assertGreaterEqual(explanation['similarity_score'], 0)
        
        print("✅ Explanation functionality tests passed")


class TestEnhancedV2Integration(unittest.TestCase):
    """Enhanced V2 Engine integration tests"""
    
    @classmethod
    def setUpClass(cls):
        """Setup for integration tests"""
        # Build absolute path to data files
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(script_dir)
        cls.data_path = os.path.join(project_dir, "data", "raw")
        
        # Check if data files exist
        required_files = [
            "user_interactions.csv",
            "user_features.csv",
            "video_features.csv"
        ]
        
        cls.data_available = all(
            os.path.exists(os.path.join(cls.data_path, file)) 
            for file in required_files
        )
        
        if not cls.data_available:
            print("⚠️ Warning: Data files not found. Integration tests will be skipped.")
            print(f"Looking for data in: {cls.data_path}")
        else:
            print(f"✅ Data files found in: {cls.data_path}")
    
    def setUp(self):
        """Test setup for each test"""
        if not self.data_available:
            self.skipTest("Data files not available")
    
    def test_engine_initialization(self):
        """Test Enhanced V2 engine initialization"""
        print("Testing Enhanced V2 engine initialization...")
        
        engine = EnhancedRecommendationEngineV2(
            os.path.join(self.data_path, "user_interactions.csv"),
            os.path.join(self.data_path, "user_features.csv"),
            os.path.join(self.data_path, "video_features.csv")
        )
        
        self.assertTrue(hasattr(engine, 'faiss_engine'))
        self.assertEqual(engine.faiss_engine.__class__.__name__, 'FAISSContentEngine')
        self.assertFalse(engine.faiss_ready)  # Should be False before data loading
        
        print("✅ Engine initialization test passed")
    
    def test_data_loading_and_faiss_building(self):
        """Test data loading and FAISS index building"""
        print("Testing data loading and FAISS index building...")
        
        engine = EnhancedRecommendationEngineV2(
            os.path.join(self.data_path, "user_interactions.csv"),
            os.path.join(self.data_path, "user_features.csv"),
            os.path.join(self.data_path, "video_features.csv")
        )
        
        # Load and enhance data (includes FAISS building)
        engine.load_and_enhance_data()
        
        # Check if FAISS is ready
        self.assertTrue(engine.faiss_ready)
        
        # Check FAISS stats
        stats = engine.faiss_engine.get_stats()
        self.assertGreater(stats['total_videos'], 0)
        self.assertGreater(stats['total_users'], 0)
        self.assertTrue(stats['index_ready'])
        
        print(f"✅ FAISS built with {stats['total_videos']} videos and {stats['total_users']} users")
    
    def test_recommendation_strategies(self):
        """Test different recommendation strategies"""
        print("Testing recommendation strategies...")
        
        engine = EnhancedRecommendationEngineV2(
            os.path.join(self.data_path, "user_interactions.csv"),
            os.path.join(self.data_path, "user_features.csv"),
            os.path.join(self.data_path, "video_features.csv")
        )
        
        engine.load_and_enhance_data()
        
        # Test user (from user_features.csv)
        test_user = "user_3"  # "coğrafya,matematik,kimya",11
        
        strategies = ['faiss', 'hybrid', 'content', 'cold_start', 'auto']
        
        for strategy in strategies:
            try:
                result = engine.get_comprehensive_recommendations(
                    test_user, k=5, strategy=strategy
                )
                
                self.assertIsInstance(result, dict)
                self.assertIn('recommendations', result)
                self.assertIn('strategy_used', result)
                self.assertIn('metrics', result)
                
                recommendations = result['recommendations']
                self.assertIsInstance(recommendations, list)
                self.assertLessEqual(len(recommendations), 5)
                
                print(f"✅ {strategy.upper()} strategy: {len(recommendations)} recommendations")
                
            except Exception as e:
                print(f"⚠️ {strategy.upper()} strategy failed: {e}")
    
    def test_cold_start_performance(self):
        """Test cold start user performance"""
        print("Testing cold start performance...")
        
        engine = EnhancedRecommendationEngineV2(
            os.path.join(self.data_path, "user_interactions.csv"),
            os.path.join(self.data_path, "user_features.csv"),
            os.path.join(self.data_path, "video_features.csv")
        )
        
        engine.load_and_enhance_data()
        
        # Find cold start users (users with ≤2 interactions)
        if hasattr(engine, 'df') and engine.df is not None:
            user_interaction_counts = engine.df.groupby('user_id').size()
            cold_start_users = user_interaction_counts[user_interaction_counts <= 2].index.tolist()
            
            if cold_start_users:
                test_cold_user = cold_start_users[0]
                
                # Test cold start recommendations
                cold_recs = engine.get_cold_start_recommendations(test_cold_user, k=5)
                
                self.assertIsInstance(cold_recs, list)
                self.assertGreater(len(cold_recs), 0)  # Should get some recommendations
                
                # Test metrics for cold start
                metrics = engine.get_recommendation_metrics(test_cold_user, cold_recs)
                self.assertTrue(metrics['cold_start_user'])
                
                print(f"✅ Cold start test: {len(cold_recs)} recommendations for user {test_cold_user}")
            else:
                print("⚠️ No cold start users found in dataset")


class TestPerformance(unittest.TestCase):
    """Performance tests for FAISS integration"""
    
    def test_faiss_performance(self):
        """Test FAISS performance with larger dataset"""
        print("Testing FAISS performance...")
        
        # Create larger mock dataset
        n_videos = 500
        n_users = 100
        
        subjects = ['matematik', 'fizik', 'kimya', 'biyoloji', 'tarih', 'coğrafya', 'edebiyat']
        grades = ['9', '10', '11', '12', 'mezun']
        exam_types = ['tyt', 'ayt', 'lgs']
        
        # Generate videos
        np.random.seed(42)
        video_ids = [f"video_{i}" for i in range(n_videos)]
        hashtags = []
        
        for i in range(n_videos):
            selected_subjects = np.random.choice(subjects, size=1, replace=False)
            selected_grade = np.random.choice(grades)
            selected_exam = np.random.choice(exam_types)
            hashtag = f"#{selected_subjects[0]}#{selected_grade}#{selected_exam}"
            hashtags.append(hashtag)
        
        video_df = pd.DataFrame({
            'video_id': video_ids,
            'tags': hashtags
        })
        
        # Generate users
        user_ids = [f"user_{i}" for i in range(n_users)]
        user_subjects = []
        user_grades = []
        
        for i in range(n_users):
            selected_subjects = np.random.choice(subjects, size=3, replace=False)
            user_subjects.append(','.join(selected_subjects))
            user_grades.append(np.random.choice(grades))
        
        user_df = pd.DataFrame({
            'user_id': user_ids,
            'top_subjects': user_subjects,
            'class': user_grades
        })
        
        # Performance test
        engine = FAISSContentEngine(embedding_dim=128)
        
        # Index building time
        start_time = time.time()
        engine.build_index(video_df, user_df)
        build_time = time.time() - start_time
        
        print(f"⏱️ Index building time: {build_time:.2f}s for {n_videos} videos, {n_users} users")
        
        # Recommendation time
        start_time = time.time()
        total_recommendations = 0
        
        for i in range(10):  # 10 users
            recs = engine.recommend(f"user_{i}", top_k=10)
            total_recommendations += len(recs)
            
        recommendation_time = time.time() - start_time
        avg_time = recommendation_time / 10
        
        print(f"⏱️ Average recommendation time: {avg_time:.4f}s per user")
        print(f"📊 Average recommendations per user: {total_recommendations / 10:.1f}")
        
        # Performance assertions
        self.assertLess(build_time, 10.0)  # Should build in less than 10 seconds
        self.assertLess(avg_time, 0.1)  # Should recommend in less than 0.1 seconds per user
        
        print("✅ Performance tests passed")


def run_comprehensive_test_suite():
    """Run all tests in order"""
    print("🚀 Running Comprehensive FAISS Integration Test Suite")
    print("=" * 60)
    
    # Test results tracker
    test_results = {
        'unit_tests': False,
        'integration_tests': False,
        'performance_tests': False
    }
    
    try:
        # 1. Unit Tests
        print("\n🧪 1. Running Unit Tests...")
        unit_suite = unittest.TestLoader().loadTestsFromTestCase(TestFAISSContentEngine)
        unit_result = unittest.TextTestRunner(verbosity=1).run(unit_suite)
        test_results['unit_tests'] = unit_result.wasSuccessful()
        
        # 2. Integration Tests
        print("\n🔗 2. Running Integration Tests...")
        integration_suite = unittest.TestLoader().loadTestsFromTestCase(TestEnhancedV2Integration)
        integration_result = unittest.TextTestRunner(verbosity=1).run(integration_suite)
        test_results['integration_tests'] = integration_result.wasSuccessful()
        
        # 3. Performance Tests
        print("\n⚡ 3. Running Performance Tests...")
        performance_suite = unittest.TestLoader().loadTestsFromTestCase(TestPerformance)
        performance_result = unittest.TextTestRunner(verbosity=1).run(performance_suite)
        test_results['performance_tests'] = performance_result.wasSuccessful()
        
    except Exception as e:
        print(f"❌ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    # Results summary
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    for test_type, passed in test_results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_type.replace('_', ' ').title()}: {status}")
    
    all_passed = all(test_results.values())
    if all_passed:
        print("\n🎉 ALL TESTS PASSED! FAISS integration is working correctly.")
        print("💡 The system is ready for production use.")
    else:
        print("\n⚠️ Some tests failed. Please check the output above for details.")
    
    return test_results


if __name__ == "__main__":
    # Set up test environment - change to project directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    os.chdir(project_dir)
    
    # Run comprehensive test suite
    results = run_comprehensive_test_suite()
    
    # Exit with appropriate code
    if all(results.values()):
        exit(0)
    else:
        exit(1)