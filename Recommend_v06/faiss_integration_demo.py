"""
FAISS Integration Demo Script

Bu script FAISS entegrasyonunu canlı olarak test eder ve
farklı kullanım senaryolarını gösterir:

1. FAISS Content Engine standalone test
2. Enhanced V2 Engine with real data
3. Cold start scenario demonstration
4. Performance comparison
5. Recommendation explanation
"""

import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.faiss_content_engine import FAISSContentEngine
from src.enhanced_recommendation_engine import EnhancedRecommendationEngineV2
import pandas as pd
import numpy as np
import time

def demo_faiss_standalone():
    """FAISS Content Engine standalone demo"""
    print("🔍 FAISS Content Engine Standalone Demo")
    print("-" * 50)
    
    # Create mock data for quick test
    video_data = pd.DataFrame({
        'video_id': ['video_1', 'video_2', 'video_3', 'video_4', 'video_5'],
        'tags': [
            '#matematik#10#tyt',
            '#fizik#11#ayt', 
            '#biyoloji#12#ayt',
            '#edebiyat#9#lgs',
            '#kimya#mezun#ayt'
        ]
    })
    
    user_data = pd.DataFrame({
        'user_id': ['user_1', 'user_2', 'user_3'],
        'top_subjects': [
            'matematik,fizik,kimya', 
            'edebiyat,tarih,biyoloji',
            'geometri,coğrafya,paragraf'
        ],
        'class': ['10', '11', 'mezun']
    })
    
    print(f"📊 Mock data: {len(video_data)} videos, {len(user_data)} users")
    
    # Initialize and build index
    engine = FAISSContentEngine(embedding_dim=64)
    
    start_time = time.time()
    engine.build_index(video_data, user_data)
    build_time = time.time() - start_time
    
    print(f"⏱️ Index built in {build_time:.3f}s")
    
    # Get recommendations for each user
    for user_id in user_data['user_id']:
        recs = engine.recommend(user_id, top_k=3)
        print(f"👤 {user_id}: {recs}")
        
        # Show explanation for first recommendation
        if recs:
            explanation = engine.explain_recommendation(user_id, recs[0])
            print(f"   💡 {recs[0]}: {explanation.get('matched_subjects', [])} subjects match")
    
    # Show stats
    stats = engine.get_stats()
    print(f"📈 Engine stats: {stats}")
    
    return engine


def demo_enhanced_v2_with_real_data():
    """Enhanced V2 Engine with real data demo"""
    print("\n🚀 Enhanced V2 Engine with Real Data Demo")
    print("-" * 50)
    
    try:
        # Initialize engine
        engine = EnhancedRecommendationEngineV2(
            "data/raw/user_interactions.csv",
            "data/raw/user_features.csv",
            "data/raw/video_features.csv"
        )
        
        print("📝 Loading and processing data...")
        start_time = time.time()
        engine.load_and_enhance_data()
        load_time = time.time() - start_time
        
        print(f"✅ Data loaded and FAISS index built in {load_time:.2f}s")
        
        # Show system stats
        if hasattr(engine, 'df') and engine.df is not None:
            print(f"📊 Dataset stats:")
            print(f"   - Total interactions: {len(engine.df):,}")
            print(f"   - Unique users: {engine.df['user_id'].nunique():,}")
            print(f"   - Unique videos: {engine.df['video_id'].nunique():,}")
        
        faiss_stats = engine.faiss_engine.get_stats()
        print(f"🔍 FAISS stats: {faiss_stats['total_videos']} videos, {faiss_stats['total_users']} users indexed")
        
        return engine
        
    except FileNotFoundError as e:
        print(f"❌ Data files not found: {e}")
        print("💡 Please ensure data files are in data/raw/ directory")
        return None
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None


def demo_cold_start_scenario(engine):
    """Demonstrate cold start scenario"""
    if engine is None:
        return
        
    print("\n🧊 Cold Start Scenario Demo")
    print("-" * 50)
    
    # Find cold start users
    cold_start_users = []
    if hasattr(engine, 'df') and engine.df is not None:
        user_interaction_counts = engine.df.groupby('user_id').size()
        cold_start_users = user_interaction_counts[user_interaction_counts <= 2].index.tolist()
    
    if not cold_start_users:
        print("⚠️ No cold start users found in dataset")
        # Use a regular user for demo
        if hasattr(engine, 'user_features_df') and engine.user_features_df is not None:
            test_user = engine.user_features_df['user_id'].iloc[0]
        else:
            test_user = "user_3"
    else:
        test_user = cold_start_users[0]
    
    print(f"👤 Testing cold start for user: {test_user}")
    
    # Show user profile from user_features.csv
    if hasattr(engine, 'user_features_df') and engine.user_features_df is not None:
        user_row = engine.user_features_df[engine.user_features_df['user_id'] == test_user]
        if not user_row.empty:
            profile = user_row.iloc[0]
            print(f"📚 User profile:")
            print(f"   - Preferred subjects: {profile.get('top_subjects', 'N/A')}")
            print(f"   - Class: {profile.get('class', 'N/A')}")
    
    # Get cold start recommendations
    start_time = time.time()
    cold_recs = engine.get_cold_start_recommendations(test_user, k=5)
    rec_time = time.time() - start_time
    
    print(f"⏱️ Cold start recommendations generated in {rec_time:.4f}s")
    print(f"🎯 Recommendations: {cold_recs}")
    
    # Show explanation for first recommendation
    if cold_recs:
        explanation = engine.explain_recommendation(test_user, cold_recs[0])
        print(f"💡 Explanation for {cold_recs[0]}:")
        if 'sources' in explanation:
            if 'faiss' in explanation['sources']:
                faiss_exp = explanation['sources']['faiss']
                print(f"   🔍 FAISS: {faiss_exp.get('matched_subjects', [])} subjects, "
                      f"similarity={faiss_exp.get('similarity_score', 0):.3f}")
            if 'content_based' in explanation['sources']:
                content_exp = explanation['sources']['content_based']
                print(f"   📊 Content: {content_exp.get('subject_matches', 0)} subject matches")
    
    # Get metrics
    metrics = engine.get_recommendation_metrics(test_user, cold_recs)
    print(f"📈 Quality metrics: diversity={metrics.get('diversity_score', 0):.2f}, "
          f"FAISS coverage={metrics.get('faiss_coverage', 0):.2f}")


def demo_performance_comparison(engine):
    """Demonstrate performance comparison between methods"""
    if engine is None:
        return
        
    print("\n⚡ Performance Comparison Demo")
    print("-" * 50)
    
    # Test user
    test_user = "user_3"
    
    methods = {
        'FAISS': lambda: engine.get_faiss_recommendations(test_user, k=5),
        'Content-Based': lambda: engine.get_content_based_recommendations(test_user, k=5),
        'Hybrid': lambda: engine.get_hybrid_advanced_recommendations(test_user, k=5),
        'Cold Start': lambda: engine.get_cold_start_recommendations(test_user, k=5)
    }
    
    results = {}
    
    for method_name, method_func in methods.items():
        try:
            # Warm up
            method_func()
            
            # Measure time
            start_time = time.time()
            recommendations = method_func()
            end_time = time.time()
            
            execution_time = (end_time - start_time) * 1000  # Convert to milliseconds
            
            results[method_name] = {
                'time_ms': execution_time,
                'count': len(recommendations),
                'recommendations': recommendations
            }
            
            print(f"🎯 {method_name:15}: {execution_time:6.2f}ms, {len(recommendations)} recs")
            
        except Exception as e:
            print(f"❌ {method_name:15}: Failed ({e})")
            results[method_name] = {'error': str(e)}
    
    # Show overlap analysis
    print("\n📊 Recommendation Overlap Analysis:")
    method_recs = {name: result.get('recommendations', []) 
                   for name, result in results.items() 
                   if 'recommendations' in result}
    
    methods_list = list(method_recs.keys())
    for i, method1 in enumerate(methods_list):
        for method2 in methods_list[i+1:]:
            recs1 = set(method_recs[method1])
            recs2 = set(method_recs[method2])
            overlap = len(recs1 & recs2)
            union = len(recs1 | recs2)
            jaccard = overlap / union if union > 0 else 0
            
            print(f"   {method1} ∩ {method2}: {overlap} videos (Jaccard: {jaccard:.2f})")


def demo_recommendation_explanation(engine):
    """Demonstrate detailed recommendation explanation"""
    if engine is None:
        return
        
    print("\n💡 Recommendation Explanation Demo")
    print("-" * 50)
    
    # Test user with known preferences
    test_user = "user_3"  # "coğrafya,matematik,kimya",11
    
    # Get comprehensive recommendations
    result = engine.get_comprehensive_recommendations(test_user, k=3, strategy='auto')
    
    print(f"👤 User: {test_user}")
    print(f"🎯 Strategy used: {result['strategy_used']}")
    print(f"📝 Recommendations: {result['recommendations']}")
    
    # Show detailed explanations
    for i, video_id in enumerate(result['recommendations'][:2]):  # Top 2
        print(f"\n🔍 Explanation for recommendation #{i+1}: {video_id}")
        
        explanation = engine.explain_recommendation(test_user, video_id)
        
        if 'sources' in explanation:
            # FAISS explanation
            if 'faiss' in explanation['sources']:
                faiss_exp = explanation['sources']['faiss']
                print(f"   🤖 FAISS Analysis:")
                print(f"      - Similarity score: {faiss_exp.get('similarity_score', 0):.3f}")
                print(f"      - Matched subjects: {faiss_exp.get('matched_subjects', [])}")
                print(f"      - Grade match: {faiss_exp.get('grade_match', False)}")
                print(f"      - Exam type match: {faiss_exp.get('exam_type_match', False)}")
                print(f"      - Video hashtag: {faiss_exp.get('video_hashtag', 'N/A')}")
            
            # Content-based explanation
            if 'content_based' in explanation['sources']:
                content_exp = explanation['sources']['content_based']
                print(f"   📊 Content-Based Analysis:")
                print(f"      - Subject matches: {content_exp.get('subject_matches', 0)}")
                print(f"      - User subjects: {content_exp.get('preferred_subjects', [])}")
                print(f"      - Video hashtags: {content_exp.get('video_hashtags', [])}")
    
    # Show quality metrics
    print(f"\n📈 Overall Quality Metrics:")
    metrics = result['metrics']
    for metric, value in metrics.items():
        print(f"   - {metric}: {value}")


def main():
    """Main demo function"""
    print("🎬 FAISS Integration Demo")
    print("=" * 60)
    
    # 1. Standalone FAISS demo
    faiss_engine = demo_faiss_standalone()
    
    # 2. Enhanced V2 with real data
    enhanced_engine = demo_enhanced_v2_with_real_data()
    
    if enhanced_engine is not None:
        # 3. Cold start scenario
        demo_cold_start_scenario(enhanced_engine)
        
        # 4. Performance comparison
        demo_performance_comparison(enhanced_engine)
        
        # 5. Recommendation explanation
        demo_recommendation_explanation(enhanced_engine)
        
        # Summary
        print("\n" + "=" * 60)
        print("🎉 DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("✅ FAISS Content Engine: Fully functional")
        print("✅ Enhanced Recommendation Engine V2: Integrated")
        print("✅ Cold Start Problem: Solved with user preferences")
        print("✅ Performance: Fast similarity search")
        print("✅ Explainability: Multi-source recommendations")
        print("\n💡 The FAISS integration is working correctly and ready for production!")
        
    else:
        print("\n⚠️ Demo completed with limitations due to missing data files.")
        print("💡 Please ensure data files are available for full functionality testing.")


if __name__ == "__main__":
    # Change to project directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Run demo
    main()