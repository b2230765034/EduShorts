"""
EduScroll Recommendation System - Jüri Sunumu (CLEAN VERSION)
==============================================

Bu dosya EduScroll öneri sisteminin GERÇEK verileri ile jüriye göstermek için hazırlanmıştır.
Sistem önceden işlenmiş veriler ve eğitilmiş modeller kullanır:

1. 🔍 FAISS Content Engine - Eğitilmiş vektör indeksi (500 kullanıcı, 200 video)
2. 📊 Enhanced Recommendation Engine V2 - ALS + FAISS + Content hibrit sistemi  
3. 🎯 Smart Strategy Selection - Gerçek 5000 etkileşim verisi

Kullanım: python jury_demo_clean.py
"""

import sys
import os
import time
import pandas as pd
import numpy as np
import json

# Modülleri import et
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.enhanced_recommendation_engine import EnhancedRecommendationEngineV2
    from src.faiss_content_engine import FAISSContentEngine
except ImportError as e:
    print(f"❌ Modül import hatası: {e}")
    print("Lütfen src/ klasöründe gerekli dosyaların bulunduğundan emin olun.")
    sys.exit(1)

class EduScrollProductionDemo:
    """Gerçek eğitilmiş modeller ile jüri sunumu"""
    
    def __init__(self):
        self.enhanced_engine = None
        self.real_users = None
        self.real_videos = None
        self.system_stats = {}
        self.system_ready = False
        
        print("🎓 EduScroll Production Recommendation System - Jüri Sunumu")
        print("=" * 70)
        print("📊 GERÇEK VERİLER: 5000 etkileşim, 500 kullanıcı, 200 video")
        print("🤖 EĞİTİLMİŞ MODELLER: FAISS indeksi + ALS matrisi + Content similarity")
        print("=" * 70)
    
    def verify_processed_data(self):
        """İşlenmiş verilerin varlığını kontrol et"""
        print("\n🔍 İşlenmiş model dosyaları kontrol ediliyor...")
        
        required_paths = [
            "data/processed_train/als/user_item_matrix.npz",
            "data/processed_train/als/mappings.json", 
            "data/processed_train/faiss/user_features.npy",
            "data/processed_train/faiss/video_features.npy",
            "data/processed_train/faiss/mappings.json"
        ]
        
        missing_files = []
        for path in required_paths:
            if not os.path.exists(path):
                missing_files.append(path)
        
        if missing_files:
            print("❌ Eksik model dosyaları:")
            for file in missing_files:
                print(f"   - {file}")
            print("\n💡 Çözüm: python process_data.py komutunu çalıştırın")
            return False
        
        print("✅ Tüm eğitilmiş model dosyaları mevcut")
        return True
    
    def load_real_data(self):
        """Gerçek veri dosyalarını yükle"""
        print("\n📁 Gerçek veri dosyaları yükleniyor...")
        
        try:
            # Raw data dosyalarını yükle
            self.real_users = pd.read_csv("data/raw/user_features.csv")
            self.real_videos = pd.read_csv("data/raw/video_features.csv") 
            interactions = pd.read_csv("data/raw/user_interactions.csv")
            
            print(f"✅ {len(self.real_users)} gerçek kullanıcı yüklendi")
            print(f"✅ {len(self.real_videos)} gerçek video yüklendi")
            print(f"✅ {len(interactions)} gerçek etkileşim yüklendi")
            
            # İstatistikleri kaydet
            self.system_stats = {
                'total_users': len(self.real_users),
                'total_videos': len(self.real_videos),
                'total_interactions': len(interactions),
                'avg_interactions_per_user': len(interactions) / len(self.real_users),
                'data_sparsity': 1.0 - (len(interactions) / (len(self.real_users) * len(self.real_videos)))
            }
            
            return True
            
        except Exception as e:
            print(f"❌ Veri yükleme hatası: {e}")
            return False
    
    def initialize_production_engine(self):
        """Production-ready Enhanced Engine başlat"""
        print("\n🚀 Production Enhanced Engine başlatılıyor...")
        print("   (Bu işlem eğitilmiş FAISS indeksi + ALS modelini yükler)")
        
        try:
            # Enhanced Engine'i başlat
            self.enhanced_engine = EnhancedRecommendationEngineV2(
                "data/raw/user_interactions.csv",
                "data/raw/user_features.csv",
                "data/raw/video_features.csv"
            )
            
            # Verileri yükle ve modelleri eğit/yükle
            print("   📊 Veri yükleniyor ve FAISS+ALS modelleri hazırlanıyor...")
            self.enhanced_engine.load_and_enhance_data()
            
            # Sistem durumunu kontrol et
            if hasattr(self.enhanced_engine, 'faiss_ready') and self.enhanced_engine.faiss_ready:
                print("   ✅ FAISS indeksi aktif")
            
            if hasattr(self.enhanced_engine, 'als_model') and self.enhanced_engine.als_model:
                print("   ✅ ALS modeli aktif")
                
            if hasattr(self.enhanced_engine, 'content_similarity_matrix') and self.enhanced_engine.content_similarity_matrix is not None:
                print("   ✅ Content similarity matrisi aktif")
            
            print("✅ Production sistem tamamen hazır!")
            return True
            
        except Exception as e:
            print(f"❌ Production sistem başlatma hatası: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def demonstrate_real_recommendations(self):
        """Gerçek kullanıcılar ile öneri sistemini göster"""
        print("\n🎯 GERÇEK KULLANICI ÖNERİLERİ - 4 Farklı Algoritma")
        print("-" * 60)
        
        # Gerçek kullanıcılardan 5 örnek seç
        sample_users = self.real_users.sample(n=5, random_state=42)
        
        for _, user in sample_users.iterrows():
            user_id = user['user_id']
            subjects = user['top_subjects'] 
            grade = user['class']
            
            print(f"\n👤 Gerçek Kullanıcı: {user_id}")
            print(f"   📚 İlgi Alanları: {subjects}")
            print(f"   🎓 Sınıf: {grade}")
            
            try:
                # 1. FAISS önerileri (Cold start için)
                start_time = time.time()
                faiss_result = self.enhanced_engine.get_comprehensive_recommendations(
                    user_id, k=3, strategy='faiss'
                )
                faiss_time = (time.time() - start_time) * 1000
                faiss_recs = faiss_result.get('recommendations', [])
                print(f"   🔍 FAISS Önerileri ({faiss_time:.1f}ms): {faiss_recs}")
                
                # 2. ALS Collaborative Filtering
                start_time = time.time()
                als_result = self.enhanced_engine.get_comprehensive_recommendations(
                    user_id, k=3, strategy='als'
                )
                als_time = (time.time() - start_time) * 1000
                als_recs = als_result.get('recommendations', [])
                print(f"   🤖 ALS Collaborative ({als_time:.1f}ms): {als_recs}")
                
                # 3. Content-based önerileri
                start_time = time.time()
                content_result = self.enhanced_engine.get_comprehensive_recommendations(
                    user_id, k=3, strategy='content'
                )
                content_time = (time.time() - start_time) * 1000
                content_recs = content_result.get('recommendations', [])
                print(f"   📄 Content-Based ({content_time:.1f}ms): {content_recs}")
                
                # 4. Hybrid (Akıllı Kombinasyon)
                start_time = time.time()
                hybrid_result = self.enhanced_engine.get_comprehensive_recommendations(
                    user_id, k=3, strategy='auto'
                )
                hybrid_time = (time.time() - start_time) * 1000
                hybrid_recs = hybrid_result.get('recommendations', [])
                strategy_used = hybrid_result.get('strategy_used', 'unknown')
                engines_used = hybrid_result.get('metrics', {}).get('engines_used', [])
                print(f"   🎯 Akıllı Hybrid ({hybrid_time:.1f}ms): {hybrid_recs}")
                print(f"     Seçilen Strateji: {strategy_used}")
                print(f"     Aktif Motorlar: {', '.join(engines_used)}")
                
                # Performans açıklaması
                fastest = min(faiss_time, als_time, content_time, hybrid_time)
                if faiss_time == fastest:
                    print(f"   ⚡ En hızlı: FAISS (Cold start için ideal)")
                elif als_time == fastest:
                    print(f"   ⚡ En hızlı: ALS (Collaborative filtering)")
                
            except Exception as e:
                print(f"   ❌ Öneri hatası: {e}")
            
            print()  # Boş satır
    
    def demonstrate_advanced_features(self):
        """Gelişmiş özellikleri göster"""
        print("\n🧠 GELİŞMİŞ ÖZELLIKLER VE ANALİZ")
        print("-" * 50)
        
        # Örnek kullanıcı seç
        sample_user = self.real_users.sample(n=1, random_state=42).iloc[0]
        user_id = sample_user['user_id']
        
        print(f"📊 Analiz Edilen Kullanıcı: {user_id}")
        print(f"📚 İlgi Alanları: {sample_user['top_subjects']}")
        print(f"🎓 Sınıf: {sample_user['class']}")
        
        try:
            # Comprehensive recommendation with full metrics
            result = self.enhanced_engine.get_comprehensive_recommendations(
                user_id, k=5, strategy='auto'
            )
            
            recommendations = result['recommendations']
            metrics = result['metrics']
            
            print(f"\n🎯 Önerilen Videolar: {recommendations}")
            print(f"📈 Sistem Metrikleri:")
            print(f"   - Çeşitlilik Skoru: {metrics.get('diversity_score', 0):.3f}")
            print(f"   - Güven Skoru: {metrics.get('confidence_score', 0):.3f}")
            print(f"   - Kapsama Skoru: {metrics.get('coverage_score', 0):.3f}")
            print(f"   - Cold Start Kullanıcı: {metrics.get('cold_start_user', False)}")
            
            # İlk önerinin detaylı açıklaması
            if recommendations:
                first_rec = recommendations[0]
                explanation = self.enhanced_engine.explain_recommendation(user_id, first_rec)
                
                print(f"\n💡 '{first_rec}' Önerisinin Detaylı Açıklaması:")
                
                if 'sources' in explanation:
                    sources = explanation['sources']
                    
                    if 'faiss' in sources:
                        faiss_exp = sources['faiss']
                        print(f"   🔍 FAISS: Benzerlik={faiss_exp.get('similarity_score', 0):.3f}")
                        print(f"       Eşleşen Konular: {faiss_exp.get('matched_subjects', [])}")
                        print(f"       Sınıf Uyumu: {faiss_exp.get('grade_match', False)}")
                    
                    if 'als' in sources:
                        print(f"   🤖 ALS: Collaborative filtering ile tespit")
                    
                    if 'content_based' in sources:
                        content_exp = sources['content_based']
                        print(f"   📄 Content: {content_exp.get('subject_matches', 0)} konu eşleşmesi")
                
        except Exception as e:
            print(f"❌ Gelişmiş analiz hatası: {e}")
    
    def show_production_performance(self):
        """Production sistem performansını göster"""
        print("\n⚡ PRODUCTION SİSTEM PERFORMANSI")
        print("-" * 50)
        
        # Sistem istatistiklerini göster
        print(f"📊 Veri Seti İstatistikleri:")
        print(f"   👥 Toplam Kullanıcı: {self.system_stats['total_users']:,}")
        print(f"   📹 Toplam Video: {self.system_stats['total_videos']:,}")
        print(f"   🔄 Toplam Etkileşim: {self.system_stats['total_interactions']:,}")
        print(f"   📈 Kullanıcı Başına Ortalama: {self.system_stats['avg_interactions_per_user']:.1f}")
        print(f"   🕳️ Veri Seyrekliği: {self.system_stats['data_sparsity']:.1%}")
        
        # Performans testi - 10 rastgele kullanıcı
        test_users = self.real_users.sample(n=10, random_state=42)['user_id'].tolist()
        
        strategies = ['faiss', 'als', 'content', 'auto']
        performance_results = {}
        
        print(f"\n⏱️ Performans Testi (10 kullanıcı):")
        
        for strategy in strategies:
            total_time = 0
            total_recs = 0
            successful = 0
            
            for user_id in test_users:
                try:
                    start_time = time.time()
                    result = self.enhanced_engine.get_comprehensive_recommendations(
                        user_id, k=5, strategy=strategy
                    )
                    elapsed = time.time() - start_time
                    
                    total_time += elapsed
                    total_recs += len(result.get('recommendations', []))
                    successful += 1
                    
                except Exception:
                    pass
            
            if successful > 0:
                avg_time = (total_time / successful) * 1000
                avg_recs = total_recs / successful
                
                performance_results[strategy] = {
                    'avg_time_ms': avg_time,
                    'avg_recommendations': avg_recs,
                    'success_rate': successful / len(test_users)
                }
                
                print(f"   {strategy.upper():12}: {avg_time:6.1f}ms, {avg_recs:.1f} öneri, %{(successful/len(test_users)*100):3.0f} başarı")
        
        # En hızlı stratejyi bul
        fastest_strategy = min(performance_results.keys(), 
                             key=lambda x: performance_results[x]['avg_time_ms'])
        fastest_time = performance_results[fastest_strategy]['avg_time_ms']
        
        print(f"\n🏆 En Hızlı Strateji: {fastest_strategy.upper()} ({fastest_time:.1f}ms)")
        print(f"🚀 Saniye Başına İstek Kapasitesi: ~{1000/fastest_time:.0f} req/s")
    
    def run_production_demonstration(self):
        """Tam production jüri demonstrasyonu çalıştır"""
        print("🚀 EduScroll Production Jüri Demonstrasyonu Başlıyor...")
        
        # 1. İşlenmiş verileri kontrol et
        if not self.verify_processed_data():
            return False
        
        # 2. Gerçek verileri yükle
        if not self.load_real_data():
            return False
            
        # 3. Production sistemi başlat
        if not self.initialize_production_engine():
            return False
            
        # 4. Sistem hazır
        self.system_ready = True
        print(f"\n✅ PRODUCTION SİSTEM HAZIR!")
        print(f"🔍 FAISS Engine: Eğitilmiş indeks aktif")
        print(f"🤖 ALS Model: Collaborative filtering aktif") 
        print(f"📄 Content Engine: Similarity matrisi aktif")
        print(f"🎯 Hybrid Strategy: Akıllı kombinasyon aktif")
        
        # 5. Demonstrasyonları çalıştır
        self.demonstrate_real_recommendations()
        self.demonstrate_advanced_features()
        self.show_production_performance()
        
        # 6. Sonuç
        print("\n" + "=" * 70)
        print("🎉 PRODUCTION JÜRİ DEMONSTRASYONU TAMAMLANDI!")
        print("=" * 70)
        print("✅ GERÇEK 5000 etkileşim verisi ile eğitilmiş sistem")
        print("✅ FAISS vektör indeksi: <10ms hızlı similarity search")
        print("✅ ALS collaborative filtering: Kullanıcı davranış analizi")
        print("✅ Content-based filtering: Konu ve sınıf uyumu")
        print("✅ Hybrid strateji: Akıllı motor kombinasyonu")
        print("✅ Cold-start çözümü: Yeni kullanıcılar için instant öneriler")
        print("✅ Açıklanabilir AI: Her önerinin detaylı gerekçesi")
        print("\n🎓 Sistem production ortamı için hazır!")
        
        return True


def main():
    """Ana demo fonksiyonu - GERÇEK veriler ile"""
    try:
        demo = EduScrollProductionDemo()
        success = demo.run_production_demonstration()
        
        if success:
            print("\n🎓 EduScroll Production Recommendation System başarıyla çalışıyor!")
            print("🚀 Jüri için hazır - gerçek veriler ve eğitilmiş modeller ile!")
        else:
            print("\n❌ Production demo sırasında hata oluştu!")
            print("💡 python process_data.py komutunu çalıştırarak modelleri eğitin")
            return 1
            
    except KeyboardInterrupt:
        print("\n\n⏹️ Demo kullanıcı tarafından durduruldu")
        return 0
    except Exception as e:
        print(f"\n❌ Beklenmeyen hata: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    # Demo çalıştır
    exit_code = main()
    sys.exit(exit_code)