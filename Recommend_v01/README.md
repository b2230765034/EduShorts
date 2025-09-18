EduShorts Öneri Sistemi (MVP) 🧠
Bu proje, TikTok tarzı eğitim platformu EduShorts için geliştirilen öneri motorunun ilk versiyonunu (v0.1) içerir. Temel amacı, kullanıcıların izleme geçmişini ve tercihlerini analiz ederek kişiselleştirilmiş video önerileri sunmaktır.

🚀 Temel Özellikler
Veri İşleme ve Profil Çıkarma: Kullanıcı etkileşim verileri (interactions.csv) işlenerek kapsamlı kullanıcı profilleri ve video özetleri oluşturulur.

Ağırlıklı Etkileşim Puanı: İzleme oranı, quiz cevapları ve beğeniler gibi etkileşimlere göre kullanıcı davranışının önemini ölçen bir puanlama sistemi (weight) kullanılır.

Temel Öneri Algoritması: Öneriler, üç ana faktöre göre sıralanır:

Popülerlik: En çok izlenen veya etkileşim alan videolar.

Güncellik (Recency): Yakın zamanda yüklenen veya etkileşime giren videolar.

Kişiselleştirme: Kullanıcının en çok ilgi gösterdiği hashtag'lerle videoların eşleşme oranı.

🎯 Proje Hedefleri (v0.1)
Bu versiyon, yapay zeka entegrasyonu için sağlam bir temel oluşturmayı hedefler. Backend ekibi, bu motordan gelen öneri listesini doğrudan mobil uygulamaya sunabilir.

⚙️ Kullanım
Öneri motoru, backend tarafından çağrılmak üzere tasarlanmış basit bir Python fonksiyonu içerir.

get_recommendations() Fonksiyonu
Kullanıcı için öneri listesi oluşturur.

Python

get_recommendations(user_id, k=20)
Parametreler:

user_id (string): Öneri istenecek kullanıcının ID'si.

k (integer): İstenen öneri sayısı.

Dönüş Değeri:

list: Önerilen video ID'lerinin listesi.

📊 Veri Setleri
Bu kodun çalışabilmesi için recommender_system/data/raw klasöründe aşağıdaki üç CSV dosyasının bulunması gerekir:

interactions.csv: Kullanıcıların videolarla olan etkileşimleri.

user_features.csv: Kullanıcıların statik bilgileri (sınıf, ilgi alanları vb.).

video_features.csv: Videoların meta verileri (süre, hashtag'ler vb.).

Kod, bu ham verileri işleyerek recommender_system/data/processed klasörüne çıktı dosyaları kaydeder.