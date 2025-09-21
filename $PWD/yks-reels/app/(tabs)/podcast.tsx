// app/(tabs)/podcast.tsx — commented only (no logic changes)
// -----------------------------------------------------------------------------
// Purpose: Podcast ekranı. Kategoriler, popüler podcast kartları ve bölüm listesi
// içerir. Arama ve kategori filtreleme desteklenir. Bu sürümde SADECE yorum eklenmiştir.
// -----------------------------------------------------------------------------

import React, { useState } from "react";
import {
  View,
  Text,
  FlatList,
  Pressable,
  TextInput,
  ScrollView,
  Image,
  Dimensions,
  StyleSheet,
  Alert,
} from "react-native";
import { useSafeAreaInsets } from "react-native-safe-area-context";

// Ekran genişliği bazı layout hesapları için kullanılabilir (şu an sabit stiller var)
const { width: screenWidth } = Dimensions.get("window");

// -----------------------------------------------------------------------------
// Mock data (gerçek API bağlanana kadar placeholder)
// categories: chips; trendingPodcasts: yatay kart listesi; episodes: dikey bölüm listesi
// -----------------------------------------------------------------------------
const podcastData = {
  categories: ["Tümü", "Eğitim", "Teknoloji", "Bilim", "Tarih", "Felsefe"],
  trendingPodcasts: [
    {
      id: 1,
      title: "YKS Hazırlık Podcast",
      host: "Eğitim Merkezi",
      duration: "45:30",
      listeners: "12.5K",
      thumbnail: "https://via.placeholder.com/120x120/667eea/ffffff?text=YKS",
      category: "Eğitim",
      description: "YKS sınavına hazırlanan öğrenciler için özel içerikler",
    },
    {
      id: 2,
      title: "Fizik Dünyası",
      host: "Dr. Ahmet Yılmaz",
      duration: "38:15",
      listeners: "8.2K",
      thumbnail: "https://via.placeholder.com/120x120/764ba2/ffffff?text=Fizik",
      category: "Bilim",
      description: "Fizik konularını eğlenceli bir şekilde öğrenin",
    },
    {
      id: 3,
      title: "Matematik Sohbetleri",
      host: "Prof. Mehmet Kaya",
      duration: "52:20",
      listeners: "15.3K",
      thumbnail: "https://via.placeholder.com/120x120/4ecdc4/ffffff?text=Mat",
      category: "Eğitim",
      description: "Matematik problemlerini çözme teknikleri",
    },
  ],
  episodes: [
    {
      id: 1,
      title: "Limit ve Süreklilik Konuları",
      podcast: "YKS Hazırlık Podcast",
      duration: "28:45",
      date: "2 gün önce",
      thumbnail: "https://via.placeholder.com/80x80/667eea/ffffff?text=L",
      isPlaying: false,
      progress: 0,
    },
    {
      id: 2,
      title: "Newton'un Hareket Yasaları",
      podcast: "Fizik Dünyası",
      duration: "35:20",
      date: "3 gün önce",
      thumbnail: "https://via.placeholder.com/80x80/764ba2/ffffff?text=N",
      isPlaying: true,
      progress: 65, // oynatma çubuğunda yüzde ilerleme (sadece görsel)
    },
    {
      id: 3,
      title: "Türev ve İntegral",
      podcast: "Matematik Sohbetleri",
      duration: "42:10",
      date: "5 gün önce",
      thumbnail: "https://via.placeholder.com/80x80/4ecdc4/ffffff?text=T",
      isPlaying: false,
      progress: 0,
    },
    {
      id: 4,
      title: "Paragraf Çözme Teknikleri",
      podcast: "YKS Hazırlık Podcast",
      duration: "31:55",
      date: "1 hafta önce",
      thumbnail: "https://via.placeholder.com/80x80/667eea/ffffff?text=P",
      isPlaying: false,
      progress: 0,
    },
    {
      id: 5,
      title: "Elektrik ve Manyetizma",
      podcast: "Fizik Dünyası",
      duration: "48:30",
      date: "1 hafta önce",
      thumbnail: "https://via.placeholder.com/80x80/764ba2/ffffff?text=E",
      isPlaying: false,
      progress: 0,
    },
  ],
};

export default function PodcastScreen() {
  const insets = useSafeAreaInsets();

  // Arama metni ve seçili kategori (chip) state'leri
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedCategory, setSelectedCategory] = useState("Tümü");

  /**
   * Bölümü oynatmayı onaylamak için basit bir dialog.
   * Şimdilik sadece console.log; gerçek app'te player'ı tetikleyin.
   */
  const handlePlayEpisode = (episode: any) => {
    Alert.alert(
      "Podcast Oynat",
      `"${episode.title}" bölümünü oynatmak istiyor musunuz?`,
      [
        { text: "İptal", style: "cancel" },
        { text: "Oynat", onPress: () => console.log("Playing:", episode.title) }
      ]
    );
  };

  /**
   * Kategori chip'i (yatay scroll). Seçileni vurgular.
   */
  const renderCategoryChip = ({ item }: { item: string }) => (
    <Pressable
      style={[
        styles.categoryChip,
        selectedCategory === item && styles.categoryChipActive
      ]}
      onPress={() => setSelectedCategory(item)}
    >
      <Text style={[
        styles.categoryChipText,
        selectedCategory === item && styles.categoryChipTextActive
      ]}>
        {item}
      </Text>
    </Pressable>
  );

  /**
   * Popüler (trend) podcast kartı — yatay listede gösterilir.
   * Kart tıklanınca örnek olarak handlePlayEpisode tetiklenir.
   */
  const renderTrendingPodcast = ({ item }: { item: any }) => (
    <Pressable
      style={styles.trendingCard}
      onPress={() => handlePlayEpisode(item)}
    >
      <Image source={{ uri: item.thumbnail }} style={styles.trendingThumbnail} />
      <View style={styles.trendingContent}>
        <Text style={styles.trendingTitle} numberOfLines={2}>
          {item.title}
        </Text>
        <Text style={styles.trendingHost}>{item.host}</Text>
        <View style={styles.trendingStats}>
          <Text style={styles.trendingDuration}>⏱️ {item.duration}</Text>
          <Text style={styles.trendingListeners}>👥 {item.listeners}</Text>
        </View>
        <Text style={styles.trendingDescription} numberOfLines={2}>
          {item.description}
        </Text>
      </View>
    </Pressable>
  );

  /**
   * Bölüm kartı — dikey listede gösterilir. Oynatma durumu varsa alt çubuk görünür.
   */
  const renderEpisode = ({ item }: { item: any }) => (
    <Pressable
      style={styles.episodeCard}
      onPress={() => handlePlayEpisode(item)}
    >
      <Image source={{ uri: item.thumbnail }} style={styles.episodeThumbnail} />
      <View style={styles.episodeContent}>
        <Text style={styles.episodeTitle} numberOfLines={2}>
          {item.title}
        </Text>
        <Text style={styles.episodePodcast}>{item.podcast}</Text>
        <View style={styles.episodeMeta}>
          <Text style={styles.episodeDuration}>⏱️ {item.duration}</Text>
          <Text style={styles.episodeDate}>📅 {item.date}</Text>
        </View>
        {item.isPlaying && (
          <View style={styles.playingIndicator}>
            {/* Arka plan bar (mavi), üstüne progress (turkuaz) bindiriliyor */}
            <View style={styles.playingBar} />
            <View style={[styles.playingProgress, { width: `${item.progress}%` }]} />
          </View>
        )}
      </View>
      {/* Basit oynat/pausa simgesi — gerçek player'a bağlanınca kontrol değişir */}
      <View style={styles.playButton}>
        <Text style={styles.playIcon}>
          {item.isPlaying ? "⏸️" : "▶️"}
        </Text>
      </View>
    </Pressable>
  );

  // ---------------------------------------------------------------------------
  // Arama ve kategoriye göre bölüm filtreleme
  //  - Arama: başlık veya podcast adı içinde geçiyorsa eşleşir
  //  - Kategori: "Tümü" hariçse, bölümün ait olduğu podcast'in kategorisi ile eşleşmeli
  //    (kategori bilgisi trendingPodcasts koleksiyonundan çekiliyor)
  // ---------------------------------------------------------------------------
  const filteredEpisodes = podcastData.episodes.filter(episode => {
    const matchesSearch = episode.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
                         episode.podcast.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesCategory = selectedCategory === "Tümü" || 
                           podcastData.trendingPodcasts.find(p => p.title === episode.podcast)?.category === selectedCategory;
    return matchesSearch && matchesCategory;
  });

  return (
    <View style={[styles.container, { paddingTop: insets.top }]}>
      {/* Header */}
      <View style={styles.header}>
        <Text style={styles.headerTitle}>Podcast</Text>
        <Text style={styles.headerSubtitle}>
          {filteredEpisodes.length} bölüm bulundu
        </Text>
      </View>

      {/* Search Bar */}
      <View style={styles.searchContainer}>
        <TextInput
          style={styles.searchInput}
          placeholder="Podcast ara..."
          placeholderTextColor="#666"
          value={searchQuery}
          onChangeText={setSearchQuery}
        />
      </View>

      {/* Categories (yatay chip listesi) */}
      <View style={styles.categoriesSection}>
        <FlatList
          data={podcastData.categories}
          renderItem={renderCategoryChip}
          keyExtractor={(item) => item}
          horizontal
          showsHorizontalScrollIndicator={false}
          contentContainerStyle={styles.categoriesContainer}
        />
      </View>

      {/* Trending Podcasts (yatay kart listesi) */}
      <View style={styles.trendingSection}>
        <Text style={styles.sectionTitle}>🔥 Popüler Podcast'ler</Text>
        <FlatList
          data={podcastData.trendingPodcasts}
          renderItem={renderTrendingPodcast}
          keyExtractor={(item) => `trending-${item.id}`}
          horizontal
          showsHorizontalScrollIndicator={false}
          contentContainerStyle={styles.trendingContainer}
        />
      </View>

      {/* Episodes (dikey liste) */}
      <View style={styles.episodesSection}>
        <Text style={styles.sectionTitle}>📻 Son Bölümler</Text>
        <FlatList
          data={filteredEpisodes}
          renderItem={renderEpisode}
          keyExtractor={(item) => item.id.toString()}
          showsVerticalScrollIndicator={false}
          contentContainerStyle={styles.episodesContainer}
          ListEmptyComponent={
            <View style={styles.emptyState}>
              <Text style={styles.emptyStateText}>
                Aradığınız kriterlere uygun podcast bulunamadı
              </Text>
            </View>
          }
        />
      </View>
    </View>
  );
}

// -----------------------------------------------------------------------------
// Styles — koyu tema, kart bazlı layout
// Not: Bu dosyada stiller component içinde; proje büyürse ortak stilleri ayırmak mantıklı.
// -----------------------------------------------------------------------------
const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#000",
  },
  header: {
    paddingHorizontal: 20,
    paddingVertical: 16,
  },
  headerTitle: {
    fontSize: 32,
    fontWeight: "700",
    color: "#fff",
    marginBottom: 4,
  },
  headerSubtitle: {
    fontSize: 16,
    color: "#666",
  },
  searchContainer: {
    paddingHorizontal: 20,
    marginBottom: 20,
  },
  searchInput: {
    backgroundColor: "#1a1a1a",
    borderRadius: 12,
    paddingHorizontal: 16,
    paddingVertical: 12,
    fontSize: 16,
    color: "#fff",
    borderWidth: 1,
    borderColor: "#333",
  },
  categoriesSection: {
    marginBottom: 20,
  },
  categoriesContainer: {
    paddingHorizontal: 20,
    gap: 8,
  },
  categoryChip: {
    backgroundColor: "#1a1a1a",
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 20,
    borderWidth: 1,
    borderColor: "#333",
  },
  categoryChipActive: {
    backgroundColor: "#0ea5e9",
    borderColor: "#0ea5e9",
  },
  categoryChipText: {
    color: "#fff",
    fontSize: 14,
    fontWeight: "500",
  },
  categoryChipTextActive: {
    color: "#fff",
    fontWeight: "600",
  },
  trendingSection: {
    marginBottom: 20,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: "600",
    color: "#fff",
    marginBottom: 12,
    paddingHorizontal: 20,
  },
  trendingContainer: {
    paddingHorizontal: 20,
    gap: 12,
  },
  trendingCard: {
    width: 200,
    backgroundColor: "#1a1a1a",
    borderRadius: 16,
    padding: 12,
    borderWidth: 1,
    borderColor: "#333",
  },
  trendingThumbnail: {
    width: "100%",
    height: 120,
    borderRadius: 12,
    marginBottom: 12,
  },
  trendingContent: {
    flex: 1,
  },
  trendingTitle: {
    fontSize: 14,
    fontWeight: "600",
    color: "#fff",
    marginBottom: 4,
    lineHeight: 18,
  },
  trendingHost: {
    fontSize: 12,
    color: "#0ea5e9",
    marginBottom: 8,
  },
  trendingStats: {
    flexDirection: "row",
    justifyContent: "space-between",
    marginBottom: 8,
  },
  trendingDuration: {
    fontSize: 11,
    color: "#666",
  },
  trendingListeners: {
    fontSize: 11,
    color: "#666",
  },
  trendingDescription: {
    fontSize: 11,
    color: "#999",
    lineHeight: 14,
  },
  episodesSection: {
    flex: 1,
  },
  episodesContainer: {
    paddingHorizontal: 20,
    paddingBottom: 20,
  },
  episodeCard: {
    flexDirection: "row",
    backgroundColor: "#1a1a1a",
    borderRadius: 12,
    padding: 12,
    marginBottom: 12,
    borderWidth: 1,
    borderColor: "#333",
    alignItems: "center",
  },
  episodeThumbnail: {
    width: 60,
    height: 60,
    borderRadius: 8,
    marginRight: 12,
  },
  episodeContent: {
    flex: 1,
  },
  episodeTitle: {
    fontSize: 16,
    fontWeight: "600",
    color: "#fff",
    marginBottom: 4,
    lineHeight: 20,
  },
  episodePodcast: {
    fontSize: 14,
    color: "#0ea5e9",
    marginBottom: 6,
  },
  episodeMeta: {
    flexDirection: "row",
    justifyContent: "space-between",
  },
  episodeDuration: {
    fontSize: 12,
    color: "#666",
  },
  episodeDate: {
    fontSize: 12,
    color: "#666",
  },
  playingIndicator: {
    height: 3,
    backgroundColor: "#333",
    borderRadius: 2,
    marginTop: 8,
    overflow: "hidden",
  },
  playingBar: {
    height: "100%",
    backgroundColor: "#0ea5e9",
    borderRadius: 2,
  },
  playingProgress: {
    height: "100%",
    backgroundColor: "#4ecdc4",
    borderRadius: 2,
    position: "absolute",
    top: 0,
    left: 0,
  },
  playButton: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: "#0ea5e9",
    justifyContent: "center",
    alignItems: "center",
    marginLeft: 12,
  },
  playIcon: {
    fontSize: 16,
  },
  emptyState: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    paddingVertical: 60,
  },
  emptyStateText: {
    color: "#666",
    fontSize: 16,
    textAlign: "center",
  },
});
