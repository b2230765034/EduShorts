

import React, { memo, useCallback, useMemo } from "react";
import {
  View,
  Text,
  FlatList,
  Pressable,
  ScrollView,
  Image,
  StyleSheet,
  Alert,
  AccessibilityInfo,
} from "react-native";
import { useSafeAreaInsets } from "react-native-safe-area-context";
import { router } from "expo-router";

// -----------------------------------------------------------------------------
// Types
// -----------------------------------------------------------------------------

interface Achievement {
  id: number;
  title: string;
  description: string;
  icon: string; // emoji or remote icon
  earned: boolean;
}

interface ActivityItem {
  id: number;
  title: string;
  subject: string;
  time: string; // localized relative time string
  completed: boolean;
}

interface UserData {
  name: string;
  email: string;
  avatar: string; // url
  level: number;
  xp: number;
  nextLevelXp: number;
  studyStreak: number; // days
  totalVideosWatched: number;
  totalStudyTime: string; // human readable
  achievements: Achievement[];
  recentActivity: ActivityItem[];
}

// -----------------------------------------------------------------------------
// Mock Data (replace with real data source / selector later)
// -----------------------------------------------------------------------------

const MOCK_USER: UserData = {
  name: "Ahmet Yılmaz",
  email: "ahmet@example.com",
  avatar: "https://via.placeholder.com/100",
  level: 12,
  xp: 2450,
  nextLevelXp: 3000,
  studyStreak: 7,
  totalVideosWatched: 156,
  totalStudyTime: "42 saat",
  achievements: [
    { id: 1, title: "İlk Video", description: "İlk videonuzu izlediniz", icon: "🎬", earned: true },
    { id: 2, title: "Haftalık Hedef", description: "7 gün üst üste çalıştınız", icon: "🔥", earned: true },
    { id: 3, title: "Fizik Ustası", description: "50 fizik videosu izlediniz", icon: "⚡", earned: false },
  ],
  recentActivity: [
    { id: 1, title: "Basınç Konusu", subject: "Fizik", time: "2 saat önce", completed: true },
    { id: 2, title: "Limit Konusu", subject: "Matematik", time: "1 gün önce", completed: true },
    { id: 3, title: "Paragraf Konusu", subject: "Türkçe", time: "2 gün önce", completed: false },
  ],
};

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

/** Clamp a number between 0 and 100 to avoid layout glitches. */
const clampPercent = (value: number) => Math.max(0, Math.min(100, value));

// -----------------------------------------------------------------------------
// Small presentational components
// -----------------------------------------------------------------------------

interface StatCardProps {
  title: string;
  value: string;
  icon: string;
}

const StatCard = memo(({ title, value, icon }: StatCardProps) => (
  <View style={styles.statCard} accessibilityRole="summary" accessibilityLabel={`${title} ${value}`}>
    <Text style={styles.statIcon}>{icon}</Text>
    <Text style={styles.statValue}>{value}</Text>
    <Text style={styles.statTitle}>{title}</Text>
  </View>
));

interface AchievementItemProps {
  item: Achievement;
}

const AchievementRow = memo(({ item }: AchievementItemProps) => (
  <View style={[styles.achievementCard, !item.earned && styles.achievementLocked]}
        accessibilityRole="image"
        accessibilityLabel={`${item.title} — ${item.description} ${item.earned ? "kazanıldı" : "kilitli"}`}>
    <Text style={styles.achievementIcon}>{item.icon}</Text>
    <View style={styles.achievementContent}>
      <Text style={[styles.achievementTitle, !item.earned && styles.achievementTitleLocked]}>
        {item.title}
      </Text>
      <Text style={[styles.achievementDescription, !item.earned && styles.achievementDescriptionLocked]}>
        {item.description}
      </Text>
    </View>
    {item.earned && <Text style={styles.achievementCheck}>✓</Text>}
  </View>
));

interface ActivityRowProps { item: ActivityItem }

const ActivityRow = memo(({ item }: ActivityRowProps) => (
  <View style={styles.activityItem}
        accessibilityRole="summary"
        accessibilityLabel={`${item.subject} — ${item.title} — ${item.time}`}>
    <View style={[styles.activityIndicator, item.completed && styles.activityCompleted]} />
    <View style={styles.activityContent}>
      <Text style={styles.activityTitle}>{item.title}</Text>
      <Text style={styles.activitySubject}>{item.subject}</Text>
    </View>
    <Text style={styles.activityTime}>{item.time}</Text>
  </View>
));

interface MenuOptionProps {
  title: string;
  icon: string;
  onPress: () => void;
}

const MenuOption = memo(({ title, icon, onPress }: MenuOptionProps) => (
  <Pressable
    style={styles.menuOption}
    onPress={onPress}
    hitSlop={8}
    accessibilityRole="button"
    accessibilityLabel={`${title} aç`}
  >
    <Text style={styles.menuIcon}>{icon}</Text>
    <Text style={styles.menuTitle}>{title}</Text>
    <Text style={styles.menuArrow}>›</Text>
  </Pressable>
));

// -----------------------------------------------------------------------------
// Screen
// -----------------------------------------------------------------------------

export default function ExploreScreen() {
  const insets = useSafeAreaInsets();

  // In a real app, swap MOCK_USER with state/selector.
  const user = MOCK_USER;

  const progressPercentage = useMemo(() => {
    const raw = (user.xp / Math.max(1, user.nextLevelXp)) * 100;
    return clampPercent(raw);
  }, [user.xp, user.nextLevelXp]);

  const handleEditProfile = useCallback(() => {
    Alert.alert("Profil Düzenle", "Profil düzenleme özelliği yakında eklenecek!");
  }, []);

  const handleLogout = useCallback(() => {
    Alert.alert(
      "Çıkış Yap",
      "Hesabınızdan çıkmak istediğinizden emin misiniz?",
      [
        { text: "İptal", style: "cancel" },
        {
          text: "Çıkış Yap",
          style: "destructive",
          onPress: () => {
            AccessibilityInfo.announceForAccessibility?.("Çıkış yapılıyor");
            router.push("/Login");
          },
        },
      ]
    );
  }, []);

  return (
    <ScrollView
      style={[styles.container, { paddingTop: insets.top }]}
      showsVerticalScrollIndicator={false}
    >
      {/* Profile Header */}
      <View style={styles.profileHeader}>
        <View style={styles.profileContent}>
          <View style={styles.avatarContainer}>
            <Image source={{ uri: user.avatar }} style={styles.avatar} />
            <View style={styles.levelBadge}>
              <Text style={styles.levelText}>{user.level}</Text>
            </View>
          </View>
          <Text style={styles.userName}>{user.name}</Text>
          <Text style={styles.userEmail}>{user.email}</Text>

          {/* XP Progress */}
          <View style={styles.xpContainer}>
            <View style={styles.xpHeader}>
              <Text style={styles.xpText}>Seviye {user.level}</Text>
              <Text style={styles.xpValue}>
                {user.xp}/{user.nextLevelXp} XP
              </Text>
            </View>
            <View style={styles.xpProgressBar}>
              <View style={[styles.xpProgress, { width: `${progressPercentage}%` }]} />
            </View>
          </View>
        </View>
      </View>

      {/* Stats Cards */}
      <View style={styles.statsContainer}>
        <StatCard title="Günlük Seri" value={`${user.studyStreak} gün`} icon="🔥" />
        <StatCard title="İzlenen Video" value={String(user.totalVideosWatched)} icon="📺" />
        <StatCard title="Toplam Süre" value={user.totalStudyTime} icon="⏱️" />
      </View>

      {/* Recent Activity */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Son Aktiviteler</Text>
        <FlatList
          data={user.recentActivity}
          renderItem={({ item }) => <ActivityRow item={item} />}
          keyExtractor={(i) => i.id.toString()}
          scrollEnabled={false}
        />
      </View>

      {/* Achievements */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Başarımlar</Text>
        <FlatList
          data={user.achievements}
          renderItem={({ item }) => <AchievementRow item={item} />}
          keyExtractor={(i) => i.id.toString()}
          scrollEnabled={false}
        />
      </View>

      {/* Menu Options */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Hesap</Text>
        <MenuOption title="Profil Düzenle" icon="✏️" onPress={handleEditProfile} />
        <MenuOption title="Bildirimler" icon="🔔" onPress={() => Alert.alert("Bildirimler", "Bildirim ayarları yakında!")} />
        <MenuOption title="Yardım & Destek" icon="❓" onPress={() => Alert.alert("Yardım", "Yardım sayfası yakında!")} />
        <MenuOption title="Çıkış Yap" icon="🚪" onPress={handleLogout} />
      </View>

      <View style={styles.bottomSpacer} />
    </ScrollView>
  );
}

// -----------------------------------------------------------------------------
// Styles
// -----------------------------------------------------------------------------

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#000",
  },
  profileHeader: {
    paddingHorizontal: 20,
    paddingVertical: 30,
    marginBottom: 20,
    backgroundColor: "#667eea",
  },
  profileContent: {
    alignItems: "center",
  },
  avatarContainer: {
    position: "relative",
    marginBottom: 16,
  },
  avatar: {
    width: 100,
    height: 100,
    borderRadius: 50,
    borderWidth: 4,
    borderColor: "rgba(255, 255, 255, 0.3)",
  },
  levelBadge: {
    position: "absolute",
    bottom: -5,
    right: -5,
    backgroundColor: "#ff6b6b",
    width: 32,
    height: 32,
    borderRadius: 16,
    justifyContent: "center",
    alignItems: "center",
    borderWidth: 3,
    borderColor: "#fff",
  },
  levelText: {
    color: "#fff",
    fontSize: 14,
    fontWeight: "700",
  },
  userName: {
    fontSize: 24,
    fontWeight: "700",
    color: "#fff",
    marginBottom: 4,
  },
  userEmail: {
    fontSize: 16,
    color: "rgba(255, 255, 255, 0.8)",
    marginBottom: 20,
  },
  xpContainer: {
    width: "100%",
  },
  xpHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
    marginBottom: 8,
  },
  xpText: {
    color: "#fff",
    fontSize: 14,
    fontWeight: "600",
  },
  xpValue: {
    color: "rgba(255, 255, 255, 0.8)",
    fontSize: 14,
  },
  xpProgressBar: {
    height: 8,
    backgroundColor: "rgba(255, 255, 255, 0.2)",
    borderRadius: 4,
    overflow: "hidden",
  },
  xpProgress: {
    height: "100%",
    backgroundColor: "#4ecdc4",
    borderRadius: 4,
  },
  statsContainer: {
    flexDirection: "row",
    paddingHorizontal: 20,
    marginBottom: 30,
    gap: 12,
  },
  statCard: {
    flex: 1,
    backgroundColor: "#1a1a1a",
    borderRadius: 16,
    padding: 16,
    alignItems: "center",
    borderWidth: 1,
    borderColor: "#333",
  },
  statIcon: {
    fontSize: 24,
    marginBottom: 8,
  },
  statValue: {
    fontSize: 20,
    fontWeight: "700",
    color: "#fff",
    marginBottom: 4,
  },
  statTitle: {
    fontSize: 12,
    color: "#666",
    textAlign: "center",
  },
  section: {
    marginBottom: 30,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: "700",
    color: "#fff",
    marginBottom: 16,
    paddingHorizontal: 20,
  },
  activityItem: {
    flexDirection: "row",
    alignItems: "center",
    paddingHorizontal: 20,
    paddingVertical: 12,
    backgroundColor: "#1a1a1a",
    marginHorizontal: 20,
    marginBottom: 8,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: "#333",
  },
  activityIndicator: {
    width: 12,
    height: 12,
    borderRadius: 6,
    backgroundColor: "#666",
    marginRight: 12,
  },
  activityCompleted: {
    backgroundColor: "#4ecdc4",
  },
  activityContent: {
    flex: 1,
  },
  activityTitle: {
    fontSize: 16,
    fontWeight: "600",
    color: "#fff",
    marginBottom: 2,
  },
  activitySubject: {
    fontSize: 14,
    color: "#0ea5e9",
  },
  activityTime: {
    fontSize: 12,
    color: "#666",
  },
  achievementCard: {
    flexDirection: "row",
    alignItems: "center",
    paddingHorizontal: 20,
    paddingVertical: 16,
    backgroundColor: "#1a1a1a",
    marginHorizontal: 20,
    marginBottom: 8,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: "#333",
  },
  achievementLocked: {
    opacity: 0.5,
  },
  achievementIcon: {
    fontSize: 24,
    marginRight: 16,
  },
  achievementContent: {
    flex: 1,
  },
  achievementTitle: {
    fontSize: 16,
    fontWeight: "600",
    color: "#fff",
    marginBottom: 2,
  },
  achievementTitleLocked: {
    color: "#666",
  },
  achievementDescription: {
    fontSize: 14,
    color: "#666",
  },
  achievementDescriptionLocked: {
    color: "#444",
  },
  achievementCheck: {
    fontSize: 20,
    color: "#4ecdc4",
    fontWeight: "700",
  },
  menuOption: {
    flexDirection: "row",
    alignItems: "center",
    paddingHorizontal: 20,
    paddingVertical: 16,
    backgroundColor: "#1a1a1a",
    marginHorizontal: 20,
    marginBottom: 8,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: "#333",
  },
  menuIcon: {
    fontSize: 20,
    marginRight: 16,
  },
  menuTitle: {
    flex: 1,
    fontSize: 16,
    fontWeight: "500",
    color: "#fff",
  },
  menuArrow: {
    fontSize: 20,
    color: "#666",
  },
  bottomSpacer: {
    height: 40,
  },
});
