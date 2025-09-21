
import React, { useEffect, useState } from "react";
import { View, Text, Pressable, ActivityIndicator, StyleSheet } from "react-native";
import { router } from "expo-router";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { useSafeAreaInsets } from "react-native-safe-area-context";

type User = { name: string; email: string };

export default function AccountScreen() {
  const insets = useSafeAreaInsets();
  const [loading, setLoading] = useState(true);
  const [user, setUser] = useState<User | null>(null);

  useEffect(() => {
    (async () => {
      try {
        const raw = await AsyncStorage.getItem("user");
        setUser(raw ? JSON.parse(raw) : null);
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  const logout = async () => {
    await AsyncStorage.removeItem("user");
    setUser(null);
  };

  if (loading) {
    return (
      <View style={[styles.container, { paddingTop: insets.top }]}>
        <View style={styles.loadingContainer}>
          <ActivityIndicator color="#fff" size="large" />
          <Text style={styles.loadingText}>Yükleniyor...</Text>
        </View>
      </View>
    );
  }

  return (
    <View style={[styles.container, { paddingTop: insets.top }]}>
      <View style={styles.header}>
        <Text style={styles.headerTitle}>Giriş</Text>
      </View>

      {!user ? (
        <View style={styles.authContainer}>
          <View style={styles.welcomeSection}>
            <Text style={styles.welcomeTitle}>Hoş Geldiniz!</Text>
            <Text style={styles.welcomeSubtitle}>
              Eğitim videolarınızı keşfetmek için giriş yapın
            </Text>
          </View>

          <View style={styles.buttonContainer}>
            <Pressable
              onPress={() => router.push("/Login")}
              style={({ pressed }) => [
                styles.primaryButton,
                { opacity: pressed ? 0.85 : 1 }
              ]}
            >
              <Text style={styles.primaryButtonText}>Giriş Yap</Text>
            </Pressable>

            <Pressable
              onPress={() => router.push("/register")}
              style={({ pressed }) => [
                styles.secondaryButton,
                { opacity: pressed ? 0.85 : 1 }
              ]}
            >
              <Text style={styles.secondaryButtonText}>Kayıt Ol</Text>
            </Pressable>
          </View>

          <View style={styles.featuresSection}>
            <Text style={styles.featuresTitle}>Özellikler:</Text>
            <View style={styles.featureItem}>
              <Text style={styles.featureIcon}>📚</Text>
              <Text style={styles.featureText}>Binlerce eğitim videosu</Text>
            </View>
            <View style={styles.featureItem}>
              <Text style={styles.featureIcon}>🎯</Text>
              <Text style={styles.featureText}>Kişiselleştirilmiş öneriler</Text>
            </View>
            <View style={styles.featureItem}>
              <Text style={styles.featureIcon}>📊</Text>
              <Text style={styles.featureText}>İlerleme takibi</Text>
            </View>
          </View>
        </View>
      ) : (
        <View style={styles.userContainer}>
          <View style={styles.profileSection}>
            <View style={styles.avatarContainer}>
              <Text style={styles.avatarText}>
                {user.name.charAt(0).toUpperCase()}
              </Text>
            </View>
            <Text style={styles.userName}>{user.name}</Text>
            <Text style={styles.userEmail}>{user.email}</Text>
          </View>

          <View style={styles.menuSection}>
            <Pressable style={styles.menuItem}>
              <Text style={styles.menuIcon}>📚</Text>
              <Text style={styles.menuText}>İzleme Geçmişi</Text>
              <Text style={styles.menuArrow}>›</Text>
            </Pressable>

            <Pressable style={styles.menuItem}>
              <Text style={styles.menuIcon}>❤️</Text>
              <Text style={styles.menuText}>Beğenilen Videolar</Text>
              <Text style={styles.menuArrow}>›</Text>
            </Pressable>

            <Pressable style={styles.menuItem}>
              <Text style={styles.menuIcon}>💾</Text>
              <Text style={styles.menuText}>Kaydedilen Videolar</Text>
              <Text style={styles.menuArrow}>›</Text>
            </Pressable>

            <Pressable style={styles.menuItem}>
              <Text style={styles.menuIcon}>⚙️</Text>
              <Text style={styles.menuText}>Ayarlar</Text>
              <Text style={styles.menuArrow}>›</Text>
            </Pressable>
          </View>

          <Pressable
            onPress={logout}
            style={({ pressed }) => [
              styles.logoutButton,
              { opacity: pressed ? 0.85 : 1 }
            ]}
          >
            <Text style={styles.logoutButtonText}>Çıkış Yap</Text>
          </Pressable>
        </View>
      )}
    </View>
  );
}

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
  },
  loadingContainer: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    gap: 16,
  },
  loadingText: {
    color: "#fff",
    fontSize: 16,
  },
  authContainer: {
    flex: 1,
    paddingHorizontal: 20,
    justifyContent: "center",
  },
  welcomeSection: {
    marginBottom: 40,
    alignItems: "center",
  },
  welcomeTitle: {
    fontSize: 28,
    fontWeight: "700",
    color: "#fff",
    marginBottom: 8,
    textAlign: "center",
  },
  welcomeSubtitle: {
    fontSize: 16,
    color: "#666",
    textAlign: "center",
    lineHeight: 22,
  },
  buttonContainer: {
    gap: 16,
    marginBottom: 40,
  },
  primaryButton: {
    backgroundColor: "#0ea5e9",
    paddingVertical: 16,
    borderRadius: 12,
    alignItems: "center",
  },
  primaryButtonText: {
    color: "#fff",
    fontSize: 16,
    fontWeight: "600",
  },
  secondaryButton: {
    backgroundColor: "#22c55e",
    paddingVertical: 16,
    borderRadius: 12,
    alignItems: "center",
  },
  secondaryButtonText: {
    color: "#04140a",
    fontSize: 16,
    fontWeight: "700",
  },
  featuresSection: {
    gap: 16,
  },
  featuresTitle: {
    fontSize: 18,
    fontWeight: "600",
    color: "#fff",
    marginBottom: 8,
  },
  featureItem: {
    flexDirection: "row",
    alignItems: "center",
    gap: 12,
  },
  featureIcon: {
    fontSize: 20,
  },
  featureText: {
    color: "#ccc",
    fontSize: 16,
  },
  userContainer: {
    flex: 1,
    paddingHorizontal: 20,
  },
  profileSection: {
    alignItems: "center",
    paddingVertical: 32,
    borderBottomWidth: 1,
    borderBottomColor: "#333",
    marginBottom: 32,
  },
  avatarContainer: {
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: "#0ea5e9",
    justifyContent: "center",
    alignItems: "center",
    marginBottom: 16,
  },
  avatarText: {
    color: "#fff",
    fontSize: 32,
    fontWeight: "700",
  },
  userName: {
    fontSize: 24,
    fontWeight: "600",
    color: "#fff",
    marginBottom: 4,
  },
  userEmail: {
    fontSize: 16,
    color: "#666",
  },
  menuSection: {
    gap: 8,
    marginBottom: 32,
  },
  menuItem: {
    flexDirection: "row",
    alignItems: "center",
    paddingVertical: 16,
    paddingHorizontal: 16,
    backgroundColor: "#1a1a1a",
    borderRadius: 12,
    gap: 16,
  },
  menuIcon: {
    fontSize: 20,
  },
  menuText: {
    flex: 1,
    color: "#fff",
    fontSize: 16,
    fontWeight: "500",
  },
  menuArrow: {
    color: "#666",
    fontSize: 20,
  },
  logoutButton: {
    backgroundColor: "#ef4444",
    paddingVertical: 16,
    borderRadius: 12,
    alignItems: "center",
  },
  logoutButtonText: {
    color: "#fff",
    fontSize: 16,
    fontWeight: "600",
  },
});