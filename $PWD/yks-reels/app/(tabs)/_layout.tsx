// app/(tabs)/_layout.tsx
import { Tabs } from "expo-router";
import { Ionicons } from "@expo/vector-icons";
import { useColorScheme } from "react-native";

export default function TabsLayout() {
  const scheme = useColorScheme();
  return (
    <Tabs
      initialRouteName="index" // Keşfet
      screenOptions={{
        headerShown: false,
        tabBarStyle: { backgroundColor: scheme === "dark" ? "#111" : "#fff" },
        tabBarActiveTintColor: scheme === "dark" ? "#4CC9F0" : "#0EA5E9",
        tabBarInactiveTintColor: scheme === "dark" ? "#9CA3AF" : "#6B7280",
        tabBarLabelStyle: { fontSize: 12 },
      }}
    >
      {/* 1) Keşfet -> index.tsx */}
      <Tabs.Screen
        name="index"
        options={{
          title: "Keşfet",
          tabBarIcon: ({ color, size }) => (
            <Ionicons name="compass-outline" size={size} color={color} />
          ),
        }}
      />

      {/* 2) Hesabım -> explore.tsx */}
      <Tabs.Screen
        name="explore"
        options={{
          title: "Hesabım",
          tabBarIcon: ({ color, size }) => (
            <Ionicons name="person-circle-outline" size={size} color={color} />
          ),
        }}
      />

      {/* 3) Podcast -> podcast.tsx */}
      <Tabs.Screen
        name="podcast"
        options={{
          title: "Podcast",
          tabBarIcon: ({ color, size }) => (
            <Ionicons name="mic-outline" size={size} color={color} />
          ),
        }}
      />

      {/* 4) Giriş -> account.tsx */}
      <Tabs.Screen
        name="account"
        options={{
          title: "Giriş",
          tabBarIcon: ({ color, size }) => (
            <Ionicons name="log-in-outline" size={size} color={color} />
          ),
        }}
      />
    </Tabs>
  );
}
