// app/login.tsx
import React, { useState } from "react";
import { View, Text, TextInput, Pressable, Alert } from "react-native";
import { router } from "expo-router";
import AsyncStorage from "@react-native-async-storage/async-storage";

export default function LoginScreen() {
  const [email, setEmail] = useState("");
  const [pass, setPass] = useState("");

  const submit = async () => {
    if (!email || !pass) return Alert.alert("Uyarı", "E-posta ve şifre gerekli.");
    // demo: kullanıcıyı kaydet varmış gibi kabul ediyoruz
    await AsyncStorage.setItem("user", JSON.stringify({ name: email.split("@")[0], email }));
    router.replace("/(tabs)/explore");
  };

  return (
    <View style={{ flex: 1, backgroundColor: "#000", padding: 20, gap: 12, justifyContent: "center" }}>
      <Text style={{ color: "#fff", fontSize: 24, fontWeight: "700", marginBottom: 8 }}>Giriş Yap</Text>

      <TextInput
        placeholder="E-posta"
        placeholderTextColor="#888"
        style={{ backgroundColor: "#111", color: "#fff", borderRadius: 10, padding: 12 }}
        autoCapitalize="none"
        keyboardType="email-address"
        value={email}
        onChangeText={setEmail}
      />
      <TextInput
        placeholder="Şifre"
        placeholderTextColor="#888"
        style={{ backgroundColor: "#111", color: "#fff", borderRadius: 10, padding: 12 }}
        secureTextEntry
        value={pass}
        onChangeText={setPass}
      />

      <Pressable
        onPress={submit}
        style={({ pressed }) => ({
          backgroundColor: "#0ea5e9",
          paddingVertical: 14,
          borderRadius: 12,
          opacity: pressed ? 0.85 : 1,
          marginTop: 6,
        })}
      >
        <Text style={{ color: "#fff", textAlign: "center", fontSize: 16, fontWeight: "600" }}>Giriş yap</Text>
      </Pressable>

      <Pressable onPress={() => router.replace("/register")} style={{ marginTop: 10 }}>
        <Text style={{ color: "#9ae6ff", textAlign: "center" }}>Hesabın yok mu? Kayıt ol</Text>
      </Pressable>
    </View>
  );
}
