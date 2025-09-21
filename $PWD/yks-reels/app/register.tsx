// app/register.tsx
import React, { useState } from "react";
import { View, Text, TextInput, Pressable, Alert } from "react-native";
import { router } from "expo-router";
import AsyncStorage from "@react-native-async-storage/async-storage";

export default function RegisterScreen() {
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [pass, setPass] = useState("");

  const submit = async () => {
    if (!name || !email || !pass) return Alert.alert("Uyarı", "Tüm alanları doldurun.");
    await AsyncStorage.setItem("user", JSON.stringify({ name, email }));
    router.replace("/(tabs)/explore");
  };

  return (
    <View style={{ flex: 1, backgroundColor: "#000", padding: 20, gap: 12, justifyContent: "center" }}>
      <Text style={{ color: "#fff", fontSize: 24, fontWeight: "700", marginBottom: 8 }}>Kayıt Ol</Text>

      <TextInput
        placeholder="Ad Soyad"
        placeholderTextColor="#888"
        style={{ backgroundColor: "#111", color: "#fff", borderRadius: 10, padding: 12 }}
        value={name}
        onChangeText={setName}
      />
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
          backgroundColor: "#22c55e",
          paddingVertical: 14,
          borderRadius: 12,
          opacity: pressed ? 0.85 : 1,
          marginTop: 6,
        })}
      >
        <Text style={{ color: "#04140a", textAlign: "center", fontSize: 16, fontWeight: "700" }}>
          Kayıt ol
        </Text>
      </Pressable>

      <Pressable onPress={() => router.replace("/Login")} style={{ marginTop: 10 }}>
        <Text style={{ color: "#9ae6ff", textAlign: "center" }}>Zaten hesabın var mı? Giriş yap</Text>
      </Pressable>
    </View>
  );
}
