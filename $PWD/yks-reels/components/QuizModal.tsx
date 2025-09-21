// components/QuizModal.tsx — Cleaned & Commented
// -----------------------------------------------------------------------------
// Purpose: Basit bir quiz modalı. Bir soru ve şıklardan oluşur; seçimde
// doğru/yanlış bilgisini `onClose(isCorrect)` ile yukarıya bildirir.
// -----------------------------------------------------------------------------

import React, { useCallback, useMemo, useRef } from "react";
import { Modal, View, Text, Pressable, StyleSheet } from "react-native";
import type { Quiz } from "../constants/videos";

export type QuizModalProps = {
  visible: boolean;
  onClose: (isCorrect: boolean | null) => void;
  quiz?: Quiz;
  videoId: number;
  userId: string;
};

export default function QuizModal({ visible, onClose, quiz, videoId, userId }: QuizModalProps) {
  // Quiz yoksa modal'ı göstermiyoruz (erken çıkış)
  if (!quiz) return null;

  // Aynı anda birden fazla submit'i engellemek için basit kilit
  const submittingRef = useRef(false);

  
  const submit = useCallback(async (choiceIdx: number) => {
    if (submittingRef.current) return;
    submittingRef.current = true;

    const is_correct = choiceIdx === quiz.correct;
   

    onClose(is_correct);
    submittingRef.current = false;
  }, [onClose, quiz, userId, videoId]);

  const choices = useMemo(() => quiz.a, [quiz.a]);

  return (
    <Modal visible={visible} transparent animationType="fade">
      <View style={styles.backdrop}>
        <View style={styles.card}>
          <Text style={styles.question} accessibilityRole="header">
            {quiz.q}
          </Text>

          {choices.map((opt, i) => (
            <Pressable
              key={`${i}-${opt}`}
              onPress={() => submit(i)}
              style={({ pressed }) => [styles.choice, pressed && styles.choicePressed]}
              accessibilityRole="button"
              accessibilityLabel={`Şık ${i + 1}: ${opt}`}
              hitSlop={6}
            >
              <Text style={styles.choiceText}>{opt}</Text>
            </Pressable>
          ))}

          <Pressable
            onPress={() => onClose(null)}
            style={({ pressed }) => [styles.closeBtn, pressed && styles.closeBtnPressed]}
            accessibilityRole="button"
            accessibilityLabel="Kapat"
            hitSlop={6}
          >
            <Text style={styles.closeText}>Kapat</Text>
          </Pressable>
        </View>
      </View>
    </Modal>
  );
}

// -----------------------------------------------------------------------------
// Styles
// -----------------------------------------------------------------------------
const styles = StyleSheet.create({
  backdrop: {
    flex: 1,
    backgroundColor: "rgba(0,0,0,0.6)",
    justifyContent: "center",
    padding: 16,
  },
  card: {
    backgroundColor: "#fff",
    borderRadius: 12,
    padding: 16,
  },
  question: {
    fontSize: 18,
    fontWeight: "600",
    marginBottom: 12,
    color: "#111",
  },
  choice: {
    padding: 12,
    borderWidth: 1,
    borderColor: "#ddd",
    borderRadius: 8,
    marginBottom: 8,
    backgroundColor: "#fff",
  },
  choicePressed: {
    backgroundColor: "#f3f4f6",
  },
  choiceText: {
    color: "#111",
    fontSize: 15,
  },
  closeBtn: {
    padding: 12,
    marginTop: 4,
    alignSelf: "center",
  },
  closeBtnPressed: {
    opacity: 0.7,
  },
  closeText: {
    textAlign: "center",
    color: "#111",
    fontSize: 15,
    fontWeight: "500",
  },
});
