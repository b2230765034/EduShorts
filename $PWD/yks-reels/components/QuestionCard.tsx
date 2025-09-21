// components/QuestionCard.tsx
import React, { useState } from "react";
import {
  View,
  Text,
  Pressable,
  StyleSheet,
  Dimensions,
} from "react-native";
import { QuestionItem } from "../constants/videos";

const { width: screenWidth, height: screenHeight } = Dimensions.get("window");

type Props = {
  item: QuestionItem;
  onAnswer: (isCorrect: boolean) => void;
  onSkip: () => void;
};

export default function QuestionCard({ item, onAnswer, onSkip }: Props) {
  const [selectedAnswer, setSelectedAnswer] = useState<number | null>(null);
  const [showResult, setShowResult] = useState(false);

  const handleAnswerSelect = (answerIndex: number) => {
    if (showResult) return;
    
    setSelectedAnswer(answerIndex);
    setShowResult(true);
    
    // Call onAnswer after a short delay to show the result
    setTimeout(() => {
      onAnswer(answerIndex === item.question.correct);
    }, 1500);
  };

  const handleSkip = () => {
    onSkip();
  };

  const getAnswerStyle = (index: number) => {
    if (!showResult) {
      return [
        styles.answerButton,
        selectedAnswer === index && styles.selectedAnswer
      ];
    }
    
    if (index === item.question.correct) {
      return [styles.answerButton, styles.correctAnswer];
    }
    
    if (index === selectedAnswer && index !== item.question.correct) {
      return [styles.answerButton, styles.wrongAnswer];
    }
    
    return [styles.answerButton, styles.disabledAnswer];
  };

  const getAnswerTextStyle = (index: number) => {
    if (!showResult) {
      return [
        styles.answerText,
        selectedAnswer === index && styles.selectedAnswerText
      ];
    }
    
    if (index === item.question.correct) {
      return [styles.answerText, styles.correctAnswerText];
    }
    
    if (index === selectedAnswer && index !== item.question.correct) {
      return [styles.answerText, styles.wrongAnswerText];
    }
    
    return [styles.answerText, styles.disabledAnswerText];
  };

  return (
    <View style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <View style={styles.subjectBadge}>
          <Text style={styles.subjectText}>{item.subject}</Text>
        </View>
        <Text style={styles.topicText}>{item.topic}</Text>
      </View>

      {/* Question */}
      <View style={styles.questionContainer}>
        <Text style={styles.questionText}>{item.question.q}</Text>
      </View>

      {/* Answers */}
      <View style={styles.answersContainer}>
        {item.question.a.map((answer, index) => (
          <Pressable
            key={index}
            style={getAnswerStyle(index)}
            onPress={() => handleAnswerSelect(index)}
            disabled={showResult}
          >
            <Text style={getAnswerTextStyle(index)}>
              {String.fromCharCode(65 + index)}. {answer}
            </Text>
          </Pressable>
        ))}
      </View>

      {/* Skip Button */}
      {!showResult && (
        <View style={styles.skipContainer}>
          <Pressable style={styles.skipButton} onPress={handleSkip}>
            <Text style={styles.skipText}>Geç</Text>
          </Pressable>
        </View>
      )}

      {/* Result Message */}
      {showResult && (
        <View style={styles.resultContainer}>
          <Text style={styles.resultText}>
            {selectedAnswer === item.question.correct ? "✅ Doğru!" : "❌ Yanlış!"}
          </Text>
          <Text style={styles.explanationText}>
            Doğru cevap: {String.fromCharCode(65 + item.question.correct)}. {item.question.a[item.question.correct]}
          </Text>
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#000",
    paddingHorizontal: 20,
    paddingVertical: 40,
    justifyContent: "center",
  },
  header: {
    alignItems: "center",
    marginBottom: 40,
  },
  subjectBadge: {
    backgroundColor: "#0ea5e9",
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 20,
    marginBottom: 8,
  },
  subjectText: {
    color: "#fff",
    fontSize: 14,
    fontWeight: "600",
  },
  topicText: {
    color: "#666",
    fontSize: 16,
  },
  questionContainer: {
    marginBottom: 40,
  },
  questionText: {
    color: "#fff",
    fontSize: 24,
    fontWeight: "700",
    textAlign: "center",
    lineHeight: 32,
  },
  answersContainer: {
    gap: 16,
    marginBottom: 40,
  },
  answerButton: {
    backgroundColor: "#1a1a1a",
    paddingVertical: 16,
    paddingHorizontal: 20,
    borderRadius: 12,
    borderWidth: 2,
    borderColor: "#333",
  },
  selectedAnswer: {
    borderColor: "#0ea5e9",
    backgroundColor: "#1e3a8a",
  },
  correctAnswer: {
    borderColor: "#22c55e",
    backgroundColor: "#14532d",
  },
  wrongAnswer: {
    borderColor: "#ef4444",
    backgroundColor: "#7f1d1d",
  },
  disabledAnswer: {
    opacity: 0.5,
  },
  answerText: {
    color: "#fff",
    fontSize: 16,
    fontWeight: "500",
  },
  selectedAnswerText: {
    color: "#fff",
    fontWeight: "600",
  },
  correctAnswerText: {
    color: "#fff",
    fontWeight: "600",
  },
  wrongAnswerText: {
    color: "#fff",
    fontWeight: "600",
  },
  disabledAnswerText: {
    color: "#666",
  },
  skipContainer: {
    alignItems: "center",
  },
  skipButton: {
    backgroundColor: "#333",
    paddingVertical: 12,
    paddingHorizontal: 24,
    borderRadius: 20,
  },
  skipText: {
    color: "#fff",
    fontSize: 16,
    fontWeight: "500",
  },
  resultContainer: {
    alignItems: "center",
    marginTop: 20,
  },
  resultText: {
    color: "#fff",
    fontSize: 20,
    fontWeight: "700",
    marginBottom: 8,
  },
  explanationText: {
    color: "#666",
    fontSize: 14,
    textAlign: "center",
  },
});
