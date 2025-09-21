// components/RightSideActions.tsx — updated
import React, { memo } from "react";
import { View, Pressable, Text } from "react-native";
import { Ionicons, Feather, MaterialCommunityIcons } from "@expo/vector-icons";

// Added optional onQuiz prop so ReelsScreen can trigger quiz manually
export type Props = {
  liked: boolean;
  saved: boolean;
  likeCount?: number;
  commentCount?: number;
  onLike: () => void;
  onSave: () => void;
  onComment: () => void;
  onShare: () => void | Promise<void>;
  tint?: "light" | "dark";
  /** Optional: show a quiz button when provided */
  onQuiz?: () => void;
};

const RightSideActions = ({
  liked,
  saved,
  likeCount = 0,
  commentCount = 0,
  onLike,
  onSave,
  onComment,
  onShare,
  tint = "light",
  onQuiz,
}: Props) => {
  const color = tint === "light" ? "#fff" : "#000";

  return (
    <View style={{ alignItems: "center", gap: 18 }}>
      <Pressable onPress={onLike} style={{ alignItems: "center" }} hitSlop={10} accessibilityRole="button" accessibilityLabel={liked ? "Beğeniyi kaldır" : "Beğen"}>
        <Ionicons name={liked ? "heart" : "heart-outline"} size={30} color={liked ? "#ff2d55" : color} />
        <Text style={{ color, fontSize: 12, marginTop: 4 }}>{likeCount}</Text>
      </Pressable>

      <Pressable onPress={onSave} style={{ alignItems: "center" }} hitSlop={10} accessibilityRole="button" accessibilityLabel={saved ? "Kaydı kaldır" : "Kaydet"}>
        <MaterialCommunityIcons name={saved ? "bookmark" : "bookmark-outline"} size={30} color={color} />
        <Text style={{ color, fontSize: 12, marginTop: 4 }}>{saved ? "Kaydedildi" : "Kaydet"}</Text>
      </Pressable>

      <Pressable onPress={onComment} style={{ alignItems: "center" }} hitSlop={10} accessibilityRole="button" accessibilityLabel="Yorumlar">
        <Feather name="message-circle" size={28} color={color} />
        <Text style={{ color, fontSize: 12, marginTop: 4 }}>{commentCount}</Text>
      </Pressable>

      <Pressable onPress={onShare} style={{ alignItems: "center" }} hitSlop={10} accessibilityRole="button" accessibilityLabel="Paylaş">
        <Feather name="share-2" size={28} color={color} />
        <Text style={{ color, fontSize: 12, marginTop: 4 }}>Paylaş</Text>
      </Pressable>

      {/* Optional quiz trigger */}
      {onQuiz && (
        <Pressable onPress={onQuiz} style={{ alignItems: "center" }} hitSlop={10} accessibilityRole="button" accessibilityLabel="Quiz aç">
          <MaterialCommunityIcons name="brain" size={30} color={color} />
          <Text style={{ color, fontSize: 12, marginTop: 4 }}>Quiz</Text>
        </Pressable>
      )}
    </View>
  );
};

export default memo(RightSideActions);
