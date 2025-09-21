// components/Mp4Card.tsx — Cleaned & Commented
// -----------------------------------------------------------------------------
// Purpose: Tek bir MP4 kaynağını tam ekran (konteyneri kaplayacak şekilde)
// oynatan basit bir kart. Dışarıdan "active" geldiğinde otomatik oynatır,
// kullanıcı dokunursa oynat/duraklat arasında geçiş yapar.
//  - Dış API değişmedi: { uri, height, active }
//  - Ufak iyileştirmeler: props tipi, useCallback, unmount'ta güvenli durdurma,
//    erişilebilirlik etiketleri ve açıklayıcı yorumlar
// -----------------------------------------------------------------------------

import React, { memo, useCallback, useEffect, useState } from "react";
import { View, Pressable, StyleSheet, AccessibilityInfo } from "react-native";
import { useVideoPlayer, VideoView } from "expo-video";

// İsteğe bağlı: dış bileşenlerde import kolaylığı için tip export edelim
export interface Mp4CardProps {
  uri: string;       // MP4 dosyası URL'si
  height: number;    // Kartın yüksekliği (genelde ekran yüksekliği)
  active: boolean;   // Bu kart görünürde mi? (Aktifse auto-play)
}

function Mp4Card({ uri, height, active }: Mp4CardProps) {
  // Player'ı ilklendiriyoruz. İlk kurulumda loop, ses ayarları vb.
  const player = useVideoPlayer(uri, (p) => {
    p.loop = true;
    p.muted = false; // ses açık
    p.volume = 1.0;
  });

  // Kullanıcının manuel durdurma tercihini koruruz. Active olsa da bu tercih baskın.
  const [pausedByUser, setPausedByUser] = useState(false);

  // Active/pausedByUser değişince oynat/duraklat durumunu güncelle.
  useEffect(() => {
    if (!player) return;
    if (active && !pausedByUser) player.play();
    else player.pause();
  }, [active, pausedByUser, player]);

  // Bileşen unmount olurken player'ı güvenle durdur.
  useEffect(() => {
    return () => {
      try { player?.pause(); } catch {}
    };
  }, [player]);

  // Tek dokunuşla kullanıcı tercihini değiştir (oynat/duraklat toggle)
  const onTogglePlay = useCallback(() => {
    setPausedByUser((prev) => {
      const next = !prev;
      // Erişilebilirlik servisine duyuru (VoiceOver/TalkBack)
      AccessibilityInfo.announceForAccessibility?.(next ? "Video duraklatıldı" : "Video oynatılıyor");
      return next;
    });
  }, []);

  return (
    <View style={[styles.container, { height }]}
          accessibilityRole="image"
          accessibilityLabel="Video kartı">
      <VideoView
        style={StyleSheet.absoluteFill}   // kabı komple doldur
        player={player}
        contentFit="cover"                // KIRPARAK DOLDUR
        nativeControls={false}
        allowsPictureInPicture={false}
      />

      {/* Tek dokunuş: oynat/duraklat */}
      <Pressable
        onPress={onTogglePlay}
        hitSlop={8}
        accessibilityRole="button"
        accessibilityLabel="Oynat veya duraklat"
        style={styles.touchOverlay}
      />
    </View>
  );
}

export default memo(Mp4Card);

// -----------------------------------------------------------------------------
// Styles
// -----------------------------------------------------------------------------
const styles = StyleSheet.create({
  container: {
    width: "100%",
    overflow: "hidden",
    backgroundColor: "#000",
  },
  touchOverlay: {
    position: "absolute",
    top: 0,
    right: 0,
    bottom: 0,
    left: 0,
  },
});
