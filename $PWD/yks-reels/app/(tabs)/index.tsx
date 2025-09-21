

import React, { useRef, useState, useMemo, useEffect, useCallback } from "react";
import {
  View,
  Text,
  FlatList,
  Pressable,
  ViewToken,
  useWindowDimensions,
  StyleSheet,
  Platform,
  Share,
  Alert,
} from "react-native";
import { useSafeAreaInsets } from "react-native-safe-area-context";
import { setAudioModeAsync } from "expo-audio";
import { router } from "expo-router";

import Mp4Card from "../../components/Mp4Card";
import QuizModal from "../../components/QuizModal";
import QuestionCard from "../../components/QuestionCard";
import RightSideActions from "../../components/RightSideActions";

import {
  videos as initialVideos,
  type VideoItem,
  type Quiz,
  type FeedItem,
  createFeedWithQuestions,
} from "../../constants/videos";

// -----------------------------------------------------------------------------
// Constants & helpers
// -----------------------------------------------------------------------------

const USER_ID = "demo-user-1";
const AUTO_MIN = 5; // min video sayısı (sorudan önce)
const AUTO_MAX = 10; // max video sayısı (sorudan önce)
const AUTO_DELAY_MS = 800; // quiz modal açma gecikmesi

const randInt = (min: number, max: number) => Math.floor(Math.random() * (max - min + 1)) + min;

/** Güvenli yüzde (0–100). Progress bar/width için layout taşmalarını önler. */
const clampPercent = (x: number) => Math.max(0, Math.min(100, x));

// Quiz modal state
type QuizState = { videoId: number; quiz?: Quiz } | null;

// -----------------------------------------------------------------------------
// Screen
// -----------------------------------------------------------------------------

export default function ReelsScreen() {
  const insets = useSafeAreaInsets();
  const { height: winH } = useWindowDimensions();

  // Üst boşluk (kamera/çentik vs.)
  const TOP_GAP: number = Platform.select({
    ios: Math.max(12, Math.floor(insets.top * 0.4)),
    android: 12,
    default: 12,
  }) ?? 12;

  // Her item tam ekran görünür; snap için yükseklik
  const ITEM_HEIGHT = useMemo(() => winH - insets.top - insets.bottom, [winH, insets.top, insets.bottom]);

  // Açıklama aç/kapa durumları
  const [expandedDesc, setExpandedDesc] = useState<Set<number>>(new Set());
  const toggleDesc = useCallback((id: number) => {
    setExpandedDesc((prev) => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });
  }, []);

  // Feed (videolar + aralara serpiştirilmiş sorular)
  const [feed, setFeed] = useState<FeedItem[]>(() => createFeedWithQuestions(initialVideos, 3, 3));

  // Sonsuz scroll için id klonlayıcı
  const nextIdRef = useRef<number>(initialVideos.length + 1);
  const cloneBatch = useCallback((): FeedItem[] => {
    const newVideos: VideoItem[] = initialVideos.map((v) => ({ ...v, id: nextIdRef.current++ }));
    return createFeedWithQuestions(newVideos, 3, 3);
  }, []);

  const handleEndReached = useCallback(() => setFeed((prev) => [...prev, ...cloneBatch()]), [cloneBatch]);

  // Quiz ilişkili durumlar
  const [quizOpenFor, setQuizOpenFor] = useState<QuizState>(null);
  const [activeId, setActiveId] = useState<number | null>(null);
  const askedRef = useRef<Set<number>>(new Set()); // bu videoda soru soruldu mu?
  const sinceLastAutoRef = useRef(0); // son otomatik sorudan beri izlenen video sayısı
  const thresholdRef = useRef(randInt(AUTO_MIN, AUTO_MAX));
  const pendingAutoRef = useRef(false); // sonraki uygun videoda soru gösterilmesi bekleniyor mu?
  const autoTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const prevActiveRef = useRef<number | null>(null);

  // Görünürlük ayarı: bir item %80 göründüğünde aktif say
  const viewabilityConfig = useMemo(() => ({ itemVisiblePercentThreshold: 80 }), []);

  // RN tipleri için güvenli onViewableItemsChanged (ref gerekli)
  const onViewableItemsChanged = useRef(
    ({ viewableItems }: { viewableItems: Array<ViewToken> }) => {
      const first = viewableItems[0];
      const id = (first?.item as FeedItem | undefined) && (first!.item as any).id;
      if (typeof id === "number") setActiveId(id);
    }
  );

  // Otomatik quiz zamanlaması — aktif item her değiştiğinde çalışır
  useEffect(() => {
    if (activeId == null) return;
    if (prevActiveRef.current === activeId) return; // aynı item ise atla
    prevActiveRef.current = activeId;

    sinceLastAutoRef.current += 1;
    if (autoTimerRef.current) {
      clearTimeout(autoTimerRef.current);
      autoTimerRef.current = null;
    }

    const current = feed.find((v) => v.id === activeId);
    const isVideo = current && !("type" in current && current.type === "question");
    const hasQuiz = Boolean(isVideo && (current as VideoItem).quiz?.length);
    const notAskedYet = !askedRef.current.has(activeId);
    const canAskHere = Boolean(hasQuiz && notAskedYet);

    const openQuizForCurrent = () => {
      const q = (current as VideoItem).quiz![0];
      setQuizOpenFor({ videoId: current!.id, quiz: q });
      askedRef.current.add(current!.id);
      sinceLastAutoRef.current = 0;
      thresholdRef.current = randInt(AUTO_MIN, AUTO_MAX);
      pendingAutoRef.current = false;
    };

    if (pendingAutoRef.current && canAskHere) {
      autoTimerRef.current = setTimeout(openQuizForCurrent, AUTO_DELAY_MS);
      return;
    }

    if (sinceLastAutoRef.current >= thresholdRef.current) {
      if (canAskHere) {
        autoTimerRef.current = setTimeout(openQuizForCurrent, AUTO_DELAY_MS);
      } else {
        // Şu anki item soru için uygun değil; bir sonrakinde dene
        pendingAutoRef.current = true;
      }
    }
  }, [activeId, feed]);

  // Unmount'ta timer temizliği
  useEffect(() => () => {
    if (autoTimerRef.current) clearTimeout(autoTimerRef.current);
  }, []);

  // Sessiz anahtarda da ses çal (iOS)
  useEffect(() => {
    (async () => {
      try {
        await setAudioModeAsync({ playsInSilentMode: true });
      } catch (e) {
        console.warn("setAudioModeAsync failed", e);
      }
    })();
  }, []);

  const openQuizManually = useCallback((v: VideoItem) => {
    if (!v.quiz?.length) return;
    if (autoTimerRef.current) {
      clearTimeout(autoTimerRef.current);
      autoTimerRef.current = null;
    }
    setQuizOpenFor({ videoId: v.id, quiz: v.quiz[0] });
    askedRef.current.add(v.id);
    sinceLastAutoRef.current = 0;
    thresholdRef.current = randInt(AUTO_MIN, AUTO_MAX);
    pendingAutoRef.current = false;
  }, []);

  const onCloseQuiz = useCallback((_ok: boolean | null) => setQuizOpenFor(null), []);

  // Sağ aksiyon rayı durumları
  const [likedIds, setLikedIds] = useState<Set<number>>(new Set());
  const [savedIds, setSavedIds] = useState<Set<number>>(new Set());
  const [likeCounts, setLikeCounts] = useState<Record<number, number>>({});
  const [commentCounts] = useState<Record<number, number>>({});

  const toggleLike = useCallback((id: number) => {
    setLikedIds((prev) => {
      const next = new Set(prev);
      const isLiked = next.has(id);
      if (isLiked) next.delete(id);
      else next.add(id);

      setLikeCounts((c) => ({ ...c, [id]: Math.max(0, (c[id] ?? 0) + (isLiked ? -1 : 1)) }));
      return next;
    });
  }, []);

  const toggleSave = useCallback((id: number) => {
    setSavedIds((prev) => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });
  }, []);

  const openComments = useCallback((video: VideoItem) => {
    Alert.alert("Yorumlar", `"${video.title}" için yorum ekranı burada açılacak.`);
  }, []);

  const shareVideo = useCallback(async (video: VideoItem) => {
    try {
      const url = video.platform === "mp4" ? (video as any).url ?? "" : `https://youtu.be/${(video as any).youtubeId ?? ""}`;
      const message = `Bunu beğeneceğini düşündüm:\n${video.title}\n${url}`.trim();
      await Share.share({ message });
    } catch {
      
    }
  }, []);

  return (
    <View style={{ flex: 1, backgroundColor: "#000" }}>
      <FlatList
        data={feed}
        keyExtractor={(i) => i.id.toString()}
        pagingEnabled
        snapToInterval={ITEM_HEIGHT}
        snapToAlignment="start"
        decelerationRate="fast"
        showsVerticalScrollIndicator={false}
        viewabilityConfig={viewabilityConfig}
        onViewableItemsChanged={onViewableItemsChanged.current as any}
        onEndReached={handleEndReached}
        onEndReachedThreshold={0.6}
        renderItem={({ item }) => {
          const isActive = item.id === activeId;

          // Soru kartı mı?
          if ("type" in item && item.type === "question") {
            return (
              <View style={{ height: ITEM_HEIGHT }}>
                <QuestionCard
                  item={item}
                  onAnswer={(isCorrect) => {
                    console.log("Question answered:", isCorrect);
                    // TODO: skor/istatistik güncelle
                  }}
                  onSkip={() => {
                    console.log("Question skipped");
                    // TODO: skip davranışı
                  }}
                />
              </View>
            );
          }

          // Video kartı
          const videoItem = item as VideoItem;
          const isWhiteBg = videoItem.backgroundColor === "#ffffff" || videoItem.backgroundColor === "#fff";
          const buttonTint: "light" | "dark" = isWhiteBg ? "dark" : "light";

          return (
            <View style={{ height: ITEM_HEIGHT }}>
              {/* Video */}
              <View style={{ height: ITEM_HEIGHT, paddingTop: TOP_GAP }}>
                <Mp4Card uri={videoItem.url} height={ITEM_HEIGHT - TOP_GAP} active={isActive} />
              </View>

              {/* Sol-alt: başlık + açıklama */}
              <View style={{ position: "absolute", left: 0, bottom: 0, maxWidth: "100%", zIndex: 10 }}>
                <View
                  style={{
                    backgroundColor: "rgba(0,0,0,0.35)",
                    paddingHorizontal: 14,
                    paddingVertical: 10,
                    borderTopLeftRadius: 0,
                    borderBottomLeftRadius: 0,
                    borderTopRightRadius: 14,
                    borderBottomRightRadius: 14,
                    gap: 6,
                  }}
                >
                  <Text style={{ color: "#fff", fontSize: 20, fontWeight: "700" }} numberOfLines={2}>
                    {videoItem.title}
                  </Text>

                  {!!videoItem.desc && (
                    <Pressable onPress={() => toggleDesc(videoItem.id)}>
                      <Text
                        style={{ color: "#eee", fontSize: 14, lineHeight: 18 }}
                        numberOfLines={expandedDesc.has(videoItem.id) ? undefined : 2}
                        ellipsizeMode="tail"
                      >
                        {videoItem.desc}
                      </Text>
                      <Text style={{ color: "#ddd", fontSize: 12, marginTop: 2 }}>
                        {expandedDesc.has(videoItem.id) ? "daraltmak için dokun" : "devamını görmek için dokun"}
                      </Text>
                    </Pressable>
                  )}

                  <Text style={{ color: "#ddd", fontSize: 13 }}>
                    {videoItem.subject} • {videoItem.topic}
                  </Text>
                </View>
              </View>

              {/* Sağ aksiyon paneli */}
              <View pointerEvents="box-none" style={StyleSheet.absoluteFill}>
                <View style={{ position: "absolute", right: 10, bottom: insets.bottom + 120 }}>
                  <RightSideActions
                    liked={likedIds.has(videoItem.id)}
                    saved={savedIds.has(videoItem.id)}
                    likeCount={likeCounts[videoItem.id] ?? 0}
                    commentCount={commentCounts[videoItem.id] ?? 0}
                    onLike={() => toggleLike(videoItem.id)}
                    onSave={() => toggleSave(videoItem.id)}
                    onComment={() => openComments(videoItem)}
                    onShare={() => shareVideo(videoItem)}
                    tint={buttonTint}
                    onQuiz={() => openQuizManually(videoItem)}
                  />
                </View>
              </View>
            </View>
          );
        }}
      />

      {/* Quiz Modal */}
      <QuizModal
        visible={!!quizOpenFor}
        onClose={onCloseQuiz}
        quiz={quizOpenFor?.quiz}
        videoId={quizOpenFor?.videoId ?? -1}
        userId={USER_ID}
      />
    </View>
  );
}

