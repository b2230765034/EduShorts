import { useMemo } from "react";
import { View, useWindowDimensions } from "react-native";
import YoutubePlayer from "react-native-youtube-iframe";

type Props = { id: string; height: number; active: boolean };

export default function VideoCard({ id, height, active }: Props) {
  const { width } = useWindowDimensions();

  // Ekranı dolduracak kadar yakınlaştır (cihaza göre 1.15–1.35 iyi çalışır)
  const scale = useMemo(() => 8.0, []);

  // YouTube player içindeki <video> ve kaplamaları büyütüp gizlemek için
  const injected = `
    (function() {
      const apply = () => {
        // alt barları, başlık/önerileri gizle
        const hide = (sel) => { const el = document.querySelector(sel); if (el) el.style.display='none'; };
        hide('.ytp-chrome-bottom');
        hide('.ytp-gradient-bottom');
        hide('.ytp-show-cards-title');
        hide('.ytp-pause-overlay');

        // video konteynerini ekrana yay
        const player = document.querySelector('.html5-video-player');
        if (player) {
          player.style.position='fixed';
          player.style.top='0';
          player.style.left='0';
          player.style.width='100vw';
          player.style.height='100vh';
          player.style.backgroundColor='black';
        }
        const vc = document.querySelector('.html5-video-container');
        if (vc) {
          vc.style.transform='scale(${scale})';
          vc.style.transformOrigin='center center';
          vc.style.willChange='transform';
        }
      };
      apply();
      // YouTube dinamik yüklediği için periyodik tekrar uygula
      setInterval(apply, 700);
    })();
    true;
  `;

  return (
    <View style={{ height, width, backgroundColor: "#000", overflow: "hidden" }}>
      <YoutubePlayer
        key={id}                       // kart değişince player resetlensin
        height={height}
        width={width}
        play={active}                  // sadece aktif kart oynasın
        mute={true}                    // iOS autoplay için gerekli
        videoId={id}
        initialPlayerParams={{
          controls: false,             // toolbar kapalı
          modestbranding: true,
          rel: false,
          playsInline: true,
          loop: true,
          playlist: id,                // loop hilesi
        }}
        webViewProps={{
          allowsInlineMediaPlayback: true,
          mediaPlaybackRequiresUserAction: false,
          injectedJavaScript: injected, // <-- asıl sihir
          androidLayerType: "hardware",
          // Mobile UA zorlayalım (bazı cihazlarda masaüstü yerleşimi gelmesin)
          userAgent:
            "Mozilla/5.0 (iPhone; CPU iPhone OS 15_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Mobile/15E148 Safari/604.1",
        }}
        webViewStyle={{ backgroundColor: "black" }}
      />
    </View>
  );
}
