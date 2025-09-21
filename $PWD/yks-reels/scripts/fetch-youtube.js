// scripts/fetch-youtube.js
const fs = require("fs");
const path = require("path");
const axios = require("axios");

const API_KEY = process.env.YT_API_KEY;

// 🔎 Konu başlıkları (istediğin kadar ekleyebilirsin)
const QUERIES = [
  { subject: "Matematik", topic: "Limit",   q: "YKS matematik limit shorts" },
  { subject: "Türkçe",    topic: "Paragraf", q: "TYT türkçe paragraf #shorts" },
  { subject: "Fizik",     topic: "Newton",   q: "TYT fizik newton shorts" },
];

// Shorts odaklı arama
async function searchShorts(q) {
  const { data } = await axios.get("https://www.googleapis.com/youtube/v3/search", {
    params: {
      key: API_KEY,
      part: "snippet",
      type: "video",
      maxResults: 25,
      q,                             // içinde "shorts" / "#shorts" var
      regionCode: "TR",
      relevanceLanguage: "tr",
      safeSearch: "strict",
      videoEmbeddable: "true",
      videoDuration: "short",        // < 4 dk (Shorts < 60 sn’nin alt kümesi)
      // order: "relevance",         // istersen "date" deneyebilirsin
    },
    timeout: 15000,
  });

  return (data.items || [])
    .filter((it) => it.id?.videoId)
    .map((it) => ({
      youtubeId: it.id.videoId,
      title: it.snippet.title,
      channelTitle: it.snippet.channelTitle,
    }));
}

(async () => {
  if (!API_KEY) {
    console.error("YT_API_KEY yok. PowerShell: $env:YT_API_KEY=\"KEY\"; node scripts\\fetch-youtube.js");
    process.exit(1);
  }

  const out = [];
  for (const { subject, topic, q } of QUERIES) {
    console.log("Shorts arıyor:", q);
    const items = await searchShorts(q);
    console.log(" -> bulundu:", items.length);
    for (const v of items) {
      out.push({
        subject, topic,
        tag: `${subject.toLowerCase()}-${topic.toLowerCase()}`,
        platform: "youtube",
        youtubeId: v.youtubeId,
        title: v.title,
      });
    }
  }

  const file = path.join(__dirname, "../data/youtube.json");
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, JSON.stringify(out, null, 2), "utf8");
  console.log("Saved:", file, "count:", out.length);
})();
