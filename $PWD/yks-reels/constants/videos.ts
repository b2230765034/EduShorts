// constants/videos.ts
export type Quiz = { q: string; a: string[]; correct: number; tag: string };

export type VideoItem = {
  id: number;
  title: string;
  subject: string;
  topic: string;
  tag: string;
  platform: "mp4";
  url: string;  
  desc?: string;       // Asset'ten gelen URI
  quiz?: Quiz[];
  backgroundColor?: string; // Video background color
};

export type QuestionItem = {
  id: number;
  type: "question";
  question: Quiz;
  subject: string;
  topic: string;
  tag: string;
};

export type FeedItem = VideoItem | QuestionItem;

// Function to create questions from video quizzes
export function createQuestionFromVideo(video: VideoItem, questionIndex: number): QuestionItem {
  const quiz = video.quiz?.[questionIndex];
  if (!quiz) throw new Error("No quiz found");
  
  return {
    id: video.id * 1000 + questionIndex, // Unique ID for questions
    type: "question",
    question: quiz,
    subject: video.subject,
    topic: video.topic,
    tag: video.tag,
  };
}

// Function to create a mixed feed with questions at intervals
export function createFeedWithQuestions(videos: VideoItem[], minInterval: number = 5, maxInterval: number = 9): FeedItem[] {
  const feed: FeedItem[] = [];
  let questionIdCounter = 10000; // Start from high number to avoid conflicts
  
  for (let i = 0; i < videos.length; i++) {
    feed.push(videos[i]);
    
    // Check if we should add a question after this video
    const shouldAddQuestion = (i + 1) % Math.floor(Math.random() * (maxInterval - minInterval + 1) + minInterval) === 0;
    
    if (shouldAddQuestion && i < videos.length - 1) { // Don't add question after last video
      // Find a video with quiz questions
      const videosWithQuiz = videos.filter(v => v.quiz && v.quiz.length > 0);
      if (videosWithQuiz.length > 0) {
        const randomVideo = videosWithQuiz[Math.floor(Math.random() * videosWithQuiz.length)];
        const randomQuestionIndex = Math.floor(Math.random() * (randomVideo.quiz?.length || 1));
        
        if (randomVideo.quiz && randomVideo.quiz[randomQuestionIndex]) {
          const question: QuestionItem = {
            id: questionIdCounter++,
            type: "question",
            question: randomVideo.quiz[randomQuestionIndex],
            subject: randomVideo.subject,
            topic: randomVideo.topic,
            tag: randomVideo.tag,
          };
          feed.push(question);
        }
      }
    }
  }
  
  return feed;
}

import { Asset } from "expo-asset";

// ⬇️ BURAYA kendi dosya adlarını yazıyorsun
const v1 = Asset.fromModule(require("../assets/images/videos/basınç.mp4")).uri;
const v2 = Asset.fromModule(require("../assets/images/videos/limit1.mp4")).uri;
const v3 = Asset.fromModule(require("../assets/images/videos/limit2.mp4")).uri;
const v4 = Asset.fromModule(require("../assets/images/videos/newton.mp4")).uri;
const v5 = Asset.fromModule(require("../assets/images/videos/paragraf1.mp4")).uri;
const v6 = Asset.fromModule(require("../assets/images/videos/paragraf2.mp4")).uri;
const v7 = Asset.fromModule(require("../assets/images/videos/paragraf3.mp4")).uri;
const v8 = Asset.fromModule(require("../assets/images/videos/paragraf4.mp4")).uri;
const v9 = Asset.fromModule(require("../assets/images/videos/limit3.mp4")).uri;
const v10 = Asset.fromModule(require("../assets/images/videos/fizik1.mp4")).uri;
const v11 = Asset.fromModule(require("../assets/images/videos/limit4.mp4")).uri;
const v12 = Asset.fromModule(require("../assets/images/videos/limit5.mp4")).uri;
const v13 = Asset.fromModule(require("../assets/images/videos/limit6.mp4")).uri;
const v14 = Asset.fromModule(require("../assets/images/videos/fizik2.mp4")).uri;
const v15 = Asset.fromModule(require("../assets/images/videos/fizik3.mp4")).uri;
// gerekirse ekle: const v3 = Asset.fromModule(require("../assets/videos/..")).uri;

export const videos: VideoItem[] = [
  {
    id: 1,
    title: "Basınç",
    subject: "Fizik",
    topic: "Basınç",
    tag: "fizik-basınç",
    platform: "mp4",
    url: v1,
    desc: "Kaldırma kuvveti, bir sıvı (veya gaz) içine bırakılan cisme, yer değiştirdiği akışkanın ağırlığı kadar yukarı yönlü etki eden kuvvettir. Büyüklüğü F= ρgV(batan) ile verilir. F>G ise cisim yüzer, F=G ise denge halinde kalır, F<G ise batar. #Fizik #KaldırmaKuvveti #ArşimetPrensibi #Hidrostatik #Basınç #TYT #AYT #SıvıBasıncı",
    quiz: [
      {
        q: "Basınç hangi birimle ölçülür?",
        a: ["Pascal", "Newton", "Joule", "Watt"],
        correct: 0,
        tag: "fizik-basınç"
      },
      {
        q: "Arşimet prensibi hangi kuvveti açıklar?",
        a: ["Kaldırma kuvveti", "Sürtünme kuvveti", "Yerçekimi kuvveti", "Elastik kuvvet"],
        correct: 0,
        tag: "fizik-basınç"
      }
    ],
    backgroundColor: "#ffffff",
  },
  {
    id: 2,
    title: "Limit",
    subject: "Matematik",
    topic: "Limit",
    tag: "matematik-limit",
    platform: "mp4",
    url: v2,
    desc: "Limit, bir fonksiyonun girdi değeri belli bir sayıya yaklaşırken çıktı değerinin 'hangi sayıya yöneldiğini' anlatır. Noktadaki gerçek değer önemli olmayabilir; hatta o noktada tanımlı olmasa bile yaklaşımın gittiği sayı limittir. Soldan ve sağdan yaklaşımlar aynıysa limit vardır. Sonsuzda limit ise girdi çok büyürken (veya çok küçülürken) çıktının nasıl davrandığını inceler. Hesapta genelde sadeleştirme, çarpanlara ayırma ya da eşleniği kullanma gibi küçük düzenlemeler işe yarar. #Matematik #Limit #Analiz #Süreklilik #AYT #TYT",
    quiz: [
      {
        q: "lim(x→2) (x²-4)/(x-2) değeri kaçtır?",
        a: ["0", "2", "4", "Tanımsız"],
        correct: 2,
        tag: "matematik-limit"
      },
      {
        q: "Bir fonksiyonun limitinin var olması için hangi koşul gerekir?",
        a: ["Soldan limit = Sağdan limit", "Fonksiyon tanımlı olmalı", "Sürekli olmalı", "Türevlenebilir olmalı"],
        correct: 0,
        tag: "matematik-limit"
      }
    ],
    backgroundColor: "#ffffff",
  },
  {
    id: 3,
    title: "L'Hospital",
    subject: "Matematik",
    topic: "Limit",
    tag: "matematik-limit",
    platform: "mp4",
    url: v3,
    desc: "L’Hospital (Lopital) kuralı, limit hesabında 0/0 veya ∞/∞ gibi belirsizliklerle karşılaştığımızda kullanılan pratik bir yöntemdir. Amaç, pay ve paydanın değerlerini değil, bu fonksiyonların ilgili noktadaki değişim hızlarını karşılaştırmaktır: payı ve paydayı ayrı ayrı türevleyip yeni oluşan oranının limitini alırız; belirsizlik sürerse aynı işlemi tekrarlarız. Kuralın geçerli olması için fonksiyonların o noktada (ya da yakın çevresinde) türevlenebilir olması gerekir. “0·∞”, “∞−∞”, “1^∞” gibi diğer belirsizliklerse önce uygun cebirsel dönüşümlerle 0/0 ya da ∞/∞ formuna getirilir; bazı durumlarda sadeleştirme veya eşlenik alma gibi klasik yöntemler daha hızlı sonuç verebilir. #Matematik #Limit #LHospital #Türev #ayt #tyt #KPSS",
    quiz: [],
  },
  {
    id: 4,
    title: "Newton Hareket Kanunları",
    subject: "Fizik",
    topic: "Newton",
    tag: "fizik-newton",
    platform: "mp4",
    url: v4,
    desc: "Hız–zaman grafiğinde hızın işareti hareket yönünü, eğimin (türevin) işareti ise ivmeyi gösterir. Net kuvvet yönü her zaman ivme yönüyle aynıdır. Dolayısıyla kuvvetin hız vektörüne ters olduğu durumlar, hız ile ivmenin (yani eğimin) zıt işaretli olduğu aralıklardır: cisim +hızla giderken grafik aşağı eğimliyse veya –hızla giderken grafik yukarı eğimliyse hızının büyüklüğü azalır (yavaşlar) ve net kuvvet hızın ters yönündedir. Grafiğin altında/üstünde kalmak yalnızca yer değiştirmeyi (alan) verir; kuvvet–hız yön ilişkisini belirleyen tek şey grafiğin eğimidir. #Fizik #HızZamanGrafiği #İvme #NetKuvvet #Newton #Kinematik #Hareket #Yavaşlama #TYT #AYT #YKS ",
    quiz: [],
  },
  {
    id: 5,
    title: "Paragraf",
    subject: "Türkçe",
    topic: "Paragraf",
    tag: "türkçe-paragraf",
    platform: "mp4",
    url: v5,
    desc: "Paragraf sorularında metni okumadan seçeneklere atlamak, kökteki “değildir/çıkarılamaz” gibi olumsuzlukları gözden kaçırmak ve kendi bilgini metne katmak en yaygın hatalardır. Anahtar kelime avcılığı yapmak yerine bağlamı izle; “daima/asla” gibi mutlak ifadelere temkinli yaklaş. Takıldığında tek soruya zaman gömmek yerine işaretle-geç; döndüğünde bağlaç ve karşıtlık ipuçlarını yeniden kontrol et. Her seçeneği metinden kanıtla doğrulamadan işaretleme.#Türkçe #Paragraf #OkuduğunuAnlama #SoruÇözme #SınavTeknikleri #TYT #AYT #YKS #DikkatHataları #HızlıOkuma #Çeldirici #DenemeSınavı",
    quiz: [],
  },
  {
    id: 6,
    title: "Paragraf",
    subject: "Türkçe",
    topic: "Paragraf",
    tag: "türkçe-paragraf",
    platform: "mp4",
    url: v6,
    desc: "Paragraf sorularında önce ana düşünceyi yakala, sonra yardımcı fikirleri ve yapıyı (giriş–gelişme–sonuç, neden–sonuç, karşılaştırma vb.) ayırt et. Soru kökünü dikkatle okuyup olumsuzluk/isteme ifadelerini işaretle. Seçeneklerde anahtar kelimeler yerine bağlamı ve tutarlılığı kontrol et; aşırı genelleme, metinde olmayan ek bilgi ve çelişen ifadelerden kaçın. Zamanı verimli kullanmak için önce kısa ve net soruları çöz, uzun metinlerde paragrafı parçalara bölerek ilerle. #Türkçe #Paragraf #OkuduğunuAnlama #AnaDüşünce #YardımcıDüşünce #SoruÇözme #SınavTeknikleri #TYT #AYT #YKS #DilBilgisi #HızlıOkuma",
    quiz: [],
  },
  {
    id: 7,
    title: "Paragraf",
    subject: "Türkçe",
    topic: "Paragraf",
    tag: "türkçe-paragraf",
    platform: "mp4",
    url: v7,
    desc: "Paragrafta “düşünce akışını bozan cümle”, metnin ana konusu ve mantıksal örgüsüyle uyumlu olmayan, konu dışına sapan ya da bağlaç–zamir–zaman uyumunu koparan cümledir. Tespit için önce ana düşünceyi ve paragrafın planını (giriş–gelişme–sonuç, neden–sonuç, karşılaştırma, sıralama) belirle; ardından her cümlenin öncesi–sonrasıyla bağdaşmasına bak. Şu ipuçları bozan cümleyi ele verir: aniden yeni alt konu açma, zamirlerin (bu, o, bunlar) gönderiminin kaybolması, zaman/kişi değişmesi, bağlacın (oysa, ancak, çünkü) gerektirdiği mantığın kurulmaması, kronoloji kırılması ve örnek–genel ifade uyumsuzluğu. #Türkçe #Paragraf #DüşünceAkışı #BozanCümle #OkuduğunuAnlama #SoruTekniği #TYT #AYT #YKS",
    quiz: [],
  },
  {
    id: 8,
    title: "Paragraf",
    subject: "Türkçe",
    topic: "Paragraf",
    tag: "türkçe-paragraf",
    platform: "mp4",
    url: v8,
    desc: "Bu tip sorular “altı çizili sözün bağlamdaki anlamı”nı ölçer. Yöntem: Önce paragrafın ana duygusunu/konusunu yakala, sonra çizili ifadeyi yerine düz bir cümle koyarak dene; kelime kelime değil, deyimsel anlamı ararsın. İpucunu yakın cümlelerdeki benzetmeler, duygular ve zaman ifadeleri verir. Örneğin metindeki “demini beklemek” çay benzetmesinden gelir; “zamanla olgunlaşmasını, kıvamını bulmasını beklemek” demektir. Bu yüzden doğru seçenek, “olgunlaşması için zamana bırakmak” anlamını karşılayan şık olmalıdır (kelime anlamına takılma). #Türkçe #Paragraf #AltıÇiziliSöz #BağlamdanAnlam #Deyimler #SözcükteAnlam #OkuduğunuAnlama #SoruTekniği #TYT #AYT #YKS",
    quiz: [],
  },
  {
    id: 9,
    title: "Limit",
    subject: "Matematik",
    topic: "Limit",
    tag: "matematik-limit",
    platform: "mp4",
    url: v9,
    desc:"Soldan–sağdan limit şöyle düşünülür: Bir noktaya soldan yaklaşırken grafiğin o noktanın sol tarafındaki parçasını izleriz; sağdan yaklaşırken sağ tarafını izleriz. İki taraftan ulaştığımız değer aynıysa limit vardır; farklıysa yoktur. Grafikte noktadaki dolu işaret, fonksiyonun o noktadaki gerçek değerini gösterir ve bu değer limitten farklı olabilir. Bu çizimde üç değerine soldan yaklaşınca yaklaşık bir, sağdan yaklaşınca yaklaşık dört görülüyor; yani iki taraf aynı olmadığı için limit yok, fakat fonksiyonun üçteki değeri dört.#Matematik #Limit #SoldanSağdanLimit #Süreklilik #Analiz #AYT #TYT #YKS #ÇalışmaNotu #GrafikOkuma",
    quiz: [],
  }, 
  {
    id: 10,
    title: "Newton Beşiği",
    subject: "Fizik",
    topic: "Fizik",
    tag: "fizik-fizik",
    platform: "mp4",
    url: v11,
    desc:"Newton beşiği, çarpışmalarda momentumun ve (yaklaşık) mekanik enerjinin korunumunu görsel olarak gösteren düzendir. Kenardaki bilyeyi kaldırıp bıraktığında, momentum çubuklar boyunca iletilir; ortadakiler neredeyse sabit kalırken karşı uçtaki tek bilye aynı hızla dışarı fırlar. Sürtünme ve hava direnci küçükse toplam momentum değişmez; bu yüzden iç çarpışmalar esnek kabul edilir. Birden çok bilye kaldırırsan, karşıda aynı sayıda bilye hareket eder: “giren momentum = çıkan momentum”. #Fizik #NewtonBeşiği #MomentumKorunumu #EnerjiKorunumu #EsnekÇarpışma #İtme #Eylemsizlik #TYT #AYT #YKS #STEM",
    quiz: [],
  },
  {
    id: 11,
    title: "Limit",
    subject: "Matematik",
    topic: "Limit",
    tag: "matematik-limit",
    platform: "mp4",
    url: v12,
    desc:"Limit, bir fonksiyonun girdi bir sayıya yaklaşırken çıktı olarak yöneldiği değerdir. İki taraftan yaklaşımlar aynıysa limit vardır; bu değer noktadaki gerçek fonksiyon değerinden farklı olabilir. Belirsizlikte sadeleştirme/eşlenik, gerekirse L’Hospital ve Sıkıştırma Teoremi kullanılır. Sonsuzda limit, değişken çok büyürken/küçülürken davranışı inceler.#Limit #Matematik #Analiz #Süreklilik #SoldanSağdanLimit #Belirsizlik #LHospital #SıkıştırmaTeoremi #SonsuzdaLimit #GrafikOkuma #TYT #AYT #YKS #ÇalışmaNotu",
    quiz: [],
  },
  {
    id: 12,
    title: "Limit",
    subject: "Matematik",
    topic: "Limit",
    tag: "matematik-limit",
    platform: "mp4",
    url: v13,
    desc:"Limitte hızlı yorum için: Sürekli noktada doğrudan değer ver; parçalı ve mutlak değerli ifadelerde doğru aralığı seç. Sonsuzda en yüksek derece belirleyicidir; payda daha büyükse sonuç sıfıra yaklaşır, dereceler eşitse katsayı oranı fikri geçerlidir. Kök ve trig belirsizliklerinde eşlenik ve küçük açı yaklaşımları, olmadı sıkıştırma ya da türev temelli yöntemler işe yarar.#Limit #Matematik #FonksiyonYorumları #Süreklilik #ParçalıFonksiyon #MutlakDeğer #SonsuzdaLimit #Asimptot #Köklüİfadeler #Trigonometri #SıkıştırmaTeoremi #LHospital #TYT #AYT #YKS #ÇalışmaNotu",
    quiz: [],
  },
  {
    id: 13,
    title: "Fizik",
    subject: "Fizik",
    topic: "Fizik",
    tag: "fizik-fizik",
    platform: "mp4",
    url: v14,
    desc:"Sürtünme kuvveti, temas eden yüzeyler arasında hareketi başlatmaya veya sürdürmeye karşı koyan kuvvettir; her zaman hareket yönüne ters yönde etki eder. Statik sürtünme harekete geçene kadar artar ve bir üst sınıra kadar destek olur; kinetik (kayma) sürtünme hareket başladıktan sonra yaklaşık sabit kalır; yuvarlanma sürtünmesi genelde en küçüğüdür. Büyüklük yüzeylerin türüne ve yüzeylerin birbirine bastırılma şiddetine (normal kuvvet) bağlıdır. Mekanik enerjinin bir kısmını ısıya çevirir; yürümeyi, frenlemeyi mümkün kılar, ama verimi düşürebilir.#Fizik #SürtünmeKuvveti #StatikSürtünme #KinetikSürtünme #YuvarlanmaSürtünmesi #Hareket #EnerjiKayıpları #TYT #AYT #YKS",
    quiz: [],
  },
  {
    id: 14,
    title: "Fizik",
    subject: "Fizik",
    topic: "Fizik",
    tag: "fizik-fizik",
    platform: "mp4",
    url: v15,
    desc: "Sürtünmeli yatay düzlemde cisimler ok yönünde hareket ederken sürtünme kuvveti ok yönüne ters olur. K ve L özdeş ve uygulanan kuvvetlerin büyüklükleri eşit. Eğer K’ye uygulanan kuvvet ok yönünün tersiyse K’nin net kuvveti ters yönde F + sürtünme olur ve cisim yavaşlar. L’ye uygulanan kuvvet ok yönüyle aynıysa L’nin net kuvveti F − sürtünme olur ve cisim hızlanır (F sürtünmeden büyükse). Böylece net kuvvet büyüklüğü K’de L’den büyüktür. Sonuç: I, II ve III doğrudur. #Fizik #SürtünmeKuvveti #NetKuvvet #İvme #Hareket #KuvvetDenge #TYT #AYT #YKS",
    quiz: [],
  },
  // v3, v4 ... ekleyebilirsin
];
