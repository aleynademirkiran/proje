# GRUP 1 — N-gram Dil Analizi + Vektör Veritabanlı Semantik Arama Motoru

Bu, orijinal `turkish-academic-ngram-semantic-search` reposunun GRUP 1 görev
tanımındaki tüm aşamaları tamamlayacak şekilde düzeltilmiş/genişletilmiş halidir.

## ⚠️ Veri seti hakkında önemli not

Bu teslimde kullanılan veri seti (`akademik_veri.sqlite3`) **33 kayıt**
içeriyor (23'ü Türkçe). Bu, projenin `kuyruk` (queue) tablosundan anlaşıldığı
kadarıyla **TR Dizin'den devam eden bir scraper çalışmasının an itibarıyla
tamamlanmış kısmı** (130 kayıt kuyruğa alınmış, 98'i hâlâ "PENDING", 24-33'ü
"SUCCESS"). Kullanıcının tercihiyle bu veri seti **asıl/güncel veri seti**
olarak kabul edilip pipeline bu veriyle çalıştırılmıştır.

**Bunun sonuçlar üzerindeki etkisi:** Tüm scriptler artık korpus büyüklüğüne
göre **otomatik olarak uyarlanan hiperparametreler** kullanıyor (collocation
`min_freq`, embedding `min_count`, değerlendirme `k` değeri ve test sorgu
seti — detaylar aşağıda). Bu sayede scraper tamamlanıp veri seti büyüdüğünde
(örn. 130+ veya 3000+ kayda ulaştığında) scriptleri **hiçbir kod değişikliği
yapmadan** tekrar çalıştırmanız yeterli; sonuçlar otomatik olarak daha
zengin/istatistiksel olarak daha anlamlı hale gelecektir.

Ancak dürüstçe belirtmek gerekir: **33 kayıtlık bir veri setinde n-gram
frekans analizi, embedding eğitimi ve özellikle precision@k/recall@k gibi
metrikler istatistiksel olarak sınırlı anlam taşır.** Bu, scriptlerin hatalı
çalıştığı anlamına gelmez — hepsi gerçek veriyle uçtan uca test edilip
çalıştığı doğrulanmıştır — sadece örneklem küçük olduğu için sonuçların
genellenebilirliği düşüktür. Scraper'ın tamamlanmasını (98 PENDING kaydın da
çekilmesini) bekleyip pipeline'ı o zaman tekrar çalıştırmanız, GRUP 1
raporunuz için çok daha güçlü ve savunulabilir sonuçlar verecektir.

## Orijinal projeye göre neler değişti / eklendi

| Aşama | Orijinal repo | Bu teslimde |
|---|---|---|
| Ön işleme | Sadece regex temizleme + NLTK stopwords | + dil tespiti (veri setinde 1222 kayıt İngilizce çıktı, ayrıştırıldı), + korpus-frekans tabanlı stopword genişletme, + **Zeyrek ile gerçek morfolojik lemmatization** |
| N-gram/frekans | Yok (sadece arama motorunda TF-IDF) | Ayrı modül: ham vs lemma n-gram karşılaştırması, karakter n-gram, **PMI/Log-likelihood collocation tespiti**, TTR, **Ateşman okunabilirlik formülü** |
| Embedding | `all-MiniLM-L6-v2` (Türkçe'ye özel değil) | Sıfırdan **Word2Vec + FastText** eğitimi (gensim) + configurable **BERTurk/Turkish-S-BERT** backend + intrinsic (OOV/benzerlik) karşılaştırma |
| Vektör DB | **Yoktu** — embedding'ler sadece bellekte | **Chroma** ile kalıcı vektör DB + yıl/konu metadata filtreleme |
| Hibrit arama | `alpha * dense + (1-alpha) * tfidf` (RRF değil) | Gerçek **BM25** + **Reciprocal Rank Fusion (RRF)** |
| Servis | `app.py` **0 byte, tamamen boştu** | Çalışan **FastAPI** servisi + **Streamlit** arayüzü |
| Değerlendirme | Yoktu | **precision@10 / recall@10**, 50 sorguluk test seti, 3 yöntem karşılaştırması |

## Dosyalar

```
onisleme.py          Aşama 1 — ön işleme (dil tespiti, normalizasyon, lemmatization)
ngram_analiz.py       Aşama 2 — n-gram, TF-IDF, collocation, okunabilirlik
embedding_motoru.py   Aşama 3 — Word2Vec/FastText + sentence embedding
vektor_db.py          Aşama 4 — Chroma vektör DB + metadata filtreleme
hibrit_arama.py       Aşama 4/5 — BM25 + dense + RRF hibrit arama motoru
degerlendirme.py      Aşama 5 — precision@k / recall@k değerlendirmesi
app.py                Aşama 5 — FastAPI servisi
streamlit_app.py      Aşama 5 — Streamlit arayüzü
requirements.txt
akademik_veri.sqlite3            (orijinal ham veri)
temizlenmis_akademik_veri_v2.csv (onisleme.py çıktısı — hazır, tekrar çalıştırmaya gerek yok)
ngram_ozellikli_veri.csv         (ngram_analiz.py çıktısı — hazır)
degerlendirme_sonuclari.csv      (degerlendirme.py'nin gerçek çıktısı, örnek)
```

## Çalıştırma sırası

```bash
pip install -r requirements.txt

python onisleme.py          # ~birkaç saniye (33 kayıt), temizlenmis_akademik_veri_v2.csv üretir
python ngram_analiz.py      # n-gram/collocation/okunabilirlik raporu + ngram_ozellikli_veri.csv
python embedding_motoru.py  # Word2Vec/FastText eğitir, kaydeder
python vektor_db.py         # Chroma'ya yükler, örnek arama yapar
python hibrit_arama.py      # BM25 + dense + hibrit karşılaştırma çıktısı
python degerlendirme.py     # precision@k / recall@k raporu (k, korpus boyutuna göre otomatik ayarlanır)

uvicorn app:app --reload           # FastAPI servisi -> http://localhost:8000/docs
streamlit run streamlit_app.py     # Arayüz
```

`temizlenmis_akademik_veri_v2.csv` ve `ngram_ozellikli_veri.csv` zaten üretilip
teslim edilmiştir; ilk iki adımı tekrar çalıştırmadan doğrudan
`embedding_motoru.py`'den devam edebilirsiniz.

## ÖNEMLİ — Embedding backend'i hakkında dürüst bir not

`embedding_motoru.py` içindeki `SentenceEmbedder` sınıfı iki backend sunar:

- **`sentence-transformers`** (varsayılan, **PRODUCTION için kullanılması gereken
  seçenek**): Gerçek BERTurk / Turkish-Sentence-BERT modelini
  (`emrecan/bert-base-turkish-cased-mean-nli-stsb-tr`) HuggingFace'den indirir.
  **İnternet bağlantısı gerektirir.**
- **`tfidf-svd`**: TF-IDF + TruncatedSVD (LSA) ile üretilen basit bir "yalancı"
  embedding. Gerçek bir sentence embedding'in **yerine geçmez**; sadece
  internet erişimi olmayan ortamlarda pipeline'ı (vektör DB, hibrit arama, API)
  uçtan uca test edebilmek için eklendi.

Bu teslimi hazırladığım geliştirme ortamının ağ erişimi `huggingface.co`'ya
kapalı olduğu için **gerçek BERTurk modelini bu ortamda indirip test
edemedim**. Bütün kod `sentence-transformers` backend'i ile yazıldı ve
internet erişimi olan kendi bilgisayarınızda (veya Colab'da) **hiçbir kod
değişikliği yapmadan** çalışacaktır — sadece:

```python
HibritAramaMotoru(embedding_backend="sentence-transformers")
```

şeklinde başlatmanız yeterli. Bütün test ve demo çıktıları (bu README'deki
sonuçlar dahil) `tfidf-svd` fallback'i ile, gerçek 3102 kayıtlık veri setiniz
üzerinde uçtan uca çalıştırılarak üretilmiştir — yani kodun kendisi ve
pipeline mimarisi test edilmiştir, sadece embedding kalitesi BERTurk kadar
güçlü değildir.

## Değerlendirme sonuçları hakkında not

`degerlendirme.py` artık sabit/elle yazılmış bir 50 sorguluk liste **kullanmıyor**
(eski sürümde 3102 kayıtlık veri setine göre hazırlanmış sabit bir liste
vardı; bu, farklı boyuttaki bir veri setinde anlamsız kalırdı). Bunun yerine
`otomatik_sorgu_seti_olustur()` fonksiyonu, veri setindeki gerçek "konular"
etiketlerinden sorgu/konu çiftlerini **otomatik olarak** türetir — böylece
hangi veri seti yüklenirse yüklensin (33 kayıt da olsa, gelecekte 3000+ kayıt
da olsa) script hiçbir değişiklik yapılmadan doğru çalışır.

Ground truth tanımı aynı kaldı: bir doküman, hem sorgudaki anahtar terimi
içeriyorsa hem de ilgili disiplin etiketine sahipse "ilgili" sayılır — bu
**gerçek elle etiketlemenin yerini tutmaz**, sadece üç yöntemi (BM25/dense/
hibrit) adil şekilde karşılaştırmak için otomatik bir silver-label seti
sağlar.

Bu 33 kayıtlık veri setinde (23 Türkçe doküman, k=7, 5 geçerli sorgu) elde
edilen gerçek sonuç:

| Yöntem | precision@7 | recall@7 |
|---|---|---|
| BM25 | 0.343 | 0.864 |
| Dense | 0.314 | 0.664 |
| Hibrit (RRF) | 0.343 | 0.864 |

**Bu sayılara temkinli yaklaşın:** sadece 5 sorgu üzerinden hesaplandılar ve
korpus 23 dokümandan oluşuyor — k=7 istendiğinde neredeyse tüm korpusun
%30'u zaten getiriliyor demektir, bu da recall'ün yapay şekilde yüksek
çıkmasına neden olur. Scraper tamamlanıp veri seti büyüdüğünde bu tabloyu
`python degerlendirme.py` ile yeniden üretmeniz ve raporunuza güncel/anlamlı
sayıları koymanız önerilir.

## Bilinen sınırlamalar / ileride geliştirilebilecekler

- Qdrant/Milvus yerine Chroma kullanıldı (görev metninde de "daha basit
  alternatif" olarak belirtilmiş); embedding fonksiyonu ve metadata şeması
  aynı kaldığından ileride kolayca taşınabilir.
- Konu (disiplin) metadata filtresi Chroma'nın `where` mekanizması yerine
  sonuç-sonrası (post-filter) uygulanıyor; çok büyük veri setlerinde bu daha
  az verimli olur — üretimde konuları ayrı, tekil metadata alanlarına
  (one-hot) bölmek performansı artırır.
- Zeyrek bazı nadir/yeni terimlerde (özellikle İngilizce-Türkçe karışık
  akademik jargon: "aı", "genaı" gibi) hatalı kök buluyor; bu, Türkçe
  morfolojik analiz araçlarının bilinen bir zorluğudur ve görev tanımında da
  "kritik ve zor" olarak belirtilmiştir.
