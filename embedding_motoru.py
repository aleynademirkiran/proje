# -*- coding: utf-8 -*-
"""
AŞAMA 3 — Embedding Üretimi
============================
GRUP 1 gereksinimleri:
  - Kendi korpusundan sıfırdan Word2Vec/FastText eğitimi (gensim)
  - Hazır BERTurk / Turkish Sentence-BERT ile cümle embedding'i
  - Intrinsic (benzerlik/analoji) ve extrinsic (arama kalitesi) karşılaştırma

NOT (önemli, şeffaflık için): Bu geliştirme ortamının ağ erişimi
huggingface.co'ya kapalı olduğundan, BERTurk/Turkish-S-BERT modelleri BU
SANDBOX içinde indirilip test edilemedi. Kod, kendi bilgisayarınızda
(internet erişimi olan bir ortamda) sorunsuz çalışacak şekilde yazıldı.
Bu modül, iki backend sunar:

  1) "sentence-transformers" -> gerçek BERTurk/Turkish-S-BERT (varsayılan,
     PRODUCTION için kullanılması gereken seçenek, internet gerektirir)
  2) "tfidf-svd"              -> yalnızca çevrimdışı geliştirme/test için,
     TF-IDF + TruncatedSVD (LSA) ile üretilen basit "yalancı" embedding.
     Bu, gerçek bir sentence embedding YERİNE geçmez; sadece pipeline'ın
     (vektör DB, hibrit arama, API) internetsiz ortamlarda da uçtan uca
     test edilebilmesini sağlar.

Kendi korpusundan eğitilen Word2Vec/FastText (gensim) kısmı internet
gerektirmediği için TAMAMEN gerçek veriyle eğitilip test edilmiştir.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from gensim.models import Word2Vec, FastText

# Varsayılan olarak gerçek Türkçe Sentence-BERT modeli. İnternet erişimi olan
# bir makinede çalıştırdığınızda başka bir şey yapmanıza gerek yok.
TURKISH_SBERT_MODEL = "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr"


# ----------------------------------------------------------------------
# 1) Sıfırdan Word2Vec / FastText eğitimi
# ----------------------------------------------------------------------
def word2vec_egit(tokenli_cumleler: list[list[str]], vector_size=100, window=5, min_count=None, epochs=20) -> Word2Vec:
    if min_count is None:
        # Küçük korpuslarda (örn. 33 doküman) min_count=3 sözlüğün neredeyse
        # tamamını eler; büyük korpuslarda (3000+) ise 3 makul bir gürültü eşiğidir.
        min_count = 1 if len(tokenli_cumleler) < 200 else 3
    model = Word2Vec(
        sentences=tokenli_cumleler, vector_size=vector_size, window=window,
        min_count=min_count, workers=4, sg=1, epochs=epochs,
    )
    return model


def fasttext_egit(tokenli_cumleler: list[list[str]], vector_size=100, window=5, min_count=None, epochs=20) -> FastText:
    if min_count is None:
        min_count = 1 if len(tokenli_cumleler) < 200 else 3
    model = FastText(
        sentences=tokenli_cumleler, vector_size=vector_size, window=window,
        min_count=min_count, workers=4, sg=1, epochs=epochs,
    )
    return model


def dokuman_vektoru(model, tokenlar: list[str]) -> np.ndarray:
    """Bir dokümanın kelime vektörlerinin ortalaması (basit ama etkili doküman temsili)."""
    vektorler = [model.wv[t] for t in tokenlar if t in model.wv]
    if not vektorler:
        return np.zeros(model.vector_size)
    return np.mean(vektorler, axis=0)


# ----------------------------------------------------------------------
# 2) Sentence embedding backend'leri
# ----------------------------------------------------------------------
class SentenceEmbedder:
    """
    backend="sentence-transformers": BERTurk/Turkish-S-BERT (internet gerekir)
    backend="tfidf-svd"            : çevrimdışı geliştirme fallback'i
    """

    def __init__(self, backend: str = "sentence-transformers",
                 model_name: str = TURKISH_SBERT_MODEL, svd_boyut: int = 200):
        self.backend = backend
        self.model_name = model_name
        self.svd_boyut = svd_boyut
        self._model = None
        self._vectorizer = None
        self._svd = None

    def fit(self, metinler: list[str]):
        if self.backend == "sentence-transformers":
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        elif self.backend == "tfidf-svd":
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.decomposition import TruncatedSVD
            self._vectorizer = TfidfVectorizer(max_features=30000, ngram_range=(1, 2))
            tfidf = self._vectorizer.fit_transform(metinler)
            self._svd = TruncatedSVD(n_components=min(self.svd_boyut, tfidf.shape[1] - 1), random_state=42)
            self._svd.fit(tfidf)
        else:
            raise ValueError(f"Bilinmeyen backend: {self.backend}")
        return self

    def encode(self, metinler: list[str]) -> np.ndarray:
        if self.backend == "sentence-transformers":
            return np.asarray(self._model.encode(metinler, show_progress_bar=False))
        elif self.backend == "tfidf-svd":
            tfidf = self._vectorizer.transform(metinler)
            return self._svd.transform(tfidf)
        raise RuntimeError("Önce fit() çağırılmalı.")


# ----------------------------------------------------------------------
# 3) Intrinsic değerlendirme: benzerlik / analoji testleri
# ----------------------------------------------------------------------
def intrinsic_benzerlik_testi(model, kelime_ciftleri: list[tuple[str, str]]):
    """Word2Vec/FastText'in en_benzer() fonksiyonu ile kavramsal yakınlığı gösterir.
    Örn. 'yapay' ~ 'zek' gibi domain-özel yakınlıkların yakalanıp yakalanmadığını
    inceler. FastText, alt-kelime (subword) bilgisi kullandığı için Türkçe'nin
    zengin çekim/yapım ekleri karşısında Word2Vec'ten daha dayanıklıdır -
    bu fonksiyon iki modeli aynı kelime çiftleri üzerinde karşılaştırmak için
    kullanılabilir.
    """
    sonuclar = []
    for kelime, hedef in kelime_ciftleri:
        try:
            benzerlik = model.wv.similarity(kelime, hedef)
        except KeyError:
            benzerlik = None
        sonuclar.append({"kelime": kelime, "hedef": hedef, "benzerlik": benzerlik})
    return pd.DataFrame(sonuclar)


def en_yakin_kelimeler(model, kelime: str, top_n: int = 10):
    try:
        return model.wv.most_similar(kelime, topn=top_n)
    except KeyError:
        return []


# ----------------------------------------------------------------------
# Ana akış
# ----------------------------------------------------------------------
def ana_akis(csv_yolu: str = "ngram_ozellikli_veri.csv", backend: str = "tfidf-svd"):
    df = pd.read_csv(csv_yolu)
    tokenli_cumleler = [str(t).split() for t in df["lemma_ozet"].fillna("")]

    print("Word2Vec (kendi korpusundan, sıfırdan) eğitiliyor...")
    w2v = word2vec_egit(tokenli_cumleler)
    print("FastText (kendi korpusundan, sıfırdan) eğitiliyor...")
    ft = fasttext_egit(tokenli_cumleler)

    print("\n--- 'yapay' kelimesine en yakın kelimeler ---")
    print("Word2Vec :", en_yakin_kelimeler(w2v, "yapay", 8))
    print("FastText :", en_yakin_kelimeler(ft, "yapay", 8))

    # OOV örneği: FastText, alt-kelime bilgisiyle hiç görmediği bir çekimli
    # forma bile vektör üretebilir; Word2Vec KeyError verir.
    oov_kelime = "öğrenmesindeki"
    print(f"\nOOV testi ('{oov_kelime}' korpusta muhtemelen yok):")
    try:
        w2v.wv[oov_kelime]
        print("Word2Vec: vektör var (beklenmedik)")
    except KeyError:
        print("Word2Vec: KeyError (OOV kelimeler için vektör üretemiyor)")
    try:
        vek = ft.wv[oov_kelime]
        print(f"FastText: OOV kelime için de vektör üretebildi (subword sayesinde), boyut={vek.shape}")
    except KeyError:
        print("FastText: bu kelime için de üretemedi")

    print("\nDoküman vektörleri (Word2Vec ortalaması) hesaplanıyor...")
    doc_vektorleri_w2v = np.array([dokuman_vektoru(w2v, tok) for tok in tokenli_cumleler])
    print("Boyut:", doc_vektorleri_w2v.shape)

    print(f"\nSentence embedding backend='{backend}' ile cümle embedding'i üretiliyor...")
    embedder = SentenceEmbedder(backend=backend)
    embedder.fit(df["temiz_ozet"].fillna("").tolist())
    sbert_vektorleri = embedder.encode(df["temiz_ozet"].fillna("").tolist()[:5])
    print("Örnek (ilk 5 doküman) embedding boyutu:", sbert_vektorleri.shape)

    np.save("w2v_doc_vektorleri.npy", doc_vektorleri_w2v)
    w2v.save("w2v_model.bin")
    ft.save("fasttext_model.bin")
    print("\n✓ Word2Vec/FastText modelleri ve doküman vektörleri kaydedildi.")
    return w2v, ft, embedder


if __name__ == "__main__":
    ana_akis()
