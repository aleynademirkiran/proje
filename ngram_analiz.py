# -*- coding: utf-8 -*-
"""
AŞAMA 2 — N-gram Çıkarımı ve Frekans Analizi
=============================================
GRUP 1 gereksinimleri:
  - Kelime bazlı 2/3/4-gram (ham vs lemmatize edilmiş metin karşılaştırması)
  - Karakter n-gram + OOV tartışması
  - Ham frekans / TF-IDF / PMI-Log-likelihood ile collocation ("yapay zeka" gibi
    anlamlı öbeklerin rastgele n-gram'lardan ayrıştırılması)
  - Dil özellikleri: ortalama cümle/kelime uzunluğu, type-token ratio,
    Ateşman okunabilirlik formülü (Türkçe)
"""
from __future__ import annotations

import re
from collections import Counter

import pandas as pd
from nltk import ngrams
from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder
from sklearn.feature_extraction.text import TfidfVectorizer

VOWELS = set("aeıioöuü")


# ----------------------------------------------------------------------
# Kelime / karakter n-gram çıkarımı
# ----------------------------------------------------------------------
def kelime_ngram(metin: str, n: int) -> list[str]:
    tokenlar = metin.split() if isinstance(metin, str) else []
    return [" ".join(g) for g in ngrams(tokenlar, n)] if len(tokenlar) >= n else []


def karakter_ngram(metin: str, n: int = 3) -> list[str]:
    metin = re.sub(r"\s+", "", metin) if isinstance(metin, str) else ""
    return [metin[i:i + n] for i in range(len(metin) - n + 1)]


def ham_vs_lemma_ngram_karsilastir(df: pd.DataFrame, n: int = 2, top_k: int = 15) -> pd.DataFrame:
    """Ham (temiz_ozet) ve lemmatize (lemma_ozet) metin üzerinde n-gram frekanslarını
    karşılaştırır. Lemmatize metinde çekim eklerinden arındığı için aynı kavramın
    farklı yüzey formları (öğrenme/öğrenmesi/öğrenmenin) tek bir n-gram altında
    toplanır -> frekanslar daha az parçalı, daha yüksek çıkar."""
    ham_sayac, lemma_sayac = Counter(), Counter()
    for ham, lemma in zip(df["temiz_ozet"].fillna(""), df["lemma_ozet"].fillna("")):
        ham_sayac.update(kelime_ngram(ham, n))
        lemma_sayac.update(kelime_ngram(lemma, n))
    ham_top = pd.DataFrame(ham_sayac.most_common(top_k), columns=["ham_ngram", "ham_frekans"])
    lemma_top = pd.DataFrame(lemma_sayac.most_common(top_k), columns=["lemma_ngram", "lemma_frekans"])
    return pd.concat([ham_top, lemma_top], axis=1)


# ----------------------------------------------------------------------
# TF-IDF n-gram matrisi
# ----------------------------------------------------------------------
def tfidf_ngram_matrisi(metinler: list[str], ngram_range=(1, 3), max_features=20000):
    vec = TfidfVectorizer(ngram_range=ngram_range, max_features=max_features)
    matris = vec.fit_transform(metinler)
    return vec, matris


# ----------------------------------------------------------------------
# Collocation tespiti: PMI / log-likelihood
# "yapay zeka", "makine öğrenmesi" gibi ifadelerin, yüksek frekanslı ama
# anlamsal olarak bağlantısız rastgele n-gram'lardan ayrıştırılması.
# Sadece ham frekans yüksek n-gram'lar arasında "yapay zeka" gibi anlamlı
# ifadeler kaybolabilir (örn. "bu çalışma" da sık geçer ama collocation değildir);
# PMI/log-likelihood, iki kelimenin birlikte, ayrı ayrı beklenenden çok daha sık
# görülüp görülmediğini ölçerek bunu ayırt eder.
# ----------------------------------------------------------------------
def collocation_bul(tum_tokenlar: list[str], min_freq: int = 15, top_k: int = 25):
    finder = BigramCollocationFinder.from_words(tum_tokenlar)
    finder.apply_freq_filter(min_freq)

    pmi_sonuc = finder.nbest(BigramAssocMeasures.pmi, top_k)
    llr_sonuc = finder.nbest(BigramAssocMeasures.likelihood_ratio, top_k)
    raw_freq_sonuc = finder.nbest(BigramAssocMeasures.raw_freq, top_k)

    return {
        "pmi": [" ".join(b) for b in pmi_sonuc],
        "log_likelihood": [" ".join(b) for b in llr_sonuc],
        "ham_frekans": [" ".join(b) for b in raw_freq_sonuc],
    }


# ----------------------------------------------------------------------
# Dil özellikleri: TTR, ortalama cümle/kelime uzunluğu, Ateşman okunabilirlik
# ----------------------------------------------------------------------
def hece_say(kelime: str) -> int:
    """Türkçe'de hece sayısı ~ ünlü harf sayısına eşittir (yaygın yaklaşım)."""
    return max(1, sum(1 for h in kelime if h in VOWELS))


def type_token_ratio(metin: str) -> float:
    tokenlar = metin.split() if isinstance(metin, str) else []
    if not tokenlar:
        return 0.0
    return len(set(tokenlar)) / len(tokenlar)


def atesman_okunabilirlik(ham_metin: str, cumle_sayisi: int) -> float | None:
    """
    Ateşman (1997) Türkçe okunabilirlik formülü:
        Puan = 198.825 - 40.175 * (ort. hece/kelime) - 2.610 * (ort. kelime/cümle)
    Yüksek puan = daha kolay okunur metin.
    """
    if not isinstance(ham_metin, str) or cumle_sayisi == 0:
        return None
    kelimeler = re.findall(r"[a-zA-ZçğıöşüÇĞİÖŞÜ]+", ham_metin)
    if not kelimeler:
        return None
    ort_hece = sum(hece_say(k) for k in kelimeler) / len(kelimeler)
    ort_kelime_cumle = len(kelimeler) / cumle_sayisi
    return 198.825 - 40.175 * ort_hece - 2.610 * ort_kelime_cumle


def dil_ozellikleri_hesapla(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ttr"] = df["temiz_ozet"].apply(type_token_ratio)
    df["atesman_puani"] = df.apply(
        lambda r: atesman_okunabilirlik(r["ozet"], r.get("cumle_sayisi", 0) or 1)
        if r.get("dil") == "tr" else None, axis=1,
    )
    df["ort_kelime_uzunlugu"] = df["temiz_ozet"].apply(
        lambda t: (sum(len(k) for k in t.split()) / len(t.split())) if isinstance(t, str) and t.split() else None
    )
    return df


# ----------------------------------------------------------------------
# Ana akış
# ----------------------------------------------------------------------
def ana_akis(csv_yolu: str = "temizlenmis_akademik_veri_v2.csv"):
    df = pd.read_csv(csv_yolu)
    df_tr = df[df["dil"] == "tr"].copy()
    print(f"Türkçe kayıt sayısı: {len(df_tr)} / toplam {len(df)}")

    print("\n--- Ham vs Lemmatize 2-gram karşılaştırması (ilk 15) ---")
    print(ham_vs_lemma_ngram_karsilastir(df_tr, n=2))

    print("\n--- TF-IDF matrisi oluşturuluyor (1-3 gram) ---")
    vec, matris = tfidf_ngram_matrisi(df_tr["temiz_ozet"].fillna("").tolist())
    print("TF-IDF matris boyutu:", matris.shape)

    print("\n--- Karakter 3-gram örneği (OOV/bilinmeyen kelime durumunda faydası) ---")
    ornek = df_tr["temiz_ozet"].iloc[0]
    print("Örnek metin:", ornek[:80])
    print("Karakter 3-gram (ilk 10):", karakter_ngram(ornek, 3)[:10])
    print("Not: karakter n-gram, sözlükte bulunmayan (OOV) yeni/nadir akademik terimler")
    print("için bile alt-kelime düzeyinde benzerlik kurulmasına imkân tanır; örn.")
    print("'öğrenmesi' hiç görülmemiş olsa bile 'öğren' alt-dizisi başka kelimelerle eşleşebilir.")

    print("\n--- Collocation tespiti (PMI / Log-Likelihood / Ham frekans) ---")
    tum_tokenlar = " ".join(df_tr["temiz_ozet"].fillna("")).split()
    # min_freq, korpus büyüklüğüne göre uyarlanır: küçük veri setlerinde (örn. 33
    # doküman) sabit min_freq=15 hiçbir bigram bulamaz; büyük veri setlerinde
    # (3000+ doküman) ise çok düşük bir eşik anlamsız/gürültülü sonuçlar üretir.
    min_freq = max(2, len(df_tr) // 10)
    print(f"(Korpus büyüklüğüne göre uyarlanan min_freq={min_freq}, doküman sayısı={len(df_tr)})")
    colloc = collocation_bul(tum_tokenlar, min_freq=min_freq, top_k=20)
    for yontem, sonuc in colloc.items():
        print(f"\n[{yontem}] ilk 10:")
        print(sonuc[:10])

    print("\n--- Dil özellikleri (TTR, Ateşman okunabilirlik, ort. kelime uzunluğu) ---")
    df_ozellik = dil_ozellikleri_hesapla(df_tr)
    print(df_ozellik[["ttr", "atesman_puani", "ort_kelime_uzunlugu"]].describe())

    df_ozellik.to_csv("ngram_ozellikli_veri.csv", index=False)
    print("\n✓ Kaydedildi: ngram_ozellikli_veri.csv")
    return df_ozellik, colloc


if __name__ == "__main__":
    ana_akis()
