# -*- coding: utf-8 -*-
"""
AŞAMA 1 — Ön İşleme
====================
GRUP 1 gereksinimleri:
  - Türkçe'ye özgü normalizasyon (İ/i, I/ı sorunu)
  - Korpustan frekansla genişletilen Türkçe stopword listesi (körü körüne hazır liste değil)
  - Zeyrek (Python) ile morfolojik analiz / kök bulma (lemmatization)
  - Cümle ve kelime tokenizasyonu (NLTK punkt + Türkçe'ye özel ek kurallar)
  - Dil tespiti: veri setinde İngilizce özetler de bulunduğu için, Türkçe'ye özgü
    normalizasyon/lemmatization sadece Türkçe metinlere uygulanır (aksi halde
    yanlış/anlamsız kök bulma sonuçları üretilir).

Bu modül, projede bulunan `onisleme_test.py` / `01_veri_ve_onisleme.ipynb`
dosyalarındaki basit temizleme fonksiyonunun üzerine inşa edilmiştir ve onu
GRUP 1'in istediği seviyeye tamamlar.
"""
from __future__ import annotations

import re
import sqlite3
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

warnings.filterwarnings("ignore")

for pkg in ("stopwords", "punkt", "punkt_tab"):
    try:
        nltk.data.find(f"corpora/{pkg}" if pkg == "stopwords" else f"tokenizers/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

try:
    import logging
    logging.disable(logging.WARNING)  # zeyrek'in gürültülü "APPENDING RESULT" loglarını sustur
    import zeyrek
    _ZEYREK_VAR = True
except Exception:  # pragma: no cover - zeyrek kurulu değilse
    _ZEYREK_VAR = False

try:
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 42
    _LANGDETECT_VAR = True
except Exception:  # pragma: no cover
    _LANGDETECT_VAR = False


TURKCE_HARFLER = "abcçdefgğhıijklmnoöprsştuüvyzxwq"


@dataclass
class OnislemeSonucu:
    ham: str
    dil: str
    normalize: str
    tokenlar: list = field(default_factory=list)
    lemma: str = ""
    cumle_sayisi: int = 0


class TurkceOnisleyici:
    """Türkçe akademik özetler için ön işleme sınıfı."""

    def __init__(self, ekstra_stopwords: set[str] | None = None):
        self.temel_stopwords = set(stopwords.words("turkish"))
        if ekstra_stopwords:
            self.temel_stopwords.update(ekstra_stopwords)
        self._zeyrek = zeyrek.MorphAnalyzer() if _ZEYREK_VAR else None
        self._lemma_cache: dict[str, str] = {}

    # ------------------------------------------------------------------
    # 1) Dil tespiti — sadece Türkçe metinlere Türkçe normalizasyon uygula
    # ------------------------------------------------------------------
    def dil_tespit_et(self, metin: str) -> str:
        if not isinstance(metin, str) or not metin.strip():
            return "unk"
        if _LANGDETECT_VAR:
            try:
                return detect(metin[:500])
            except Exception:
                pass
        # langdetect yoksa basit ısı-fitresi: Türkçe'ye özgü harf oranı
        turkce_karakter = sum(metin.lower().count(c) for c in "çğıöşü")
        return "tr" if turkce_karakter > 0 else "en"

    # ------------------------------------------------------------------
    # 2) Normalizasyon — Türkçe büyük/küçük harf (İ/i, I/ı) + noktalama
    # ------------------------------------------------------------------
    def normalize_et(self, metin: str) -> str:
        if not isinstance(metin, str):
            return ""
        metin = metin.replace("İ", "i").replace("I", "ı")
        metin = metin.lower()
        metin = re.sub(r"[^a-zçğıöşü\s]", " ", metin)
        metin = re.sub(r"\s+", " ", metin).strip()
        return metin

    # ------------------------------------------------------------------
    # 3) Tokenizasyon — cümle + kelime (Türkçe kısaltmalarda basit düzeltme)
    # ------------------------------------------------------------------
    def cumlelere_ayir(self, ham_metin: str) -> list[str]:
        if not isinstance(ham_metin, str) or not ham_metin.strip():
            return []
        # NLTK punkt İngilizce eğitildiği için Türkçe'ye özgü kısaltmalarda
        # (Dr., vb., Prof.) hatalı bölünmeyi azaltmak için basit bir koruma:
        korunan = re.sub(r"\b(Dr|Prof|vb|vs|Doç)\.", r"\1<NOKTA>", ham_metin)
        cumleler = sent_tokenize(korunan, language="english")
        return [c.replace("<NOKTA>", ".") for c in cumleler]

    def kelimelere_ayir(self, normalize_metin: str) -> list[str]:
        return [t for t in word_tokenize(normalize_metin) if t not in self.temel_stopwords]

    # ------------------------------------------------------------------
    # 4) Lemmatization — Zeyrek ile morfolojik kök bulma
    # ------------------------------------------------------------------
    def lemmatize(self, tokenlar: list[str]) -> str:
        if not self._zeyrek:
            return " ".join(tokenlar)  # zeyrek yoksa ham tokenlarla devam
        sonuc = []
        for tok in tokenlar:
            if tok in self._lemma_cache:
                sonuc.append(self._lemma_cache[tok])
                continue
            try:
                analiz = self._zeyrek.lemmatize(tok)
                lemma = analiz[0][1][0] if analiz and analiz[0][1] else tok
            except Exception:
                lemma = tok
            self._lemma_cache[tok] = lemma
            sonuc.append(lemma)
        return " ".join(sonuc)

    # ------------------------------------------------------------------
    # Tüm adımları uçtan uca uygulayan yardımcı
    # ------------------------------------------------------------------
    def isle(self, ham_metin: str) -> OnislemeSonucu:
        dil = self.dil_tespit_et(ham_metin)
        cumleler = self.cumlelere_ayir(ham_metin) if isinstance(ham_metin, str) else []
        if dil != "tr":
            # İngilizce/başka dil: Türkçe'ye özgü adımları uygulamadan
            # sadece küçük harfe çevirip noktalamaları temizliyoruz.
            normalize = re.sub(r"\s+", " ", re.sub(r"[^a-z\s]", " ", str(ham_metin).lower())).strip()
            tokenlar = [t for t in normalize.split() if t]
            lemma = " ".join(tokenlar)
        else:
            normalize = self.normalize_et(ham_metin)
            tokenlar = self.kelimelere_ayir(normalize)
            lemma = self.lemmatize(tokenlar)
        return OnislemeSonucu(
            ham=ham_metin, dil=dil, normalize=normalize,
            tokenlar=tokenlar, lemma=lemma, cumle_sayisi=len(cumleler),
        )


def korpustan_stopword_genislet(metinler: list[str], mevcut_stopwords: set[str],
                                  ust_yuzde: float = 0.5, min_uzunluk: int = 2) -> set[str]:
    """
    'Hazır listeyi körü körüne kullanmak yerine korpustan frekansla genişletme.'
    Belgelerin ust_yuzde'sinden fazlasında geçen (ve zaten anlamsız/yaygın olan)
    kelimeleri stopword adayı olarak işaretler. Akademik jargonun (örn. 'model',
    'yöntem' gibi konudan bağımsız ama gerçekten sık kullanılan genel kelimeler)
    stopword'e dönüşüp dönüşmeyeceğine karar verilebilmesi için aday listesi
    döndürülür; otomatik olarak stopword listesine eklenmez.
    """
    from collections import Counter
    doc_freq = Counter()
    for metin in metinler:
        if not isinstance(metin, str):
            continue
        kelimeler = set(re.findall(r"[a-zçğıöşü]+", metin.lower()))
        doc_freq.update(kelimeler)
    n_doc = len(metinler)
    adaylar = {
        kelime for kelime, df in doc_freq.items()
        if df / n_doc >= ust_yuzde and len(kelime) >= min_uzunluk and kelime not in mevcut_stopwords
    }
    return adaylar


def veriyi_yukle(sqlite_yolu: str = "akademik_veri.sqlite3") -> pd.DataFrame:
    conn = sqlite3.connect(sqlite_yolu)
    df = pd.read_sql_query(
        "SELECT id, yayin_id, baslik, ozet, yayin_yili, konular FROM makaleler", conn
    )
    conn.close()
    return df


def ana_akis(sqlite_yolu: str = "akademik_veri.sqlite3",
             cikti_csv: str = "temizlenmis_akademik_veri_v2.csv") -> pd.DataFrame:
    print("Veri yükleniyor...")
    df = veriyi_yukle(sqlite_yolu)
    print(f"Toplam kayıt: {len(df)}")

    # Ek stopword adaylarını korpustan tespit et (raporlama amaçlı)
    ek_manuel = {"vb", "vs", "veya", "göre", "kadar", "üzerine", "yapılan", "bulunmaktadır", "bu", "ile"}
    onisleyici = TurkceOnisleyici(ekstra_stopwords=ek_manuel)
    adaylar = korpustan_stopword_genislet(df["ozet"].tolist(), onisleyici.temel_stopwords)
    print(f"Korpus-frekans tabanlı stopword adayı sayısı (>%50 belgede geçen): {len(adaylar)}")
    print("Örnek adaylar:", list(adaylar)[:15])

    print("Ön işleme uygulanıyor (dil tespiti + normalizasyon + lemmatization)...")
    sonuclar = df["ozet"].apply(onisleyici.isle)
    df["dil"] = sonuclar.apply(lambda s: s.dil)
    df["temiz_ozet"] = sonuclar.apply(lambda s: s.normalize)
    df["lemma_ozet"] = sonuclar.apply(lambda s: s.lemma)
    df["cumle_sayisi"] = sonuclar.apply(lambda s: s.cumle_sayisi)
    df["kelime_sayisi"] = df["temiz_ozet"].apply(lambda t: len(t.split()) if t else 0)

    print("\nDil dağılımı:")
    print(df["dil"].value_counts())

    df.to_csv(cikti_csv, index=False)
    print(f"\n✓ Kaydedildi: {cikti_csv}")
    return df


if __name__ == "__main__":
    ana_akis()
