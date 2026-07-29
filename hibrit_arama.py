# -*- coding: utf-8 -*-
"""
AŞAMA 4/5 — Hibrit Arama Motoru (arama_motoru.py'nin GRUP 1 gereksinimlerine göre
tamamlanmış / düzeltilmiş hâli)
=====================================================================
Orijinal arama_motoru.py'deki eksikler ve burada yapılan düzeltmeler:
  1) Orijinalde vektörler sadece bellekte tutuluyordu, kalıcı bir vektör DB
     yoktu -> burada vektor_db.py (Chroma) kullanılıyor.
  2) Orijinalde "hibrit arama" = ağırlıklı skor toplamıydı (alpha * dense +
     (1-alpha) * tfidf) -> görevde açıkça istenen YÖNTEM olan
     "Reciprocal Rank Fusion (RRF)" burada gerçek anlamıyla uygulanıyor.
  3) Orijinalde embedding modeli 'all-MiniLM-L6-v2' (Türkçe'ye özel değil,
     çok dilli genel amaçlı bir model) kullanılıyordu -> burada
     embedding_motoru.SentenceEmbedder üzerinden BERTurk/Turkish-S-BERT
     kullanılabiliyor (bkz. embedding_motoru.py başındaki not: bu sandbox'ta
     internet erişimi olmadığından gerçek model indirilemedi, offline
     "tfidf-svd" fallback ile uçtan uca test edildi).
  4) Metadata filtreleme (yıl, konu) eklendi.
  5) N-gram tabanlı arama artık gerçek BM25 (rank_bm25) ile yapılıyor;
     TF-IDF kosinüs benzerliği yerine, klasik bilgi erişiminin standart
     algoritması kullanılıyor.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from embedding_motoru import SentenceEmbedder
from vektor_db import AkademikVektorDB


def reciprocal_rank_fusion(siralamalar: list[list[str]], k: int = 60) -> dict[str, float]:
    """
    RRF(d) = sum_over_rankings( 1 / (k + rank(d)) )
    Birden çok arama yönteminin (BM25, dense vektör) sonuç sıralamalarını,
    ham skorların ölçeğiyle uğraşmadan (BM25 skoru ile kosinüs mesafesi
    farklı ölçeklerdedir, doğrudan toplanamaz) birleştirmenin standart yolu.
    k=60, literatürde (Cormack et al. 2009) önerilen ve yaygın kullanılan
    varsayılan sabittir.
    """
    skorlar: dict[str, float] = {}
    for siralama in siralamalar:
        for rank, doc_id in enumerate(siralama):
            skorlar[doc_id] = skorlar.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return skorlar


class HibritAramaMotoru:
    def __init__(self, csv_yolu: str = "ngram_ozellikli_veri.csv",
                 embedding_backend: str = "tfidf-svd",
                 chroma_yolu: str = "./chroma_db"):
        print("🔍 Veri yükleniyor...")
        df = pd.read_csv(csv_yolu)
        self.df = df[df["dil"] == "tr"].reset_index(drop=True)
        self.id_to_row = {str(r["id"]): r for _, r in self.df.iterrows()}

        print("📚 BM25 (lexical/n-gram tabanlı) index oluşturuluyor...")
        self.tokenli_korpus = [str(t).split() for t in self.df["temiz_ozet"].fillna("")]
        self.bm25 = BM25Okapi(self.tokenli_korpus)
        self.bm25_ids = [str(i) for i in self.df["id"].tolist()]

        print("🧠 Embedding + vektör DB hazırlanıyor...")
        self.embedder = SentenceEmbedder(backend=embedding_backend)
        self.embedder.fit(self.df["temiz_ozet"].fillna("").tolist())
        vektorler = self.embedder.encode(self.df["temiz_ozet"].fillna("").tolist())
        self.vdb = AkademikVektorDB(kalici_yol=chroma_yolu)
        if self.vdb.sayim() != len(self.df):
            self.vdb.yukle(self.df, vektorler)
        print("✅ Hibrit arama motoru hazır!")

    # ------------------------------------------------------------------
    def _bm25_sirala(self, sorgu: str, top_k: int) -> list[str]:
        sorgu_tok = sorgu.lower().split()
        skorlar = self.bm25.get_scores(sorgu_tok)
        top_idx = np.argsort(skorlar)[::-1][:top_k]
        return [self.bm25_ids[i] for i in top_idx]

    def _dense_sirala(self, sorgu: str, top_k: int,
                       yil_araligi=None, konu_filtre=None) -> list[str]:
        sorgu_vek = self.embedder.encode([sorgu])[0]
        sonuclar = self.vdb.ara(sorgu_vek, top_k=top_k, yil_araligi=yil_araligi, konu_filtre=konu_filtre)
        return [s["id"] for s in sonuclar]

    def ara(self, sorgu: str, top_k: int = 5, yontem: str = "hibrit",
            yil_araligi: tuple[int, int] | None = None, konu_filtre: str | None = None) -> list[dict]:
        """
        yontem: 'bm25' | 'dense' | 'hibrit' (RRF ile BM25 + dense birleşimi)
        """
        genis_k = max(top_k * 4, 20)

        if yontem == "bm25":
            siralama = self._bm25_sirala(sorgu, top_k)
        elif yontem == "dense":
            siralama = self._dense_sirala(sorgu, top_k, yil_araligi, konu_filtre)
        elif yontem == "hibrit":
            bm25_sira = self._bm25_sirala(sorgu, genis_k)
            dense_sira = self._dense_sirala(sorgu, genis_k, yil_araligi, konu_filtre)
            rrf_skorlar = reciprocal_rank_fusion([bm25_sira, dense_sira])
            siralama = [doc_id for doc_id, _ in sorted(rrf_skorlar.items(), key=lambda x: x[1], reverse=True)]
            # yıl/konu filtresi bm25 tarafında uygulanmadığı için hibritte de post-filter
            if yil_araligi or konu_filtre:
                siralama = [i for i in siralama if i in set(dense_sira)]
            siralama = siralama[:top_k]
        else:
            raise ValueError("yontem: 'bm25' | 'dense' | 'hibrit' olmalı")

        sonuc = []
        for doc_id in siralama:
            row = self.id_to_row.get(doc_id)
            if row is None:
                continue
            sonuc.append({
                "id": doc_id,
                "baslik": row["baslik"],
                "ozet": row["ozet"][:300],
                "yayin_yili": int(row["yayin_yili"]) if pd.notna(row["yayin_yili"]) else None,
                "konular": row["konular"],
            })
        return sonuc


if __name__ == "__main__":
    motor = HibritAramaMotoru()
    for yontem in ("bm25", "dense", "hibrit"):
        print(f"\n=== '{yontem}' yöntemi ile 'yapay zeka eğitim' araması ===")
        for r in motor.ara("yapay zeka eğitim", top_k=5, yontem=yontem):
            print(f"- [{r['yayin_yili']}] {r['baslik'][:70]}")
