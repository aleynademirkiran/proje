# -*- coding: utf-8 -*-
"""
AŞAMA 4 — Vektör Veritabanı Entegrasyonu
==========================================
GRUP 1 gereksinimleri:
  - Qdrant/Milvus yerine burada Chroma kullanıldı (görev metninde de belirtildiği
    gibi "Chroma daha basit" alternatif; kurulum/işletim yükü çok daha az,
    Qdrant/Milvus'a geçiş embedding fonksiyonu ve metadata şeması aynı kaldığı
    için ileride kolayca yapılabilir).
  - Metadata filtreleme desteği (yıl, disiplin/konu, üniversite yerine bu veri
    setinde üniversite bilgisi olmadığından "konular" alanı kullanıldı)
  - Hibrit arama için gereken dense-vektör tarafı burada sağlanır; BM25 tarafı
    hibrit_arama.py içinde ayrıca eklenir.

Not: chromadb varsayılan embedding fonksiyonu da HuggingFace'den model indirir;
bunun önüne geçmek için burada KENDİ embedding vektörlerimizi (embedding_motoru.py
çıktısı) chromadb'ye elle veriyoruz (embedding_function=None, embeddings=...).
Bu sayede vektör DB, hangi embedding kaynağını kullandığınızdan (BERTurk, kendi
Word2Vec'iniz, vs.) bağımsız ve internet erişimi gerektirmeden çalışır.
"""
from __future__ import annotations

import chromadb
import numpy as np
import pandas as pd


KOLEKSIYON_ADI = "akademik_makaleler"


class AkademikVektorDB:
    def __init__(self, kalici_yol: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(path=kalici_yol)
        self.koleksiyon = self.client.get_or_create_collection(
            name=KOLEKSIYON_ADI, metadata={"hnsw:space": "cosine"}
        )

    def yukle(self, df: pd.DataFrame, embeddings: np.ndarray, batch_size: int = 256):
        """
        df: en az ['id','baslik','ozet','yayin_yili','konular'] sütunlarını içermeli.
        embeddings: df ile aynı sırada, (n_doc, boyut) şeklinde önceden hesaplanmış vektörler.
        """
        assert len(df) == len(embeddings), "df ve embeddings uzunlukları eşleşmiyor"

        ids = [str(i) for i in df["id"].tolist()]
        dokumanlar = df["ozet"].fillna("").tolist()
        metadatalar = [
            {
                "baslik": str(row["baslik"]),
                "yayin_yili": int(row["yayin_yili"]) if pd.notna(row["yayin_yili"]) else -1,
                "konular": str(row.get("konular", "Belirtilmemiş")),
            }
            for _, row in df.iterrows()
        ]

        for i in range(0, len(ids), batch_size):
            self.koleksiyon.upsert(
                ids=ids[i:i + batch_size],
                embeddings=embeddings[i:i + batch_size].tolist(),
                documents=dokumanlar[i:i + batch_size],
                metadatas=metadatalar[i:i + batch_size],
            )
        print(f"✓ {len(ids)} doküman vektör veritabanına yüklendi (koleksiyon: {KOLEKSIYON_ADI}).")

    def ara(self, sorgu_vektoru: np.ndarray, top_k: int = 10,
            yil_araligi: tuple[int, int] | None = None, konu_filtre: str | None = None):
        """Metadata filtreleme destekli anlamsal (dense) arama."""
        # Not: Chroma'nın metadata "where" filtresi eşitlik/aralık türü koşullar için
        # uygundur ancak alt-dize (substring) araması desteklemez; bu yüzden
        # "konular" alanındaki serbest metin filtresi burada sonuç-sonrası
        # (post-filter) olarak uygulanır (aşağıda), yıl aralığı ise doğrudan
        # Chroma'nın "where" mekanizmasıyla veritabanı seviyesinde filtrelenir.
        kosullar = []
        if yil_araligi:
            kosullar.append({"yayin_yili": {"$gte": yil_araligi[0]}})
            kosullar.append({"yayin_yili": {"$lte": yil_araligi[1]}})
        where_clause = None
        if len(kosullar) == 2:  # yıl aralığı
            where_clause = {"$and": kosullar}
        elif len(kosullar) == 1:
            where_clause = kosullar[0]

        sonuc = self.koleksiyon.query(
            query_embeddings=[sorgu_vektoru.tolist()],
            n_results=top_k * 3 if konu_filtre else top_k,  # konu filtresi sonradan uygulanacaksa geniş çek
            where=where_clause,
        )

        kayitlar = []
        for i in range(len(sonuc["ids"][0])):
            meta = sonuc["metadatas"][0][i]
            if konu_filtre and konu_filtre.lower() not in meta.get("konular", "").lower():
                continue
            kayitlar.append({
                "id": sonuc["ids"][0][i],
                "baslik": meta.get("baslik"),
                "yayin_yili": meta.get("yayin_yili"),
                "konular": meta.get("konular"),
                "ozet": sonuc["documents"][0][i],
                "mesafe": sonuc["distances"][0][i],
            })
            if len(kayitlar) >= top_k:
                break
        return kayitlar

    def sayim(self) -> int:
        return self.koleksiyon.count()


def ana_akis():
    from embedding_motoru import SentenceEmbedder

    df = pd.read_csv("ngram_ozellikli_veri.csv")
    df_tr = df[df["dil"] == "tr"].reset_index(drop=True)

    print("Embedding üretiliyor (offline demo backend='tfidf-svd')...")
    embedder = SentenceEmbedder(backend="tfidf-svd", svd_boyut=200)
    embedder.fit(df_tr["temiz_ozet"].fillna("").tolist())
    vektorler = embedder.encode(df_tr["temiz_ozet"].fillna("").tolist())

    print("Vektör veritabanına yükleniyor (Chroma, kalıcı disk)...")
    db = AkademikVektorDB()
    db.yukle(df_tr, vektorler)
    print("Toplam kayıt:", db.sayim())

    print("\n--- Örnek arama: 'yapay zeka eğitim' (2024-2026, konu filtresi yok) ---")
    sorgu_vek = embedder.encode(["yapay zeka eğitim"])[0]
    sonuclar = db.ara(sorgu_vek, top_k=5, yil_araligi=(2024, 2026))
    for s in sonuclar:
        print(f"- [{s['yayin_yili']}] {s['baslik'][:70]} (mesafe={s['mesafe']:.3f})")

    print("\n--- Aynı sorgu, konu filtresi='Hukuk' ---")
    sonuclar2 = db.ara(sorgu_vek, top_k=5, konu_filtre="Hukuk")
    for s in sonuclar2:
        print(f"- [{s['konular']}] {s['baslik'][:70]}")


if __name__ == "__main__":
    ana_akis()
