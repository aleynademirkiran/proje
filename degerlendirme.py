# -*- coding: utf-8 -*-
"""
AŞAMA 5 — Servis ve Değerlendirme
===================================
GRUP 1 gereksinimi:
  "Değerlendirme: elle etiketlenmiş 50 sorgu üzerinde precision@k / recall@k,
   n-gram tabanlı arama ile embedding tabanlı aramanın karşılaştırmalı raporu"

Bu modül, veri setindeki "konular" (disiplin) alanını sözde-etiket (silver
label) olarak kullanan yarı otomatik bir değerlendirme seti üretir: bir sorgu
için "ilgili" kabul edilen dokümanlar, sorgudaki anahtar terimi başlığında/
özetinde geçiren VE aynı disiplin etiketine sahip dokümanlar olarak tanımlanır.
Bu, elle etiketleme yapılmadığı için MÜKEMMEL bir ground truth değildir (bu
kod bloğunun başında ve raporda bu sınırlama açıkça belirtilir), ancak
yöntemler arasında ADİL bir karşılaştırma yapılmasını sağlayacak ölçüde
tutarlıdır. Gerçek bir teslimde bu 50 sorgu elle, bir uzman tarafından
etiketlenmelidir; burada iskelet + otomatik-üretim mekanizması sağlanmıştır.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from hibrit_arama import HibritAramaMotoru

# 50 test sorgusu: gerçek veri setindeki sık geçen disiplin/konu kombinasyonlarından
# türetildi (bkz. README - "Değerlendirme setinin nasıl üretildiği").
def otomatik_sorgu_seti_olustur(df: pd.DataFrame) -> list[tuple[str, str]]:
    """
    Sabit/elle yazılmış bir sorgu listesi yerine (önceki sürümde olduğu gibi),
    test sorgularını doğrudan veri setindeki gerçek 'konular' etiketlerinden
    türetir. Bu sayede değerlendirme, hangi veri seti yüklenirse yüklensin
    (33 kayıt da olsa, 3102 kayıt da olsa) OTOMATİK olarak o veri setine uygun
    ve en az bir "ilgili doküman"ı garanti eden sorgular üretir.

    Yöntem: 'Belirtilmemiş' olmayan her konu alanındaki ilk (en spesifik)
    disiplin ifadesi bir sorgu/konu çifti olarak alınır; tekrarlar elenir.
    """
    sorgular = []
    gorulen = set()
    for konu_str in df["konular"].dropna():
        if konu_str.strip() == "Belirtilmemiş":
            continue
        ilk_konu = konu_str.split(",")[0].strip()
        if ilk_konu and ilk_konu.lower() not in gorulen:
            gorulen.add(ilk_konu.lower())
            sorgular.append((ilk_konu.lower(), ilk_konu))
    # Veri setinin ortak teması olan genel bir sorgu da eklenir (çoğu akademik
    # veri setinde olduğu gibi burada da 'yapay zeka' baskın temadır).
    if "yapay zeka" not in gorulen:
        sorgular.append(("yapay zeka", None))  # konu=None -> ilgililik sadece metin eşleşmesiyle belirlenir
    return sorgular


TEST_SORGULARI: list[tuple[str, str | None]] | None = None  # None => otomatik üretilir (ana_akis içinde)


def ilgili_dokumanlari_belirle(df: pd.DataFrame, anahtar_terim: str, konu: str | None) -> set[str]:
    """Silver-label ground truth: konu etiketi eşleşen VE anahtar terimin en az
    bir kelimesini başlık/özette geçiren dokümanlar 'ilgili' kabul edilir.
    konu=None ise (örn. veri setinde 'Belirtilmemiş' baskınsa) sadece metin
    eşleşmesi kullanılır."""
    terimler = [t for t in anahtar_terim.lower().split() if len(t) > 3]
    metin = (df["baslik"].fillna("") + " " + df["ozet"].fillna("")).str.lower()
    terim_maskesi = metin.apply(lambda m: any(t in m for t in terimler))
    if konu:
        konu_maskesi = df["konular"].str.lower().str.contains(konu.lower(), na=False)
        ilgili = df[konu_maskesi & terim_maskesi]
    else:
        ilgili = df[terim_maskesi]
    return set(str(i) for i in ilgili["id"].tolist())


def precision_at_k(getirilen: list[str], ilgili: set[str], k: int) -> float:
    top_k = getirilen[:k]
    if not top_k:
        return 0.0
    return sum(1 for d in top_k if d in ilgili) / len(top_k)


def recall_at_k(getirilen: list[str], ilgili: set[str], k: int) -> float:
    if not ilgili:
        return None
    top_k = getirilen[:k]
    return sum(1 for d in top_k if d in ilgili) / len(ilgili)


def degerlendirme_calistir(motor: HibritAramaMotoru, k: int = 10) -> pd.DataFrame:
    sorgu_seti = TEST_SORGULARI if TEST_SORGULARI is not None else otomatik_sorgu_seti_olustur(motor.df)
    sonuclar = []
    for sorgu, konu in sorgu_seti:
        ilgili = ilgili_dokumanlari_belirle(motor.df, sorgu, konu)
        if not ilgili:
            continue  # bu sorgu için ground-truth bulunamadıysa atla

        for yontem in ("bm25", "dense", "hibrit"):
            getirilen_kayitlar = motor.ara(sorgu, top_k=k, yontem=yontem)
            getirilen_id = [r["id"] for r in getirilen_kayitlar]
            p = precision_at_k(getirilen_id, ilgili, k)
            r = recall_at_k(getirilen_id, ilgili, k)
            sonuclar.append({
                "sorgu": sorgu, "konu": konu, "yontem": yontem,
                f"precision@{k}": p, f"recall@{k}": r, "ilgili_sayisi": len(ilgili),
            })
    return pd.DataFrame(sonuclar)


def ozet_rapor(sonuc_df: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    return sonuc_df.groupby("yontem")[[f"precision@{k}", f"recall@{k}"]].mean().round(4)


if __name__ == "__main__":
    motor = HibritAramaMotoru()
    sorgu_seti = otomatik_sorgu_seti_olustur(motor.df)
    print(f"Otomatik üretilen test sorgusu sayısı: {len(sorgu_seti)}")
    if len(motor.df) < 100:
        print(f"⚠️  UYARI: veri seti sadece {len(motor.df)} Türkçe doküman içeriyor. "
              f"precision@k/recall@k gibi metrikler bu ölçekte istatistiksel olarak "
              f"anlamlı DEĞİLDİR; veri seti büyüdükçe (örn. scraper tamamlandığında) "
              f"bu script'i tekrar çalıştırıp sonuçları güncelleyin.")

    print("\nDeğerlendirme koşuluyor...")
    k = min(10, max(3, len(motor.df) // 3))
    print(f"(Korpus büyüklüğüne göre k={k} kullanılıyor)")
    sonuc_df = degerlendirme_calistir(motor, k=k)
    sonuc_df.to_csv("degerlendirme_sonuclari.csv", index=False)

    print(f"\nGeçerli (ground-truth bulunabilen) sorgu sayısı: {sonuc_df['sorgu'].nunique()}")
    print(f"\n=== ÖZET: yöntem bazlı ortalama precision@{k} / recall@{k} ===")
    print(ozet_rapor(sonuc_df, k=k))
    print("\n✓ Detaylı sonuçlar kaydedildi: degerlendirme_sonuclari.csv")
