# -*- coding: utf-8 -*-
"""
AŞAMA 5 — Servis: FastAPI Arama Servisi
=========================================
Orijinal projede bu dosya (app.py) BOŞTU. GRUP 1'in istediği
"FastAPI ile arama servisi" burada tamamlandı.

Çalıştırma:
    uvicorn app:app --reload --port 8000

Örnek istek:
    GET /ara?sorgu=yapay+zeka+eğitim&top_k=5&yontem=hibrit
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Query
from pydantic import BaseModel

from hibrit_arama import HibritAramaMotoru

_motor: HibritAramaMotoru | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _motor
    print("Arama motoru başlatılıyor (modeller yükleniyor)...")
    _motor = HibritAramaMotoru()
    print("Servis hazır.")
    yield


app = FastAPI(
    title="Akademik Hibrit Arama Servisi",
    description="N-gram (BM25) + anlamsal (embedding) hibrit arama — GRUP 1",
    version="1.0.0",
    lifespan=lifespan,
)


class SonucKaydi(BaseModel):
    id: str
    baslik: str
    ozet: str
    yayin_yili: int | None
    konular: str


@app.get("/", tags=["health"])
def anasayfa():
    return {"durum": "çalışıyor", "belge_sayisi": _motor.vdb.sayim() if _motor else 0}


@app.get("/ara", response_model=list[SonucKaydi], tags=["arama"])
def ara(
    sorgu: str = Query(..., description="Arama sorgusu, örn. 'yapay zeka eğitim'"),
    top_k: int = Query(5, ge=1, le=50),
    yontem: str = Query("hibrit", pattern="^(bm25|dense|hibrit)$"),
    yil_min: int | None = Query(None, description="Yıl aralığı alt sınırı"),
    yil_max: int | None = Query(None, description="Yıl aralığı üst sınırı"),
    konu: str | None = Query(None, description="Konu/disiplin filtresi, örn. 'Hukuk'"),
):
    yil_araligi = (yil_min, yil_max) if (yil_min and yil_max) else None
    sonuclar = _motor.ara(sorgu, top_k=top_k, yontem=yontem, yil_araligi=yil_araligi, konu_filtre=konu)
    return sonuclar


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
