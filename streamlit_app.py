# -*- coding: utf-8 -*-
"""
AŞAMA 5 — Servis: Streamlit Arayüzü
======================================
Çalıştırma:
    streamlit run streamlit_app.py
"""
import streamlit as st

from hibrit_arama import HibritAramaMotoru

st.set_page_config(page_title="Akademik Hibrit Arama", page_icon="📚", layout="wide")


@st.cache_resource(show_spinner="Arama motoru yükleniyor (ilk çalıştırmada biraz sürebilir)...")
def motor_yukle():
    return HibritAramaMotoru()


st.title("📚 Türkçe Akademik Veri Seti — N-gram + Anlamsal Hibrit Arama")
st.caption("GRUP 1: N-gram Dil Analizi + Vektör Veritabanlı Semantik Arama Motoru")

motor = motor_yukle()

col1, col2, col3 = st.columns([3, 1, 1])
with col1:
    sorgu = st.text_input("Arama sorgusu", value="yapay zeka eğitim")
with col2:
    yontem = st.selectbox("Yöntem", ["hibrit", "bm25", "dense"], index=0,
                           help="bm25 = klasik n-gram tabanlı, dense = anlamsal embedding, hibrit = RRF ile ikisinin birleşimi")
with col3:
    top_k = st.slider("Sonuç sayısı", 1, 20, 5)

with st.expander("Gelişmiş filtreler (metadata)"):
    fc1, fc2 = st.columns(2)
    with fc1:
        yillar = motor.df["yayin_yili"].dropna()
        yil_min_veri, yil_max_veri = (int(yillar.min()), int(yillar.max())) if len(yillar) else (2020, 2026)
        if yil_min_veri == yil_max_veri:
            yil_max_veri += 1  # st.slider aynı min/max değeriyle çalışmaz
        yil_araligi = st.slider("Yayın yılı aralığı", yil_min_veri, yil_max_veri, (yil_min_veri, yil_max_veri))
    with fc2:
        konu_secenekleri = ["(filtre yok)"] + sorted(
            {k.strip() for konu_str in motor.df["konular"].dropna() for k in konu_str.split(",")}
        )
        konu = st.selectbox("Konu/Disiplin", konu_secenekleri)

if st.button("🔍 Ara", type="primary") or sorgu:
    konu_filtre = None if konu == "(filtre yok)" else konu
    with st.spinner("Aranıyor..."):
        sonuclar = motor.ara(
            sorgu, top_k=top_k, yontem=yontem,
            yil_araligi=yil_araligi if yontem != "bm25" else None,
            konu_filtre=konu_filtre if yontem != "bm25" else None,
        )

    st.subheader(f"{len(sonuclar)} sonuç bulundu ({yontem})")
    for i, s in enumerate(sonuclar, 1):
        with st.container(border=True):
            st.markdown(f"**{i}. {s['baslik']}**")
            st.caption(f"📅 {s['yayin_yili']} · 🏷️ {s['konular']}")
            st.write(s["ozet"])

st.divider()
st.caption(
    "Not: 'bm25' klasik n-gram/TF-IDF tabanlı sözcük eşleşmesine, 'dense' anlamsal "
    "embedding benzerliğine, 'hibrit' ise Reciprocal Rank Fusion (RRF) ile ikisinin "
    "birleşimine dayanır. Metadata filtreleri (yıl/konu) yalnızca dense ve hibrit "
    "modlarında etkindir."
)
