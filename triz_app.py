# -*- coding: utf-8 -*-
"""
TRIZ — Streamlit Prototip Arayüzü
==================================
Çalıştırmak için:
    pip install streamlit pandas simplemma scikit-learn openpyxl
    streamlit run triz_app.py

triz_cohere.py ile AYNI klasörde olmalı.
"""

import streamlit as st
import pandas as pd
import re
import simplemma
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import urllib.request
import json
import time

# ─────────────────────────────────────────────────────────────
# AYARLAR
# ─────────────────────────────────────────────────────────────
import os
DOSYA = os.path.join(os.path.dirname(__file__), "Tez_Triz_Veri_14nisan.xlsx")
COHERE_API_KEY = "NeuB3jL9eQf0imVgamBGT5quYHOXOYP4GgioF5Tf"
COHERE_MODEL   = "command-a-03-2025"
TOP_K_ILKE     = 3

# ─────────────────────────────────────────────────────────────
# SAYFA AYARI
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TRIZ İlke Öneri Sistemi",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# ÖZEL STİL
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

/* Arka plan */
.stApp {
    background-color: #0f1117;
    color: #e8eaf0;
}

/* Başlık bandı */
.hero-band {
    background: linear-gradient(135deg, #1a1f2e 0%, #0d1b2a 60%, #112240 100%);
    border-left: 4px solid #00d4aa;
    border-radius: 2px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
}
.hero-band h1 {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.9rem;
    color: #00d4aa;
    margin: 0 0 0.3rem 0;
    letter-spacing: -0.5px;
}
.hero-band p {
    color: #8892a4;
    font-size: 0.95rem;
    margin: 0;
}

/* Kart */
.triz-card {
    background: #161b27;
    border: 1px solid #232b3e;
    border-radius: 6px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}

/* İlke badge */
.ilke-badge {
    display: inline-block;
    background: #0d2137;
    border: 1px solid #00d4aa44;
    color: #00d4aa;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    padding: 0.25rem 0.7rem;
    border-radius: 3px;
    margin: 0.2rem 0.2rem 0.2rem 0;
}

/* Skor çubuğu */
.skor-row {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    margin: 0.5rem 0;
}
.skor-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    color: #cdd6f4;
    min-width: 220px;
}
.skor-bar-bg {
    flex: 1;
    background: #1e2535;
    border-radius: 2px;
    height: 8px;
    overflow: hidden;
}
.skor-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, #00d4aa, #0096ff);
    border-radius: 2px;
}
.skor-pct {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    color: #00d4aa;
    min-width: 45px;
    text-align: right;
}

/* AI açıklama kutusu */
.ai-box {
    background: #0d1b2a;
    border-left: 3px solid #0096ff;
    border-radius: 0 6px 6px 0;
    padding: 1.2rem 1.4rem;
    font-size: 0.93rem;
    line-height: 1.7;
    color: #c9d1d9;
    white-space: pre-wrap;
}

/* Benzer problem */
.benzer-item {
    border-left: 2px solid #232b3e;
    padding-left: 0.9rem;
    margin: 0.6rem 0;
    color: #8892a4;
    font-size: 0.88rem;
}
.benzer-pct {
    color: #00d4aa;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0d1117;
    border-right: 1px solid #1e2535;
}

/* Buton */
.stButton > button {
    background: #00d4aa;
    color: #0f1117;
    border: none;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
    font-size: 0.9rem;
    padding: 0.6rem 2rem;
    border-radius: 3px;
    width: 100%;
    transition: background 0.2s;
}
.stButton > button:hover {
    background: #00f0c4;
    color: #0f1117;
}

/* Textarea */
.stTextArea textarea {
    background: #161b27 !important;
    border: 1px solid #2d3748 !important;
    color: #e8eaf0 !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 0.95rem !important;
    border-radius: 4px !important;
}

/* Slider */
.stSlider > div { color: #8892a4; }

/* Metric */
[data-testid="metric-container"] {
    background: #161b27;
    border: 1px solid #232b3e;
    border-radius: 6px;
    padding: 0.8rem 1rem;
}

/* Divider */
hr { border-color: #1e2535; }

/* Sekme */
.stTabs [data-baseweb="tab-list"] {
    background: #0d1117;
    border-bottom: 1px solid #1e2535;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
    color: #8892a4;
}
.stTabs [aria-selected="true"] {
    color: #00d4aa !important;
    border-bottom: 2px solid #00d4aa !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# YARDIMCI FONKSİYONLAR
# ─────────────────────────────────────────────────────────────
_stop = {
    "bir","ve","ile","bu","de","da","için","olan","o","şu","ben","sen",
    "biz","siz","onlar","ki","mi","mı","mu","mü","ya","yok","var","ama",
    "fakat","ancak","hem","veya","ne","nasıl","neden","çünkü","artırılması",
    "azaltılması","istenirken","gerekmektedir","olmaktadır","sağlamak",
    "kullanmak","yapılması","edilmesi","getirilmesi","hedeflenmektedir",
    "olarak","gibi","daha","çok","az","durum","nedeniyle","ilişkin","yönelik",
    "çalışma","ilgili","sahip","bağlı","sistem","proses","süreç","uygulama",
    "yöntem","tasarım","yapı","etki","analiz","problem","sorun",
}

def temizle(m):
    if not isinstance(m, str): return ""
    m = m.replace("İ","i").replace("I","ı").lower()
    m = re.sub(r"[^\w\sçğıöşü]", " ", m)
    m = re.sub(r"\s+", " ", m).strip()
    return " ".join(
        simplemma.lemmatize(w, lang="tr")
        for w in m.split() if w not in _stop and len(w) > 1
    )

@st.cache_resource(show_spinner="Veri tabanı yükleniyor...")
def yukle_model():
    pd.read_excel(DOSYA)
    df.columns = df.columns.str.strip()
    st.write(df.columns.tolist())  # geçici debug
    df["Problem"] = df["Problem"].fillna("").astype(str).str.strip()
    df["Önerilen Buluş İlkeleri (Ad)"] = (
        df["Önerilen Buluş İlkeleri (Ad)"].fillna("").astype(str).str.strip()
    )
    df = df[
        (df["Problem"] != "") &
        (df["Önerilen Buluş İlkeleri (Ad)"] != "") &
        (df["Önerilen Buluş İlkeleri (Ad)"] != "nan")
    ]
    grouped = (
        df.groupby("Problem", sort=False)
        .agg(ilke_seti=("Önerilen Buluş İlkeleri (Ad)",
                        lambda x: set(v for v in x.astype(str).str.strip() if v and v != "nan")))
        .reset_index()
    )
    grouped = grouped[grouped["ilke_seti"].apply(len) > 0].reset_index(drop=True)
    grouped["temiz"] = grouped["Problem"].apply(temizle)

    vec = TfidfVectorizer(ngram_range=(1, 2))
    X   = vec.fit_transform(grouped["temiz"])
    knn = NearestNeighbors(n_neighbors=min(5, len(grouped)), metric="cosine")
    knn.fit(X)
    return grouped, vec, X, knn


def tahmin_et(problem_metni, grouped, vec, X, knn, top_k):
    temiz = temizle(problem_metni)
    if not temiz.strip():
        return [], []
    sv      = vec.transform([temiz])
    skorlar = cosine_similarity(sv, X)[0]
    _, idxl = knn.kneighbors(sv)

    ilke_ag = {}
    for j in idxl[0]:
        for ilke in grouped.iloc[j]["ilke_seti"]:
            if ilke and str(ilke).strip() and str(ilke).strip() != "nan":
                ilke_ag[ilke] = ilke_ag.get(ilke, 0.0) + skorlar[j]

    if not ilke_ag:
        return [], []

    maks   = max(ilke_ag.values())
    sirali = sorted(ilke_ag.items(), key=lambda x: -x[1])
    oneriler = [(ilke, round(s / maks * 100, 1)) for ilke, s in sirali[:top_k]]

    dist, idxl2 = knn.kneighbors(sv, n_neighbors=min(3, len(grouped)))
    benzerler = [
        {
            "problem"  : grouped.iloc[j]["Problem"],
            "ilkeler"  : grouped.iloc[j]["ilke_seti"],
            "benzerlik": round((1 - d) * 100, 1),
        }
        for d, j in zip(dist[0], idxl2[0])
    ]
    return oneriler, benzerler


def cohere_acikla(problem, ilkeler):
    temiz_ilkeler = [str(p).strip() for p in ilkeler if p and str(p).strip()]
    if not temiz_ilkeler:
        return "Geçerli ilke bulunamadı."
    principle_str = ", ".join(sorted(temiz_ilkeler))
    prompt = f"""Sen deneyimli bir TRIZ uzmanısın.

Aşağıdaki mühendislik problemi için TRIZ sistemi şu buluş ilkelerini önerdi:
  {principle_str}

Görevin:
1. Her ilke için problem bağlamında KISA ve NET bir başlık yaz.
2. O ilkeyi bu probleme nasıl uygulayabileceğini 2-3 cümleyle, somut ve mühendislik diliyle açıkla.
3. En sona "Özet Öneri" başlığı altında tüm ilkeleri birleştiren 1-2 cümlelik bütünleşik bir çözüm yolu yaz.

Yanıtını YALNIZCA Türkçe ver.

PROBLEM: {problem}
"""
    payload = json.dumps({
        "model"      : COHERE_MODEL,
        "messages"   : [{"role": "user", "content": prompt}],
        "temperature": 0.4,
        "max_tokens" : 900,
    }).encode("utf-8")
    try:
        req = urllib.request.Request(
            "https://api.cohere.com/v2/chat",
            data=payload,
            headers={
                "Content-Type" : "application/json",
                "Authorization": f"Bearer {COHERE_API_KEY}",
                "X-Client-Name": "triz-streamlit",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=40) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["message"]["content"][0]["text"].strip()
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        return f"[Cohere HTTP hatası {e.code}]: {body[:300]}"
    except Exception as e:
        return f"[Bağlantı hatası]: {e}"

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Ayarlar")
    st.divider()

    top_k = st.slider("Önerilen ilke sayısı (Top-K)", 1, 5, TOP_K_ILKE)
    ai_aciklama = st.toggle("🤖 AI açıklaması üret", value=True)
    detay_mod   = st.toggle("📚 Benzer problemleri göster", value=False)

    st.divider()
    st.markdown("""
<div style='color:#8892a4; font-size:0.78rem; line-height:1.6'>
<b style='color:#00d4aa'>V4 Modeli</b><br>
TF-IDF + KNN (n=5)<br>
Ağırlıklı kosinüs oylama<br>
Problem bazlı split<br><br>
<b style='color:#00d4aa'>AI</b><br>
Cohere · command-a-03-2025
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# BAŞLIK
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-band">
  <h1>🔬 TRIZ İlke Öneri Sistemi</h1>
  <p>Mühendislik probleminizi girin — sistem en uygun TRIZ buluş ilkelerini önerir ve AI destekli çözüm açıklaması üretir.</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# MODEL YÜKLEME
# ─────────────────────────────────────────────────────────────
try:
    grouped, vec, X, knn = yukle_model()
    st.success(f"✓ Veri tabanı yüklendi — {len(grouped)} problem", icon="✅")
except FileNotFoundError:
    st.error(f"Excel dosyası bulunamadı: `{DOSYA}`\n\nLütfen `DOSYA` yolunu `triz_app.py` içinde güncelleyin.")
    st.stop()
except Exception as e:
    st.error(f"Yükleme hatası: {e}")
    st.stop()

# ─────────────────────────────────────────────────────────────
# ANA SEKME YAPISI
# ─────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🎯  Tahmin", "📊  Veri Tabanı"])

# ══════════════════════════════════════════════════════════════
# TAB 1 — TAHMİN
# ══════════════════════════════════════════════════════════════
with tab1:
    col_gir, col_sag = st.columns([3, 2], gap="large")

    with col_gir:
        st.markdown("#### Mühendislik Problemini Girin")
        problem_metni = st.text_area(
            label="",
            placeholder="Örnek: Bir pompanın verimi artırılmak istenirken enerji tüketiminin de azaltılması gerekmektedir...",
            height=160,
            label_visibility="collapsed",
        )
        ara_btn = st.button("🔍  TRIZ İlkelerini Bul", use_container_width=True)

    with col_sag:
        st.markdown("#### Nasıl Çalışır?")
        st.markdown("""
<div class="triz-card" style="font-size:0.87rem; color:#8892a4; line-height:1.8">
1. Probleminiz temizlenir ve TF-IDF vektörüne dönüştürülür<br>
2. Veri tabanındaki en benzer 5 problem bulunur<br>
3. Kosinüs benzerliği ile ilkeler ağırlıklı oylanır<br>
4. Top-K ilke önerilir, AI açıklaması üretilir
</div>
""", unsafe_allow_html=True)

    # ── Sonuçlar ──────────────────────────────────────────────
    if ara_btn:
        if not problem_metni.strip():
            st.warning("Lütfen bir problem metni girin.")
        else:
            with st.spinner("Analiz ediliyor..."):
                oneriler, benzerler = tahmin_et(
                    problem_metni, grouped, vec, X, knn, top_k
                )

            if not oneriler:
                st.error("Öneri üretilemedi. Daha açıklayıcı bir problem tanımı deneyin.")
            else:
                st.divider()

                # ── İlke skorları ─────────────────────────────
                st.markdown("#### 🎯 Önerilen TRIZ İlkeleri")
                ilke_adlari = []
                bars_html = ""
                for sira, (ilke, skor) in enumerate(oneriler, 1):
                    ilke_adlari.append(ilke)
                    bars_html += f"""
<div class="skor-row">
  <div class="skor-label">{sira}. {ilke}</div>
  <div class="skor-bar-bg">
    <div class="skor-bar-fill" style="width:{skor}%"></div>
  </div>
  <div class="skor-pct">%{skor:.0f}</div>
</div>"""
                st.markdown(f'<div class="triz-card">{bars_html}</div>', unsafe_allow_html=True)

                # ── Metrikler ─────────────────────────────────
                m1, m2, m3 = st.columns(3)
                m1.metric("Önerilen İlke", len(oneriler))
                m2.metric("En Yüksek Skor", f"%{oneriler[0][1]:.0f}")
                if benzerler:
                    m3.metric("En Benzer Problem", f"%{benzerler[0]['benzerlik']:.0f}")

                # ── AI Açıklaması ─────────────────────────────
                if ai_aciklama:
                    st.markdown("#### 🤖 AI Destekli Çözüm Açıklaması")
                    with st.spinner("Cohere AI açıklama üretiyor..."):
                        aciklama = cohere_acikla(problem_metni, ilke_adlari)
                    st.markdown(
                        f'<div class="ai-box">{aciklama}</div>',
                        unsafe_allow_html=True
                    )

                # ── Benzer Problemler ─────────────────────────
                if detay_mod and benzerler:
                    st.markdown("#### 📚 Veri Tabanındaki En Benzer Problemler")
                    for i, b in enumerate(benzerler, 1):
                        ilke_str = "  ".join(
                            f'<span class="ilke-badge">{ilke}</span>'
                            for ilke in sorted(b["ilkeler"])
                        )
                        st.markdown(f"""
<div class="triz-card">
  <div class="benzer-pct">%{b['benzerlik']:.1f} benzerlik</div>
  <div style="color:#cdd6f4; margin:0.3rem 0 0.5rem 0; font-size:0.9rem">{b['problem'][:200]}</div>
  <div>{ilke_str}</div>
</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# TAB 2 — VERİ TABANI
# ══════════════════════════════════════════════════════════════
with tab2:
    st.markdown("#### 📊 Veri Tabanı Özeti")

    # İstatistikler
    tum_ilkeler = []
    for s in grouped["ilke_seti"]:
        tum_ilkeler.extend(s)
    ilke_sayilari = pd.Series(tum_ilkeler).value_counts().reset_index()
    ilke_sayilari.columns = ["İlke", "Frekans"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Toplam Problem", len(grouped))
    c2.metric("Toplam İlke Çeşidi", ilke_sayilari["İlke"].nunique())
    c3.metric("Ortalama İlke/Problem",
              f"{pd.Series([len(s) for s in grouped['ilke_seti']]).mean():.1f}")

    st.divider()

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("##### En Sık Kullanılan İlkeler")
        st.bar_chart(ilke_sayilari.set_index("İlke").head(15))

    with col_b:
        st.markdown("##### İlke Frekans Tablosu")
        st.dataframe(
            ilke_sayilari,
            use_container_width=True,
            height=380,
        )

    st.divider()
    st.markdown("##### Tüm Problemler")
    arama = st.text_input("🔍 Problem ara...", placeholder="anahtar kelime")
    goster = grouped.copy()
    goster["ilke_seti"] = goster["ilke_seti"].apply(lambda s: ", ".join(sorted(s)))
    if arama.strip():
        goster = goster[goster["Problem"].str.contains(arama, case=False, na=False)]
    st.dataframe(
        goster[["Problem", "ilke_seti"]].rename(columns={"ilke_seti": "İlkeler"}),
        use_container_width=True,
        height=400,
    )
