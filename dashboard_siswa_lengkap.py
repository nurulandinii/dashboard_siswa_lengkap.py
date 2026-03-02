import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ==========================================================
# KONFIGURASI HALAMAN
# ==========================================================
st.set_page_config(page_title="Dashboard Analisis Hasil Tes", layout="wide")
st.title("📊 Dashboard Analisis Hasil Tes Siswa")
st.markdown("Analisis berbasis data untuk evaluasi pembelajaran")

# ==========================================================
# LOAD DATA
# ==========================================================
df = pd.read_excel("data_simulasi_50_siswa_20_soal (1).xlsx")

nama_kolom = df.columns[0]
soal = df.columns[1:]

df[soal] = df[soal].apply(pd.to_numeric, errors="coerce").fillna(0)

# Hitung Total Benar
df["Total Benar"] = df[soal].sum(axis=1)

# ==========================================================
# KPI UTAMA
# ==========================================================
rata_rata = df["Total Benar"].mean()
tertinggi = df["Total Benar"].max()
terendah = df["Total Benar"].min()

col1, col2, col3 = st.columns(3)
col1.metric("📈 Rata-rata Nilai", f"{rata_rata:.2f}")
col2.metric("🏆 Nilai Tertinggi", int(tertinggi))
col3.metric("📉 Nilai Terendah", int(terendah))

st.divider()

# ==========================================================
# DISTRIBUSI NILAI
# ==========================================================
st.header("1️⃣ Distribusi Nilai Siswa")

fig1, ax1 = plt.subplots(figsize=(6,4))
ax1.hist(df["Total Benar"], bins=10)
ax1.set_xlabel("Total Benar")
ax1.set_ylabel("Jumlah Siswa")
ax1.set_title("Distribusi Skor")
st.pyplot(fig1)

st.divider()

# ==========================================================
# ANALISIS SOAL (TINGKAT KESULITAN)
# ==========================================================
st.header("2️⃣ Analisis Tingkat Kesulitan Soal")

rata_soal = df[soal].mean()

fig2, ax2 = plt.subplots(figsize=(8,4))
ax2.bar(rata_soal.index, rata_soal.values)
ax2.set_ylabel("Rata-rata Jawaban Benar")
ax2.set_title("Tingkat Kesulitan Soal")
plt.xticks(rotation=90)
st.pyplot(fig2)

soal_tersulit = rata_soal.idxmin()
st.info(f"📌 Soal paling sulit: {soal_tersulit}")

st.divider()

# ==========================================================
# SEGMENTASI SISWA
# ==========================================================
st.header("3️⃣ Segmentasi Performa Siswa")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[soal])

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)

cluster_mean = df.groupby("Cluster")["Total Benar"].mean().sort_values(ascending=False)

segment_map = {
    cluster_mean.index[0]: "Tinggi",
    cluster_mean.index[1]: "Sedang",
    cluster_mean.index[2]: "Rendah"
}

df["Kategori"] = df["Cluster"].map(segment_map)

st.dataframe(df[[nama_kolom, "Total Benar", "Kategori"]])

st.success("📌 Dashboard Analisis Selesai – Siap untuk Evaluasi Pembelajaran")
