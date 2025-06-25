import streamlit as st
import pandas as pd
import shap
import pickle
import matplotlib.pyplot as plt

# ============================
# === LOAD MODEL & DATASET ===
# ============================

@st.cache_resource
def load_model():
    return pickle.load(open("model_xgb.pkl", "rb"))

@st.cache_data
def load_data():
    df = pd.read_csv("dataset_mahasiswa_812.csv")  # Pastikan nama file sesuai
    df['status_akademik_terakhir'] = df['status_akademik_terakhir'].map({
        'IPK < 2.5': 0,
        'IPK 2.5 - 3.0': 1,
        'IPK > 3.0': 2
    })
    return df

model = load_model()
data = load_data()

# ===================
# === LOGIN SYSTEM ==
# ===================

def login():
    st.sidebar.title("🔐 Login Mahasiswa")
    nama_list = data["Nama"].unique()
    selected_nama = st.sidebar.selectbox("Pilih Nama Mahasiswa", nama_list)
    nim = st.sidebar.text_input("Masukkan NIM Mahasiswa", type="password")

    if st.sidebar.button("Login"):
        user_row = data[
            (data["Nama"] == selected_nama) &
            (data["ID Mahasiswa"].astype(str) == nim)
        ]
        if not user_row.empty:
            st.session_state["logged_in"] = True
            st.session_state["username"] = selected_nama
            st.session_state["user_data"] = user_row
            st.rerun()
        else:
            st.sidebar.error("❌ NIM tidak cocok dengan nama yang dipilih.")

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if not st.session_state["logged_in"]:
    login()
    st.stop()

# =====================
# === SIDEBAR MENU ====
# =====================

st.sidebar.success(f"Login sebagai {st.session_state['username']}")
menu = st.sidebar.radio("📚 Menu Navigasi", ["Dashboard", "Prediksi Dropout & Visualisasi", "Logout"])

# ====================
# ==== DASHBOARD =====
# ====================

if menu == "Dashboard":
    mahasiswa = st.session_state["user_data"].iloc[0]

    nama = mahasiswa["Nama"]
    total_login = int(mahasiswa.get("total_login", 0))
    materi_selesai = int(mahasiswa.get("materi_selesai", 0))
    kemajuan = int(mahasiswa.get("kemajuan_kelas", 0))

    st.markdown(f"<h2>🎓 LMS Mahasiswa - {nama}</h2>", unsafe_allow_html=True)
    st.markdown(f"<h4>👋 Selamat datang, {nama}!</h4>", unsafe_allow_html=True)
    st.markdown("---")

    # 3 KARTU DASHBOARD (Tanpa IPK/Sales)
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Status Login**")
        st.metric(label="", value="Aktif")

    with col2:
        st.markdown("**Total Login**")
        st.metric(label="", value=f"{total_login}x")

    with col3:
        st.markdown("**Materi Selesai**")
        st.metric(label="", value=f"{materi_selesai}")

    # Hitung persentase kemajuan berdasarkan total_login
    progress = int(min(total_login, 50) / 50 * 100)

    st.markdown("### 📈 Kemajuan Kelas (berdasarkan login)")
    st.progress(progress)

# ================================================
# === GABUNG: PREDIKSI DROPOUT & VISUALISASI =====
# ================================================

elif menu == "Prediksi Dropout & Visualisasi":
    st.title("🧠 Prediksi & Visualisasi Risiko Dropout")

    # ===== Prediksi =====
    st.subheader("🤖 Hasil Prediksi")
    mahasiswa = st.session_state["user_data"]
    nama = mahasiswa["Nama"].values[0]
    st.write("**Nama Mahasiswa:**", nama)

    X = mahasiswa.drop(columns=["ID Mahasiswa", "Nama", "dropout"])
    prediksi = model.predict(X)[0]
    proba = model.predict_proba(X)[0][1]

    st.write("**Status Prediksi:**", "Dropout" if prediksi == 1 else "Tidak Dropout")
    st.write("**Probabilitas Risiko Dropout:**", f"{proba:.2%}")

    if proba < 0.2:
        st.success("✅ Mahasiswa ini sangat kecil kemungkinannya untuk dropout.")
    elif proba > 0.7:
        st.error("⚠️ Mahasiswa ini berisiko tinggi dropout.")
    else:
        st.warning("⚠️ Mahasiswa ini memiliki kemungkinan dropout sedang.")

    # ===== SHAP =====
    st.markdown("---")
    st.subheader("📊 Visualisasi SHAP (Penjelasan Prediksi)")

    explainer = shap.Explainer(model)
    shap_values = explainer(X)
    shap.plots.waterfall(shap_values[0])
    st.pyplot(plt.gcf())
    
    # Tambahkan keterangan interpretasi SHAP
        st.markdown("### ℹ️ Penjelasan Visualisasi SHAP")
        st.markdown("""
        Visualisasi di atas menunjukkan bagaimana masing-masing fitur mempengaruhi prediksi dropout mahasiswa:
        
        - 🔵 Warna biru menunjukkan fitur yang **mengurangi risiko dropout**.
        - 🔴 Warna merah menunjukkan fitur yang **meningkatkan risiko dropout**.
        - Panjang batang menunjukkan seberapa besar pengaruh fitur tersebut terhadap prediksi akhir.
        
        Contoh interpretasi:
        - Fitur `durasi_total_akses = 58.6` memiliki nilai negatif besar (-1.35), artinya **menurunkan risiko dropout secara signifikan**.
        - Sebaliknya, `skor_kuis_rata2 = 40.45` sedikit **meningkatkan risiko dropout**.
        
        Nilai prediksi akhir (`f(x)`) digerakkan dari rata-rata prediksi (`E[f(x)]`) oleh kontribusi setiap fitur.
        """)


# ====================
# ===== LOGOUT =======
# ====================

elif menu == "Logout":
    st.session_state.clear()
    st.rerun()
