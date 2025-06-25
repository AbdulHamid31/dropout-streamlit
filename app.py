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
   # Simpan label teks IPK untuk dashboard
    df["ipk_label"] = df["status_akademik_terakhir"]

    # Lalu mapping ke angka untuk model prediksi
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
    status_login = "Aktif"
    total_login = mahasiswa.get("total_login", 0)
    materi_selesai = mahasiswa.get("materi_selesai", 0)
    status_ipk = mahasiswa.get("status_akademik_terakhir", "IPK < 2.5")

    # Konversi status IPK ke nilai numerik
    ipk_lookup = {
        "IPK < 2.5": 2.25,
        "IPK 2.5 - 3.0": 2.75,
        "IPK > 3.0": 3.25
    }
    ipk = ipk_lookup.get(status_ipk, 0.0)
    progress = mahasiswa.get("kemajuan_kelas", 0)

    st.markdown(f"<h2>🎓 LMS Mahasiswa - {nama}</h2>", unsafe_allow_html=True)
    st.markdown(f"<h4>👋 Selamat datang, {nama}!</h4>", unsafe_allow_html=True)
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("**Status Login**")
        st.metric(label="", value=status_login)

    with col2:
        st.markdown("**Total Login**")
        st.metric(label="", value=f"{total_login}x", delta="+5% sejak minggu lalu")

    with col3:
        st.markdown("**Materi Selesai**")
        st.metric(label="", value=f"{materi_selesai}", delta="+2 modul")

    with col4:
        st.markdown("**IPK Terakhir**")
        st.metric(label="", value=f"{ipk:.2f}", delta="+0.10")

    st.markdown("### Kemajuan Kelas")
    st.progress(int(progress))


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

    # ===== Pie Chart Distribusi =====
    st.markdown("---")
    st.subheader("📈 Distribusi Dropout Mahasiswa")

    dropout_counts = data['dropout'].value_counts()
    labels = ['Tidak Dropout', 'Dropout']
    colors = ['#28a745', '#dc3545']

    fig1, ax1 = plt.subplots()
    ax1.pie(dropout_counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    ax1.axis('equal')
    st.pyplot(fig1)

# ====================
# ===== LOGOUT =======
# ====================

elif menu == "Logout":
    st.session_state.clear()
    st.rerun()
