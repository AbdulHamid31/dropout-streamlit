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
    df = pd.read_csv("dataset_mahasiswa.csv")  # Ganti sesuai nama file final di deploy
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
    password = st.sidebar.text_input("Masukkan NIM Mahasiswa", type="password")

    if st.sidebar.button("Login"):
        user_row = data[
            (data["Nama"] == selected_nama) &
            (data["ID Mahasiswa"].astype(str) == password)
        ]
        if not user_row.empty:
            st.session_state["logged_in"] = True
            st.session_state["username"] = selected_nama
            st.session_state["user_data"] = user_row
            st.experimental_rerun()
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
menu = st.sidebar.radio("📚 Menu Navigasi", ["Dashboard", "Prediksi Dropout", "Visualisasi", "Logout"])

# ====================
# ==== DASHBOARD =====
# ====================

if menu == "Dashboard":
    st.title("🎓 Dashboard LMS Mahasiswa")
    st.markdown("Selamat datang di sistem prediksi risiko dropout mahasiswa.")
    mahasiswa = st.session_state["user_data"].iloc[0]

    st.markdown("### 👤 Informasi Mahasiswa")
    st.write("**Nama:**", mahasiswa["Nama"])
    st.write("**ID Mahasiswa:**", mahasiswa["ID Mahasiswa"])
    st.write("**Status Akademik Terakhir:**", mahasiswa["status_akademik_terakhir"])

    st.markdown("---")
    st.dataframe(data[["Nama", "status_akademik_terakhir", "dropout"]].sample(5), use_container_width=True)

# ====================
# ==== PREDIKSI ======
# ====================

elif menu == "Prediksi Dropout":
    st.title("🤖 Prediksi Risiko Dropout")
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

# ============================
# ==== VISUALISASI ==========
# ============================

elif menu == "Visualisasi":
    st.title("📈 Visualisasi Data Mahasiswa")

    st.subheader("Distribusi Dropout Mahasiswa")
    dropout_counts = data['dropout'].value_counts()
    labels = ['Tidak Dropout', 'Dropout']
    colors = ['#28a745', '#dc3545']

    fig1, ax1 = plt.subplots()
    ax1.pie(dropout_counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    ax1.axis('equal')
    st.pyplot(fig1)

    st.subheader("Visualisasi SHAP (Penjelasan Prediksi)")
    mahasiswa = st.session_state["user_data"]
    X = mahasiswa.drop(columns=["ID Mahasiswa", "Nama", "dropout"])

    explainer = shap.Explainer(model)
    shap_values = explainer(X)
    shap.plots.waterfall(shap_values[0])
    st.pyplot(plt.gcf())

# ====================
# ===== LOGOUT =======
# ====================

elif menu == "Logout":
    st.session_state.clear()
    st.experimental_rerun()
