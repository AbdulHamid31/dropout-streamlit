import streamlit as st
import pandas as pd
import shap
import pickle
import matplotlib.pyplot as plt

# =========================
# === LOAD MODEL & DATA ===
# =========================
@st.cache_resource
def load_model():
    return pickle.load(open("model_xgb.pkl", "rb"))

@st.cache_data
def load_data():
    df = pd.read_csv("dataset_mahasiswa_812 (3).csv")
    df['status_akademik_terakhir'] = df['status_akademik_terakhir'].map({
        'IPK < 2.5': 0,
        'IPK 2.5 - 3.0': 1,
        'IPK > 3.0': 2
    })
    return df

model = load_model()
data = load_data()

# =======================
# === LOGIN FUNCTION ====
# =======================
def login():
    st.sidebar.title("🔐 Login Mahasiswa")
    username = st.sidebar.text_input("Nama Mahasiswa (username)")
    password = st.sidebar.text_input("ID Mahasiswa (password)", type="password")

    if st.sidebar.button("Login"):
        user_row = data[(data["Nama"] == username) & (data["ID Mahasiswa"].astype(str) == password)]
        if not user_row.empty:
            st.session_state["logged_in"] = True
            st.session_state["username"] = username
            st.session_state["user_data"] = user_row
            st.experimental_rerun()
        else:
            st.sidebar.error("Nama atau ID salah.")

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if not st.session_state["logged_in"]:
    login()
    st.stop()

# =======================
# ==== SIDEBAR MENU =====
# =======================
st.sidebar.success(f"Login sebagai {st.session_state['username']}")
menu = st.sidebar.radio("📚 Menu", ["Dashboard", "Prediksi Dropout", "Visualisasi", "Logout"])

# =======================
# ==== DASHBOARD =========
# =======================
if menu == "Dashboard":
    st.title("📊 Dashboard Mahasiswa")
    mahasiswa = st.session_state["user_data"].iloc[0]

    st.write("### Informasi Mahasiswa")
    st.write("**Nama:**", mahasiswa["Nama"])
    st.write("**ID Mahasiswa:**", mahasiswa["ID Mahasiswa"])
    st.write("**Status Akademik Terakhir:**", mahasiswa["status_akademik_terakhir"])

    st.markdown("---")
    st.write("📌 Gunakan menu di samping untuk melakukan prediksi atau melihat visualisasi.")

# =======================
# ==== PREDIKSI ==========
# =======================
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
        st.error("⚠️ Mahasiswa ini berisiko tinggi dropout. Perlu perhatian khusus.")
    else:
        st.warning("⚠️ Mahasiswa ini memiliki kemungkinan dropout sedang.")

# =======================
# ==== VISUALISASI ======
# =======================
elif menu == "Visualisasi":
    st.title("📈 Visualisasi Data Dropout")

    # Distribusi dropout keseluruhan
    dropout_counts = data['dropout'].value_counts()
    labels = ['Tidak Dropout', 'Dropout']
    colors = ['#28a745', '#dc3545']

    fig1, ax1 = plt.subplots()
    ax1.pie(dropout_counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    ax1.axis('equal')
    st.subheader("Distribusi Dropout Mahasiswa")
    st.pyplot(fig1)

    # Visualisasi SHAP Waterfall
    st.subheader("Penjelasan Prediksi dengan SHAP")
    mahasiswa = st.session_state["user_data"]
    X = mahasiswa.drop(columns=["ID Mahasiswa", "Nama", "dropout"])

    explainer = shap.Explainer(model)
    shap_values = explainer(X)

    shap.plots.waterfall(shap_values[0])
    st.pyplot(plt.gcf())

# =======================
# ==== LOGOUT ===========
# =======================
elif menu == "Logout":
    st.session_state.clear()
    st.experimental_rerun()
