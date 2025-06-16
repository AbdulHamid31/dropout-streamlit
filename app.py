import streamlit as st
import pandas as pd
import shap
import pickle
import matplotlib.pyplot as plt

# Load model dan data
model = pickle.load(open("model_xgb.pkl", "rb"))
data = pd.read_csv("dataset_mahasiswa_812.csv")

# Encode status_akademik_terakhir
data['status_akademik_terakhir'] = data['status_akademik_terakhir'].map({
    'IPK < 2.5': 0, 'IPK 2.5 - 3.0': 1, 'IPK > 3.0': 2
})

# Hitung statistik dropout
jumlah_mahasiswa = len(data)
jumlah_dropout = data['dropout'].sum()
jumlah_tidak = jumlah_mahasiswa - jumlah_dropout
persentase_dropout = (jumlah_dropout / jumlah_mahasiswa) * 100

# Header
st.title("🎓 Prediksi Dropout Mahasiswa")
st.markdown(f"**Total Mahasiswa:** {jumlah_mahasiswa}")
st.markdown(f"**Dropout:** {jumlah_dropout} mahasiswa ({persentase_dropout:.1f}%)")
st.markdown("---")

# Sidebar - Pilih Mahasiswa
st.sidebar.header("🎯 Prediksi Individu")
selected = st.sidebar.selectbox("Pilih Mahasiswa", data["Nama"])
mahasiswa = data[data["Nama"] == selected]

# Prediksi
X = mahasiswa.drop(columns=["ID Mahasiswa", "Nama", "dropout"])
prediksi = model.predict(X)[0]
proba = model.predict_proba(X)[0][1]

# Tampilkan hasil
st.subheader("📌 Hasil Prediksi Mahasiswa")
st.write("**Nama Mahasiswa:**", selected)
st.write("**Status Prediksi:**", "❌ Dropout" if prediksi == 1 else "✅ Tidak Dropout")
st.write("**Probabilitas Risiko Dropout:**", f"{proba:.2%}")

# Visualisasi SHAP
st.subheader("📊 Penjelasan Prediksi (SHAP)")
explainer = shap.Explainer(model)
shap_values = explainer(X)

# Plot waterfall
shap.plots.waterfall(shap_values[0])
st.pyplot(plt.gcf())


