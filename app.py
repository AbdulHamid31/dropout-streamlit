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

# Sidebar - Pilih Mahasiswa
st.sidebar.title("Prediksi Dropout Mahasiswa")
selected = st.sidebar.selectbox("Pilih Mahasiswa", data["Nama"])
mahasiswa = data[data["Nama"] == selected]

# Tampilkan informasi
st.title("Hasil Prediksi Dropout")

# Hitung statistik dropout
jumlah_mahasiswa = len(data)
jumlah_dropout = data['dropout'].sum()
persentase_dropout = (jumlah_dropout / jumlah_mahasiswa) * 100

# Tampilkan di halaman utama
st.markdown(f"### 📊 Total Mahasiswa: {jumlah_mahasiswa}")
st.markdown(f"### ❌ Jumlah Dropout: {jumlah_dropout} ({persentase_dropout:.1f}%)")

st.write("**Nama Mahasiswa:**", selected)

# Persiapkan data untuk prediksi
X = mahasiswa.drop(columns=["ID Mahasiswa", "Nama", "dropout"])

# Prediksi
prediksi = model.predict(X)[0]
proba = model.predict_proba(X)[0][1]

st.write("**Status Prediksi:**", "Dropout" if prediksi == 1 else "Tidak Dropout")
st.write("**Probabilitas Risiko Dropout:**", f"{proba:.2%}")

# Interpretasi otomatis berdasarkan probabilitas
if proba < 0.2:
    st.success("✅ Mahasiswa ini sangat kecil kemungkinannya untuk dropout.")
elif proba > 0.7:
    st.error("⚠️ Mahasiswa ini berisiko tinggi dropout. Perlu perhatian khusus.")
else:
    st.warning("⚠️ Mahasiswa ini memiliki kemungkinan dropout sedang.")


# Interpretasi dengan SHAP
st.subheader("Penjelasan Prediksi (Visualisasi SHAP)")
explainer = shap.Explainer(model)
shap_values = explainer(X)

shap.plots.waterfall(shap_values[0])
st.pyplot(plt.gcf())

import matplotlib.pyplot as plt

# Hitung jumlah dropout dan tidak
dropout_counts = data['dropout'].value_counts()
labels = ['Tidak Dropout', 'Dropout']
colors = ['#28a745', '#dc3545']

fig1, ax1 = plt.subplots()
ax1.pie(dropout_counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

st.subheader("Distribusi Dropout Mahasiswa")
st.pyplot(fig1)

from graphviz import Digraph

dot = Digraph(comment='Diagram Alur Sistem Streamlit')

# Gaya umum
dot.attr(rankdir='TB', size='8,10')
dot.attr('node', shape='box', style='filled', fillcolor='#f0f8ff', fontname='Arial')

# Node alur
dot.node('A', 'Mulai')
dot.node('B', 'User membuka aplikasi Streamlit')
dot.node('C', 'User memilih mahasiswa dari sidebar')
dot.node('D', 'Ambil data fitur mahasiswa dari CSV')
dot.node('E', 'Preprocessing data (encode IPK)')
dot.node('F', 'Model XGBoost memproses data')
dot.node('G', 'Hitung probabilitas dropout')
dot.node('H', 'Tampilkan hasil prediksi')
dot.node('I1', 'Status (Dropout / Tidak)')
dot.node('I2', 'Probabilitas Dropout (%)')
dot.node('I3', 'Visualisasi SHAP / Tabel')
dot.node('Z', 'Selesai')

# Panah antar langkah
dot.edges(['AB', 'BC', 'CD', 'DE', 'EF', 'FG', 'GH'])
dot.edge('H', 'I1')
dot.edge('H', 'I2')
dot.edge('H', 'I3')
dot.edge('I1', 'Z')
dot.edge('I2', 'Z')
dot.edge('I3', 'Z')

# Tampilkan atau simpan
dot.render('diagram_alur_streamlit', format='png', cleanup=True)
        

