import streamlit as st
import google.generativeai as genai
import re

# --- 1. KONFIGURASI KEAMANAN (STREAMLIT SECRETS) ---
# Di Streamlit Cloud, masukkan GEMINI_API_KEY di menu Settings > Secrets
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
except:
    API_KEY = "KOSONG" # Untuk running lokal jika belum ada secrets

genai.configure(api_key=API_KEY)

# --- 2. INSTRUKSI SISTEM (LOGIKA SOKRATIK) ---
SYSTEM_PROMPT = """
Anda adalah "Kak Guru AI", tutor matematika interaktif jenjang SD-SMP.
PRINSIP UTAMA:
1. JANGAN PERNAH memberikan jawaban akhir.
2. JANGAN PERNAH menghitungkan untuk siswa.
3. Gunakan Metode Sokratik: Berikan pertanyaan pemantik agar siswa berpikir.
4. Gunakan LaTeX untuk rumus (Contoh: $$L = \\pi r^2$$).
5. Bahasa: Ramah, sabar, dan gunakan analogi sederhana.

ALUR:
- Identifikasi Rumus -> Tanya Variabel (r, p, l, t) -> Minta input angka ke rumus -> Bimbing hitung parsial -> Selesai.
"""

# --- 3. UI & AESTHETIC DESIGN ---
st.set_page_config(page_title="Kak Guru AI", page_icon="📐", layout="wide")

# CSS untuk mempercantik tampilan (Vibe Edukasi Modern)
st.markdown("""
    <style>
    .main { background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); }
    .stChatMessage { background-color: white; border-radius: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); padding: 15px; margin-bottom: 10px; }
    .stButton>button { border-radius: 30px; background-color: #4F46E5; color: white; }
    h1 { color: #1E293B; font-family: 'Poppins', sans-serif; }
    </style>
    """, unsafe_allow_html=True)

# --- 4. SIDEBAR & PROGRESS ---
with st.sidebar:
    st.title("🚀 Panel Belajar")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Hitung Progress sederhana
    prog_value = min(len(st.session_state.messages) * 10, 100)
    st.write(f"Progres Logika: {prog_value}%")
    st.progress(prog_value)
    
    st.divider()
    with st.expander("📝 Intip Rumus Penting"):
        st.write("**Lingkaran:** $$L = \\pi r^2$$")
        st.write("**Segitiga:** $$L = \\frac{1}{2} a t$$")
        st.write("**Kubus (Vol):** $$V = s^3$$")

    if st.button("🗑️ Hapus Sesi & Mulai Baru"):
        st.session_state.messages = []
        st.rerun()

# --- 5. LOGIKA CHAT AI ---
st.title("🧑‍🏫 Kak Guru AI: Tutor Matematika")
st.write("Halo! Masukkan soal matematika SD/SMP-mu, Kak Guru bantu pahami langkahnya ya!")

# Inisialisasi model
if "model" not in st.session_state:
    st.session_state.model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        system_instruction=SYSTEM_PROMPT
    )
    st.session_state.chat = st.session_state.model.start_chat(history=[])

# Menampilkan chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input siswa
if API_KEY == "KOSONG":
    st.warning("⚠️ API Key belum terdeteksi. Masukkan di Secrets Streamlit.")
else:
    if prompt := st.chat_input("Tulis soalmu... (Contoh: Cari luas segitiga jika alas 10 dan tinggi 5)"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                response = st.session_state.chat.send_message(prompt)
                st.markdown(response.text)
                st.session_state.messages.append({"role": "assistant", "content": response.text})
                
                # Fitur Tambahan: Deteksi Otomatis Rumus untuk ditampilkan besar
                latex_match = re.findall(r'\$\$(.*?)\$\$', response.text)
                if latex_match:
                    st.info("💡 **Rumus Terdeteksi:**")
                    for f in latex_match:
                        st.latex(f)
            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")
