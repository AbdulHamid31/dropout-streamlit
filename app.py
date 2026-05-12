import streamlit as st
import google.generativeai as genai
import re

# --- 1. KONFIGURASI KEAMANAN & API ---
def init_api():
    if "GEMINI_API_KEY" in st.secrets:
        # .strip() menghapus spasi atau karakter aneh yang tidak sengaja tersalin
        api_key = st.secrets["GEMINI_API_KEY"].strip()
        genai.configure(api_key=api_key)
        return True
    else:
        st.error("❌ GEMINI_API_KEY tidak ditemukan di Secrets Streamlit Cloud!")
        st.info("💡 Pastikan Anda sudah memasukkan API Key di menu Settings > Secrets pada Dashboard Streamlit.")
        return False

# --- 2. SYSTEM INSTRUCTION (LOGIKA GURU SOKRATIK) ---
SYSTEM_PROMPT = """
Anda adalah "Kak Guru AI", tutor matematika interaktif jenjang SD-SMP.
TUJUAN: Membimbing siswa memahami LOGIKA, bukan memberi JAWABAN AKHIR.

ATURAN KETAT:
1. JANGAN PERNAH memberikan jawaban angka akhir (misal: "Hasilnya adalah 154").
2. Jika siswa bertanya soal, jawab dengan:
   - Identifikasi Rumus menggunakan format LaTeX $$...$$.
   - Tanya variabel yang diketahui (seperti jari-jari, alas, tinggi, dll).
   - Minta siswa memasukkan angka ke dalam rumus.
   - Bimbing hitungan langkah demi langkah dengan pertanyaan pemantik.
3. Gunakan bahasa Indonesia yang ceria, santai, dan memotivasi (Gunakan "Kak Guru" dan "Kamu").
4. Jika siswa salah hitung, berikan semangat dan petunjuk (clue), jangan langsung koreksi angkanya.
"""

# --- 3. UI & AESTHETIC CONFIG ---
st.set_page_config(page_title="Kak Guru AI: Math Tutor", page_icon="📐", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #F8FAFC; }
    .stChatMessage { border-radius: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.05); margin-bottom: 15px; }
    .stButton>button { width: 100%; border-radius: 20px; background-color: #4F46E5; color: white; border: none; padding: 10px; }
    .stButton>button:hover { background-color: #4338CA; border: none; }
    </style>
    """, unsafe_allow_html=True)

# --- 4. SIDEBAR (PROGRESS & TOOLS) ---
with st.sidebar:
    st.title("🚀 Panel Belajar")
    st.markdown("Progres pengerjaan soal saat ini:")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    prog = min(len(st.session_state.messages) * 10, 100)
    st.progress(prog)
    st.write(f"Level Pemahaman: {prog}%")
    
    st.divider()
    with st.expander("💡 Tips Belajar"):
        st.write("1. Sebutkan angka yang kamu tahu dari soal.")
        st.write("2. Coba hitung pelan-pelan di kertas ya!")
        st.write("3. Jangan malu tanya kalau bingung rumus.")
    
    if st.button("🔄 Mulai Soal Baru"):
        st.session_state.messages = []
        if "chat" in st.session_state:
            del st.session_state.chat
        st.rerun()

# --- 5. LOGIKA UTAMA CHAT ---
st.title("🧑‍🏫 Kak Guru AI: Inovasi Matematika")
st.caption("Khusus jenjang SD - SMP | Metode Sokratik")

if init_api():
    # Perbaikan Nama Model agar tidak 404
    if "model" not in st.session_state:
        st.session_state.model = genai.GenerativeModel(
            model_name="gemini-1.5-flash-latest", # Nama model yang lebih kompatibel
            system_instruction=SYSTEM_PROMPT
        )
    
    if "chat" not in st.session_state:
        st.session_state.chat = st.session_state.model.start_chat(history=[])

    # Tampilkan History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input Siswa
    if prompt := st.chat_input("Tulis soalmu di sini..."):
        st
