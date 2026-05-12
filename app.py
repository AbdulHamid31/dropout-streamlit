import streamlit as st
import google.generativeai as genai
import re

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Kak Guru AI", page_icon="📐", layout="centered")

# CSS untuk tampilan lebih bersih
st.markdown("""
    <style>
    .stApp { background-color: #F8FAFC; }
    .stChatMessage { border-radius: 15px; margin-bottom: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. FUNGSI KONFIGURASI API ---
def setup_gemini():
    if "GEMINI_API_KEY" in st.secrets:
        key = st.secrets["GEMINI_API_KEY"].strip()
        genai.configure(api_key=key)
        return True
    return False

# --- 3. SYSTEM INSTRUCTION ---
SYSTEM_PROMPT = (
    "Kamu adalah 'Kak Guru AI', tutor matematika SD-SMP yang ramah. "
    "Tugasmu: Bantu siswa memahami konsep, JANGAN beri jawaban akhir. "
    "Gunakan metode Sokratik (tanya balik untuk memancing logika). "
    "Gunakan LaTeX $$ untuk rumus matematika agar rapi."
)

# --- 4. LOGIKA SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 5. SIDEBAR ---
with st.sidebar:
    st.title("🎒 Menu Belajar")
    if st.button("🗑️ Reset Percakapan"):
        st.session_state.messages = []
        if "chat_session" in st.session_state:
            del st.session_state.chat_session
        st.rerun()

# --- 6. UTAMA ---
st.title("🧑‍🏫 Kak Guru AI")
st.caption("Tutor Matematika Interaktif")

if not setup_gemini():
    st.error("❌ API Key belum terpasang di Secrets Streamlit!")
    st.stop()

# Inisialisasi Chat Session dengan Nama Model yang Lebih Stabil
if "chat_session" not in st.session_state:
    # Kita coba gunakan 'gemini-1.5-flash', jika gagal sistem akan lari ke 'except'
    try:
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash", 
            system_instruction=SYSTEM_PROMPT
        )
        st.session_state.chat_session = model.start_chat(history=[])
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        st.stop()

# Tampilkan riwayat chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input dari User
if prompt := st.chat_input("Halo Kak Guru, mau tanya soal..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            # Mengirim pesan ke AI
            response = st.session_state.chat_session.send_message(prompt)
            full_text = response.text
            
            st.markdown(full_text)
            st.session_state.messages.append({"role": "assistant", "content": full_text})
            
            # Deteksi rumus LaTeX untuk ditampilkan lebih cantik
            if "$$" in full_text:
                formulas = re.findall(r'\$\$(.*?)\$\$', full_text)
                for f in formulas:
                    st.latex(f)

        except Exception as e:
            error_str = str(e)
            if "404" in error_str:
                st.error("⚠️ Model sedang tidak tersedia. Coba klik 'Reset Percakapan' di sidebar.")
            elif "API key not valid" in error_str:
                st.error("❌ API Key salah. Tolong buat Key baru di Google AI Studio dan update di Secrets.")
            else:
                st.error(f"❌ Gangguan teknis: {error_str}")
