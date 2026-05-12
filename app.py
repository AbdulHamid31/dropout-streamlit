import streamlit as st
import google.generativeai as genai
import re

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Kak Guru AI", page_icon="📐", layout="centered")

# --- 2. CSS CUSTOM (Aesthetic & Ramah Anak) ---
st.markdown("""
    <style>
    .stApp { background-color: #F0F2F6; }
    .stChatMessage { border-radius: 15px; border: 1px solid #E0E0E0; margin-bottom: 10px; }
    .stButton>button { border-radius: 20px; background-color: #4F46E5; color: white; border: none; }
    .stChatInputContainer { padding-bottom: 20px; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. FUNGSI INISIALISASI API ---
def setup_gemini():
    if "GEMINI_API_KEY" in st.secrets:
        key = st.secrets["GEMINI_API_KEY"].strip()
        genai.configure(api_key=key)
        return True
    return False

# --- 4. SISTEM INSTRUCTION ---
SYSTEM_PROMPT = (
    "Kamu adalah 'Kak Guru AI', tutor matematika yang asyik untuk anak SD-SMP. "
    "PRINSIP: Jangan pernah memberi jawaban akhir. Gunakan metode Sokratik. "
    "Jika siswa bertanya soal, tanya balik: 'Apa yang kita ketahui dulu?' atau 'Menurutmu rumus apa yang pas?'. "
    "Gunakan LaTeX $$ untuk rumus. Bahasa harus ceria, gunakan 'Kak Guru' dan 'Kamu'."
)

# --- 5. LOGIKA SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 6. SIDEBAR ---
with st.sidebar:
    st.title("🎒 Menu Belajar")
    st.write("Progres kamu hari ini:")
    prog = min(len(st.session_state.messages) * 10, 100)
    st.progress(prog)
    
    st.divider()
    if st.button("🗑️ Hapus Percakapan / Mulai Baru"):
        st.session_state.messages = []
        if "chat_session" in st.session_state:
            del st.session_state.chat_session
        st.rerun()

# --- 7. JALANKAN APLIKASI ---
st.title("🧑‍🏫 Kak Guru AI")
st.markdown("---")

if not setup_gemini():
    st.error("❌ API Key tidak ditemukan! Masukkan di Secrets Streamlit Cloud dengan nama: GEMINI_API_KEY")
    st.stop()

# Inisialisasi Model & Chat jika belum ada
if "chat_session" not in st.session_state:
    try:
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash-latest",
            system_instruction=SYSTEM_PROMPT
        )
        st.session_state.chat_session = model.start_chat(history=[])
    except Exception as e:
        st.error(f"Gagal memulai AI: {e}")
        st.stop()

# Tampilkan history chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input Chat
if prompt := st.chat_input("Tanya soal matematika ke Kak Guru..."):
    # Tampilkan input user
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Respon AI
    with st.chat_message("assistant"):
        try:
            # Kirim pesan ke API
            response = st.session_state.chat_session.send_message(prompt)
            full_text = response.text
            
            st.markdown(full_text)
            st.session_state.messages.append({"role": "assistant", "content": full_text})
            
            # Efek balon jika AI memuji
            if any(puji in full_text.lower() for puji in ["hebat", "pintar", "benar sekali", "bagus"]):
                st.balloons()
                
        except Exception as e:
            err_msg = str(e)
            if "API key not valid" in err_msg:
                st.error("❌ API Key kamu salah atau tidak valid. Silakan ganti dengan yang baru di Secrets.")
            elif "quota" in err_msg.lower():
                st.error("❌ Kuota gratis harian kamu habis. Coba lagi besok ya!")
            else:
                st.error(f"❌ Aduh, Kak Guru sedikit bingung. Error: {err_msg}")
                # Tombol bantu reset jika error berkelanjutan
                if st.button("Klik untuk Reset Sesi"):
                    st.session_state.messages = []
                    del st.session_state.chat_session
                    st.rerun()
