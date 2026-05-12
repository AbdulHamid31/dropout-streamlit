import streamlit as st
import google.generativeai as genai
import re

# --- 1. KONFIGURASI KEAMANAN ---
# Mengambil API Key dari Streamlit Secrets (Aman untuk GitHub)
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=API_KEY)
except:
    st.error("⚠️ API Key tidak ditemukan! Masukkan GEMINI_API_KEY di Secrets Streamlit Cloud.")
    st.stop()

# --- 2. SYSTEM INSTRUCTION (LOGIKA GURU) ---
SYSTEM_PROMPT = """
Anda adalah "Kak Guru AI", tutor matematika interaktif jenjang SD-SMP.
TUJUAN: Membimbing siswa memahami LOGIKA, bukan memberi JAWABAN AKHIR.

ATURAN KETAT:
1. JANGAN PERNAH memberikan jawaban angka akhir (misal: "Hasilnya adalah 154").
2. Jika siswa bertanya soal, jawab dengan:
   - Identifikasi Rumus (Gunakan LaTeX $$...$$).
   - Tanya variabel yang diketahui (r, a, t, p, l, s).
   - Minta siswa memasukkan angka ke rumus.
   - Bimbing hitungan langkah demi langkah.
3. Gunakan bahasa yang ceria, santai, dan memotivasi.
4. Jika siswa salah hitung, berikan semangat dan petunjuk (clue).
"""

# --- 3. UI & AESTHETIC CONFIG ---
st.set_page_config(page_title="Kak Guru AI: Math Tutor", page_icon="📐", layout="wide")

# Custom CSS agar tampilan menarik untuk anak sekolah
st.markdown("""
    <style>
    .stApp { background-color: #F8FAFC; }
    .stChatMessage { border-radius: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.05); }
    .stButton>button { width: 100%; border-radius: 20px; background-color: #4F46E5; color: white; }
    .reportview-container .main .block-container { padding-top: 2rem; }
    </style>
    """, unsafe_allow_html=True)

# --- 4. SIDEBAR (PROGRESS & TOOLS) ---
with st.sidebar:
    st.title("🚀 Panel Belajar")
    st.markdown("Progres pengerjaan soal saat ini:")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Progress bar berdasarkan jumlah chat
    prog = min(len(st.session_state.messages) * 12, 100)
    st.progress(prog)
    st.write(f"Level Pemahaman: {prog}%")
    
    st.divider()
    with st.expander("💡 Tips Belajar"):
        st.write("1. Baca soal dengan teliti.")
        st.write("2. Sebutkan angka yang diketahui ke Kak Guru.")
        st.write("3. Siapkan kertas coret-coretan ya!")
    
    if st.button("🔄 Mulai Soal Baru"):
        st.session_state.messages = []
        st.session_state.chat = st.session_state.model.start_chat(history=[])
        st.rerun()

# --- 5. LOGIKA UTAMA CHAT ---
st.title("🧑‍🏫 Kak Guru AI: Inovasi Matematika")
st.info("AI ini didesain untuk membantumu berpikir, bukan sekadar menyalin jawaban. Yuk, mulai!")

# Inisialisasi Model
if "model" not in st.session_state:
    st.session_state.model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        system_instruction=SYSTEM_PROMPT
    )
    st.session_state.chat = st.session_state.model.start_chat(history=[])

# Menampilkan Riwayat Chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input dari Siswa
if prompt := st.chat_input("Contoh: Kak, gimana cara cari luas segitiga alas 10 tinggi 8?"):
    # Simpan chat user
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Respon AI
    with st.chat_message("assistant"):
        try:
            response = st.session_state.chat.send_message(prompt)
            st.markdown(response.text)
            st.session_state.messages.append({"role": "assistant", "content": response.text})
            
            # FITUR VISUAL: Jika ada rumus di dalam jawaban AI, tampilkan di kotak khusus
            latex_found = re.findall(r'\$\$(.*?)\$\$', response.text)
            if latex_found:
                with st.container():
                    st.success("📌 **Gunakan Rumus Ini:**")
                    for formula in latex_found:
                        st.latex(formula)
            
            # Efek Selebrasi jika AI mendeteksi keberhasilan (kata kunci tertentu)
            if any(word in response.text.lower() for word in ["hebat", "pintar", "tepat sekali", "selamat"]):
                st.balloons()
                
        except Exception as e:
            st.error(f"Aduh, ada gangguan teknis: {e}")
