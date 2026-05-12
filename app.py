import streamlit as st
import google.generativeai as genai
import re

# --- 1. KONFIGURASI KEAMANAN & API ---
# Fungsi untuk inisialisasi API dengan pembersihan karakter spasi
def init_api():
    if "GEMINI_API_KEY" in st.secrets:
        # .strip() sangat penting untuk menghapus spasi/newline tak sengaja dari Secrets
        api_key = st.secrets["GEMINI_API_KEY"].strip()
        genai.configure(api_key=api_key)
        return True
    else:
        st.error("❌ GEMINI_API_KEY tidak ditemukan di Secrets Streamlit Cloud!")
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

# CSS untuk mempercantik Chat dan Sidebar
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
    
    # Inisialisasi history jika belum ada
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Progress bar dinamis
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

# Jalankan Inisialisasi API
if init_api():
    # Inisialisasi Model & Chat Session
    if "model" not in st.session_state:
        st.session_state.model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            system_instruction=SYSTEM_PROMPT
        )
    
    if "chat" not in st.session_state:
        st.session_state.chat = st.session_state.model.start_chat(history=[])

    # Menampilkan Riwayat Chat
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input dari Siswa
    if prompt := st.chat_input("Tulis soalmu di sini..."):
        # Tampilkan chat user
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Respon AI
        with st.chat_message("assistant"):
            try:
                # Mengirim pesan ke model
                response = st.session_state.chat.send_message(prompt)
                full_response = response.text
                
                st.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
                # FITUR VISUAL RUMUS: Ekstrak LaTeX $$...$$ untuk ditampilkan besar
                latex_found = re.findall(r'\$\$(.*?)\$\$', full_response)
                if latex_found:
                    st.info("📌 **Rumus yang kita pakai:**")
                    for formula in latex_found:
                        st.latex(formula)
                
                # Efek Selebrasi (Gamification)
                pujian = ["hebat", "pintar", "tepat", "selamat", "benar", "100", "jos"]
                if any(word in full_response.lower() for word in pujian):
                    st.balloons()
                    
            except Exception as e:
                # Menangkap error 400 atau error API lainnya
                st.error(f"❌ Terjadi gangguan teknis: {e}")
                st.warning("Pastikan API Key di Secrets sudah benar dan tidak ada spasi.")
