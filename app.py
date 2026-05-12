import streamlit as st
import google.generativeai as genai
import re

# --- 1. KONFIGURASI API ---
def init_api():
    try:
        if "GEMINI_API_KEY" in st.secrets:
            api_key = st.secrets["GEMINI_API_KEY"].strip()
            genai.configure(api_key=api_key)
            return True
        else:
            st.warning("⚠️ Konfigurasi API belum lengkap di Secrets.")
            return False
    except Exception as e:
        st.error(f"Masalah koneksi: {e}")
        return False

# --- 2. SISTEM GURU (PROMPT) ---
SYSTEM_PROMPT = """
Anda adalah "Kak Guru AI", tutor matematika SD-SMP yang ramah.
- Jika siswa menyapa (Halo, Hai, dsb), balas dengan hangat dan ajak mereka belajar.
- Jika siswa memberi soal, gunakan metode Sokratik: jangan beri jawaban, tapi beri pertanyaan pemandu.
- Gunakan LaTeX $$...$$ hanya jika ada rumus.
- Selalu gunakan bahasa Indonesia yang sopan dan ceria.
"""

# --- 3. UI CONFIG ---
st.set_page_config(page_title="Kak Guru AI", page_icon="📐")

# --- 4. LOGIKA CHAT ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Inisialisasi AI (Hanya jika belum ada)
if init_api():
    if "chat" not in st.session_state:
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash-latest",
            system_instruction=SYSTEM_PROMPT
        )
        st.session_state.chat = model.start_chat(history=[])

# --- TAMPILAN ---
st.title("🧑‍🏫 Kak Guru AI")
st.write("Halo! Kak Guru siap bantu kamu belajar matematika. Mau bahas soal apa hari ini?")

# Tampilkan history chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input Siswa
if prompt := st.chat_input("Tanya soal atau sapa Kak Guru..."):
    # Simpan chat user
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Respon Assistant
    with st.chat_message("assistant"):
        try:
            # Kirim ke AI
            response = st.session_state.chat.send_message(prompt)
            output = response.text
            
            st.markdown(output)
            st.session_state.messages.append({"role": "assistant", "content": output})
            
            # Deteksi Rumus secara aman
            formulas = re.findall(r'\$\$(.*?)\$\$', output)
            if formulas:
                for f in formulas:
                    st.latex(f)
            
            # Balon jika jawaban benar/pujian
            if any(x in output.lower() for x in ["hebat", "benar", "pintar", "bagus"]):
                st.balloons()
                
        except Exception as e:
            # JIKA TERJADI ERROR, AI TIDAK AKAN CRASH
            st.error("Aduh, Kak Guru sedang berpikir terlalu keras. Bisa ulangi pertanyaannya?")
            print(f"Error Detail: {e}") # Log untuk kamu
