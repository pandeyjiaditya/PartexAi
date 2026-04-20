from dotenv import load_dotenv
import os
import io
import hashlib
import subprocess
import tempfile
import shutil

load_dotenv()

# ---------------- LLM ----------------
from langchain_groq import ChatGroq
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.5
)

# ---------------- MONGODB ----------------
from pymongo import MongoClient
import certifi

client = MongoClient(
    os.getenv("MONGODB_URL"),
    tls=True,
    tlsCAFile=certifi.where()
)

db = client["patient_info"]
patients_col = db["info"]
consult_col = db["consultations"]

# ---------------- EMBEDDINGS ----------------
from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# ---------------- RAG ----------------
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors.chain_extract import LLMChainExtractor

# ---------------- AUDIO ----------------
from groq import Groq
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ---------------- PDF ----------------
import pdfplumber

def transcribe_audio_bytes(audio_bytes, filename="chunk.wav"):
    audio_buffer = io.BytesIO(audio_bytes)
    audio_buffer.name = filename

    transcription = groq_client.audio.transcriptions.create(
        file=audio_buffer,
        model="whisper-large-v3",
        prompt=(
            "Transcribe this multilingual medical conversation accurately. "
            "Auto-detect Hindi, Marathi, and English."
        ),
    )

    text = getattr(transcription, "text", "")
    if not text and isinstance(transcription, dict):
        text = transcription.get("text", "")

    return text.strip() if text else ""


def convert_wav_to_mp3(wav_path, mp3_path):
    ffmpeg_executable = shutil.which("ffmpeg")

    if not ffmpeg_executable:
        raise FileNotFoundError("FFmpeg was not found on this machine")

    subprocess.run(
        [
            ffmpeg_executable,
            "-y",
            "-i",
            wav_path,
            "-codec:a",
            "libmp3lame",
            "-q:a",
            "4",
            mp3_path,
        ],
        check=True,
        capture_output=True,
    )


def save_recorded_audio_as_mp3(recorded_audio):
    audio_bytes = recorded_audio.getvalue() if hasattr(recorded_audio, "getvalue") else recorded_audio.read()

    if not audio_bytes:
        raise ValueError("Recorded audio is empty")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as wav_file:
        wav_file.write(audio_bytes)
        wav_path = wav_file.name

    mp3_handle = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    mp3_path = mp3_handle.name
    mp3_handle.close()

    try:
        convert_wav_to_mp3(wav_path, mp3_path)
    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)

    return mp3_path


def extract_pdf_text(pdf_file):
    """Extract medical findings from test report"""
    report_text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                report_text += page_text + "\n"
    return report_text.strip()


def summarize_conversation(messages):
    transcript = "\n".join([f"{msg['speaker']}: {msg['text']}" for msg in messages])

    summary_prompt = PromptTemplate(
        template="""
You are a medical assistant.

Summarize this doctor-patient conversation clearly and briefly.
Focus on symptoms, concerns, advice, decisions, and next steps.

Conversation:
{conversation}

Summary:
""",
        input_variables=["conversation"]
    )

    parser = StrOutputParser()
    chain = summary_prompt | llm | parser

    return chain.invoke({"conversation": transcript})


def save_text_to_patient_store(patient_id, text):
    patient_store_path = f"faiss/{patient_id}"
    os.makedirs(patient_store_path, exist_ok=True)

    index_path = os.path.join(patient_store_path, "index.faiss")

    if os.path.exists(index_path):
        try:
            vector_store = FAISS.load_local(
                patient_store_path,
                embeddings,
                allow_dangerous_deserialization=True
            )
            vector_store.add_texts([text])
        except Exception:
            vector_store = FAISS.from_texts([text], embeddings)
    else:
        vector_store = FAISS.from_texts([text], embeddings)

    vector_store.save_local(patient_store_path)


def load_latest_conversation(patient_id):
    latest_conversation = consult_col.find_one(
        {"patient_id": patient_id, "type": "conversation"},
        sort=[("date", -1)]
    )

    if not latest_conversation:
        return [], ""

    return latest_conversation.get("conversation", []), latest_conversation.get("conversation_summary", "")

# ---------------- STREAMLIT ----------------
import streamlit as st
from datetime import datetime

# SESSION
if "selected_patient" not in st.session_state:
    st.session_state.selected_patient = None

if "transcribed_text" not in st.session_state:
    st.session_state.transcribed_text = ""

if "final_input" not in st.session_state:
    st.session_state.final_input = ""

if "recorded_audio_path" not in st.session_state:
    st.session_state.recorded_audio_path = ""

if "recorded_audio_hash" not in st.session_state:
    st.session_state.recorded_audio_hash = ""

if "clinical_audio_path" not in st.session_state:
    st.session_state.clinical_audio_path = ""

if "clinical_audio_hash" not in st.session_state:
    st.session_state.clinical_audio_hash = ""

if "conversation_messages" not in st.session_state:
    st.session_state.conversation_messages = {}

if "conversation_summary" not in st.session_state:
    st.session_state.conversation_summary = {}

if "active_section" not in st.session_state:
    st.session_state.active_section = "Dashboard"

# ================= SIDEBAR =================
st.sidebar.title("👤 Patients")

st.sidebar.markdown("### Sections")
if st.sidebar.button("📄 Dashboard"):
    st.session_state.active_section = "Dashboard"
if st.sidebar.button("💬 Chat"):
    st.session_state.active_section = "Chat"

# Add Patient
name = st.sidebar.text_input("Name")
age = st.sidebar.number_input("Age", min_value=0)

if st.sidebar.button("➕ Add Patient"):
    if name:
        patients_col.insert_one({
            "name": name,
            "age": age,
            "created_at": datetime.now()
        })
        st.sidebar.success("Patient Added ✅")

# Search
search = st.sidebar.text_input("🔍 Search Patient")

query = {}
if search:
    query["name"] = {"$regex": search, "$options": "i"}

patients = list(patients_col.find(query))

st.sidebar.markdown("### 📋 Patient List")

for p in patients:
    if st.sidebar.button(p["name"], key=str(p["_id"])):
        st.session_state.selected_patient = p

# ================= INPUT =================
st.sidebar.markdown("### 🎙️ Upload MP3")
st.sidebar.caption("Upload an MP3 file and transcribe it into text.")
mp3_file = st.sidebar.file_uploader("📤 Upload MP3", type=["mp3"])

if st.sidebar.button("Transcribe MP3"):
    if mp3_file:
        try:
            st.sidebar.info("Transcribing MP3...")
            uploaded_bytes = mp3_file.read()
            text = transcribe_audio_bytes(uploaded_bytes, filename=mp3_file.name or "audio.mp3")

            if text:
                st.session_state.transcribed_text = text
                st.session_state.final_input = text
                st.sidebar.success("MP3 transcribed ✅")
                st.sidebar.text_area("MP3 Transcript", value=text, height=180)
            else:
                st.sidebar.warning("No speech detected in the MP3.")
        except Exception as exc:
            st.sidebar.error(f"MP3 transcription failed: {exc}")
    else:
        st.sidebar.warning("Upload an MP3 file first.")

st.sidebar.markdown("### 🎙️ Record from Mic")
st.sidebar.caption("Click the mic, speak, then save the recording as MP3 and convert it to text.")

recorded_audio = st.sidebar.audio_input("🎤 Record Audio")

if recorded_audio:
    try:
        recorded_bytes = recorded_audio.getvalue() if hasattr(recorded_audio, "getvalue") else recorded_audio.read()
        audio_hash = hashlib.sha256(recorded_bytes).hexdigest()

        if audio_hash != st.session_state.recorded_audio_hash:
            if st.session_state.recorded_audio_path and os.path.exists(st.session_state.recorded_audio_path):
                os.remove(st.session_state.recorded_audio_path)

            st.session_state.recorded_audio_path = save_recorded_audio_as_mp3(recorded_audio)
            st.session_state.recorded_audio_hash = audio_hash

        st.sidebar.audio(st.session_state.recorded_audio_path)
        st.sidebar.success("Recording saved as MP3 ✅")
        st.sidebar.caption(f"Saved at: {st.session_state.recorded_audio_path}")

        if st.sidebar.button("Convert Recorded MP3 to Text"):
            with open(st.session_state.recorded_audio_path, "rb") as audio_file_handle:
                text = transcribe_audio_bytes(audio_file_handle.read(), filename=os.path.basename(st.session_state.recorded_audio_path))

            if text:
                st.session_state.transcribed_text = text
                st.session_state.final_input = text
                st.sidebar.success("Recorded audio transcribed ✅")
                st.sidebar.text_area("Recorded Audio Transcript", value=text, height=180)
            else:
                st.sidebar.warning("No speech detected in the recording.")

        if st.sidebar.button("Clear Recording"):
            if st.session_state.recorded_audio_path and os.path.exists(st.session_state.recorded_audio_path):
                os.remove(st.session_state.recorded_audio_path)
            st.session_state.recorded_audio_path = ""
            st.session_state.recorded_audio_hash = ""
            st.rerun()
    except Exception as exc:
        st.sidebar.error(f"Mic recording failed: {exc}")
else:
    st.sidebar.info("No recording yet. Click the mic and speak, then stop recording.")


# ================= TEST REPORTS UPLOAD =================
st.sidebar.markdown("### 🏥 Upload Test Reports")
st.sidebar.caption("Upload your test report and automatically extract findings.")
test_report_file = st.sidebar.file_uploader("📤 Upload Report", type=["pdf"])

if st.sidebar.button("Extract Test Report"):
    if test_report_file:
        try:
            st.sidebar.info("Reading test report...")
            report_text = extract_pdf_text(test_report_file)

            if report_text:
                st.session_state.transcribed_text = report_text
                st.session_state.final_input = report_text
                st.sidebar.success("Test report extracted ✅")
                preview = report_text[:500] + "..." if len(report_text) > 500 else report_text
                st.sidebar.text_area("Report Findings", value=preview, height=180, disabled=True)
            else:
                st.sidebar.warning("No findings extracted from report.")
        except Exception as exc:
            st.sidebar.error(f"Report extraction failed: {exc}")
    else:
        st.sidebar.warning("Please upload a test report first.")

# Edit Transcribed
manual_text_sidebar = st.sidebar.text_area(
    "Edit Transcribed Text",
    value=st.session_state.transcribed_text,
    height=120
)

# Use Text
direct_text = ""

if st.sidebar.button("✅ Use This Text"):

    if direct_text.strip():
        st.session_state.final_input = direct_text
    elif manual_text_sidebar.strip():
        st.session_state.final_input = manual_text_sidebar
    else:
        st.sidebar.warning("Enter text or audio first")
        st.stop()

    st.sidebar.success("Text Ready ✅")

if st.session_state.active_section == "Chat":
    st.title("💬 Doctor - Patient Chat")

    if not st.session_state.selected_patient:
        st.info("👈 Select a patient from the sidebar first")
        st.stop()

    patient = st.session_state.selected_patient
    patient_id = str(patient["_id"])

    if patient_id not in st.session_state.conversation_messages or not st.session_state.conversation_messages[patient_id]:
        loaded_messages, loaded_summary = load_latest_conversation(patient["_id"])
        st.session_state.conversation_messages[patient_id] = loaded_messages
        st.session_state.conversation_summary[patient_id] = loaded_summary

    patient_messages = st.session_state.conversation_messages.get(patient_id, [])
    patient_summary = st.session_state.conversation_summary.get(patient_id, "")

    st.caption("Chat here. The summary updates automatically after each submitted message.")

    speaker = st.radio(
        "Speaker",
        ["Doctor", "Patient"],
        horizontal=True,
        key=f"chat_speaker_{patient_id}"
    )

    new_message = st.chat_input("Type a message and press Enter")

    if new_message:
        patient_messages.append({
            "speaker": speaker,
            "text": new_message.strip(),
            "time": datetime.now()
        })
        st.session_state.conversation_messages[patient_id] = patient_messages

        try:
            with st.spinner("Generating live summary..."):
                conversation_summary = summarize_conversation(patient_messages)

            st.session_state.conversation_summary[patient_id] = conversation_summary

            consult_col.insert_one({
                "patient_id": patient["_id"],
                "date": datetime.now(),
                "type": "conversation",
                "description": "Doctor-patient conversation",
                "conversation": patient_messages,
                "conversation_summary": conversation_summary
            })

            save_text_to_patient_store(patient_id, f"Conversation Summary: {conversation_summary}")
            st.success("Message saved and summary updated ✅")
            st.rerun()
        except Exception as exc:
            st.error(f"Conversation summary failed: {exc}")

    if patient_messages:
        st.markdown("#### Chat Log")
        for message in patient_messages:
            stamp = message["time"].strftime("%H:%M") if message.get("time") else ""
            st.write(f"**{message['speaker']}** {stamp}")
            st.write(message["text"])
            st.divider()

    if patient_summary:
        st.markdown("#### Live AI Summary")
        st.write(patient_summary)

    clear_chat_col, back_col = st.columns(2)
    with clear_chat_col:
        if st.button("Clear Chat", key=f"clear_chat_{patient_id}"):
            st.session_state.conversation_messages[patient_id] = []
            st.session_state.conversation_summary[patient_id] = ""
            st.rerun()

    with back_col:
        if st.button("Back to Dashboard", key=f"back_dashboard_{patient_id}"):
            st.session_state.active_section = "Dashboard"
            st.rerun()

    st.stop()

# ================= MAIN =================
col1, col2 = st.columns([1, 3])

with col1:
    st.write("### 👤 Selected Patient")
    if st.session_state.selected_patient:
        st.write(st.session_state.selected_patient["name"])
        st.write(f"Age: {st.session_state.selected_patient['age']}")
    else:
        st.write("No patient selected")

with col2:
    st.title("📄 Patient Dashboard")

    if st.session_state.selected_patient:
        patient = st.session_state.selected_patient
        patient_id = str(patient["_id"])
        patient_messages = st.session_state.conversation_messages.get(patient_id, [])
        patient_summary = st.session_state.conversation_summary.get(patient_id, "")

        # ================= HISTORY =================
        st.subheader("📜 History")

        consultations = list(
            consult_col.find({"patient_id": patient["_id"]}).sort("date", -1)
        )

        full_context = ""
        latest_saved_conversation_summary = ""

        for c in consultations:
            desc = c.get("description", "")
            st.write(f"🗓 {c['date'].strftime('%Y-%m-%d')}")
            if c.get("type") == "conversation" and c.get("conversation_summary"):
                st.write("💬 Conversation Summary")
                st.write(c["conversation_summary"])
                if not latest_saved_conversation_summary:
                    latest_saved_conversation_summary = c["conversation_summary"]
            else:
                st.write(f"📝 {desc}")
            st.divider()
            full_context += c.get("conversation_summary", desc) + "\n"

        if not patient_summary:
            patient_summary = latest_saved_conversation_summary

        if patient_summary:
            st.info("Latest conversation summary")
            st.write(patient_summary)

        # ================= FINAL DESCRIPTION =================
        st.subheader("✍ Clinical Summary / Physician Prescription Notes")

        st.caption("Type clinical notes directly, or record audio to append transcript.")
        clinical_audio = st.audio_input("🎤 Record Clinical Summary Audio", key="clinical_summary_audio_input")

        if clinical_audio:
            try:
                clinical_bytes = clinical_audio.getvalue() if hasattr(clinical_audio, "getvalue") else clinical_audio.read()
                clinical_hash = hashlib.sha256(clinical_bytes).hexdigest()

                if clinical_hash != st.session_state.clinical_audio_hash:
                    if st.session_state.clinical_audio_path and os.path.exists(st.session_state.clinical_audio_path):
                        os.remove(st.session_state.clinical_audio_path)

                    st.session_state.clinical_audio_path = save_recorded_audio_as_mp3(clinical_audio)
                    st.session_state.clinical_audio_hash = clinical_hash

                st.audio(st.session_state.clinical_audio_path)

                if st.button("Transcribe Clinical Audio", key="transcribe_clinical_audio"):
                    with open(st.session_state.clinical_audio_path, "rb") as audio_file_handle:
                        clinical_text = transcribe_audio_bytes(
                            audio_file_handle.read(),
                            filename=os.path.basename(st.session_state.clinical_audio_path)
                        )

                    if clinical_text:
                        existing_text = st.session_state.get("final_input", "").strip()
                        st.session_state.final_input = (
                            f"{existing_text}\n{clinical_text}".strip() if existing_text else clinical_text
                        )
                        st.success("Clinical audio transcribed and added ✅")
                    else:
                        st.warning("No speech detected in clinical audio.")

                if st.button("Clear Clinical Audio", key="clear_clinical_audio"):
                    if st.session_state.clinical_audio_path and os.path.exists(st.session_state.clinical_audio_path):
                        os.remove(st.session_state.clinical_audio_path)
                    st.session_state.clinical_audio_path = ""
                    st.session_state.clinical_audio_hash = ""
                    st.rerun()
            except Exception as exc:
                st.error(f"Clinical audio processing failed: {exc}")

        final_desc = st.session_state.get("final_input", "")

        final_desc = st.text_area(
            "Edit Clinical Summary Before Saving",
            value=final_desc,
            height=150
        )

        st.session_state.final_input = final_desc

        # ================= SAVE =================
        if st.button("ADD"):

            if not final_desc:
                st.warning("Please enter text or audio")
                st.stop()

            consult_col.insert_one({
                "patient_id": patient["_id"],
                "date": datetime.now(),
                "description": final_desc
            })

            save_text_to_patient_store(patient_id, final_desc)

            # CLEAR AFTER SAVE
            st.session_state.final_input = ""

            st.success("Saved + Embedded ✅")

        # ================= AUTO ANALYSIS =================
        st.subheader("🤖 Auto Analysis")

        if full_context:
            auto_prompt = PromptTemplate(
                template="""
You are a medical assistant.

Analyze the patient's condition based on history.

History:
{context}

Give a short summary.
""",
                input_variables=["context"]
            )

            parser = StrOutputParser()
            chain = auto_prompt | llm | parser

            auto_answer = chain.invoke({"context": full_context})

            st.write("🧠 Summary:")
            st.write(auto_answer)

        # ================= QUERY =================
        st.subheader("🔍 Ask Question")

        query = st.text_input("Ask about patient")

        if st.button("Ask AI"):

            if not os.path.exists(f"faiss/{patient_id}"):
                st.warning("No data available")
            else:
                vector_store = FAISS.load_local(
                    f"faiss/{patient_id}",
                    embeddings,
                    allow_dangerous_deserialization=True
                )

                base_retriever = vector_store.as_retriever(search_kwargs={"k": 3})
                compressor = LLMChainExtractor.from_llm(llm)
                compression_retriever = ContextualCompressionRetriever(
                    base_compressor=compressor,
                    base_retriever=base_retriever,
                )

                docs = compression_retriever.invoke(query)

                rag_context = "\n\n".join([doc.page_content for doc in docs])
                final_context = rag_context + "\n\n" + full_context

                prompt = PromptTemplate(
                    template="""
You are a medical assistant.

Use the patient data carefully.

Context:
{context}

Question:
{question}
""",
                    input_variables=["context", "question"]
                )

                parser = StrOutputParser()
                chain = prompt | llm | parser

                answer = chain.invoke({
                    "context": final_context,
                    "question": query
                })

                st.write("🤖 Answer:")
                st.write(answer)

    else:
        st.info("👈 Select a patient from sidebar")