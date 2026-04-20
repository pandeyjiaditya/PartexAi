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
st.set_page_config(page_title="Medical Console", page_icon="🩺", layout="wide")


def init_session_state():
    defaults = {
        "selected_patient": None,
        "transcribed_text": "",
        "final_input": "",
        "recorded_audio_path": "",
        "recorded_audio_hash": "",
        "clinical_audio_path": "",
        "clinical_audio_hash": "",
        "conversation_messages": {},
        "conversation_summary": {},
        "active_section": "Dashboard",
        "ui_theme": "Light",
        "auto_analysis_cache": {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def apply_theme(theme_name):
    if theme_name == "Dark":
        palette = {
            "bg": "#0f172a",
            "surface": "#111827",
            "surface_soft": "#1f2937",
            "border": "#334155",
            "text": "#f8fafc",
            "muted": "#cbd5e1",
            "accent": "#38bdf8",
            "accent_soft": "rgba(56, 189, 248, 0.18)",
        }
    else:
        palette = {
            "bg": "#f3f6fb",
            "surface": "#ffffff",
            "surface_soft": "#eef2ff",
            "border": "#d6deed",
            "text": "#0b1324",
            "muted": "#334155",
            "accent": "#0ea5e9",
            "accent_soft": "rgba(14, 165, 233, 0.14)",
        }

    st.markdown(
        f"""
<style>
    .stApp {{
        background: radial-gradient(circle at 10% 0%, {palette['surface_soft']} 0%, {palette['bg']} 45%);
        color: {palette['text']};
    }}
    h1, h2, h3, h4, h5, h6, p, label, li, span, div {{
        color: {palette['text']};
    }}
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {palette['surface']} 0%, {palette['surface_soft']} 100%);
        border-right: 1px solid {palette['border']};
    }}
    [data-testid="stSidebar"] * {{
        color: {palette['text']};
    }}
    [data-testid="stMetric"] {{
        background: {palette['surface']};
        border: 1px solid {palette['border']};
        border-radius: 12px;
        padding: 8px 10px;
    }}
    .app-panel {{
        background: {palette['surface']};
        border: 1px solid {palette['border']};
        border-radius: 16px;
        padding: 16px;
        margin-bottom: 14px;
        box-shadow: 0 10px 30px rgba(2, 6, 23, 0.08);
    }}
    .app-title {{
        font-size: 1.95rem;
        font-weight: 700;
        letter-spacing: 0.2px;
        margin-bottom: 4px;
    }}
    .app-subtitle {{
        color: {palette['muted']};
        margin-bottom: 10px;
    }}
    .pill {{
        display: inline-block;
        font-size: 0.8rem;
        font-weight: 600;
        color: {palette['accent']};
        background: {palette['accent_soft']};
        border: 1px solid {palette['border']};
        border-radius: 999px;
        padding: 5px 12px;
        margin-right: 8px;
    }}
    .stButton > button {{
        border-radius: 10px;
        border: 1px solid {palette['border']};
    }}
    .stTextArea textarea, .stTextInput input {{
        border-radius: 10px;
        border: 1px solid {palette['border']};
        background: {palette['surface']};
        color: {palette['text']};
    }}
</style>
""",
        unsafe_allow_html=True,
    )


def render_sidebar():
    st.sidebar.markdown("## Patient Console")
    st.sidebar.caption("Manage records, capture notes, and drive AI workflows.")

    is_dark_mode = st.sidebar.toggle(
        "Dark mode",
        value=(st.session_state.ui_theme == "Dark"),
        help="Switch between light and dark themes.",
    )
    st.session_state.ui_theme = "Dark" if is_dark_mode else "Light"

    st.sidebar.markdown("---")
    st.session_state.active_section = st.sidebar.radio(
        "Navigation",
        options=["Dashboard", "Chat"],
        index=0 if st.session_state.active_section == "Dashboard" else 1,
        horizontal=True,
    )

    with st.sidebar.expander("Add Patient", expanded=True):
        with st.form("add_patient_form", clear_on_submit=True):
            name = st.text_input("Name")
            age = st.number_input("Age", min_value=0, value=0)
            add_clicked = st.form_submit_button("Add Patient")
        if add_clicked:
            if name.strip():
                patients_col.insert_one(
                    {
                        "name": name.strip(),
                        "age": int(age),
                        "created_at": datetime.now(),
                    }
                )
                st.success("Patient added")
            else:
                st.warning("Please provide patient name.")

    with st.sidebar.expander("Patient List", expanded=True):
        search = st.text_input("Search by name")
        query_filter = {}
        if search:
            query_filter["name"] = {"$regex": search, "$options": "i"}

        patients = list(patients_col.find(query_filter).sort("created_at", -1))
        if not patients:
            st.info("No patients found.")
        else:
            labels = [f"{p['name']} (Age {p.get('age', 0)})" for p in patients]
            current_label = None
            if st.session_state.selected_patient:
                current_label = f"{st.session_state.selected_patient['name']} (Age {st.session_state.selected_patient.get('age', 0)})"
            default_index = labels.index(current_label) if current_label in labels else 0
            selected_label = st.selectbox("Choose patient", labels, index=default_index)
            selected_index = labels.index(selected_label)
            st.session_state.selected_patient = patients[selected_index]

    with st.sidebar.expander("Capture Clinical Input", expanded=False):
        st.caption("Upload or record audio, upload report, and finalize text input.")

        mp3_file = st.file_uploader("Upload MP3", type=["mp3"], key="sidebar_mp3_upload")
        if st.button("Transcribe MP3", key="btn_transcribe_mp3"):
            if mp3_file:
                try:
                    uploaded_bytes = mp3_file.read()
                    text = transcribe_audio_bytes(uploaded_bytes, filename=mp3_file.name or "audio.mp3")
                    if text:
                        st.session_state.transcribed_text = text
                        st.session_state.final_input = text
                        st.success("MP3 transcribed")
                        st.text_area("MP3 Transcript", value=text, height=160)
                    else:
                        st.warning("No speech detected in the MP3.")
                except Exception as exc:
                    st.error(f"MP3 transcription failed: {exc}")
            else:
                st.warning("Upload an MP3 file first.")

        recorded_audio = st.audio_input("Record Audio", key="sidebar_audio_input")
        if recorded_audio:
            try:
                recorded_bytes = recorded_audio.getvalue() if hasattr(recorded_audio, "getvalue") else recorded_audio.read()
                audio_hash = hashlib.sha256(recorded_bytes).hexdigest()
                if audio_hash != st.session_state.recorded_audio_hash:
                    if st.session_state.recorded_audio_path and os.path.exists(st.session_state.recorded_audio_path):
                        os.remove(st.session_state.recorded_audio_path)
                    st.session_state.recorded_audio_path = save_recorded_audio_as_mp3(recorded_audio)
                    st.session_state.recorded_audio_hash = audio_hash

                st.audio(st.session_state.recorded_audio_path)
                if st.button("Convert Recording to Text", key="btn_recording_to_text"):
                    with open(st.session_state.recorded_audio_path, "rb") as audio_file_handle:
                        text = transcribe_audio_bytes(
                            audio_file_handle.read(),
                            filename=os.path.basename(st.session_state.recorded_audio_path),
                        )
                    if text:
                        st.session_state.transcribed_text = text
                        st.session_state.final_input = text
                        st.success("Recorded audio transcribed")
                        st.text_area("Recorded Transcript", value=text, height=150)
                    else:
                        st.warning("No speech detected in recording.")

                if st.button("Clear Recording", key="btn_clear_recording"):
                    if st.session_state.recorded_audio_path and os.path.exists(st.session_state.recorded_audio_path):
                        os.remove(st.session_state.recorded_audio_path)
                    st.session_state.recorded_audio_path = ""
                    st.session_state.recorded_audio_hash = ""
                    st.rerun()
            except Exception as exc:
                st.error(f"Mic recording failed: {exc}")

        report_file = st.file_uploader("Upload Test Report (PDF)", type=["pdf"], key="sidebar_report_upload")
        if st.button("Extract Report Findings", key="btn_extract_report"):
            if report_file:
                try:
                    report_text = extract_pdf_text(report_file)
                    if report_text:
                        st.session_state.transcribed_text = report_text
                        st.session_state.final_input = report_text
                        preview = report_text[:500] + "..." if len(report_text) > 500 else report_text
                        st.success("Report findings extracted")
                        st.text_area("Report Preview", value=preview, height=160, disabled=True)
                    else:
                        st.warning("No findings extracted from report.")
                except Exception as exc:
                    st.error(f"Report extraction failed: {exc}")
            else:
                st.warning("Please upload a report first.")

        manual_text_sidebar = st.text_area(
            "Edit Transcribed Text",
            value=st.session_state.transcribed_text,
            height=120,
            key="manual_transcribed_sidebar",
        )
        direct_text = st.text_area(
            "Or type direct text",
            value="",
            height=90,
            key="direct_text_sidebar",
        )
        if st.button("Use This Text", key="btn_use_sidebar_text"):
            if direct_text.strip():
                st.session_state.final_input = direct_text.strip()
            elif manual_text_sidebar.strip():
                st.session_state.final_input = manual_text_sidebar.strip()
            else:
                st.warning("Enter text or capture audio first.")
                return
            st.success("Text prepared for clinical note.")


def render_chat_view(patient):
    st.markdown('<div class="app-title">Doctor-Patient Chat</div>', unsafe_allow_html=True)
    st.caption("Message history is summarized after every submitted message.")

    patient_id = str(patient["_id"])
    if patient_id not in st.session_state.conversation_messages or not st.session_state.conversation_messages[patient_id]:
        loaded_messages, loaded_summary = load_latest_conversation(patient["_id"])
        st.session_state.conversation_messages[patient_id] = loaded_messages
        st.session_state.conversation_summary[patient_id] = loaded_summary

    patient_messages = st.session_state.conversation_messages.get(patient_id, [])
    patient_summary = st.session_state.conversation_summary.get(patient_id, "")

    st.markdown(
        f'<span class="pill">Patient: {patient["name"]}</span><span class="pill">Age: {patient.get("age", 0)}</span>',
        unsafe_allow_html=True,
    )

    speaker = st.radio(
        "Speaker",
        ["Doctor", "Patient"],
        horizontal=True,
        key=f"chat_speaker_{patient_id}",
    )
    new_message = st.chat_input("Type a message and press Enter")

    if new_message and new_message.strip():
        patient_messages.append(
            {
                "speaker": speaker,
                "text": new_message.strip(),
                "time": datetime.now(),
            }
        )
        st.session_state.conversation_messages[patient_id] = patient_messages
        try:
            with st.spinner("Generating live summary..."):
                conversation_summary = summarize_conversation(patient_messages)
            st.session_state.conversation_summary[patient_id] = conversation_summary
            consult_col.insert_one(
                {
                    "patient_id": patient["_id"],
                    "date": datetime.now(),
                    "type": "conversation",
                    "description": "Doctor-patient conversation",
                    "conversation": patient_messages,
                    "conversation_summary": conversation_summary,
                }
            )
            save_text_to_patient_store(patient_id, f"Conversation Summary: {conversation_summary}")
            st.success("Message saved and summary updated.")
            st.rerun()
        except Exception as exc:
            st.error(f"Conversation summary failed: {exc}")

    with st.container(border=True):
        st.subheader("Chat Log")
        if not patient_messages:
            st.info("No messages yet. Start the conversation below.")
        for message in patient_messages:
            stamp = message["time"].strftime("%H:%M") if message.get("time") else ""
            avatar = "🧑‍⚕️" if message.get("speaker") == "Doctor" else "🧑"
            with st.chat_message(name=message.get("speaker", "Patient"), avatar=avatar):
                st.markdown(f"{message.get('text', '')}")
                if stamp:
                    st.caption(stamp)

    if patient_summary:
        with st.container(border=True):
            st.subheader("Live AI Summary")
            st.write(patient_summary)

    action_col1, action_col2 = st.columns(2)
    with action_col1:
        if st.button("Clear Chat", key=f"clear_chat_{patient_id}"):
            st.session_state.conversation_messages[patient_id] = []
            st.session_state.conversation_summary[patient_id] = ""
            st.rerun()
    with action_col2:
        if st.button("Back to Dashboard", key=f"back_dashboard_{patient_id}"):
            st.session_state.active_section = "Dashboard"
            st.rerun()


def render_dashboard_view(patient):
    patient_id = str(patient["_id"])
    consultations = list(consult_col.find({"patient_id": patient["_id"]}).sort("date", -1))
    patient_messages = st.session_state.conversation_messages.get(patient_id, [])
    patient_summary = st.session_state.conversation_summary.get(patient_id, "")

    st.markdown('<div class="app-title">Patient Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-subtitle">Consolidated timeline, notes, and AI-assisted clinical insights.</div>', unsafe_allow_html=True)

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Patient", patient["name"])
    metric_col2.metric("Age", patient.get("age", 0))
    metric_col3.metric("Total Entries", len(consultations))

    with st.container(border=True):
        st.subheader("History")
        full_context = ""
        latest_saved_conversation_summary = ""
        if not consultations:
            st.info("No saved history for this patient yet.")

        for item in consultations:
            desc = item.get("description", "")
            item_date = item.get("date")
            date_text = item_date.strftime("%Y-%m-%d %H:%M") if item_date else "Unknown date"
            st.markdown(f"**{date_text}**")
            if item.get("type") == "conversation" and item.get("conversation_summary"):
                st.write(item["conversation_summary"])
                if not latest_saved_conversation_summary:
                    latest_saved_conversation_summary = item["conversation_summary"]
            else:
                st.write(desc or "No description")
            st.divider()
            full_context += item.get("conversation_summary", desc) + "\n"

        if not patient_summary:
            patient_summary = latest_saved_conversation_summary
        if patient_summary:
            st.info("Latest conversation summary")
            st.write(patient_summary)

    with st.container(border=True):
        st.subheader("Clinical Summary / Prescription Notes")
        st.caption("Write direct notes or append transcript from audio.")

        clinical_audio = st.audio_input("Record Clinical Summary Audio", key="clinical_summary_audio_input")
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
                action_a, action_b = st.columns(2)
                with action_a:
                    if st.button("Transcribe Clinical Audio", key="transcribe_clinical_audio"):
                        with open(st.session_state.clinical_audio_path, "rb") as audio_file_handle:
                            clinical_text = transcribe_audio_bytes(
                                audio_file_handle.read(),
                                filename=os.path.basename(st.session_state.clinical_audio_path),
                            )
                        if clinical_text:
                            existing_text = st.session_state.get("final_input", "").strip()
                            st.session_state.final_input = (
                                f"{existing_text}\n{clinical_text}".strip() if existing_text else clinical_text
                            )
                            st.success("Clinical audio transcribed and added.")
                        else:
                            st.warning("No speech detected in clinical audio.")
                with action_b:
                    if st.button("Clear Clinical Audio", key="clear_clinical_audio"):
                        if st.session_state.clinical_audio_path and os.path.exists(st.session_state.clinical_audio_path):
                            os.remove(st.session_state.clinical_audio_path)
                        st.session_state.clinical_audio_path = ""
                        st.session_state.clinical_audio_hash = ""
                        st.rerun()
            except Exception as exc:
                st.error(f"Clinical audio processing failed: {exc}")

        final_desc = st.text_area(
            "Edit Clinical Summary Before Saving",
            value=st.session_state.get("final_input", ""),
            height=170,
            key="clinical_note_editor",
        )
        st.session_state.final_input = final_desc

        if st.button("Save Clinical Note"):
            if not final_desc.strip():
                st.warning("Please enter text or audio first.")
            else:
                consult_col.insert_one(
                    {
                        "patient_id": patient["_id"],
                        "date": datetime.now(),
                        "description": final_desc.strip(),
                    }
                )
                save_text_to_patient_store(patient_id, final_desc.strip())
                st.session_state.final_input = ""
                st.success("Clinical note saved and embedded.")

    with st.container(border=True):
        st.subheader("Auto Analysis")
        if not full_context.strip():
            st.info("Add history or conversation to generate analysis.")
        else:
            if st.button("Generate Auto Analysis", key=f"generate_auto_analysis_{patient_id}"):
                try:
                    auto_prompt = PromptTemplate(
                        template="""
You are a medical assistant.

Analyze the patient's condition based on history.

History:
{context}

Give a short summary.
""",
                        input_variables=["context"],
                    )
                    parser = StrOutputParser()
                    chain = auto_prompt | llm | parser
                    auto_answer = chain.invoke({"context": full_context})
                    st.session_state.auto_analysis_cache[patient_id] = auto_answer
                except Exception as exc:
                    st.error(f"Auto analysis failed: {exc}")

            cached_analysis = st.session_state.auto_analysis_cache.get(patient_id, "")
            if cached_analysis:
                st.write(cached_analysis)

    with st.container(border=True):
        st.subheader("Ask AI")
        question = st.text_input("Ask about patient", key=f"ask_ai_query_{patient_id}")
        if st.button("Ask AI", key=f"ask_ai_btn_{patient_id}"):
            if not question.strip():
                st.warning("Please enter a question.")
            elif not os.path.exists(f"faiss/{patient_id}"):
                st.warning("No vector data available for this patient.")
            else:
                try:
                    vector_store = FAISS.load_local(
                        f"faiss/{patient_id}",
                        embeddings,
                        allow_dangerous_deserialization=True,
                    )
                    base_retriever = vector_store.as_retriever(search_kwargs={"k": 3})
                    compressor = LLMChainExtractor.from_llm(llm)
                    compression_retriever = ContextualCompressionRetriever(
                        base_compressor=compressor,
                        base_retriever=base_retriever,
                    )
                    docs = compression_retriever.invoke(question)
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
                        input_variables=["context", "question"],
                    )
                    parser = StrOutputParser()
                    chain = prompt | llm | parser
                    answer = chain.invoke({"context": final_context, "question": question})
                    st.write(answer)
                except Exception as exc:
                    st.error(f"Ask AI failed: {exc}")


init_session_state()
render_sidebar()
apply_theme(st.session_state.ui_theme)

if not st.session_state.selected_patient:
    st.markdown('<div class="app-title">Medical Console</div>', unsafe_allow_html=True)
    st.info("Select a patient from the sidebar to continue.")
elif st.session_state.active_section == "Chat":
    render_chat_view(st.session_state.selected_patient)
else:
    render_dashboard_view(st.session_state.selected_patient)