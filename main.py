from dotenv import load_dotenv
import os
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

# ---------------- AUDIO ----------------
from groq import Groq
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def speech_to_text(audio_file):
    with open(audio_file, "rb") as f:
        transcription = groq_client.audio.transcriptions.create(
            file=f,
            model="whisper-large-v3",
            prompt="Hindi, Marathi, English medical conversation between doctor and patient"
        )
    return transcription.text.strip()

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

# ================= SIDEBAR =================
st.sidebar.title("👤 Patients")

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
st.sidebar.markdown("### 🎤 Voice + Text Input")

audio_file = st.sidebar.file_uploader("📤 Upload Audio", type=["wav", "mp3"])
mic_audio = st.sidebar.audio_input("🎙️ Record from Mic")

# Convert Audio
if st.sidebar.button("🎤 Convert Audio to Text"):

    file_to_use = None

    if mic_audio:
        with open("temp_audio.wav", "wb") as f:
            f.write(mic_audio.read())
        file_to_use = "temp_audio.wav"

    elif audio_file:
        with open("temp_audio.wav", "wb") as f:
            f.write(audio_file.read())
        file_to_use = "temp_audio.wav"

    if file_to_use:
        st.sidebar.info("Transcribing...")
        text = speech_to_text(file_to_use)
        st.session_state.transcribed_text = text
        st.sidebar.success("Done ✅")
    else:
        st.sidebar.warning("Upload or record audio first")

# Direct Text
st.sidebar.markdown("### ✍️ Or Type Directly")

direct_text = st.sidebar.text_area(
    "Write patient symptoms / conversation",
    height=120
)

# Edit Transcribed
manual_text_sidebar = st.sidebar.text_area(
    "Edit Transcribed Text",
    value=st.session_state.transcribed_text,
    height=120
)

# Use Text
if st.sidebar.button("✅ Use This Text"):

    if direct_text.strip():
        st.session_state.final_input = direct_text
    elif manual_text_sidebar.strip():
        st.session_state.final_input = manual_text_sidebar
    else:
        st.sidebar.warning("Enter text or audio first")
        st.stop()

    st.sidebar.success("Text Ready ✅")

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

        # ================= HISTORY =================
        st.subheader("📜 History")

        consultations = list(
            consult_col.find({"patient_id": patient["_id"]}).sort("date", -1)
        )

        full_context = ""

        for c in consultations:
            desc = c.get("description", "")
            st.write(f"🗓 {c['date'].strftime('%Y-%m-%d')}")
            st.write(f"📝 {desc}")
            st.divider()
            full_context += desc + "\n"

        # ================= FINAL DESCRIPTION =================
        st.subheader("✍ Final Doctor Description (Verified)")

        final_desc = st.session_state.get("final_input", "")

        final_desc = st.text_area(
            "Edit Final Description Before Saving",
            value=final_desc,
            height=150
        )

        st.session_state.final_input = final_desc

        # ================= SAVE =================
        if st.button("Add Description"):

            if not final_desc:
                st.warning("Please enter text or audio")
                st.stop()

            consult_col.insert_one({
                "patient_id": patient["_id"],
                "date": datetime.now(),
                "description": final_desc
            })

            os.makedirs(f"faiss/{patient_id}", exist_ok=True)

            if os.path.exists(f"faiss/{patient_id}/index.faiss"):
                vector_store = FAISS.load_local(
                    f"faiss/{patient_id}",
                    embeddings,
                    allow_dangerous_deserialization=True
                )
                vector_store.add_texts([final_desc])
            else:
                vector_store = FAISS.from_texts([final_desc], embeddings)

            vector_store.save_local(f"faiss/{patient_id}")

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

                retriever = vector_store.as_retriever(search_kwargs={"k": 3})
                docs = retriever.invoke(query)

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