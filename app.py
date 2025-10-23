import streamlit as st
import pandas as pd
from langchain_openai.chat_models import ChatOpenAI

# -----------------------------------
# ⚠ Check for API Key
# -----------------------------------
if "OPENAI_API_KEY" not in st.secrets:
    st.error("❌ OPENAI_API_KEY not found. Add it in .streamlit/secrets.toml")
    st.stop()

# -----------------------------------
# 🧠 Page Setup
# -----------------------------------
st.set_page_config(page_title="Medical Data Chatbot", page_icon="🩺", layout="centered")
st.title("💬 Medical Data Chatbot (Llama 3.2 + OpenAI)")

# -----------------------------------
# 📚 Load Default Datasets
# -----------------------------------
default_files = {
    "Symptom_severity.csv": None,
    "symptom_Description.csv": None,
    "symptom_precaution.csv": None,
    "dataset.csv":None,
    "Diseases_Symptoms.csv":None
}

for file in default_files:
    try:
        default_files[file] = pd.read_csv(file)
        st.success(f"Loaded dataset: {file}")
    except Exception as e:
        st.warning(f"Could not load {file}: {e}")

# -----------------------------------
# 📁 Optional File Uploads
# -----------------------------------
uploaded_files = st.file_uploader("Upload extra CSV files (optional):", type=["csv"], accept_multiple_files=True)
if uploaded_files:
    for f in uploaded_files:
        try:
            default_files[f.name] = pd.read_csv(f)
            st.success(f"Added uploaded dataset: {f.name}")
        except Exception as e:
            st.error(f"Could not read {f.name}: {e}")

# Store datasets in session
st.session_state.dfs = default_files

# -----------------------------------
# 👀 Display Available Datasets
# -----------------------------------
st.write("### Available Datasets:")
for name, df in st.session_state.dfs.items():
    if df is not None:
        with st.expander(f"📄 {name}"):
            st.dataframe(df.head())

# -----------------------------------
# 💬 Chat History Initialization
# -----------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -----------------------------------
# 🧩 Dataset Selector (Manual + Auto)
# -----------------------------------
def select_relevant_dataset(query: str):
    """Smart keyword-based auto-selection of dataset."""
    query_lower = query.lower()

    # --- Keyword groups ---
    symptom_keywords = [
        "symptom", "feel", "pain", "ache", "discomfort", "nausea", "vomit",
        "fever", "cold", "cough", "diagnosis", "disease", "infection"
    ]
    precaution_keywords = [
        "prevent", "precaution", "care", "avoid", "treatment",
        "medicine", "cure", "remedy", "what should i do", "how to recover"
    ]
    severity_keywords = [
        "severe", "mild", "intense", "critical", "level", "scale", "serious"
    ]

    if any(word in query_lower for word in symptom_keywords):
        return "symptom_Description.csv"
    elif any(word in query_lower for word in precaution_keywords):
        return "symptom_precaution.csv"
    elif any(word in query_lower for word in severity_keywords):
        return "Symptom_severity.csv"
    else:
        return None

dataset_choice = st.selectbox(
    "📘 Choose dataset (or use Auto Detect):",
    ["Auto Detect", "symptom_Description.csv", "symptom_precaution.csv", "Symptom_severity.csv"]
)

# -----------------------------------
# 🗣 Chat Input
# -----------------------------------
user_input = st.chat_input("Ask about symptoms, severity, or precautions...")

if user_input:
    # Display user message
    st.chat_message("user").markdown(user_input)
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # Determine dataset
    if dataset_choice != "Auto Detect":
        chosen_dataset = dataset_choice
    else:
        chosen_dataset = select_relevant_dataset(user_input)

    if chosen_dataset and chosen_dataset in st.session_state.dfs:
        df = st.session_state.dfs[chosen_dataset]
        context = df.head(50).to_string(index=False)
        dataset_used = chosen_dataset
    else:
        # fallback if detection failed
        st.warning("⚠ Could not determine dataset — using all datasets for context.")
        context = "\n\n".join(
            [df.head(50).to_string(index=False) for df in st.session_state.dfs.values() if df is not None]
        )
        dataset_used = "All Datasets"

    # -----------------------------------
    # 🧠 LLM Processing
    # -----------------------------------
    prompt = f"""
You are a helpful medical assistant chatbot.
Use the dataset provided to answer user queries accurately.
Do NOT assume this data represents the user's personal health report — it's a general medical dataset.

Dataset Source: {dataset_used}

Data Preview:
{context}

Question: {user_input}

Answer clearly using only reliable dataset information.
    """

    # llm = OllamaLLM(model="llama3.2", temperature=0)
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=st.secrets["OPENAI_API_KEY"])
    try:
        response = llm.invoke(prompt)
        response_text = str(response) if not isinstance(response, dict) else response.get("content", str(response))
    except Exception as e:
        response_text = f"Error generating response: {e}"

    # Display assistant response
    st.chat_message("assistant").markdown(response_text)
    st.session_state.chat_history.append({"role": "assistant", "content": response_text})

# -----------------------------------
# 💾 Persistent Chat History Display
# -----------------------------------
if st.session_state.chat_history:
    with st.expander("🕘 Chat History"):
        for message in st.session_state.chat_history:
            role = "🧑‍💻 User" if message["role"] == "user" else "🤖 Assistant"
            st.write(f"{role}:** {message['content']}")