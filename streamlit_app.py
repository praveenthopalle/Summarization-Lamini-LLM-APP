import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
import base64
import os

# Ensure SentencePiece is installed for T5 Tokenizer
try:
    import sentencepiece
except ImportError:
    print("Installing missing 'sentencepiece' package...")
    os.system("pip install sentencepiece")

# Model and tokenizer loading
checkpoint = "LaMini-Flan-T5-248M"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
base_model = AutoModelForSeq2SeqLM.from_pretrained(
    checkpoint, device_map="auto", torch_dtype=torch.float32
)

# File loader and preprocessing
def file_preprocessing(file_path):
    """Loads and splits PDF content into smaller chunks for processing."""
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    texts = text_splitter.split_documents(pages)
    
    # Combine all extracted text
    final_texts = " ".join([text.page_content for text in texts])
    return final_texts

# LLM pipeline
def llm_pipeline(file_path):
    """Runs the summarization model on the processed text."""
    pipe_sum = pipeline(
        "summarization",
        model=base_model,
        tokenizer=tokenizer,
        max_length=5000,
        min_length=250
    )
    
    input_text = file_preprocessing(file_path)
    if not input_text.strip():
        return "⚠️ No valid text found in the document."

    result = pipe_sum(input_text)
    return result[0]["summary_text"]

@st.cache_data
def displayPDF(file_path):
    """Displays PDF files in Streamlit using base64 encoding."""
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode("utf-8")

    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="500" type="application/pdf"></iframe>'
    return pdf_display

# Streamlit UI configuration
st.set_page_config(layout="wide")

def main():
    st.title("📄 Multi-Document Summarization App using LLM")

    uploaded_files = st.file_uploader("📂 Upload your PDF files", type=["pdf"], accept_multiple_files=True)

    if uploaded_files:  # Ensure files are uploaded
        os.makedirs("data", exist_ok=True)

        summaries = []
        pdf_displays = []

        for uploaded_file in uploaded_files:  # ✅ Process EACH file
            file_path = os.path.join("data", uploaded_file.name)

            # Save uploaded file
            with open(file_path, "wb") as temp_file:
                temp_file.write(uploaded_file.read())

            try:
                summary = llm_pipeline(file_path)
                summaries.append((uploaded_file.name, summary))
                pdf_displays.append((uploaded_file.name, displayPDF(file_path)))
            except Exception as e:
                st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")

        # ✅ Display PDFs and summaries using dynamic layout
        for index, (name, pdf) in enumerate(pdf_displays):
            st.markdown(f"### 📄 {name}")  # PDF Title
            st.markdown(pdf, unsafe_allow_html=True)  # PDF Display
            st.markdown(f"**📝 Summary:** {summaries[index][1]}")  # Summary

if __name__ == "__main__":
    main()
