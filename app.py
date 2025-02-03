import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
import torch
import chromadb  # Required for Chroma's new API
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate  # Required for System Prompt
from langchain_community.document_loaders import PDFMinerLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import shutil

# Define model checkpoint
checkpoint = "LaMini-Flan-T5-248M"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Check GPU availability

model = AutoModelForSeq2SeqLM.from_pretrained(
    checkpoint, torch_dtype=torch.float32
)
model.to(device)

@st.cache_resource
def llm_pipeline():
    """Initialize the text-to-text generation pipeline for LLM."""
    pipe_sum = pipeline(
        'summarization',
        model=model,
        tokenizer=tokenizer,
        max_length=500, 
        min_length=50,
        do_sample=True,
        temperature=0.7, 
        top_p=0.9
    )
    return HuggingFacePipeline(pipeline=pipe_sum)

@st.cache_resource
def process_documents():
    """Loads and processes new PDF documents."""
    documents = []

    # Check if "docs" folder contains new PDFs
    for root, _, files in os.walk("docs"):
        for file in files:
            if file.endswith(".pdf"):
                loader = PDFMinerLoader(os.path.join(root, file))
                documents.extend(loader.load())

    # Ensure documents exist
    if not documents:
        print("⚠️ No new PDF documents found.")
        return None

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    return texts

@st.cache_resource
def qa_llm():
    """Initialize the retrieval-based question-answering model and refresh vector DB."""
    try:
        llm = llm_pipeline()
        embeddings = HuggingFaceEmbeddings(model_name="LaMini-Flan-T5-248M")
        
        # ✅ Process new PDFs
        texts = process_documents()
        if not texts:
            print("⚠️ No new text data found to index.")
            return None

        # ✅ Rebuild ChromaDB with new PDFs
        chroma_client = chromadb.PersistentClient(path="db")
        db = Chroma.from_documents(texts, embeddings, client=chroma_client)
        print("✅ ChromaDB updated with new PDFs.")

        retriever = db.as_retriever()

        # ✅ Create retrieval chain
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer, say you don't know. "
            "Use three sentences maximum and keep the answer concise."
            "\n\n"
            "{context}"
        )
        prompt = ChatPromptTemplate.from_messages([("system", system_prompt)])
        combine_docs_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
        
        return retrieval_chain

    except Exception as e:
        print(f"❌ Error initializing QA system: {str(e)}")
        return None

def process_answer(instruction):
    """Processes the user question and retrieves an answer."""
    qa = qa_llm()  # This returns a RunnableBinding object
    
    try:
        result = qa.invoke({"input": instruction})  # ✅ Use "input" instead of "query"
        answer = result.get('answer', "⚠️ No answer found.")  # ✅ Ensure correct key is used
        source_documents = result.get("source_documents", [])
        return answer, source_documents
    except Exception as e:
        return f"❌ Error: {str(e)}", []

def main():
    """Streamlit UI for PDF-based Q&A."""
    st.title("📄 Search Your PDF")
    
    with st.expander("ℹ️ About the APP"):
        st.write("Upload your PDF and ask questions about its content.")

    # User input
    question = st.text_area("Ask a question about the PDF", placeholder="Type your question here...")

    if st.button("🔍 Search"):
        if question:
            st.info(f"🔎 Searching: {question}")
            
            try:
                with st.spinner('⏳ Processing your request...'):
                    answer, metadata = process_answer(question)
                    
                    st.markdown(f"**📝 Answer:** {answer}")
                    
                    if metadata:
                        st.markdown("**📌 Source Documents:**")
                        for doc in metadata:
                            st.write(f"- {doc.metadata.get('source', 'Unknown Source')}")
                    else:
                        st.warning("⚠️ No source documents found.")
            except Exception as e:
                st.error(f"❌ An error occurred: {e}")
        else:
            st.warning("⚠️ Please enter a question.")

if __name__ == "__main__":
    main()
