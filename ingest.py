from langchain_community.document_loaders import PyPDFLoader, PDFMinerLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import chromadb  # Required for new Chroma Client API

import os

# Define directory for persistence
persist_directory = os.path.join(os.getcwd(), "db")

def main():
    documents = []  # Store all documents here

    # Iterate over PDF files in "docs" directory
    for root, dirs, files in os.walk("docs"):
        for file in files:
            if file.endswith(".pdf"):
                loader = PDFMinerLoader(os.path.join(root, file))
                documents.extend(loader.load())  # Load documents into list

    # Check if any documents were loaded
    if not documents:
        print("No PDF documents found!")
        return

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="LaMini-Flan-T5-248M")

    # Use the new Chroma Persistent Client API
    chroma_client = chromadb.PersistentClient(path=persist_directory)

    # Create vector store
    vector_store = Chroma.from_documents(
        texts, 
        embeddings, 
        persist_directory=persist_directory,
        client=chroma_client  # Use this instead of `client_settings`
    )

    # Persist and clean up
    vector_store.persist()
    vector_store = None
    print("Chroma Vector Store successfully updated and persisted!")

if __name__ == "__main__":
    main()
