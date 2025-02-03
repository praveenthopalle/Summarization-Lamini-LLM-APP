# Summarization using LaMini-LLM-App
A Streamlit application that summarizes documents using LaMini-LM.

# Development Mode
Make Sure to install requirements on your Virtualenv

# Make sure to create a folder named db and docs

# Download Lamini-T5 Model
Download Lamini-T5 model weights from Hugging Face

# Install all the necessary Packages
pip install -r requirements.txt

# Run the Ingest.py to setup chromaDB with uploaded docs
python ingest.py

# Now run the application using streamlit
streamlit run app.py

# New port will be exposed to GUI
Default port = 8501