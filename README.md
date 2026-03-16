# Cybersecurity Threat Detection with RAG

This project implements a cybersecurity threat detection system using Machine Learning and Retrieval-Augmented Generation (RAG). It provides risk prediction for cybersecurity incidents and analyzes URLs for potential threats.

## Features

- **Incident Risk Prediction**: Uses Random Forest and other classifiers to predict the risk level of cybersecurity incidents based on financial loss, affected users, response time, and more.
- **URL Threat Checker**: Analyzes URLs for structural characteristics common to phishing or malicious links.
- **RAG Assistant**: A knowledge assistant that uses a vector database (FAISS) and a local LLM (DistilGPT2) to answer cybersecurity-related questions based on provided knowledge.
- **Streamlit Dashboard**: An interactive web interface for exploring the models and tools.

## File Structure

- `complete_project.py`: The main entry point, containing data generation, model training (incident and URL), and the Streamlit dashboard.
- `rag_engine.py`: Contains the logic for building the vector store and performing RAG queries.
- `knowledge.txt`: A text file containing the knowledge base for the RAG assistant.
- `data_collection.py`, `data_preprocessing.py`, `model_training.py`, `comparison_analysis.py`, `deployment.py`: Modularized components of the project (earlier versions or specific modules).
- `requirements.txt`: Project dependencies.

## Setup and Installation

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the main project script to train models and start the dashboard:
   ```bash
   python complete_project.py
   ```
   *Note: In a Streamlit environment, use `streamlit run complete_project.py`.*

## Testing the RAG System

You can test the RAG engine independently using:
```bash
python test_rag.py
```
