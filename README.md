# 🛡️ Cybersecurity Threat Intelligence Bot

> **ML-driven incident risk prediction + Retrieval-Augmented Generation (RAG) explanations powered by a local LLM.**

A full-stack cybersecurity dashboard that predicts the severity of security incidents using ensemble ML models and then generates human-readable explanations grounded in a curated knowledge base — all running locally, with no data leaving your machine.

---

## 📑 Table of Contents

1. [Project Overview](#-project-overview)
2. [Architecture](#-architecture)
3. [Machine Learning Models](#-machine-learning-models)
4. [RAG Pipeline](#-rag-pipeline)
5. [Project Structure](#-project-structure)
6. [Setup Instructions](#-setup-instructions)
7. [Running the App](#-running-the-app)
8. [Dashboard Features](#-dashboard-features)
9. [Tech Stack](#-tech-stack)
10. [Authors](#-authors)

---

## 🎯 Project Overview

Cybersecurity teams face an ever-growing volume of security incidents. Triaging them manually is slow and error-prone. This project solves two problems:

| Problem | Solution |
|---------|----------|
| *"Is this incident high risk?"* | A **Random Forest** classifier trained on historical incident data predicts the probability of high risk given financial, operational, and contextual features. |
| *"Why is it high risk? What should we do?"* | A **RAG pipeline** retrieves relevant passages from a cybersecurity knowledge base (via ChromaDB) and feeds them to a local quantized LLM (Qwen 2.5 3B via llama.cpp) to generate a concise, grounded explanation. |

### Key Features

- **Incident Risk Prediction** — slider-based input for financial loss, affected users, response time, vulnerability score, attack type, industry, and country.
- **RAG-powered Explanations** — every prediction can be accompanied by an LLM explanation citing the knowledge base.
- **URL Threat Checker** — heuristic analysis of URLs for phishing indicators.
- **Knowledge Assistant** — free-form Q&A against the cybersecurity knowledge base.
- **100 % local** — no API keys, no cloud calls; the LLM runs on-device via llama.cpp.

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     Streamlit Dashboard (UI)                     │
│         tabs: Incident Prediction │ URL Checker │ Q&A            │
└──────────┬──────────────────────────────────────┬────────────────┘
           │  user inputs (sliders, dropdowns)    │  user question
           ▼                                      ▼
┌─────────────────────┐               ┌─────────────────────────┐
│   ML Prediction     │               │   RAG Pipeline          │
│                     │               │                         │
│  LabelEncoders      │               │  1. Embed query         │
│        ▼            │               │     (all-MiniLM-L6-v2)  │
│  StandardScaler     │               │        ▼                │
│        ▼            │               │  2. ChromaDB retrieval  │
│  Random Forest      │               │     (cosine similarity) │
│  .predict_proba()   │               │        ▼                │
│        │            │               │  3. Prompt assembly     │
│        ▼            │               │     (ChatML template)   │
│  risk_probability ──┼───────────────┼──►                      │
│                     │               │  4. LLM generation      │
└─────────────────────┘               │     (Qwen 2.5 3B GGUF  │
                                      │      via llama.cpp)     │
                                      │        ▼                │
                                      │  explanation text       │
                                      └─────────────────────────┘
                                                 │
                  ┌──────────────────────────────┐│
                  │  ChromaDB (persistent)       ││
                  │  cybersecurity_kb collection  │◄── knowledge.txt
                  │  on-disk at ./chroma_db/     ││    (ingested at startup)
                  └──────────────────────────────┘│
                                                  ▼
                                      ┌─────────────────────────┐
                                      │  Qwen 2.5 3B Instruct   │
                                      │  (Q5_K_M quantisation)  │
                                      │  ./models/*.gguf        │
                                      └─────────────────────────┘
```

### Data Flow (step by step)

1. **User** fills in incident parameters on the Streamlit dashboard.
2. **LabelEncoders** convert categorical features (attack type, industry, country) to integers.
3. **StandardScaler** normalises the feature vector.
4. **Random Forest** outputs `predict_proba()` → probability of HIGH RISK.
5. The probability + user question are sent to the **RAG pipeline**.
6. **SentenceTransformer** (`all-MiniLM-L6-v2`) embeds the query.
7. **ChromaDB** performs cosine-similarity search and returns the top-3 knowledge chunks.
8. A **ChatML prompt** is assembled with the prediction context + retrieved chunks + user question.
9. **Qwen 2.5 3B** (via `llama-cpp-python`) generates a concise, grounded explanation.
10. The risk badge + explanation are rendered on the dashboard.

---

## 🤖 Machine Learning Models

### Incident Risk Classifier

| Aspect | Detail |
|--------|--------|
| **Algorithm** | Random Forest (scikit-learn) |
| **Task** | Binary classification — *High Risk* vs *Low Risk* |
| **Input features** | `financial_loss`, `affected_users`, `response_time`, `vulnerability_score`, `attack_type` (encoded), `industry` (encoded), `country` (encoded) |
| **Output** | Probability of high risk (0.0 – 1.0) |
| **Preprocessing** | LabelEncoder (categorical → int) + StandardScaler (z-score normalisation) |
| **Serialisation** | `rf_model.pkl`, `scaler.pkl`, `le_attack.pkl`, `le_industry.pkl`, `le_country.pkl` |

### Other models trained (in `model_training.py`)

| Model | File | Purpose |
|-------|------|---------|
| Logistic Regression | `lr.pkl` | Baseline linear classifier |
| Gradient Boosting | `gb.pkl` / `gradient_boosting_model.pkl` | Boosted ensemble comparison |
| Deep Learning (Keras) | `incident_dl_model.h5` | Neural network comparison |
| URL classifier (DL) | `url_dl_model.h5` / `url_model.h5` | URL phishing detection |

The dashboard currently uses the **Random Forest** model for inference as the primary classifier.

---

## 📚 RAG Pipeline

| Component | Technology | Role |
|-----------|------------|------|
| **Vector Store** | ChromaDB (persistent, on-disk) | Stores and retrieves knowledge chunks by cosine similarity |
| **Embeddings** | `all-MiniLM-L6-v2` (SentenceTransformers) | Encodes both documents and queries into 384-dim vectors |
| **Text Splitter** | `RecursiveCharacterTextSplitter` (LangChain) | Splits `knowledge.txt` into overlapping 512-char chunks |
| **LLM** | Qwen 2.5 3B Instruct (Q5_K_M GGUF) | Generates natural-language explanations from retrieved context |
| **Runtime** | `llama-cpp-python` | Runs the quantised GGUF model on CPU with 2048-token context |

---

## 📁 Project Structure

```
Cybersecurity_ML_Project/
│
├── ui/
│   └── app.py                    # Streamlit dashboard (inference only)
│
├── src/
│   ├── __init__.py
│   ├── rag/
│   │   ├── __init__.py
│   │   └── engine.py             # ChromaDB + llama.cpp RAG engine
│   ├── ml/                       # (ML utilities)
│   └── data/                     # (data processing utilities)
│
├── models/
│   └── qwen2.5-3b-instruct-q5_k_m.gguf   # Local quantised LLM (~2.4 GB)
│
├── chroma_db/                    # Persistent vector store (auto-created)
├── knowledge.txt                 # Cybersecurity knowledge base for RAG
│
├── model_training.py             # Training script (RF, GB, DL, etc.)
├── data_preprocessing.py         # Feature engineering & cleaning
├── data_collection.py            # Dataset generation/collection
│
├── rf_model.pkl                  # Pre-trained Random Forest
├── scaler.pkl                    # StandardScaler
├── le_attack.pkl                 # LabelEncoder — attack type
├── le_industry.pkl               # LabelEncoder — industry
├── le_country.pkl                # LabelEncoder — country
│
├── requirements.txt              # Python dependencies
└── README.md                     # ← You are here
```

---

## ⚙️ Setup Instructions

### Prerequisites

- **Python 3.10+** (tested on 3.10, 3.11)
- **Git**
- ~3 GB free disk space (for the GGUF model)

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/Cybersecurity_ML_Project.git
cd Cybersecurity_ML_Project
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
.\venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the GGUF model

Download **Qwen2.5-3B-Instruct (Q5_K_M)** from HuggingFace:

🔗 https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF

Place the file in the `models/` directory:

```
models/
└── qwen2.5-3b-instruct-q5_k_m.gguf   (~2.4 GB)
```

> **Note:** The app works without the GGUF model — ML predictions still function; only the RAG explanations are disabled.

### 5. Train the ML models (if `.pkl` files are missing)

```bash
python model_training.py
```

This generates `rf_model.pkl`, `scaler.pkl`, and the label encoders.

---

## 🚀 Running the App

```bash
streamlit run ui/app.py
```

The dashboard opens at **http://localhost:8501**.

---

## 🖥️ Dashboard Features

### Tab 1 — 📊 Incident Prediction

- Adjust sliders for financial loss, affected users, response time, vulnerability score.
- Select attack type, industry, and country.
- Click **Predict & Explain** to get:
  - A **risk probability** (0–100 %).
  - A colour-coded **risk badge** (🟢 Low / 🟡 Medium / 🔴 High).
  - An **LLM-generated explanation** grounded in the knowledge base (if LLM loaded).

### Tab 2 — 🌐 URL Checker

- Paste any URL and get a heuristic-based malicious probability score.
- Checks for suspicious patterns: `@` signs, excessive hyphens/dots, keywords like `login`/`verify`/`bank`, and HTTP (vs HTTPS).

### Tab 3 — 🤖 Knowledge Assistant

- Ask free-form cybersecurity questions.
- The RAG engine retrieves relevant knowledge chunks and generates an answer.

---

## 🛠️ Tech Stack

| Layer | Technologies |
|-------|-------------|
| **Frontend** | Streamlit |
| **ML** | scikit-learn (Random Forest, Logistic Regression, Gradient Boosting), Keras/TensorFlow |
| **Vector Store** | ChromaDB |
| **Embeddings** | SentenceTransformers (`all-MiniLM-L6-v2`) |
| **LLM** | Qwen 2.5 3B Instruct (GGUF Q5_K_M) via `llama-cpp-python` |
| **Text Processing** | LangChain Text Splitters |
| **Language** | Python 3.10+ |

---

## 👥 Authors

*Cybersecurity ML Project — Semester 7*

---

> Built with ❤️ using scikit-learn, ChromaDB, llama.cpp and Streamlit.
