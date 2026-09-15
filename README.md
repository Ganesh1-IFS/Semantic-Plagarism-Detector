# Semantic Plagiarism Detection System

A Streamlit-based semantic plagiarism detector that compares meaning rather than exact wording using sentence embeddings and cosine similarity.

## Features
- PDF, DOCX, and TXT document processing
- Sentence splitting with spaCy
- Semantic embeddings using SentenceTransformers
- Similarity calculation with cosine similarity
- Paraphrase detection and matched section identification
- Overall similarity percentage and plagiarism risk flag
- Sample dataset for quick testing

## Setup

1. Create and activate a virtual environment (optional but recommended)
2. Install dependencies:

```bash
pip install streamlit pymupdf python-docx sentence-transformers scikit-learn spacy pillow pytesseract
```

3. Run the app:

```bash
streamlit run app.py
```

## Files
- `app.py` – Streamlit application
- `sample_source.txt` – example source document
- `sample_suspicious.txt` – example suspicious document

## Usage
- Upload a source document and a suspicious document, or use the sample dataset
- Click the analysis button
- Review the similarity report and matching sections
"# Semantic-Plagarism-Detector" 
