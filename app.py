import re
import io
import os
import json
from pathlib import Path

import streamlit as st
import fitz  # PyMuPDF
import numpy as np
import spacy
from docx import Document
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

try:
    import pytesseract
    from PIL import Image
except Exception:
    pytesseract = None
    Image = None


@st.cache_resource
def load_spacy():
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        nlp = spacy.blank("en")

    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
    return nlp


@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


nlp = load_spacy()
model = load_model()


class SemanticPlagiarismAgent:
    def _preprocess_and_split(self, text):
        text = re.sub(r"\s+", " ", text or "").strip()
        if not text:
            return []
        doc = nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 0]
        return [s for s in sentences if len(s.split()) >= 3]

    def _get_embeddings(self, sentences):
        return model.encode(sentences, convert_to_tensor=False, normalize_embeddings=True)

    def detect_plagiarism(self, source_text, suspicious_text, similarity_threshold=0.65):
        source_sentences = self._preprocess_and_split(source_text)
        suspicious_sentences = self._preprocess_and_split(suspicious_text)

        if not suspicious_sentences or not source_sentences:
            return None

        source_embeddings = self._get_embeddings(source_sentences)
        suspicious_embeddings = self._get_embeddings(suspicious_sentences)

        sim_matrix = cosine_similarity(suspicious_embeddings, source_embeddings)

        matched_sections = []
        seen_source_indices = set()
        total_matched_words = 0
        total_suspicious_words = sum(len(sentence.split()) for sentence in suspicious_sentences)

        for i, sus_sent in enumerate(suspicious_sentences):
            best_match_idx = int(np.argmax(sim_matrix[i]))
            best_score = float(sim_matrix[i][best_match_idx])

            if best_score >= similarity_threshold and best_match_idx not in seen_source_indices:
                source_sent = source_sentences[best_match_idx]
                word_count = len(sus_sent.split())
                total_matched_words += word_count
                seen_source_indices.add(best_match_idx)

                matched_sections.append(
                    {
                        "suspicious_sentence": sus_sent,
                        "matching_source_sentence": source_sent,
                        "semantic_similarity_score": round(best_score, 4),
                        "word_count": word_count,
                    }
                )

        similarity_percentage = (
            round((total_matched_words / total_suspicious_words) * 100, 2)
            if total_suspicious_words > 0
            else 0
        )
        is_plagiarized = similarity_percentage > 20 or len(matched_sections) >= 2

        return {
            "overall_similarity_percentage": similarity_percentage,
            "plagiarism_flag": "HIGH RISK" if is_plagiarized else "LOW RISK",
            "total_paraphrased_sections_found": len(matched_sections),
            "matching_sections": matched_sections,
        }


agent = SemanticPlagiarismAgent()


def extract_text_from_pdf(file_bytes):
    if not file_bytes:
        return ""

    doc = None
    try:
        for open_pdf in [
            lambda: fitz.open(stream=file_bytes, filetype="pdf"),
            lambda: fitz.open(stream=io.BytesIO(file_bytes), filetype="pdf"),
            lambda: fitz.open(io.BytesIO(file_bytes)),
        ]:
            try:
                doc = open_pdf()
                text = ""
                for page in doc:
                    page_text = page.get_text("text")
                    if page_text:
                        text += page_text + "\n"
                if text.strip():
                    return text.strip()
            except Exception:
                continue
            finally:
                if doc is not None:
                    try:
                        doc.close()
                    except Exception:
                        pass
                    doc = None
    except Exception:
        pass

    if pytesseract is not None and Image is not None:
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            pages = []
            for page in doc:
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                text = pytesseract.image_to_string(image, config="--psm 6")
                if text.strip():
                    pages.append(text.strip())
            if pages:
                return "\n\n".join(pages)
        except Exception:
            pass
        finally:
            if doc is not None:
                try:
                    doc.close()
                except Exception:
                    pass

    return ""


def extract_text_from_docx(file_bytes):
    doc = Document(io.BytesIO(file_bytes))
    return "\n".join(paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip())


def extract_text_from_file(uploaded_file):
    if uploaded_file is None:
        return ""

    filename = (uploaded_file.name or "").lower()
    file_bytes = uploaded_file.read()

    if filename.endswith(".pdf"):
        return extract_text_from_pdf(file_bytes)
    if filename.endswith(".docx"):
        return extract_text_from_docx(file_bytes)
    if filename.endswith(".txt"):
        return file_bytes.decode("utf-8", errors="ignore")

    return ""


def load_sample_documents():
    base_dir = Path(__file__).resolve().parent
    source_path = base_dir / "sample_source.txt"
    suspicious_path = base_dir / "sample_suspicious.txt"

    if source_path.exists() and suspicious_path.exists():
        return source_path.read_text(encoding="utf-8", errors="ignore"), suspicious_path.read_text(encoding="utf-8", errors="ignore")
    return "", ""


st.set_page_config(page_title="SemanticGuard", page_icon="🛡️", layout="wide")

st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🛡️ SemanticGuard</h1>", unsafe_allow_html=True)
st.markdown(
    "<h3 style='text-align: center; color: #7a7a7a;'>Next-Gen Plagiarism Detection based on Meaning, not just Words</h3>",
    unsafe_allow_html=True,
)
st.markdown("---")

if "source_sample" not in st.session_state or "suspicious_sample" not in st.session_state:
    sample_source, sample_suspicious = load_sample_documents()
    if sample_source and sample_suspicious:
        st.session_state["source_sample"] = sample_source
        st.session_state["suspicious_sample"] = sample_suspicious

if "source_sample" in st.session_state and "suspicious_sample" in st.session_state:
    st.info("Sample dataset is loaded and ready to use. You can also upload your own documents.")

if st.button("Load sample dataset", use_container_width=True):
    sample_source, sample_suspicious = load_sample_documents()
    if sample_source and sample_suspicious:
        st.session_state["source_sample"] = sample_source
        st.session_state["suspicious_sample"] = sample_suspicious
        st.success("Sample dataset loaded.")
    else:
        st.warning("Sample files were not found. Create sample_source.txt and sample_suspicious.txt in the project folder.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📄 Original Source Document")
    source_file = st.file_uploader("Upload Original Document", type=["pdf", "txt", "docx"], key="source")
    if "source_sample" in st.session_state and not source_file:
        st.caption("Sample source loaded and ready to compare.")

with col2:
    st.subheader("📝 Suspicious Submission")
    suspicious_file = st.file_uploader("Upload Suspicious Document", type=["pdf", "txt", "docx"], key="suspicious")
    if "suspicious_sample" in st.session_state and not suspicious_file:
        st.caption("Sample suspicious text loaded and ready to compare.")

st.markdown("### ⚙️ Detection Sensitivity")
threshold = st.slider(
    "Semantic Similarity Threshold (Lower = Stricter matching)",
    min_value=0.50,
    max_value=0.95,
    value=0.65,
    step=0.05,
)

if st.button("🔍 Run Semantic Analysis", use_container_width=True):
    if source_file and suspicious_file:
        source_text = extract_text_from_file(source_file)
        suspicious_text = extract_text_from_file(suspicious_file)
    elif "source_sample" in st.session_state and "suspicious_sample" in st.session_state:
        source_text = st.session_state["source_sample"]
        suspicious_text = st.session_state["suspicious_sample"]
    else:
        source_text = ""
        suspicious_text = ""

    if source_text and suspicious_text:
        with st.spinner("Extracting text and computing semantic embeddings..."):
            source_ok = bool((source_text or "").strip())
            suspicious_ok = bool((suspicious_text or "").strip())

            if not source_ok or not suspicious_ok:
                missing = []
                if not source_ok:
                    missing.append("source")
                if not suspicious_ok:
                    missing.append("suspicious")
                st.error(
                    "No readable text could be extracted from the selected " + ", ".join(missing) + " document(s). "
                    "Please upload searchable PDF, DOCX, or TXT files or use a sample dataset."
                )
            else:
                report = agent.detect_plagiarism(source_text, suspicious_text, threshold)

                if report is not None:
                    st.session_state["report"] = report
                    st.session_state["source_text"] = source_text
                    st.session_state["suspicious_text"] = suspicious_text
                else:
                    st.warning("The extracted text was too short to compare meaningfully. Please upload longer documents.")
    else:
        st.warning("Please upload both files or use the built-in sample dataset to proceed.")

if "report" in st.session_state:
    report = st.session_state["report"]
    flag = report["plagiarism_flag"]
    color = "#FF4B4B" if flag == "HIGH RISK" else "#4CAF50"

    st.markdown("---")
    st.markdown("<h2 style='text-align: center;'>📊 Plagiarism Report</h2>", unsafe_allow_html=True)

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    with metric_col1:
        st.metric(label="Overall Similarity", value=f"{report['overall_similarity_percentage']}%")
    with metric_col2:
        st.metric(label="Flagged Sections", value=report["total_paraphrased_sections_found"])
    with metric_col3:
        st.markdown(
            f"<div style='text-align: center; background-color: {color}; padding: 10px; border-radius: 5px; color: white; font-weight: bold; font-size: 20px;'>{flag}</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("### 🔗 Matched Sections (Paraphrase Detection)")

    if not report["matching_sections"]:
        st.info("No matching paraphrased sections met the current similarity threshold.")
    else:
        for idx, match in enumerate(report["matching_sections"], 1):
            score = match["semantic_similarity_score"] * 100
            with st.expander(f"Match #{idx} - {score:.2f}% Similarity"):
                match_col1, match_col2 = st.columns(2)
                with match_col1:
                    st.markdown("**📝 Suspicious Text:**")
                    st.info(match["suspicious_sentence"])
                with match_col2:
                    st.markdown("**📄 Original Source:**")
                    st.warning(match["matching_source_sentence"])
