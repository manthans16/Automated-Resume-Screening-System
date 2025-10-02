# app.py
# ============================================================================ 
# Resume Screening System - Streamlit App (robust XGBoost + SBERT + heuristics)
# ============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import re
from pathlib import Path
import plotly.express as px
import os
import xgboost as xgb

# NLP & embeddings
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Optional file parsing
try:
    import pdfplumber
except Exception:
    pdfplumber = None
try:
    import docx2txt
except Exception:
    docx2txt = None

# -----------------------------------------------------------------------------
# NLTK data (cached)
# -----------------------------------------------------------------------------
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)

download_nltk_data()

# -----------------------------------------------------------------------------
# Streamlit config + CSS
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Resume Screening System", page_icon="📄", layout="wide")
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1E88E5; text-align: center; padding: 12px 0; }
    .sub-header { font-size: 1.25rem; font-weight: 600; color: #333; margin-top: 18px; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Utilities: file parsing
# -----------------------------------------------------------------------------
def parse_pdf(file):
    if pdfplumber is None:
        st.warning("pdfplumber not available; can't parse PDF files. Install: pip install pdfplumber")
        return ""
    try:
        with pdfplumber.open(file) as pdf:
            return "".join(page.extract_text() or "" for page in pdf.pages)
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

def parse_docx(file):
    if docx2txt is None:
        st.warning("docx2txt not available; can't parse DOCX. Install: pip install docx2txt")
        return ""
    try:
        return docx2txt.process(file)
    except Exception as e:
        st.error(f"Error reading DOCX: {e}")
        return ""

def parse_txt(file):
    try:
        return file.read().decode('utf-8')
    except Exception:
        try:
            return file.read().decode('latin-1')
        except Exception as e:
            st.error(f"Error reading TXT: {e}")
            return ""

def extract_text_from_file(uploaded_file):
    ext = uploaded_file.name.lower().split('.')[-1]
    if ext == 'pdf':
        return parse_pdf(uploaded_file)
    elif ext in ('docx', 'doc'):
        return parse_docx(uploaded_file)
    elif ext in ('txt',):
        return parse_txt(uploaded_file)
    else:
        # try reading as binary text just in case
        try:
            return uploaded_file.getvalue().decode('utf-8', errors='ignore')
        except Exception:
            return ""

# -----------------------------------------------------------------------------
# Heuristic: is this a resume?
# - prevents classifying README/other non-resume documents.
# -----------------------------------------------------------------------------
def is_likely_resume(text: str, filename: str = "") -> bool:
    if not text or len(text.strip()) < 200:
        return False
    name = filename.lower()
    # quick filename blocklist
    if any(x in name for x in ("readme", "license", "setup", "guide", "manual", "changelog", ".md")):
        return False
    t = text.lower()
    keywords = ["education", "experience", "skills", "projects", "certification", "summary", "objective", "work experience", "contact", "degree"]
    found = sum(1 for k in keywords if k in t)
    # require at least two resume-like keywords
    return found >= 2 or len(t.split()) > 400

# -----------------------------------------------------------------------------
# Text preprocessing
# -----------------------------------------------------------------------------
def clean_resume(text: str) -> str:
    text = str(text)
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\b\d{10}\b|\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '', text)
    text = re.sub(r'[^A-Za-z0-9\s\.\,\-\n]', ' ', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_text(text: str) -> str:
    text = clean_resume(text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)

# -----------------------------------------------------------------------------
# Robust XGBoost prediction helper (handles Booster & wrapper cases)
# -----------------------------------------------------------------------------
def xgb_predict_proba(model_or_booster, X):
    """
    X: 2D array-like or sparse matrix (shape [n_samples, n_features])
    model_or_booster: either xgb.XGBClassifier (wrapper) or xgb.Booster
    Returns: (preds_array, proba_array)
      preds_array: shape (n_samples,), dtype int (class indexes)
      proba_array: shape (n_samples, n_classes)
    """
    # If model has sklearn API, use it (preferred)
    if hasattr(model_or_booster, "predict_proba"):
        proba = model_or_booster.predict_proba(X)
        preds = np.argmax(proba, axis=1) if proba.ndim > 1 else (proba > 0.5).astype(int)
        return preds, proba

    # Otherwise get Booster
    booster = None
    if isinstance(model_or_booster, xgb.Booster):
        booster = model_or_booster
    elif hasattr(model_or_booster, "get_booster"):
        booster = model_or_booster.get_booster()
    else:
        raise ValueError("Provided model is neither XGBClassifier nor Booster.")

    # Convert X to numpy 2D array
    if hasattr(X, "toarray"):
        arr = X.toarray()
    else:
        arr = np.asarray(X)

    dmat = xgb.DMatrix(arr)
    raw = booster.predict(dmat)  # shape: (n_samples,) for binary or (n_samples, n_class) for multiclass

    if raw.ndim == 1:
        # binary: raw is probability for class 1
        proba = np.vstack([1.0 - raw, raw]).T
    else:
        proba = raw

    preds = np.argmax(proba, axis=1)
    return preds, proba

# -----------------------------------------------------------------------------
# Model loading (robust)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_models():
    """Load label encoder, vectorizers, classifiers, SBERT and skills dict."""
    try:
        # Label encoder & vectorizer
        label_encoder = joblib.load("label_encoder.pkl")
        tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")

        # Prefer the fixed tfidf model if present
        model_tfidf = None
        for name in ("best_model_tfidf_fixed.pkl", "best_model_tfidf.pkl"):
            if Path(name).exists():
                model_tfidf = joblib.load(name)
                break
        if model_tfidf is None:
            raise FileNotFoundError("No best_model_tfidf*.pkl found in project folder.")

        # sbert-trained classifier (may be XGBoost or sklearn)
        model_sbert = None
        for name in ("best_model_sbert.pkl", "best_model_sbert_fixed.pkl"):
            if Path(name).exists():
                model_sbert = joblib.load(name)
                break
        if model_sbert is None:
            # not fatal: the app can still run for TF-IDF only.
            st.warning("best_model_sbert.pkl not found — SBERT-based classification will be unavailable.")
        
        # Load the SBERT transformer (embedding)
        sbert_name = "all-MiniLM-L6-v2"
        # if you have sbert_model_name.txt in repo, use it
        if Path("sbert_model_name.txt").exists():
            try:
                sbert_name = Path("sbert_model_name.txt").read_text().strip()
            except Exception:
                pass
        # ensure the caching folder is the local ./models directory
        sbert = SentenceTransformer(sbert_name, cache_folder=str(Path("models").resolve()))

        # skills dictionary
        skills_dict = {}
        if Path("skills_dict.json").exists():
            skills_dict = json.load(open("skills_dict.json", "r", encoding="utf8"))

        return {
            "label_encoder": label_encoder,
            "tfidf_vectorizer": tfidf_vectorizer,
            "model_tfidf": model_tfidf,
            "model_sbert": model_sbert,
            "sbert_model": sbert,
            "skills_dict": skills_dict,
        }

    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.info("Ensure required .pkl/.json files exist (label_encoder.pkl, tfidf_vectorizer.pkl, best_model_tfidf*.pkl).")
        return None

# -----------------------------------------------------------------------------
# Predict function (uses robust XGBoost fallback)
# -----------------------------------------------------------------------------
def predict_category(resume_text: str, models: dict):
    X_text = preprocess_text(resume_text)

    # TF-IDF features
    tfidf_vec = models["tfidf_vectorizer"].transform([X_text])  # shape (1, n_features)

    # TF-IDF model prediction (handle XGBoost booster)
    model_tfidf = models["model_tfidf"]
    if isinstance(model_tfidf, (xgb.XGBClassifier, xgb.Booster)) or not hasattr(model_tfidf, "predict_proba"):
        try:
            preds_tfidf, proba_tfidf = xgb_predict_proba(model_tfidf, tfidf_vec)
            pred_tfidf = int(preds_tfidf[0])
            proba_tfidf = proba_tfidf[0]
        except Exception as e:
            # final fallback to sklearn API if available
            if hasattr(model_tfidf, "predict"):
                pred_tfidf = int(model_tfidf.predict(tfidf_vec)[0])
                if hasattr(model_tfidf, "predict_proba"):
                    proba_tfidf = model_tfidf.predict_proba(tfidf_vec)[0]
                else:
                    proba_tfidf = np.array([1.0])  # fallback
            else:
                raise
    else:
        pred_tfidf = int(model_tfidf.predict(tfidf_vec)[0])
        proba_tfidf = model_tfidf.predict_proba(tfidf_vec)[0]

    # SBERT model (if available)
    model_sbert = models.get("model_sbert", None)
    if model_sbert is None:
        pred_sbert = pred_tfidf
        proba_sbert = proba_tfidf
    else:
        # embed the preprocessed text using sbert transformer
        sbert_emb = models["sbert_model"].encode([X_text])
        if isinstance(model_sbert, (xgb.XGBClassifier, xgb.Booster)) or not hasattr(model_sbert, "predict_proba"):
            preds_sbert, proba_sbert = xgb_predict_proba(model_sbert, sbert_emb)
            pred_sbert = int(preds_sbert[0])
            proba_sbert = proba_sbert[0]
        else:
            pred_sbert = int(model_sbert.predict(sbert_emb)[0])
            proba_sbert = model_sbert.predict_proba(sbert_emb)[0]

    # use SBERT prediction as primary if available
    label_encoder = models["label_encoder"]
    try:
        category_sbert = label_encoder.inverse_transform([pred_sbert])[0]
    except Exception:
        # safe fallback mapping in case label encoder mismatch
        classes = list(label_encoder.classes_) if hasattr(label_encoder, "classes_") else []
        category_sbert = classes[pred_sbert] if (classes and pred_sbert < len(classes)) else str(pred_sbert)

    try:
        category_tfidf = label_encoder.inverse_transform([pred_tfidf])[0]
    except Exception:
        classes = list(label_encoder.classes_) if hasattr(label_encoder, "classes_") else []
        category_tfidf = classes[pred_tfidf] if (classes and pred_tfidf < len(classes)) else str(pred_tfidf)

    return {
        "category": category_sbert,
        "confidence": float(max(proba_sbert) * 100) if np.ndim(proba_sbert) > 0 else float(proba_sbert * 100),
        "category_tfidf": category_tfidf,
        "confidence_tfidf": float(max(proba_tfidf) * 100) if np.ndim(proba_tfidf) > 0 else float(proba_tfidf * 100),
    }

# -----------------------------------------------------------------------------
# Skill extraction
# -----------------------------------------------------------------------------
def extract_skills(text: str, skills_dict: dict):
    t = text.lower()
    all_skills = [s for skills in skills_dict.values() for s in skills]
    found = {s for s in all_skills if re.search(r'\b' + re.escape(s.lower()) + r'\b', t)}
    return sorted(found)

# -----------------------------------------------------------------------------
# Main Streamlit UI
# -----------------------------------------------------------------------------
def main():
    st.markdown('<div class="main-header">📄 Automated Resume Screening System</div>', unsafe_allow_html=True)
    with st.spinner("Loading models..."):
        models = load_models()
    if models is None:
        st.stop()
    st.success("✅ Models loaded successfully!")

    # Sidebar
    st.sidebar.title("Navigation")
    mode = st.sidebar.selectbox("Choose Mode", ["Single Resume Analysis", "Bulk Resume Screening", "Job Description Matching", "About"])

    if mode == "Single Resume Analysis":
        st.markdown('<div class="sub-header">Single Resume Analysis</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("Upload resume (PDF/DOCX/TXT)", type=["pdf", "docx", "txt"])
        resume_text = st.text_area("Or paste resume text here", height=300)

        if st.button("Analyze Resume"):
            # prefer uploaded file if present
            if uploaded:
                resume_text = extract_text_from_file(uploaded)
            if not resume_text or not resume_text.strip():
                st.error("Please provide resume text or upload a file.")
            else:
                # quick resume check
                if not is_likely_resume(resume_text, uploaded.name if uploaded else ""):
                    st.warning("This document doesn't look like a resume (README/manual or too short). App will not classify it.")
                    st.info("If this is a resume, paste the content into the text box or try a clearer resume file.")
                else:
                    with st.spinner("Predicting..."):
                        result = predict_category(resume_text, models)
                        skills = extract_skills(resume_text, models.get("skills_dict", {}))
                    st.markdown("### Results")
                    st.write(f"**Predicted Category:** {result['category']}")
                    st.write(f"**Confidence:** {result['confidence']:.1f}%")
                    st.write(f"**TF-IDF category:** {result['category_tfidf']} ({result['confidence_tfidf']:.1f}%)")
                    st.write("**Extracted skills:**", ", ".join(skills) if skills else "None found")

    elif mode == "Bulk Resume Screening":
        st.markdown('<div class="sub-header">Bulk Resume Screening</div>', unsafe_allow_html=True)
        uploaded_files = st.file_uploader("Upload multiple resumes", type=["pdf", "docx", "txt"], accept_multiple_files=True)
        if uploaded_files and st.button("Process All"):
            rows = []
            for f in uploaded_files:
                txt = extract_text_from_file(f)
                if not is_likely_resume(txt, f.name):
                    rows.append({"File": f.name, "Category": "Not a resume", "Confidence": ""})
                    continue
                res = predict_category(txt, models)
                rows.append({"File": f.name, "Category": res["category"], "Confidence": f"{res['confidence']:.1f}%"})
            st.dataframe(pd.DataFrame(rows))

    elif mode == "Job Description Matching":
        st.markdown('<div class="sub-header">Job Description Matching</div>', unsafe_allow_html=True)
        resumes = st.file_uploader("Upload resumes (for matching)", type=["pdf", "docx", "txt"], accept_multiple_files=True)
        jd_text = st.text_area("Paste job description", height=200)
        if resumes and jd_text and st.button("Match JD"):
            rows = []
            for f in resumes:
                txt = extract_text_from_file(f)
                if not txt.strip():
                    continue
                score = 0.0
                try:
                    resume_emb = models["sbert_model"].encode([preprocess_text(txt)])
                    jd_emb = models["sbert_model"].encode([preprocess_text(jd_text)])
                    score = float(cosine_similarity(resume_emb, jd_emb)[0][0] * 100)
                except Exception as e:
                    st.warning(f"Could not compute SBERT similarity for {f.name}: {e}")
                rows.append({"File": f.name, "Match Score %": round(score, 2)})
            st.dataframe(pd.DataFrame(rows).sort_values("Match Score %", ascending=False))

    else:
        st.markdown('<div class="sub-header">About</div>', unsafe_allow_html=True)
        st.write("Automated Resume Screening using TF-IDF + SBERT + robust XGBoost support.")
        st.write("This version includes heuristics to avoid classifying non-resume files (like README).")

if __name__ == "__main__":
    main()
