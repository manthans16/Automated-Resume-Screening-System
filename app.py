# app.py
# ============================================================================ 
# Automated Resume Screening System - Streamlit App (robust + safe defaults)
# -----------------------------------------------------------------------------
# This is a cleaned, self-contained, and robust rewrite of your app.py.
# - Tries to load local model files if present.
# - Works gracefully if some models are missing (TF-IDF-only or SBERT-only).
# - Includes heuristics to avoid classifying README/manual files as resumes.
# - Robust handling for XGBoost Booster vs sklearn wrapper.
#
# Place this file at the root of your project and run:
#    streamlit run app.py
# ============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import re
from pathlib import Path
import os
import xgboost as xgb
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# NLP tooling
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Optional file parsing libs
try:
    import pdfplumber
except Exception:
    pdfplumber = None
try:
    import docx2txt
except Exception:
    docx2txt = None

# -----------------------------------------------------------------------------
# NLTK: download minimal corpora once (cached)
# -----------------------------------------------------------------------------
@st.cache_resource
def ensure_nltk():
    # download as needed
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)


ensure_nltk()

# -----------------------------------------------------------------------------
# UI config + small CSS
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Resume Screening System", page_icon="📄", layout="wide")
st.markdown(
    """
    <style>
    .main-header { font-size: 2.1rem; font-weight: 700; color: #1E88E5; text-align:center; padding:12px 0; }
    .sub-header { font-size: 1.1rem; font-weight:600; color:#333; margin-top:12px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# File parsing helpers
# -----------------------------------------------------------------------------
def parse_pdf(uploaded_file):
    if pdfplumber is None:
        st.warning("pdfplumber not installed — PDF parsing disabled. Install: pip install pdfplumber")
        return ""
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            texts = [p.extract_text() or "" for p in pdf.pages]
        return "\n".join(texts)
    except Exception as e:
        st.error(f"PDF parsing error: {e}")
        return ""

def parse_docx(uploaded_file):
    if docx2txt is None:
        st.warning("docx2txt not installed — DOCX parsing disabled. Install: pip install docx2txt")
        return ""
    try:
        return docx2txt.process(uploaded_file)
    except Exception as e:
        st.error(f"DOCX parsing error: {e}")
        return ""

def parse_txt(uploaded_file):
    try:
        raw = uploaded_file.read()
        # try utf-8, fallback latin1
        try:
            return raw.decode("utf-8")
        except Exception:
            return raw.decode("latin-1")
    except Exception as e:
        st.error(f"TXT parsing error: {e}")
        return ""

def extract_text_from_file(uploaded_file):
    fname = uploaded_file.name.lower()
    ext = fname.split(".")[-1]
    if ext == "pdf":
        return parse_pdf(uploaded_file)
    if ext in ("docx", "doc"):
        return parse_docx(uploaded_file)
    if ext in ("txt", "text"):
        return parse_txt(uploaded_file)
    # fallback: raw bytes decoded
    try:
        return uploaded_file.getvalue().decode("utf-8", errors="ignore")
    except Exception:
        return ""

# -----------------------------------------------------------------------------
# Heuristic to detect resume-like documents (avoid README classification)
# -----------------------------------------------------------------------------
def is_likely_resume(text: str, filename: str = "") -> bool:
    if not text or len(text.strip()) < 200:
        return False
    name = filename.lower() if filename else ""
    # common non-resume filename indicators
    if any(k in name for k in ["readme", "license", "setup", "guide", "manual", "changelog", ".md"]):
        return False
    t = text.lower()
    keywords = [
        "education", "experience", "skills", "projects", "certification",
        "work experience", "summary", "objective", "contact", "degree", "linkedin", "github"
    ]
    found = sum(1 for k in keywords if k in t)
    # at least two resume keywords OR longer document
    return found >= 2 or len(t.split()) > 400

# -----------------------------------------------------------------------------
# Text preprocessing & cleaning
# -----------------------------------------------------------------------------
def clean_text(text: str) -> str:
    text = str(text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[^A-Za-z0-9\s\.\,\-\n]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()

def preprocess_text(text: str) -> str:
    text = clean_text(text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and len(t) > 2]
    return " ".join(tokens)

# -----------------------------------------------------------------------------
# Robust XGBoost prediction helper for wrapper vs raw Booster
# -----------------------------------------------------------------------------
def xgb_predict_proba(model, X):
    """
    Accepts:
      - xgb.XGBClassifier (sklearn wrapper)
      - xgb.Booster
      - any object with predict_proba
    X should be 2D array-like (n_samples, n_features) or sparse matrix.
    Returns (preds, proba) where:
      preds: np.array shape (n_samples,)
      proba: np.array shape (n_samples, n_classes)
    """
    # if model has sklearn API use it
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.ndim == 1:  # sometimes returns shape (n_samples,)
            proba = np.vstack([1 - proba, proba]).T
        preds = np.argmax(proba, axis=1) if proba.ndim > 1 else (proba > 0.5).astype(int)
        return np.array(preds), np.array(proba)

    # if model is Booster or wrapper without sklearn proba
    booster = None
    if isinstance(model, xgb.Booster):
        booster = model
    elif hasattr(model, "get_booster"):
        booster = model.get_booster()
    else:
        raise ValueError("Model must be XGBClassifier or Booster or have predict_proba.")

    # convert X
    if hasattr(X, "toarray"):
        arr = X.toarray()
    else:
        arr = np.asarray(X)

    dmat = xgb.DMatrix(arr)
    raw = booster.predict(dmat)  # shape (n_samples,) for binary or (n_samples, n_class) for multiclass
    raw = np.asarray(raw)
    if raw.ndim == 1:
        proba = np.vstack([1 - raw, raw]).T
    else:
        proba = raw
    preds = np.argmax(proba, axis=1)
    return preds, proba

# -----------------------------------------------------------------------------
# Model loading (safe, cached)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_models():
    """
    Tries to load local files:
     - label_encoder.pkl
     - tfidf_vectorizer.pkl
     - best_model_tfidf_fixed.pkl or best_model_tfidf.pkl
     - best_model_sbert.pkl (optional)
     - skills_dict.json (optional)
     - sbert_model_name.txt (optional) otherwise uses 'all-MiniLM-L6-v2'
    Returns a dict of loaded objects or None on fatal errors.
    """
    try:
        models = {}
        # label encoder
        if Path("label_encoder.pkl").exists():
            models["label_encoder"] = joblib.load("label_encoder.pkl")
        else:
            st.warning("label_encoder.pkl not found. Category decoding will be limited.")
            models["label_encoder"] = None

        # tfidf vectorizer
        if Path("tfidf_vectorizer.pkl").exists():
            models["tfidf_vectorizer"] = joblib.load("tfidf_vectorizer.pkl")
        else:
            st.warning("tfidf_vectorizer.pkl not found. TF-IDF based features unavailable.")
            models["tfidf_vectorizer"] = None

        # TF-IDF classifier (prefer fixed)
        model_tfidf = None
        for fname in ("best_model_tfidf_fixed.pkl", "best_model_tfidf.pkl"):
            if Path(fname).exists():
                try:
                    model_tfidf = joblib.load(fname)
                    break
                except Exception as e:
                    st.warning(f"Could not load {fname}: {e}")
        models["model_tfidf"] = model_tfidf

        # SBERT-trained classifier (optional)
        model_sbert = None
        for fname in ("best_model_sbert.pkl", "best_model_sbert_fixed.pkl"):
            if Path(fname).exists():
                try:
                    model_sbert = joblib.load(fname)
                    break
                except Exception as e:
                    st.warning(f"Could not load {fname}: {e}")
        models["model_sbert"] = model_sbert

        # SBERT embedding model name
        sbert_name = "all-MiniLM-L6-v2"
        if Path("sbert_model_name.txt").exists():
            try:
                sbert_name = Path("sbert_model_name.txt").read_text().strip()
            except Exception:
                pass

        # load SentenceTransformer (caches inside ./models folder if present)
        try:
            # use local ./models cache folder if exists
            cache_folder = str(Path("models").resolve())
            models["sbert_model"] = SentenceTransformer(sbert_name, cache_folder=cache_folder)
        except Exception as e:
            st.warning(f"Could not load SentenceTransformer '{sbert_name}': {e}")
            models["sbert_model"] = None

        # skills dictionary (optional)
        if Path("skills_dict.json").exists():
            try:
                with open("skills_dict.json", "r", encoding="utf8") as f:
                    models["skills_dict"] = json.load(f)
            except Exception:
                models["skills_dict"] = {}
        else:
            models["skills_dict"] = {}

        return models

    except Exception as e:
        st.error(f"Fatal error loading models: {e}")
        return None

# -----------------------------------------------------------------------------
# Prediction wrapper that gracefully handles missing parts
# -----------------------------------------------------------------------------
def predict_category(resume_text: str, models: dict):
    """
    Returns dict:
      - category: str (final category)
      - confidence: float (0-100) for chosen model (prefer SBERT if available)
      - category_tfidf, confidence_tfidf
    """
    # preprocess
    pre = preprocess_text(resume_text)

    # TF-IDF path
    tfidf_pred = None
    tfidf_proba = None
    if models.get("tfidf_vectorizer") is not None and models.get("model_tfidf") is not None:
        try:
            X_tfidf = models["tfidf_vectorizer"].transform([pre])
            mt = models["model_tfidf"]
            if hasattr(mt, "predict_proba") or not isinstance(mt, xgb.Booster):
                # try sklearn-like
                if hasattr(mt, "predict_proba"):
                    tfidf_proba = mt.predict_proba(X_tfidf)[0]
                    tfidf_pred = int(np.argmax(tfidf_proba))
                else:
                    # fallback predict only
                    tfidf_pred = int(mt.predict(X_tfidf)[0])
                    tfidf_proba = np.array([1.0])
            else:
                # booster path
                preds, proba = xgb_predict_proba(mt, X_tfidf)
                tfidf_pred = int(preds[0])
                tfidf_proba = proba[0]
        except Exception as e:
            st.warning(f"TF-IDF prediction failed: {e}")

    # SBERT path
    sbert_pred = None
    sbert_proba = None
    if models.get("sbert_model") is not None and models.get("model_sbert") is not None:
        try:
            emb = models["sbert_model"].encode([pre])
            ms = models["model_sbert"]
            if hasattr(ms, "predict_proba"):
                sbert_proba = ms.predict_proba(emb)[0]
                sbert_pred = int(np.argmax(sbert_proba))
            else:
                preds, proba = xgb_predict_proba(ms, emb)
                sbert_pred = int(preds[0])
                sbert_proba = proba[0]
        except Exception as e:
            st.warning(f"SBERT-prediction failed: {e}")

    # Decide final prediction: prefer SBERT if available else TF-IDF else unknown
    final_pred_index = None
    final_proba = None
    final_method = "None"
    if sbert_pred is not None:
        final_pred_index = sbert_pred
        final_proba = sbert_proba
        final_method = "SBERT"
    elif tfidf_pred is not None:
        final_pred_index = tfidf_pred
        final_proba = tfidf_proba
        final_method = "TF-IDF"

    # decode labels
    label_encoder = models.get("label_encoder", None)
    def decode(idx):
        if label_encoder is None:
            return str(idx)
        try:
            return label_encoder.inverse_transform([idx])[0]
        except Exception:
            # fallback to classes_ if possible
            if hasattr(label_encoder, "classes_"):
                classes = list(label_encoder.classes_)
                return classes[idx] if 0 <= idx < len(classes) else str(idx)
            return str(idx)

    category = decode(final_pred_index) if final_pred_index is not None else "Unknown"
    confidence = float(max(final_proba) * 100) if final_proba is not None else 0.0

    category_tfidf = decode(tfidf_pred) if tfidf_pred is not None else "N/A"
    conf_tfidf = float(max(tfidf_proba) * 100) if tfidf_proba is not None else 0.0

    return {
        "category": category,
        "confidence": confidence,
        "method": final_method,
        "category_tfidf": category_tfidf,
        "confidence_tfidf": conf_tfidf,
    }

# -----------------------------------------------------------------------------
# Skills extraction (simple dictionary lookup)
# -----------------------------------------------------------------------------
def extract_skills(text: str, skills_dict: dict):
    t = text.lower()
    all_skills = []
    for k, lst in skills_dict.items():
        all_skills.extend(lst)
    found = sorted({s for s in all_skills if re.search(r"\b" + re.escape(s.lower()) + r"\b", t)})
    return found

# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------
def main():
    st.markdown('<div class="main-header">📄 Automated Resume Screening System</div>', unsafe_allow_html=True)
    with st.spinner("Loading models & resources..."):
        models = load_models()
    if models is None:
        st.error("Models failed to load. Check the console and ensure required files exist.")
        st.stop()

    st.success("✅ Models loaded (some features may be disabled if files are missing).")

    # Sidebar
    st.sidebar.title("Navigation")
    mode = st.sidebar.selectbox("Choose mode", ["Single Resume Analysis", "Bulk Resume Screening", "Job Description Matching", "About"])

    # Single resume mode
    if mode == "Single Resume Analysis":
        st.markdown('<div class="sub-header">Single Resume Analysis</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("Upload resume (PDF / DOCX / TXT)", type=["pdf", "docx", "txt"])
        resume_text = st.text_area("Or paste resume text here", height=300)

        if st.button("Analyze Resume"):
            if uploaded:
                resume_text = extract_text_from_file(uploaded)
            if not resume_text or not resume_text.strip():
                st.error("No resume text provided.")
                return

            # run heuristic
            if not is_likely_resume(resume_text, uploaded.name if uploaded else ""):
                st.warning("This document does not appear to be a resume (README/manual or too short). The app will not classify it.")
                st.info("If this is a resume, paste the text into the box or upload a clearer resume file.")
                return

            with st.spinner("Predicting..."):
                result = predict_category(resume_text, models)
                skills = extract_skills(resume_text, models.get("skills_dict", {}))

            st.markdown("### Results")
            st.write(f"**Predicted Category:** {result['category']}  (method: {result['method']})")
            st.write(f"**Confidence:** {result['confidence']:.1f}%")
            st.write(f"**TF-IDF category:** {result['category_tfidf']}  ({result['confidence_tfidf']:.1f}%)")
            st.write("**Extracted skills:**", ", ".join(skills) if skills else "None found")

    # Bulk screening
    elif mode == "Bulk Resume Screening":
        st.markdown('<div class="sub-header">Bulk Resume Screening</div>', unsafe_allow_html=True)
        uploaded_files = st.file_uploader("Upload multiple resumes", type=["pdf", "docx", "txt"], accept_multiple_files=True)

        if uploaded_files and st.button("Process All"):
            rows = []
            for f in uploaded_files:
                txt = extract_text_from_file(f)
                if not txt or not txt.strip():
                    rows.append({"File": f.name, "Category": "Empty / unreadable", "Confidence": ""})
                    continue
                if not is_likely_resume(txt, f.name):
                    rows.append({"File": f.name, "Category": "Not a resume", "Confidence": ""})
                    continue
                res = predict_category(txt, models)
                rows.append({"File": f.name, "Category": res["category"], "Confidence": f"{res['confidence']:.1f}%"})
            st.dataframe(pd.DataFrame(rows))

    # Job Description Matching
    elif mode == "Job Description Matching":
        st.markdown('<div class="sub-header">Job Description Matching</div>', unsafe_allow_html=True)
        resumes = st.file_uploader("Upload resumes", type=["pdf", "docx", "txt"], accept_multiple_files=True)
        jd_text = st.text_area("Paste job description", height=200)

        if resumes and jd_text and st.button("Match JD"):
            rows = []
            for f in resumes:
                txt = extract_text_from_file(f)
                if not txt or not txt.strip():
                    continue
                score = 0.0
                if models.get("sbert_model") is not None:
                    try:
                        r_emb = models["sbert_model"].encode([preprocess_text(txt)])
                        j_emb = models["sbert_model"].encode([preprocess_text(jd_text)])
                        score = float(cosine_similarity(r_emb, j_emb)[0][0] * 100)
                    except Exception as e:
                        st.warning(f"SBERT similarity failed for {f.name}: {e}")
                else:
                    st.warning("SBERT model not available — JD matching requires SBERT.")
                rows.append({"File": f.name, "Match Score %": round(score, 2)})
            if rows:
                st.dataframe(pd.DataFrame(rows).sort_values("Match Score %", ascending=False))
            else:
                st.info("No valid resumes to score.")

    # About
    else:
        st.markdown('<div class="sub-header">About</div>', unsafe_allow_html=True)
        st.write("Automated Resume Screening using TF-IDF + SBERT and robust XGBoost handling.")
        st.write("This app includes heuristics to avoid classifying non-resume files (like README).")
        st.write("If some model files are missing the app will still run with reduced functionality (TF-IDF-only or SBERT-only).")

if __name__ == "__main__":
    main()
