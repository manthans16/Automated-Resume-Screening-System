# ============================================================================
# Automated Resume Screening System - Streamlit App (Robust & Safe Defaults)
# ============================================================================
import streamlit as st
st.set_page_config(page_title="Resume Screening System", page_icon="📄", layout="wide")

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

try:
    import pdfplumber
except:
    pdfplumber = None
try:
    import docx2txt
except:
    docx2txt = None


# ---------------------------------------------------------------------------
# NLTK resources
# ---------------------------------------------------------------------------
@st.cache_resource
def ensure_nltk():
    try: nltk.data.find("corpora/stopwords")
    except LookupError: nltk.download("stopwords", quiet=True)
    try: nltk.data.find("tokenizers/punkt")
    except LookupError: nltk.download("punkt", quiet=True)
    try: nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)
ensure_nltk()


# ---------------------------------------------------------------------------
# File parsing
# ---------------------------------------------------------------------------
def parse_pdf(uploaded_file):
    if pdfplumber is None: return ""
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            return "\n".join([p.extract_text() or "" for p in pdf.pages])
    except:
        return ""

def parse_docx(uploaded_file):
    if docx2txt is None: return ""
    try:
        return docx2txt.process(uploaded_file)
    except:
        return ""

def parse_txt(uploaded_file):
    try:
        raw = uploaded_file.read()
        return raw.decode("utf-8", errors="ignore")
    except:
        return ""

def extract_text_from_file(uploaded_file):
    ext = uploaded_file.name.lower().split(".")[-1]
    if ext == "pdf": return parse_pdf(uploaded_file)
    if ext in ("docx", "doc"): return parse_docx(uploaded_file)
    if ext in ("txt", "text"): return parse_txt(uploaded_file)
    return uploaded_file.getvalue().decode("utf-8", errors="ignore")


# ---------------------------------------------------------------------------
# Resume heuristic
# ---------------------------------------------------------------------------
def is_likely_resume(text, filename=""):
    if not text or len(text.strip()) < 200: return False
    if any(k in filename.lower() for k in ["readme","license",".md","setup","guide"]): return False
    keywords = ["education","experience","skills","projects","certification","linkedin","github"]
    return sum(1 for k in keywords if k in text.lower()) >= 2 or len(text.split()) > 400


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def clean_text(text):
    text = re.sub(r"http\S+|www\.\S+|\S+@\S+", " ", str(text))
    text = re.sub(r"[^A-Za-z0-9\s]", " ", text).lower()
    return re.sub(r"\s+", " ", text).strip()

def preprocess_text(text):
    tokens = word_tokenize(clean_text(text))
    sw = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    return " ".join([lemmatizer.lemmatize(t) for t in tokens if t not in sw and len(t) > 2])


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------
def safe_load(path):
    if Path(path).exists():
        try:
            return joblib.load(path)
        except Exception as e:
            st.warning(f"⚠️ Could not load {path}: {e}")
    return None

@st.cache_resource
def load_models():
    models = {
        "label_encoder": safe_load("label_encoder.pkl"),
        "tfidf_vectorizer": safe_load("tfidf_vectorizer.pkl"),

        "model_tfidf": safe_load("best_model_tfidf.pkl"),
        "model_sbert": safe_load("best_model_sbert.pkl"),
        "skills_dict": {},
        "sbert_model": None,
    }
    if Path("skills_dict.json").exists():
        with open("skills_dict.json","r",encoding="utf8") as f:
            models["skills_dict"] = json.load(f)

    try:
        models["sbert_model"] = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:
        st.warning(f"⚠️ SBERT not loaded: {e}")
    return models


# ---------------------------------------------------------------------------
# Prediction - return both SBERT & TF-IDF + final decision
# ---------------------------------------------------------------------------
def predict_category(resume_text, models):
    pre = preprocess_text(resume_text)
    le = models.get("label_encoder")

    results = {"pre": pre, "tfidf": None, "sbert": None, "final": None}

    # --- TF-IDF ---
    if models["tfidf_vectorizer"] and models["model_tfidf"]:
        try:
            X_tfidf = models["tfidf_vectorizer"].transform([pre])
            if hasattr(models["model_tfidf"], "predict_proba"):
                proba = models["model_tfidf"].predict_proba(X_tfidf)[0]
                pred = int(np.argmax(proba))
                label = le.inverse_transform([pred])[0] if le is not None else str(pred)
                results["tfidf"] = {"category": label, "confidence": float(max(proba)*100)}
        except Exception as e:
            st.warning(f"TF-IDF prediction failed: {e}")

    # --- SBERT ---
    if models["sbert_model"] and models["model_sbert"]:
        try:
            emb = models["sbert_model"].encode([pre])
            if hasattr(models["model_sbert"], "predict_proba"):
                proba = models["model_sbert"].predict_proba(emb)[0]
                pred = int(np.argmax(proba))
                label = le.inverse_transform([pred])[0] if le is not None else str(pred)
                results["sbert"] = {"category": label, "confidence": float(max(proba)*100)}
        except Exception as e:
            st.warning(f"SBERT prediction failed: {e}")

    # --- Final Combined Decision ---
    if results["sbert"] and results["tfidf"]:
        if results["sbert"]["confidence"] >= results["tfidf"]["confidence"]:
            results["final"] = {"category": results["sbert"]["category"], "method": "SBERT", "confidence": results["sbert"]["confidence"]}
        else:
            results["final"] = {"category": results["tfidf"]["category"], "method": "TF-IDF", "confidence": results["tfidf"]["confidence"]}
    elif results["sbert"]:
        results["final"] = {"category": results["sbert"]["category"], "method": "SBERT", "confidence": results["sbert"]["confidence"]}
    elif results["tfidf"]:
        results["final"] = {"category": results["tfidf"]["category"], "method": "TF-IDF", "confidence": results["tfidf"]["confidence"]}

    return results


# ---------------------------------------------------------------------------
# Skills
# ---------------------------------------------------------------------------
def extract_skills(text, skills_dict):
    t = text.lower()
    return sorted({s for skills in skills_dict.values() for s in skills if s.lower() in t})


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
def main():
    st.markdown('<h1 style="text-align:center;"> Automated Resume Screening System</h1>', unsafe_allow_html=True)

    models = load_models()
    st.success("Models loaded successfully")

    mode = st.sidebar.selectbox("Choose mode", ["Single Resume Analysis","Bulk Resume Screening","Job Description Matching","About"])

    if mode == "Single Resume Analysis":
        uploaded = st.file_uploader("Upload resume (PDF/DOCX/TXT)", type=["pdf","docx","txt"])
        resume_text = st.text_area("Or paste resume text here", height=300)

        if st.button("Analyze Resume"):
            if uploaded: resume_text = extract_text_from_file(uploaded)
            if not resume_text:
                st.error("No resume text provided."); return
            if not is_likely_resume(resume_text, uploaded.name if uploaded else ""):
                st.warning("This file may not be a resume."); return

            results = predict_category(resume_text, models)
            skills = extract_skills(resume_text, models.get("skills_dict",{}))

            if results["sbert"]:
                st.write("### 🏷 SBERT Prediction:", results["sbert"]["category"])
                st.write(" Confidence:", f"{results['sbert']['confidence']:.1f}%")

            if results["tfidf"]:
                st.write("### 🏷 TF-IDF Prediction:", results["tfidf"]["category"])
                st.write("Confidence:", f"{results['tfidf']['confidence']:.1f}%")

            if results["final"]:
                st.subheader("Final Combined Decision")
                st.write(f"**Category:** {results['final']['category']} ({results['final']['method']})")
                st.write(f"**Confidence:** {results['final']['confidence']:.1f}%")

            st.write("### 🛠 Skills:", ", ".join(skills) if skills else "None detected")

            # Debug
            with st.expander("DEBUG (developer only)", expanded=False):
                st.write(" - label_encoder:", type(models.get("label_encoder")), getattr(models.get("label_encoder"), "classes_", None))
                st.write(" - sbert_model:", type(models.get("sbert_model")))
                st.write(" - model_sbert:", type(models.get("model_sbert")))
                st.write(" - tfidf_vectorizer:", type(models.get("tfidf_vectorizer")))
                st.write(" - model_tfidf:", type(models.get("model_tfidf")))
                st.write("Preprocessed text (200 chars):", results["pre"][:200])

    elif mode == "Bulk Resume Screening":
        files = st.file_uploader("Upload multiple resumes", type=["pdf","docx","txt"], accept_multiple_files=True)
        if files and st.button("Process All"):
            rows = []
            for f in files:
                txt = extract_text_from_file(f)
                if not is_likely_resume(txt, f.name):
                    rows.append({"File": f.name,"Category":"Not a resume"}); continue
                res = predict_category(txt, models)
                final = res["final"]["category"] if res["final"] else "Unknown"
                conf = f"{res['final']['confidence']:.1f}%" if res["final"] else "-"
                rows.append({"File":f.name,"Category":final,"Confidence":conf})
            st.dataframe(pd.DataFrame(rows))

    elif mode == "Job Description Matching":
        resumes = st.file_uploader("Upload resumes", type=["pdf","docx","txt"], accept_multiple_files=True)
        jd = st.text_area("Paste job description", height=200)
        if resumes and jd and st.button("Match JD"):
            rows=[]
            for f in resumes:
                txt = extract_text_from_file(f)
                if not txt: continue
                score=0
                if models["sbert_model"]:
                    r_emb = models["sbert_model"].encode([preprocess_text(txt)])
                    j_emb = models["sbert_model"].encode([preprocess_text(jd)])
                    score=float(cosine_similarity(r_emb,j_emb)[0][0]*100)
                rows.append({"File":f.name,"Match %":round(score,2)})
            st.dataframe(pd.DataFrame(rows).sort_values("Match %",ascending=False))

    else:
        st.info(" Automated Resume Screening using TF-IDF + SBERT. \n\nHandles missing models gracefully and avoids misclassifying non-resumes.")


if __name__ == "__main__":
    main()
