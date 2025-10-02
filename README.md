# Automated Resume Screening and Classification System

A resume screening system built with Machine Learning and NLP that automatically classifies resumes into job categories, extracts skills, and matches candidates to job descriptions using TF-IDF and SBERT embeddings. Includes an interactive Streamlit app for HR teams to analyze single or bulk resumes and rank candidates by relevance.
### Live Demo: [resume-screening-system-ms23.streamlit.app](https://resume-screening-system-ms23.streamlit.app/)



## Project Overview
This project implements a resume screening system that:
- Predicts job categories from resume text with ~85-90% accuracy
- Extracts technical and soft skills automatically
- Matches resumes to job descriptions using semantic similarity
- Ranks candidates based on relevance
- Provides an interactive web interface for HR teams

## Architecture
```
├── project.ipynb              # Complete ML pipeline and analysis
├── app.py                     # Streamlit web application
├── UpdatedResumeDataSet.csv   # Dataset (Category, Resume columns)
├── requirements.txt           # Python dependencies
├── models/                    # Trained models (generated)
│   ├── label_encoder.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── best_model_tfidf.pkl
│   ├── best_model_sbert.pkl
│   └── skills_dict.json
└── README.md                  # This file
```

##  Quick Start
### Prerequisites
- Python 3.8+
- pip package manager
- 4GB+ RAM (for transformer models)

### Installation
```bash
# 1. Clone the repo
git clone <your-repo-url>
cd resume-screening-system

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download spaCy model
python -m spacy download en_core_web_sm

# 4. (Optional) Run Jupyter Notebook to retrain models
Jupyter Notebook project.ipynb

# 5. Launch the Streamlit app
streamlit run app.py
```
The app will open in your browser at `http://localhost:8501`

## Requirements
`requirements.txt` should include:
```
pandas==2.0.3
numpy==1.24.3
matplotlib==3.7.2
seaborn==0.12.2
plotly==5.15.0
scikit-learn==1.3.0
xgboost==1.7.6
nltk==3.8.1
spacy==3.6.0
sentence-transformers==2.2.2
transformers==4.30.2
wordcloud==1.9.2
joblib==1.3.1
streamlit==1.25.0
pdfplumber==0.9.0
docx2txt==0.8
```

## Dataset Format
Your `UpdatedResumeDataSet.csv` should have:

| Column   | Description                           |
|----------|---------------------------------------|
| Category | Job category (Data Science, HR, etc.) |
| Resume   | Raw resume text                       |

Example:
```csv
Category, Resume
"Data Science", "Experienced data scientist with Python, ML..."
"Java Developer" ,"5 years Java development, Spring Boot..."
```

##  Machine Learning Pipeline
### 1. Data Preprocessing
- Text cleaning (remove URLs, emails, special chars)
- Tokenization and lemmatization
- Stopword removal

### 2. Feature Engineering
- **TF-IDF Vectorization**: Traditional approach
- **Sentence-BERT Embeddings**: State-of-the-art semantic representations

### 3. Models Trained
- Multinomial Naive Bayes (baseline)
- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest
- XGBoost (best performer)
- Neural models with SBERT embeddings

### 4. Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score
- Confusion matrices
- Cross-validation scores

### 5. Advanced Features
- **Skills Extraction**: 100+ technical skills database
- **JD Matching**: Cosine similarity with SBERT
- **Candidate Ranking**: Automated shortlisting

##  Streamlit App Features
### Mode 1: Single Resume Analysis
- Upload or paste an individual resume
- Get predicted job category with confidence
- Extract technical and soft skills
- Resume quality assessment

### Mode 2: Bulk Resume Screening
- Upload multiple resumes (PDF/DOCX/TXT)
- Batch processing with progress tracking
- Category distribution visualization
- Export results to CSV

### Mode 3: Job Description Matching
- Upload resumes + job description
- Calculate semantic similarity scores
- Rank candidates by relevance
- Identify matched vs missing skills
- Download shortlisted candidates

### Mode 4: About
- System information
- Model statistics
- Usage instructions

## 📈 Model Performance
| Model                    | Accuracy | F1-Score |
|--------------------------|----------|----------|
| Naive Bayes (TF-IDF)     | ~75%     | ~73%     |
| Logistic Regression (TF-IDF) | ~85% | ~84%     |
| SVM (TF-IDF)             | ~84%     | ~83%     |
| Random Forest (TF-IDF)   | ~82%     | ~81%     |
| **XGBoost (TF-IDF)**     | **~88%** | **~87%** |
| Random Forest (SBERT)    | ~86%     | ~85%     |

*Note: Results vary based on dataset quality and size*

## Deployment Options
### Option 1: Streamlit Cloud (Easiest)
1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect GitHub repo
4. Deploy with one click
5. Share the public URL

**Deployment Settings:**
```toml
# .streamlit/config.toml
[server]
headless = true
port = 8501
enableCORS = false
```
