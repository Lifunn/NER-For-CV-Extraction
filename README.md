# Named Entity Recognition for CV Extraction

This repository implements an automated system for extracting structured information from Curriculum Vitae (CV) using **Named Entity Recognition (NER)**.  
Developed as part of a research project at **Institut Teknologi Sepuluh Nopember (ITS)**.

---

## 🚀 Overview

Manual data extraction from CVs is often time-consuming and error-prone.  
This project automates that process using **SpaCy Transformer-based NER** with a **semi-supervised learning** approach (pseudo-labelling).  

The system identifies 12 entity types:

| Entity | Description |
|---------|-------------|
| NAME | Candidate name |
| MAIL | Email address |
| PHONE | Phone number |
| EDU | Institution name |
| DEGREE | Degree or study program |
| ORG | Organization or company |
| ROLE | Role or job title |
| DURATION | Time period (start–end) |
| DAT | Single date |
| SKILL | Skills or competencies |
| LANG | Language proficiency |
| ACH | Achievements, awards, or certifications |

---

## 🧠 Architecture

- **Base Model:** SpaCy Transformer (RoBERTa / XLM-RoBERTa)
- **Training Strategy:** Semi-supervised (manual annotation + pseudo-labelling)
- **Optimization:** Bayesian hyperparameter sweep (Weights & Biases)
- **Evaluation Metrics:** Precision, Recall, and F1-Score

```yaml
batch_size: 64
learning_rate: 3.18e-4
dropout: 0.07
optimizer:
  L2: 3.16e-6
  grad_clip: 4.91
epochs: 1500
```

---

## 📊 Results Summary

| Model | Precision | Recall | F1-Score |
|--------|------------|---------|-----------|
| **BERT** | 0.60 | 0.68 | **0.64** |
| **Gliner** | 0.45 | 0.39 | 0.37 |

- Best-performing configuration: **F1 = 0.662**
- Optimal dropout: ~0.07  
- Best batch size: 64

---

## ⚙️ Key Components

### `src/data_preprocessing.py`
- Loads JSON annotations  
- Cleans and trims entity spans  
- Converts to SpaCy `.spacy` binary format

### `src/pseudo_labelling.py`
- Loads base NER model to generate pseudo-labels  
- Exports predictions in Label Studio JSON format

### `src/ner_training.py`
- Handles model training and W&B sweep tuning  
- Trains SpaCy Transformer NER pipeline  
- Saves model to `output/model-last`

### `src/inference.py`
- Extracts text from PDFs using **PyMuPDF**  
- Performs NER inference  
- Optionally integrates with **Gemini 2.5 Flash** (via OpenRouter) to evaluate candidate suitability

---

## 🤖 LLM Integration Example

```python
posisi_yang_dilamar = "Staff Divisi Sponsor"
summary = get_recruiter_summary(posisi_yang_dilamar, ner_results_json)
print(summary)
```

Output format:
```
Kandidat [Nama Kandidat] dinilai COCOK / DAPAT DIPERTIMBANGKAN / TIDAK COCOK
```

---

## 🧩 Folder Structure

```
ner-cv-extraction/
│
├── README.md
├── LICENSE
├── .gitignore
│
├── configs/
│   ├── config.yaml
│   └── spacy_config.cfg
│
├── src/
│   ├── data_preprocessing.py
│   ├── pseudo_labelling.py
│   ├── ner_training.py
│   ├── inference.py
│   └── utils.py
│
├── examples/
│   ├── sample_cv.pdf
│   └── sample_output.json
│
└── docs/
    └── NER_for_CV_Extraction.pdf
```

---

## 🔒 Data Ethics & Privacy

This repository **does not contain any real or identifiable CV data**.  
All example data in `/examples/` are **synthetic and anonymized**.  
When applying to real datasets, please ensure compliance with institutional data protection and ethical guidelines.

---

## 🧪 Training Example

```bash
python src/ner_training.py
```

For hyperparameter sweep (W&B):

```bash
wandb sweep configs/config.yaml
wandb agent <sweep-id>
```

---

## 📈 Evaluation

```bash
python -m spacy evaluate output/model-best corpus/test_data.spacy
```

---

## 🛠️ Installation

```bash
git clone https://github.com/<your-username>/ner-cv-extraction.git
cd ner-cv-extraction
pip install -r requirements.txt
```

---

## 🧾 License

Released under the **MIT License**.  
You are free to use, modify, and distribute this code for research or academic purposes.
