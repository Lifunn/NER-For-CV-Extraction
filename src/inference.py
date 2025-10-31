"""
inference.py
------------
Performs Named Entity Recognition (NER) inference on PDF CV files and optionally
analyzes candidate-job fit using a Large Language Model (Gemini via OpenRouter).

Dependencies:
    pip install spacy spacy-transformers PyMuPDF requests pandas
"""

import fitz
import spacy
import json
import pandas as pd
import requests
import os

# === Configuration ===
MODEL_PATH = "./output/model-last"
PDF_PATH = "./examples/sample_cv.pdf"
OUTPUT_JSON = "./examples/sample_output.json"
USE_LLM_ANALYSIS = True  # Set False if you only want NER extraction
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")  # Securely load from env
MODEL_NAME = "google/gemini-2.5-flash-preview-05-20"


# ------------------------- Utility Functions -------------------------

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF using PyMuPDF."""
    print(f"📄 Extracting text from: {pdf_path}")
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    text = " ".join(text.split("\n"))
    return text


def perform_ner_inference(model_path: str, text: str):
    """Run SpaCy NER model on input text."""
    print("🧠 Loading NER model...")
    nlp = spacy.load(model_path)
    doc = nlp(text)

    print(f"🔍 Detected {len(doc.ents)} entities.\n")
    entities = [{"Text": ent.text, "Label": ent.label_} for ent in doc.ents]

    # Convert to JSON-like structure
    ner_json = [
        [ent.text, {"entities": [[ent.start_char, ent.end_char, ent.label_]]}]
        for ent in doc.ents
    ]

    return entities, ner_json


def get_recruiter_summary(job_position: str, ner_results: list) -> str:
    """
    Sends NER results to OpenRouter API for recruiter-style evaluation.
    Returns a critical, formatted suitability summary.
    """
    system_prompt = """
Peran: Anda adalah seorang Asisten AI Rekruter senior yang kritis.
Tugas: Evaluasi kesesuaian kandidat terhadap posisi kerja berdasarkan JSON hasil NER.
Keluarkan hasil dalam salah satu kategori: COCOK, DAPAT DIPERTIMBANGKAN, atau TIDAK COCOK.
Gunakan format seperti:
Kandidat [Nama] dinilai COCOK/DAPAT DIPERTIMBANGKAN/TIDAK COCOK untuk posisi {posisi}.
    """

    user_prompt = f"""
Posisi yang dilamar: {job_position}
JSON Entitas:
{json.dumps(ner_results, indent=2, ensure_ascii=False)}
"""

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    try:
        print("🧩 Calling OpenRouter API for recruiter analysis...")
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            data=json.dumps(payload),
            timeout=60,
        )
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        return f"❌ Error during recruiter summary generation: {e}"


# ------------------------- Main Execution -------------------------

def main():
    text = extract_text_from_pdf(PDF_PATH)
    entities, ner_json = perform_ner_inference(MODEL_PATH, text)

    # Save structured output
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(ner_json, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved extracted entities to {OUTPUT_JSON}")

    # Display sample output
    df = pd.DataFrame(entities)
    print("\n=== Extracted Entities ===")
    print(df.head(15))

    if USE_LLM_ANALYSIS:
        job_position = input("\nMasukkan posisi yang dilamar: ") or "Staff Divisi Sponsor"
        summary = get_recruiter_summary(job_position, ner_json)
        print("\n--- HASIL ANALISIS KESESUAIAN KANDIDAT ---")
        print(summary)


if __name__ == "__main__":
    main()
