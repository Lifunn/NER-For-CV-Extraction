"""
data_preprocessing.py
---------------------
Convert annotated CV JSON data into SpaCy training and validation data.

Steps:
1. Load annotated JSON file
2. Clean invalid entity spans (trim whitespaces)
3. Split data into train/test sets
4. Convert annotations into SpaCy DocBin format
"""

import json
import os
import re
from tqdm import tqdm
from spacy.tokens import DocBin
import spacy
from sklearn.model_selection import train_test_split


def trim_entity_spans(data: list) -> list:
    """
    Remove leading/trailing spaces from entity spans in annotated data.
    Args:
        data (list): List of (text, annotations) pairs
    Returns:
        list: Cleaned data with valid entity spans
    """
    invalid_span_tokens = re.compile(r'\s')
    cleaned_data = []

    for text, annotations in data:
        entities = annotations["entities"]
        valid_entities = []

        for start, end, label in entities:
            valid_start = start
            valid_end = end

            while valid_start < len(text) and invalid_span_tokens.match(text[valid_start]):
                valid_start += 1
            while valid_end > valid_start and invalid_span_tokens.match(text[valid_end - 1]):
                valid_end -= 1

            if valid_start < valid_end:
                valid_entities.append([valid_start, valid_end, label])

        cleaned_data.append([text, {"entities": valid_entities}])
    return cleaned_data


def convert_to_spacy(data, output_path: str, error_log="error.txt"):
    """
    Convert (text, annotations) into SpaCy DocBin format.
    """
    nlp = spacy.blank("en")
    db = DocBin()
    file = open(error_log, "w")

    for text, annot in tqdm(data, desc="Converting to SpaCy format"):
        doc = nlp.make_doc(text)
        ents = []
        entity_indices = []

        for start, end, label in annot["entities"]:
            if any(idx in entity_indices for idx in range(start, end)):
                continue
            entity_indices.extend(list(range(start, end)))
            span = doc.char_span(start, end, label=label, alignment_mode="strict")
            if span is not None:
                ents.append(span)
            else:
                file.write(f"Invalid span: {start}-{end} in text: {text[:60]}...\n")

        doc.ents = ents
        db.add(doc)

    db.to_disk(output_path)
    file.close()
    print(f"✅ Saved SpaCy DocBin to: {output_path}")


def main():
    # === Configuration ===
    input_path = "./data/final_fix_transformed.json"
    output_dir = "./corpus"
    test_size = 0.2

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # === Load Data ===
    print("📥 Loading dataset...")
    with open(input_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # === Clean Entity Spans ===
    print("🧹 Cleaning entity spans...")
    cleaned_data = trim_entity_spans(raw_data)

    # === Train-Test Split ===
    print("✂️ Splitting dataset...")
    train, test = train_test_split(cleaned_data, test_size=test_size, random_state=42)

    # === Convert to SpaCy format ===
    print("💾 Converting to SpaCy DocBin format...")
    convert_to_spacy(train, os.path.join(output_dir, "train_data.spacy"))
    convert_to_spacy(test, os.path.join(output_dir, "test_data.spacy"))

    print("\n✅ Data preprocessing completed successfully!")


if __name__ == "__main__":
    main()
