import os
import pandas as pd
import torch

from model_loader import device, sbert_model


# -----------------------------
# TARGET RESEARCH PROVINCES
# -----------------------------

provinces = [
    "batangas",
    "laguna",
    "cavite",
    "rizal",
    "quezon"
]


# -----------------------------
# SAFE CSV LOADER
# -----------------------------

def safe_read_csv(path):

    encodings = [
        "utf-8",
        "cp1252",
        "latin-1"
    ]

    for enc in encodings:

        try:
            return pd.read_csv(
                path,
                encoding=enc
            )

        except Exception:
            continue

    return pd.DataFrame()


# -----------------------------
# LOAD REFERENCE SENTENCES
# -----------------------------

def load_reference_sentences():

    path = "reference_sentences.csv"

    if not os.path.exists(path):

        print("reference_sentences.csv not found")

        return pd.DataFrame()

    df = safe_read_csv(path)

    if df.empty:
        return df

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
    )

    required_cols = [
        "dialect",
        "sentence"
    ]

    for col in required_cols:

        if col not in df.columns:

            raise Exception(
                f"Missing required column: {col}"
            )

    df["dialect"] = (
        df["dialect"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    df["sentence"] = (
        df["sentence"]
        .astype(str)
        .str.strip()
    )

    return df


# -----------------------------
# LOAD RISK WORDS
# -----------------------------

def load_risk_words():

    path = "risk_words.csv"

    if not os.path.exists(path):

        print("risk_words.csv not found")

        return pd.DataFrame()

    df = safe_read_csv(path)

    if df.empty:
        return df

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
    )

    # Fix possible spelling variations

    if (
        "dielect_tag" in df.columns
        and "dialect_tag" not in df.columns
    ):

        df = df.rename(
            columns={
                "dielect_tag": "dialect_tag"
            }
        )

    if (
        "context_meaning" in df.columns
        and "meaning" not in df.columns
    ):

        df = df.rename(
            columns={
                "context_meaning": "meaning"
            }
        )

    if "token" not in df.columns:

        raise Exception(
            "Missing required column: token"
        )

    return df.fillna("")


# -----------------------------
# LOAD HONORIFICS
# -----------------------------

def load_honorifics():

    path = "honorifics.csv"

    if not os.path.exists(path):

        print("honorifics.csv not found")

        return pd.DataFrame()

    df = safe_read_csv(path)

    if df.empty:

        return pd.DataFrame()

    # Clean column names

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
    )

    required_cols = [
        "word",
        "category"
    ]

    for col in required_cols:

        if col not in df.columns:

            raise Exception(
                f"Missing required column "
                f"in honorifics.csv: {col}"
            )

    # Clean words

    df["word"] = (
        df["word"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    # Clean categories

    df["category"] = (
        df["category"]
        .astype(str)
        .str.strip()
    )

    return df


# -----------------------------
# LOAD ALL CSV FILES
# -----------------------------

print("--- Loading Research Data ---")

proto_df = load_reference_sentences()

risk_df = load_risk_words()

honorifics_df = load_honorifics()

print("Research Data Loaded Successfully")


# -----------------------------
# PRECOMPUTE SBERT EMBEDDINGS
# -----------------------------

print("--- Building Province Embeddings ---")

proto_embeddings = {}


if not proto_df.empty:

    with torch.no_grad():

        for prov in provinces:

            sentences = (
                proto_df[
                    proto_df["dialect"] == prov
                ]["sentence"]
                .tolist()
            )

            if len(sentences) == 0:

                proto_embeddings[prov] = None

                continue

            embeddings = sbert_model.encode(

                sentences,

                convert_to_tensor=True,

                device=device

            )

            proto_embeddings[prov] = embeddings


print("Province embeddings complete")