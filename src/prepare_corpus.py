#!/usr/bin/env python3
"""
src/prepare_corpus.py
Downloads Spanish public domain texts from Project Gutenberg, strips legal headers/footers
and transcriber metadata, normalizes formatting to UTF-8, and partitions texts into train/val sets.
"""

import os
import re
import sys
import urllib.request
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
CLEAN_TRAIN_DIR = BASE_DIR / "data" / "clean" / "train"
CLEAN_VAL_DIR = BASE_DIR / "data" / "clean" / "val"

# Master catalog mapping
CORPUS_CATALOG = [
    # Cervantes
    {
        "author": "Miguel de Cervantes",
        "title": "Don Quijote",
        "id": 2000,
        "filename": "donquijote",
        "split": "train",
    },
    {
        "author": "Miguel de Cervantes",
        "title": "Novelas Ejemplares",
        "id": 61202,
        "filename": "novelasejemplares",
        "split": "train",
    },
    # Bécquer
    {
        "author": "Gustavo Adolfo Bécquer",
        "title": "Obras Escogidas",
        "id": 53552,
        "filename": "obrasescogidas",
        "split": "train",
    },
    # Machado
    {
        "author": "Antonio Machado",
        "title": "Poesias Completas",
        "id": 68525,
        "filename": "poesiascompletas",
        "split": "val",
    },
    {
        "author": "Antonio Machado",
        "title": "Paginas Escogidas",
        "id": 49437,
        "filename": "paginasescogidas",
        "split": "train",
    },
    # Lorca
    {
        "author": "Federico García Lorca",
        "title": "Romancero Gitano",
        "id": 72665,
        "filename": "romancerogitano",
        "split": "train",
    },
    {
        "author": "Federico García Lorca",
        "title": "Libro de poemas",
        "id": 75703,
        "filename": "librodepoemas",
        "split": "train",
    },
    # Galdós
    {
        "author": "Benito Pérez Galdós",
        "title": "Fortunata y Jacinta",
        "id": 17013,
        "filename": "fortunatayjacinta",
        "split": "train",
    },
    {
        "author": "Benito Pérez Galdós",
        "title": "Misericordia",
        "id": 21831,
        "filename": "misericordia",
        "split": "val",
    },
    {
        "author": "Benito Pérez Galdós",
        "title": "Trafalgar",
        "id": 16961,
        "filename": "trafalgar",
        "split": "train",
    },
    {
        "author": "Benito Pérez Galdós",
        "title": "Tormento",
        "id": 31465,
        "filename": "tormento",
        "split": "train",
    },
    {
        "author": "Benito Pérez Galdós",
        "title": "Marianela",
        "id": 17340,
        "filename": "marianela",
        "split": "train",
    },
    # Pardo Bazán
    {
        "author": "Emilia Pardo Bazán",
        "title": "Los pazos de Ulloa",
        "id": 18005,
        "filename": "lospazosdeulloa",
        "split": "train",
    },
    {
        "author": "Emilia Pardo Bazán",
        "title": "Insolacion y Morriña",
        "id": 52597,
        "filename": "insolacionymorrina",
        "split": "train",
    },
    {
        "author": "Emilia Pardo Bazán",
        "title": "La Tribuna",
        "id": 17491,
        "filename": "latribuna",
        "split": "val",
    },
    {
        "author": "Emilia Pardo Bazán",
        "title": "Cuentos de Amor",
        "id": 55514,
        "filename": "cuentosdeamor",
        "split": "train",
    },
    # Blasco Ibáñez
    {
        "author": "Vicente Blasco Ibáñez",
        "title": "Los cuatro jinetes del apocalipsis",
        "id": 24536,
        "filename": "loscuatrojinetesdelapocalipsis",
        "split": "train",
    },
    {
        "author": "Vicente Blasco Ibáñez",
        "title": "Mare Nostrum",
        "id": 23236,
        "filename": "marenostrum",
        "split": "train",
    },
    {
        "author": "Vicente Blasco Ibáñez",
        "title": "Los Argonautas",
        "id": 25640,
        "filename": "losargonautas",
        "split": "train",
    },
    {
        "author": "Vicente Blasco Ibáñez",
        "title": "Cañas y Barro",
        "id": 57781,
        "filename": "canasybarro",
        "split": "val",
    },
    # Leopoldo Alas
    {
        "author": "Leopoldo Alas",
        "title": "La Regenta",
        "id": 17073,
        "filename": "laregenta",
        "split": "train",
    },
    {
        "author": "Leopoldo Alas",
        "title": "El Señor y los demás son Cuentos",
        "id": 62361,
        "filename": "elsenorylosdemassoncuentos",
        "split": "train",
    },
    {
        "author": "Leopoldo Alas",
        "title": "Su Unico Hijo",
        "id": 17341,
        "filename": "suunicohijo",
        "split": "train",
    },
    # Juan Valera
    {
        "author": "Juan Valera",
        "title": "Pepita Jimenez",
        "id": 17223,
        "filename": "pepitajimenez",
        "split": "train",
    },
    {
        "author": "Juan Valera",
        "title": "Juanita la Larga",
        "id": 16484,
        "filename": "juanitalalarga",
        "split": "train",
    },
    {
        "author": "Juan Valera",
        "title": "Cuentos y Dialogos",
        "id": 27214,
        "filename": "cuentosydialogos",
        "split": "val",
    },
    # Rosalía de Castro
    {
        "author": "Rosalía de Castro",
        "title": "En las orillas del Sar",
        "id": 70984,
        "filename": "enlasorillasdelsar",
        "split": "train",
    },
    # Pereda
    {
        "author": "José María de Pereda",
        "title": "Sotileza",
        "id": 49388,
        "filename": "sotileza",
        "split": "train",
    },
    {
        "author": "José María de Pereda",
        "title": "La Puchera",
        "id": 55058,
        "filename": "lapuchera",
        "split": "train",
    },
    {
        "author": "José María de Pereda",
        "title": "La Montalvez",
        "id": 25812,
        "filename": "lamontalvez",
        "split": "train",
    },
]


def ensure_directories():
    """Create necessary directories if they do not exist."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    CLEAN_TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    CLEAN_VAL_DIR.mkdir(parents=True, exist_ok=True)


def download_work(gutenberg_id: int, raw_path: Path):
    """Download work from Gutenberg if not already cached."""
    if raw_path.exists():
        print(f"[CACHE] Found raw file: {raw_path.name}")
        return

    url = f"https://www.gutenberg.org/cache/epub/{gutenberg_id}/pg{gutenberg_id}.txt"
    print(f"[DOWNLOADING] ID {gutenberg_id} from {url}...")

    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) HPC-MiniLLM-CorpusPrep/1.0"
    }
    req = urllib.request.Request(url, headers=headers)

    try:
        with urllib.request.urlopen(req) as response:
            content = response.read()
            with open(raw_path, "wb") as f:
                f.write(content)
    except Exception as e:
        print(f"[ERROR] Failed to download ID {gutenberg_id}: {e}", file=sys.stderr)
        raise


def strip_gutenberg_headers(text: str) -> str:
    """Remove Project Gutenberg legal markers, transcriber metadata, and proofreading credits."""
    # 1. Strip main Gutenberg start header
    start_match = re.search(
        r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if start_match:
        text = text[start_match.end() :]

    # 2. Strip main Gutenberg end footer
    end_match = re.search(
        r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if end_match:
        text = text[: end_match.start()]

    # 3. Filter out transcriber / proofreading / digitization boilerplate lines near the top
    lines = text.strip().splitlines()
    preamble_keywords = [
        "e-text prepared",
        "produced by",
        "transcribed by",
        "digitized by",
        "proofreading team",
        "pgdp.net",
        "illustrations and illuminations",
        "original pages",
        "generously made available",
        "project gutenberg",
        "online distributed proofreading",
        "text enclosed",
        "note:",
        "source material",
        "e-book was taken",
        "images of the original",
    ]

    clean_lines = []

    for i, line in enumerate(lines):
        line_lower = line.lower().strip()

        # Check the first 80 lines of the input file for preamble metadata
        if i < 80:
            if any(kw in line_lower for kw in preamble_keywords):
                continue
            if (
                "http://" in line_lower
                or "https://" in line_lower
                or "www." in line_lower
            ) and ("pgdp" in line_lower or "gutenberg" in line_lower):
                continue

        clean_lines.append(line)

    return "\n".join(clean_lines).strip()


def normalize_text(text_bytes: bytes) -> str:
    """Decode raw bytes trying UTF-8 then ISO-8859-1, normalize line endings."""
    try:
        text = text_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = text_bytes.decode("latin-1")

    text = strip_gutenberg_headers(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    return text


def main():
    ensure_directories()

    total_train_bytes = 0
    total_val_bytes = 0

    print("=" * 60)
    print("Starting Project Gutenberg Spanish Corpus Download & Prep")
    print("=" * 60)

    for item in CORPUS_CATALOG:
        gid = item["id"]
        fname = f"{item['filename']}.txt"
        raw_path = RAW_DIR / fname

        target_dir = CLEAN_VAL_DIR if item["split"] == "val" else CLEAN_TRAIN_DIR
        clean_path = target_dir / fname

        download_work(gid, raw_path)

        with open(raw_path, "rb") as f:
            raw_bytes = f.read()

        clean_text = normalize_text(raw_bytes)
        clean_bytes = clean_text.encode("utf-8")

        with open(clean_path, "wb") as f:
            f.write(clean_bytes)

        size_mb = len(clean_bytes) / (1024 * 1024)
        if item["split"] == "val":
            total_val_bytes += len(clean_bytes)
        else:
            total_train_bytes += len(clean_bytes)

        print(f"[PROCESSED] {item['filename']} -> {item['split']} ({size_mb:.2f} MB)")

    train_mb = total_train_bytes / (1024 * 1024)
    val_mb = total_val_bytes / (1024 * 1024)
    total_mb = train_mb + val_mb

    print("-" * 60)
    print(f"Corpus Prep Complete:")
    print(f" - Train Corpus Size: {train_mb:.2f} MB")
    print(f" - Val Corpus Size:   {val_mb:.2f} MB")
    print(f" - Total Clean Size:  {total_mb:.2f} MB")
    print("=" * 60)


if __name__ == "__main__":
    main()
