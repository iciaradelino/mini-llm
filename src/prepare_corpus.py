#!/usr/bin/env python3
"""
src/prepare_corpus.py
Downloads Spanish public domain texts from Project Gutenberg, strips legal headers/footers
and transcriber metadata, normalizes formatting to UTF-8, and partitions texts into train/val sets.
"""

import csv
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
CLEAN_TRAIN_DIR = BASE_DIR / "data" / "clean" / "train"
CLEAN_VAL_DIR = BASE_DIR / "data" / "clean" / "val"
CATALOG_CSV = BASE_DIR / "data" / "clean" / "catalog.csv"

# Project Gutenberg asks automated downloads to use a mirror, not www.gutenberg.org
# (https://www.gutenberg.org/policy/robot_access.html). This is PG's own mirror; it
# serves the same /cache/epub/ files byte-for-byte.
GUTENBERG_MIRROR = "https://gutenberg.pglaf.org"
DOWNLOAD_PAUSE_S = 2  # be polite: pause after each real download

# Literary movement of each author (criterion: standard Spanish literature
# textbook periodisation; see data/README.md). A work can override it with its
# own "movement" key in CORPUS_CATALOG.
MOVEMENT_BY_AUTHOR = {
    "Miguel de Cervantes": "siglo_de_oro",
    "Gustavo Adolfo Bécquer": "romanticismo",
    "Rosalía de Castro": "romanticismo",
    "Benito Pérez Galdós": "realismo",
    "Juan Valera": "realismo",
    "José María de Pereda": "realismo",
    "Leopoldo Alas": "realismo",
    "Emilia Pardo Bazán": "naturalismo",
    "Vicente Blasco Ibáñez": "naturalismo",
    "Antonio Machado": "generacion_98",
    "Federico García Lorca": "generacion_27",
    "Mariano José de Larra": "romanticismo",
    "Duque de Rivas": "romanticismo",
    "José Zorrilla": "romanticismo",
    "Fernán Caballero": "romanticismo",
    "Gertrudis Gómez de Avellaneda": "romanticismo",
    "Patricio de la Escosura": "romanticismo",
    "Francisco de Quevedo": "siglo_de_oro",
    "Luis Vélez de Guevara": "siglo_de_oro",
    "Lope de Vega": "siglo_de_oro",
    "Pedro Calderón de la Barca": "siglo_de_oro",
    "Juan Ruiz de Alarcón": "siglo_de_oro",
    "Fray Luis de León": "siglo_de_oro",
    "Miguel de Unamuno": "generacion_98",
    "Ramón del Valle-Inclán": "generacion_98",
    "Rubén Darío": "modernismo",
}

# Genre of each work: "prosa" unless the work's entry or this table says otherwise
# ("mixto" = prose and verse).
GENRE_BY_FILENAME = {
    "obrasescogidas": "mixto",      # Bécquer: Rimas + Leyendas
    "poesiascompletas": "poesia",
    "romancerogitano": "poesia",
    "librodepoemas": "poesia",
    "enlasorillasdelsar": "poesia",
}

CATALOG_FIELDS = ["filename", "title", "author", "movement", "genre", "source",
                  "source_id", "split", "clean_bytes", "chars"]

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
    # ---- added for the per-movement analysis (see data/README.md) ----
    # Excluded on purpose (duplicates): "El doncel de don Enrique el doliente" tomos I-IV
    # (53587-53590) are already inside "Obras completas de Figaro, Tomo 1"; Machado's
    # "Paginas escogidas" (49437) repeats most of "Poesias completas" (val).
    # romanticismo
    {"author": "Mariano José de Larra", "title": "Obras completas de Fígaro, Tomo 1", "id": 67676, "filename": "obrascompletasdefigarotomo1", "split": "train", "genre": "prosa"},
    {"author": "Mariano José de Larra", "title": "Obras completas de Fígaro, Tomo 2", "id": 68444, "filename": "obrascompletasdefigarotomo2", "split": "train", "genre": "prosa"},
    {"author": "Duque de Rivas", "title": "Don Álvaro, o, La fuerza del Sino", "id": 59492, "filename": "donalvaroolafuerzadelsino", "split": "val", "genre": "teatro"},
    {"author": "José Zorrilla", "title": "Recuerdos Del Tiempo Viejo", "id": 53294, "filename": "recuerdosdeltiempoviejo", "split": "train", "genre": "prosa"},
    {"author": "José Zorrilla", "title": "Granada, Poema Oriental, precedido de la Leyenda de al-Hamar, Tomo 1", "id": 55480, "filename": "granadapoemaorientalprecedidodelaleyenda", "split": "train", "genre": "poesia"},
    {"author": "José Zorrilla", "title": "Granada, Poema Oriental, precedido de la Leyenda de al-Hamar, Tomo 2", "id": 58275, "filename": "granadapoemaorientalprecedidodelaleyenda58275", "split": "train", "genre": "poesia"},
    {"author": "José Zorrilla", "title": "Traidor, inconfeso y martir", "id": 76073, "filename": "traidorinconfesoymartir", "split": "train", "genre": "teatro"},
    {"author": "Fernán Caballero", "title": "La gaviota", "id": 23600, "filename": "lagaviota", "split": "train", "genre": "prosa"},
    {"author": "Fernán Caballero", "title": "Clemencia: Novela de costumbres", "id": 60284, "filename": "clemencianoveladecostumbres", "split": "train", "genre": "prosa"},
    {"author": "Gertrudis Gómez de Avellaneda", "title": "Sab (novela original)", "id": 75126, "filename": "sabnovelaoriginal", "split": "val", "genre": "prosa"},
    {"author": "Patricio de la Escosura", "title": "El conde de Candespina (1 de 2)", "id": 75133, "filename": "elcondedecandespina1de2", "split": "train", "genre": "prosa"},
    {"author": "Patricio de la Escosura", "title": "El conde de Candespina (2 de 2)", "id": 75134, "filename": "elcondedecandespina2de2", "split": "train", "genre": "prosa"},
    # siglo_de_oro
    {"author": "Miguel de Cervantes", "title": "Viage al Parnaso", "id": 16110, "filename": "viagealparnaso", "split": "train", "genre": "mixto"},
    {"author": "Miguel de Cervantes", "title": "Los entremeses", "id": 57955, "filename": "losentremeses", "split": "train", "genre": "teatro"},
    {"author": "Francisco de Quevedo", "title": "Historia de la vida del Buscón, llamado Don Pablos, ejemplo de vagabundos y espejo de tacaños", "id": 32315, "filename": "historiadelavidadelbusconllamadodonpablo", "split": "val", "genre": "prosa"},
    {"author": "Francisco de Quevedo", "title": "Los Sueños, Volume I", "id": 65999, "filename": "lossuenosvolumei", "split": "train", "genre": "prosa"},
    {"author": "Francisco de Quevedo", "title": "Los Sueños, Volume II", "id": 66262, "filename": "lossuenosvolumeii", "split": "train", "genre": "prosa"},
    {"author": "Luis Vélez de Guevara", "title": "El Diablo Cojuelo", "id": 12457, "filename": "eldiablocojuelo", "split": "train", "genre": "prosa"},
    {"author": "Lope de Vega", "title": "Comedias inéditas", "id": 57035, "filename": "comediasineditas", "split": "train", "genre": "teatro"},
    {"author": "Lope de Vega", "title": "Fuente Ovejuna", "id": 60198, "filename": "fuenteovejuna", "split": "val", "genre": "teatro"},
    {"author": "Lope de Vega", "title": "El castigo sin venganza", "id": 78160, "filename": "elcastigosinvenganza", "split": "train", "genre": "teatro"},
    {"author": "Pedro Calderón de la Barca", "title": "Teatro selecto, tomo 1 de 4", "id": 54436, "filename": "teatroselectotomo1de4", "split": "train", "genre": "teatro"},
    {"author": "Pedro Calderón de la Barca", "title": "Teatro selecto, tomo 2 de 4", "id": 55917, "filename": "teatroselectotomo2de4", "split": "train", "genre": "teatro"},
    {"author": "Pedro Calderón de la Barca", "title": "Teatro selecto, tomo 3 de 4", "id": 58643, "filename": "teatroselectotomo3de4", "split": "train", "genre": "teatro"},
    {"author": "Pedro Calderón de la Barca", "title": "Teatro selecto, tomo 4 de 4", "id": 63328, "filename": "teatroselectotomo4de4", "split": "train", "genre": "teatro"},
    {"author": "Juan Ruiz de Alarcón", "title": "La verdad sospechosa", "id": 57590, "filename": "laverdadsospechosa", "split": "train", "genre": "teatro"},
    {"author": "Juan Ruiz de Alarcón", "title": "Los favores del mundo", "id": 18580, "filename": "losfavoresdelmundo", "split": "train", "genre": "teatro"},
    {"author": "Juan Ruiz de Alarcón", "title": "Las paredes oyen", "id": 77445, "filename": "lasparedesoyen", "split": "train", "genre": "teatro"},
    {"author": "Fray Luis de León", "title": "La perfecta casada", "id": 61573, "filename": "laperfectacasada", "split": "train", "genre": "prosa"},
    # generacion_98
    {"author": "Miguel de Unamuno", "title": "La Tía Tula (Novela)", "id": 44358, "filename": "latiatulanovela", "split": "train", "genre": "prosa"},
    {"author": "Miguel de Unamuno", "title": "Abel Sánchez: Una Historia de Pasión", "id": 44512, "filename": "abelsanchezunahistoriadepasion", "split": "train", "genre": "prosa"},
    {"author": "Miguel de Unamuno", "title": "Amor y Pedagogía", "id": 49149, "filename": "amorypedagogia", "split": "train", "genre": "prosa"},
    {"author": "Miguel de Unamuno", "title": "Niebla (Nivola)", "id": 49836, "filename": "nieblanivola", "split": "val", "genre": "prosa"},
    {"author": "Miguel de Unamuno", "title": "Tres novelas ejemplares y un prólogo", "id": 55916, "filename": "tresnovelasejemplaresyunprologo", "split": "train", "genre": "prosa"},
    {"author": "Miguel de Unamuno", "title": "El espejo de la muerte", "id": 77759, "filename": "elespejodelamuerte", "split": "train", "genre": "prosa"},
    {"author": "Miguel de Unamuno", "title": "Andanzas y visiones españolas", "id": 78052, "filename": "andanzasyvisionesespanolas", "split": "train", "genre": "prosa"},
    {"author": "Miguel de Unamuno", "title": "Del sentimiento trágico de la vida", "id": 59852, "filename": "delsentimientotragicodelavida", "split": "train", "genre": "prosa"},
    {"author": "Miguel de Unamuno", "title": "Vida de Don Quijote y Sancho", "id": 75472, "filename": "vidadedonquijoteysancho", "split": "train", "genre": "prosa"},
    {"author": "Ramón del Valle-Inclán", "title": "Sonata de estío: memorias del marqués de Bradomín", "id": 42424, "filename": "sonatadeestiomemoriasdelmarquesdebradomi", "split": "train", "genre": "prosa"},
    {"author": "Ramón del Valle-Inclán", "title": "Sonata de primavera: memorias del marqués de Bradomín", "id": 42440, "filename": "sonatadeprimaveramemoriasdelmarquesdebra", "split": "train", "genre": "prosa"},
    {"author": "Ramón del Valle-Inclán", "title": "Sonata de otoño; Sonata de invierno: memorias del Marqués de Bradomín", "id": 46182, "filename": "sonatadeotonosonatadeinviernomemoriasdel", "split": "train", "genre": "prosa"},
    {"author": "Ramón del Valle-Inclán", "title": "Tirano Banderas: Novela de tierra caliente", "id": 68154, "filename": "tiranobanderasnoveladetierracaliente", "split": "train", "genre": "prosa"},
    {"author": "Ramón del Valle-Inclán", "title": "Luces de Bohemia: Esperpento", "id": 68745, "filename": "lucesdebohemiaesperpento", "split": "val", "genre": "teatro"},
    {"author": "Ramón del Valle-Inclán", "title": "Divinas palabras", "id": 69951, "filename": "divinaspalabras", "split": "train", "genre": "teatro"},
    {"author": "Ramón del Valle-Inclán", "title": "Romance de lobos, comedia barbara", "id": 10506, "filename": "romancedeloboscomediabarbara", "split": "train", "genre": "teatro"},
    {"author": "Ramón del Valle-Inclán", "title": "El Marqués de Bradomín: Coloquios Románticos", "id": 58049, "filename": "elmarquesdebradomincoloquiosromanticos", "split": "train", "genre": "teatro"},
    # modernismo
    {"author": "Rubén Darío", "title": "Azul...", "id": 52894, "filename": "azul", "split": "val", "genre": "prosa"},
    {"author": "Rubén Darío", "title": "Prosas Profanas", "id": 47650, "filename": "prosasprofanas", "split": "train", "genre": "poesia"},
    {"author": "Rubén Darío", "title": "Cantos de Vida y Esperanza, Los Cisnes y otros poemas.", "id": 50341, "filename": "cantosdevidayesperanzaloscisnesyotrospoe", "split": "train", "genre": "poesia"},
    {"author": "Rubén Darío", "title": "Cuentos y crónicas", "id": 51627, "filename": "cuentosycronicas", "split": "train", "genre": "prosa"},
    {"author": "Rubén Darío", "title": "Los Raros", "id": 50365, "filename": "losraros", "split": "train", "genre": "prosa"},
    {"author": "Rubén Darío", "title": "El Canto Errante", "id": 51711, "filename": "elcantoerrante", "split": "train", "genre": "poesia"},
]


def ensure_directories():
    """Create necessary directories if they do not exist."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    CLEAN_TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    CLEAN_VAL_DIR.mkdir(parents=True, exist_ok=True)
    # data/clean/ is fully regenerated: delete old files so a work removed from
    # CORPUS_CATALOG cannot silently stay in the corpus (data/raw/ is a cache and is kept)
    for old in list(CLEAN_TRAIN_DIR.glob("*.txt")) + list(CLEAN_VAL_DIR.glob("*.txt")):
        old.unlink()


def download_work(gutenberg_id: int, raw_path: Path):
    """Download work from Gutenberg if not already cached."""
    if raw_path.exists():
        print(f"[CACHE] Found raw file: {raw_path.name}")
        return

    url = f"{GUTENBERG_MIRROR}/cache/epub/{gutenberg_id}/pg{gutenberg_id}.txt"
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
    time.sleep(DOWNLOAD_PAUSE_S)


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

    # 2a. Transcribers' illustration markers: [Illustration: ...], [Ilustración], [imagen: ...]
    #     (may span several lines; capped at 1,500 characters so prose is never eaten)
    text = re.sub(r"\[(?:Illustration|Ilustraci[oó]n|imagen)\b[^\[\]]{0,1500}\]", "", text,
                  flags=re.IGNORECASE)

    # 2b. Older files end with a one-line footer instead: "End of Project Gutenberg's <title>"
    text = re.sub(r"\n\s*End of (?:the )?Project Gutenberg'?s?\b[^\n]*\s*$", "\n", text,
                  flags=re.IGNORECASE)

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

    unknown = sorted({i["author"] for i in CORPUS_CATALOG
                      if "movement" not in i and i["author"] not in MOVEMENT_BY_AUTHOR})
    if unknown:
        sys.exit(f"[ERROR] no movement for author(s) {unknown}: add them to MOVEMENT_BY_AUTHOR")
    names = [i["filename"] for i in CORPUS_CATALOG]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        sys.exit(f"[ERROR] duplicate filenames in CORPUS_CATALOG: {dupes}")

    catalog_rows = []
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

        movement = item.get("movement", MOVEMENT_BY_AUTHOR.get(item["author"]))
        genre = item.get("genre", GENRE_BY_FILENAME.get(item["filename"], "prosa"))
        catalog_rows.append({
            "filename": fname, "title": item["title"], "author": item["author"],
            "movement": movement, "genre": genre, "source": item.get("source", "gutenberg"),
            "source_id": gid, "split": item["split"],
            "clean_bytes": len(clean_bytes), "chars": len(clean_text),
        })

        print(f"[PROCESSED] {item['filename']} -> {item['split']} | {movement} | {genre} ({size_mb:.2f} MB)")

    with open(CATALOG_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CATALOG_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(catalog_rows)

    train_mb = total_train_bytes / (1024 * 1024)
    val_mb = total_val_bytes / (1024 * 1024)
    total_mb = train_mb + val_mb

    print("-" * 60)
    print(f"Corpus Prep Complete:")
    print(f" - Train Corpus Size: {train_mb:.2f} MB")
    print(f" - Val Corpus Size:   {val_mb:.2f} MB")
    print(f" - Total Clean Size:  {total_mb:.2f} MB")
    print(f" - Catalog:           {CATALOG_CSV.relative_to(BASE_DIR)} ({len(catalog_rows)} works)")
    print("=" * 60)


if __name__ == "__main__":
    main()
