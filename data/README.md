# Data

# Spanish Literary Corpus (`data/`)

This directory contains raw, cleaned, and tokenized datasets for training and evaluating the Spanish Mini-GPT model.

_Note: Nothing inside `raw/`, `clean/`, or `tokenized/` is committed to Git. All dataset files are fully reproducible and regenerated on demand via `src/prepare_corpus.py` (see `REPRODUCE.md`)._

---

## Directory Structure

- `data/raw/` — Unmodified, cached plain-text files directly downloaded from Project Gutenberg.
- `data/clean/` — UTF-8 normalized texts with Gutenberg legal headers, footers, transcriber metadata, and footnotes removed.
  - `data/clean/train/` — Training split (25 complete works).
  - `data/clean/val/` — Validation split (5 held-out works: 1 major novel + 4 shorter poetry/story collections).
- `data/tokenized/` — BPE-tokenized binary files (`train.bin`, `val.bin`) using vocabulary size ~8,000.

---

## Sources & Corpus Catalog

| Work Title                         | Author                 | Gutenberg ID | Clean Filename                       |   Split   |
| :--------------------------------- | :--------------------- | :----------: | :----------------------------------- | :-------: |
| Don Quijote                        | Miguel de Cervantes    |    `2000`    | `donquijote.txt`                     |  `train`  |
| Novelas Ejemplares                 | Miguel de Cervantes    |   `61202`    | `novelasejemplares.txt`              |  `train`  |
| Obras Escogidas                    | Gustavo Adolfo Bécquer |   `53552`    | `obrasescogidas.txt`                 |  `train`  |
| Poesias Completas                  | Antonio Machado        |   `68525`    | `poesiascompletas.txt`               | **`val`** |
| Paginas Escogidas                  | Antonio Machado        |   `49437`    | `paginasescogidas.txt`               |  `train`  |
| Romancero Gitano                   | Federico García Lorca  |   `72665`    | `romancerogitano.txt`                |  `train`  |
| Libro de poemas                    | Federico García Lorca  |   `75703`    | `librodepoemas.txt`                  |  `train`  |
| Fortunata y Jacinta                | Benito Pérez Galdós    |   `17013`    | `fortunatayjacinta.txt`              |  `train`  |
| Misericordia                       | Benito Pérez Galdós    |   `21831`    | `misericordia.txt`                   | **`val`** |
| Trafalgar                          | Benito Pérez Galdós    |   `16961`    | `trafalgar.txt`                      |  `train`  |
| Tormento                           | Benito Pérez Galdós    |   `31465`    | `tormento.txt`                       |  `train`  |
| Marianela                          | Benito Pérez Galdós    |   `17340`    | `marianela.txt`                      |  `train`  |
| Los pazos de Ulloa                 | Emilia Pardo Bazán     |   `18005`    | `lospazosdeulloa.txt`                |  `train`  |
| Insolacion y Morriña               | Emilia Pardo Bazán     |   `52597`    | `insolacionymorrina.txt`             |  `train`  |
| La Tribuna                         | Emilia Pardo Bazán     |   `17491`    | `latribuna.txt`                      | **`val`** |
| Cuentos de Amor                    | Emilia Pardo Bazán     |   `55514`    | `cuentosdeamor.txt`                  |  `train`  |
| Los cuatro jinetes del apocalipsis | Vicente Blasco Ibáñez  |   `24536`    | `loscuatrojinetesdelapocalipsis.txt` |  `train`  |
| Mare Nostrum                       | Vicente Blasco Ibáñez  |   `23236`    | `marenostrum.txt`                    |  `train`  |
| Los Argonautas                     | Vicente Blasco Ibáñez  |   `25640`    | `losargonautas.txt`                  |  `train`  |
| Cañas y Barro                      | Vicente Blasco Ibáñez  |   `57781`    | `canasybarro.txt`                    | **`val`** |
| La Regenta                         | Leopoldo Alas          |   `17073`    | `laregenta.txt`                      |  `train`  |
| El Señor y los demás son Cuentos   | Leopoldo Alas          |   `62361`    | `elsenorylosdemassoncuentos.txt`     |  `train`  |
| Su Unico Hijo                      | Leopoldo Alas          |   `17341`    | `suunicohijo.txt`                    |  `train`  |
| Pepita Jimenez                     | Juan Valera            |   `17223`    | `pepitajimenez.txt`                  |  `train`  |
| Juanita la Larga                   | Juan Valera            |   `16484`    | `juanitalalarga.txt`                 |  `train`  |
| Cuentos y Dialogos                 | Juan Valera            |   `27214`    | `cuentosydialogos.txt`               | **`val`** |
| En las orillas del Sar             | Rosalía de Castro      |   `70984`    | `enlasorillasdelsar.txt`             |  `train`  |
| Sotileza                           | José María de Pereda   |   `49388`    | `sotileza.txt`                       |  `train`  |
| La Puchera                         | José María de Pereda   |   `55058`    | `lapuchera.txt`                      |  `train`  |
| La Montalvez                       | José María de Pereda   |   `25812`    | `lamontalvez.txt`                    |  `train`  |

---

## Split Strategy & Leakage Prevention

- **Train Set**: 25 works (~16.59 MB clean text) representing major Spanish Realist, Romantic, and Modernist literature.
- **Validation Set**: 5 works (~1.72 MB clean text) held out in their entirety to prevent data leakage during language model evaluation.
