# Spanish Literary Corpus (`data/`)

This directory contains raw, cleaned, and tokenized datasets for training and evaluating the Spanish Mini-GPT model.

_Note: Nothing inside `raw/`, `clean/`, or `tokenized/` is committed to Git. All dataset files are fully reproducible and regenerated on demand via `src/prepare_corpus.py` (see `REPRODUCE.md`)._

---

## Directory Structure

- `data/raw/` — Unmodified, cached plain-text files directly downloaded from Project Gutenberg.
- `data/clean/` — UTF-8 normalized texts with Gutenberg legal headers, footers, transcriber metadata, and footnotes removed.
- `data/clean/train/` — Training split (69 complete works).
- `data/clean/val/` — Validation split (12 held-out works across literary movements and genres).
- `data/tokenized/` — BPE tokenizer and uint16 token streams (`train.bin`, `val.bin`, `tiny.bin`, `meta.json`, `tokenizer.json`); see _Tokenization_ below.

---

## Sources & Corpus Catalog

| Work Title                           | Author                        | Gutenberg ID | Clean Filename                                      |   Split   |
| :----------------------------------- | :---------------------------- | :----------: | :-------------------------------------------------- | :-------: |
| Don Quijote                          | Miguel de Cervantes           |    `2000`    | `donquijote.txt`                                    |  `train`  |
| Novelas Ejemplares                   | Miguel de Cervantes           |   `61202`    | `novelasejemplares.txt`                             |  `train`  |
| Viage al Parnaso                     | Miguel de Cervantes           |   `16110`    | `viagealparnaso.txt`                                |  `train`  |
| Los entremeses                       | Miguel de Cervantes           |   `57955`    | `losentremeses.txt`                                 |  `train`  |
| Obras Escogidas                      | Gustavo Adolfo Bécquer        |   `53552`    | `obrasescogidas.txt`                                |  `train`  |
| Poesias Completas                    | Antonio Machado               |   `68525`    | `poesiascompletas.txt`                              | **`val`** |
| Romancero Gitano                     | Federico García Lorca         |   `72665`    | `romancerogitano.txt`                               |  `train`  |
| Libro de poemas                      | Federico García Lorca         |   `75703`    | `librodepoemas.txt`                                 |  `train`  |
| Fortunata y Jacinta                  | Benito Pérez Galdós           |   `17013`    | `fortunatayjacinta.txt`                             |  `train`  |
| Misericordia                         | Benito Pérez Galdós           |   `21831`    | `misericordia.txt`                                  | **`val`** |
| Trafalgar                            | Benito Pérez Galdós           |   `16961`    | `trafalgar.txt`                                     |  `train`  |
| Tormento                             | Benito Pérez Galdós           |   `31465`    | `tormento.txt`                                      |  `train`  |
| Marianela                            | Benito Pérez Galdós           |   `17340`    | `marianela.txt`                                     |  `train`  |
| Los pazos de Ulloa                   | Emilia Pardo Bazán            |   `18005`    | `lospazosdeulloa.txt`                               |  `train`  |
| Insolacion y Morriña                 | Emilia Pardo Bazán            |   `52597`    | `insolacionymorrina.txt`                            |  `train`  |
| La Tribuna                           | Emilia Pardo Bazán            |   `17491`    | `latribuna.txt`                                     | **`val`** |
| Cuentos de Amor                      | Emilia Pardo Bazán            |   `55514`    | `cuentosdeamor.txt`                                 |  `train`  |
| Los cuatro jinetes del apocalipsis   | Vicente Blasco Ibáñez         |   `24536`    | `loscuatrojinetesdelapocalipsis.txt`                |  `train`  |
| Mare Nostrum                         | Vicente Blasco Ibáñez         |   `23236`    | `marenostrum.txt`                                   |  `train`  |
| Los Argonautas                       | Vicente Blasco Ibáñez         |   `25640`    | `losargonautas.txt`                                 |  `train`  |
| Cañas y Barro                        | Vicente Blasco Ibáñez         |   `57781`    | `canasybarro.txt`                                   | **`val`** |
| La Regenta                           | Leopoldo Alas                 |   `17073`    | `laregenta.txt`                                     |  `train`  |
| El Señor y los demás son Cuentos     | Leopoldo Alas                 |   `62361`    | `elsenorylosdemassoncuentos.txt`                    |  `train`  |
| Su Unico Hijo                        | Leopoldo Alas                 |   `17341`    | `suunicohijo.txt`                                   |  `train`  |
| Pepita Jimenez                       | Juan Valera                   |   `17223`    | `pepitajimenez.txt`                                 |  `train`  |
| Juanita la Larga                     | Juan Valera                   |   `16484`    | `juanitalalarga.txt`                                |  `train`  |
| Cuentos y Dialogos                   | Juan Valera                   |   `27214`    | `cuentosydialogos.txt`                              | **`val`** |
| En las orillas del Sar               | Rosalía de Castro             |   `70984`    | `enlasorillasdelsar.txt`                            |  `train`  |
| Sotileza                             | José María de Pereda          |   `49388`    | `sotileza.txt`                                      |  `train`  |
| La Puchera                           | José María de Pereda          |   `55058`    | `lapuchera.txt`                                     |  `train`  |
| La Montalvez                         | José María de Pereda          |   `25812`    | `lamontalvez.txt`                                   |  `train`  |
| Obras completas de Fígaro, Tomo 1    | Mariano José de Larra         |   `67676`    | `obrascompletasdefigarotomo1.txt`                   |  `train`  |
| Obras completas de Fígaro, Tomo 2    | Mariano José de Larra         |   `68444`    | `obrascompletasdefigarotomo2.txt`                   |  `train`  |
| Don Álvaro, o, La fuerza del Sino    | Duque de Rivas                |   `59492`    | `donalvaroolafuerzadelsino.txt`                     | **`val`** |
| Recuerdos Del Tiempo Viejo           | José Zorrilla                 |   `53294`    | `recuerdosdeltiempoviejo.txt`                       |  `train`  |
| Granada, Poema Oriental, Tomo 1      | José Zorrilla                 |   `55480`    | `granadapoemaorientalprecedidodelaleyenda.txt`      |  `train`  |
| Granada, Poema Oriental, Tomo 2      | José Zorrilla                 |   `58275`    | `granadapoemaorientalprecedidodelaleyenda58275.txt` |  `train`  |
| Traidor, inconfeso y martir          | José Zorrilla                 |   `76073`    | `traidorinconfesoymartir.txt`                       |  `train`  |
| La gaviota                           | Fernán Caballero              |   `23600`    | `lagaviota.txt`                                     |  `train`  |
| Clemencia: Novela de costumbres      | Fernán Caballero              |   `60284`    | `clemencianoveladecostumbres.txt`                   |  `train`  |
| Sab (novela original)                | Gertrudis Gómez de Avellaneda |   `75126`    | `sabnovelaoriginal.txt`                             | **`val`** |
| El conde de Candespina (1 de 2)      | Patricio de la Escosura       |   `75133`    | `elcondedecandespina1de2.txt`                       |  `train`  |
| El conde de Candespina (2 de 2)      | Patricio de la Escosura       |   `75134`    | `elcondedecandespina2de2.txt`                       |  `train`  |
| Viage al Parnaso                     | Miguel de Cervantes           |   `16110`    | `viagealparnaso.txt`                                |  `train`  |
| Los entremeses                       | Miguel de Cervantes           |   `57955`    | `losentremeses.txt`                                 |  `train`  |
| Historia de la vida del Buscón       | Francisco de Quevedo          |   `32315`    | `historiadelavidadelbusconllamadodonpablo.txt`      | **`val`** |
| Los Sueños, Volume I                 | Francisco de Quevedo          |   `65999`    | `lossuenosvolumei.txt`                              |  `train`  |
| Los Sueños, Volume II                | Francisco de Quevedo          |   `66262`    | `lossuenosvolumeii.txt`                             |  `train`  |
| El Diablo Cojuelo                    | Luis Vélez de Guevara         |   `12457`    | `eldiablocojuelo.txt`                               |  `train`  |
| Comedias inéditas                    | Lope de Vega                  |   `57035`    | `comediasineditas.txt`                              |  `train`  |
| Fuente Ovejuna                       | Lope de Vega                  |   `60198`    | `fuenteovejuna.txt`                                 | **`val`** |
| El castigo sin venganza              | Lope de Vega                  |   `78160`    | `elcastigosinvenganza.txt`                          |  `train`  |
| Teatro selecto, tomo 1 de 4          | Pedro Calderón de la Barca    |   `54436`    | `teatroselectotomo1de4.txt`                         |  `train`  |
| Teatro selecto, tomo 2 de 4          | Pedro Calderón de la Barca    |   `55917`    | `teatroselectotomo2de4.txt`                         |  `train`  |
| Teatro selecto, tomo 3 de 4          | Pedro Calderón de la Barca    |   `58643`    | `teatroselectotomo3de4.txt`                         |  `train`  |
| Teatro selecto, tomo 4 de 4          | Pedro Calderón de la Barca    |   `63328`    | `teatroselectotomo4de4.txt`                         |  `train`  |
| La verdad sospechosa                 | Juan Ruiz de Alarcón          |   `57590`    | `laverdadsospechosa.txt`                            |  `train`  |
| Los favores del mundo                | Juan Ruiz de Alarcón          |   `18580`    | `losfavoresdelmundo.txt`                            |  `train`  |
| Las paredes oyen                     | Juan Ruiz de Alarcón          |   `77445`    | `lasparedesoyen.txt`                                |  `train`  |
| La perfecta casada                   | Fray Luis de León             |   `61573`    | `laperfectacasada.txt`                              |  `train`  |
| La Tía Tula (Novela)                 | Miguel de Unamuno             |   `44358`    | `latiatulanovela.txt`                               |  `train`  |
| Abel Sánchez: Una Historia de Pasión | Miguel de Unamuno             |   `44512`    | `abelsanchezunahistoriadepasion.txt`                |  `train`  |
| Amor y Pedagogía                     | Miguel de Unamuno             |   `49149`    | `amorypedagogia.txt`                                |  `train`  |
| Niebla (Nivola)                      | Miguel de Unamuno             |   `49836`    | `nieblanivola.txt`                                  | **`val`** |
| Tres novelas ejemplares y un prólogo | Miguel de Unamuno             |   `55916`    | `tresnovelasejemplaresyunprologo.txt`               |  `train`  |
| El espejo de la muerte               | Miguel de Unamuno             |   `77759`    | `elespejodelamuerte.txt`                            |  `train`  |
| Andanzas y visiones españolas        | Miguel de Unamuno             |   `78052`    | `andanzasyvisionesespanolas.txt`                    |  `train`  |
| Del sentimiento trágico de la vida   | Miguel de Unamuno             |   `59852`    | `delsentimientotragicodelavida.txt`                 |  `train`  |
| Vida de Don Quijote y Sancho         | Miguel de Unamuno             |   `75472`    | `vidadedonquijoteysancho.txt`                       |  `train`  |
| Sonata de estío                      | Ramón del Valle-Inclán        |   `42424`    | `sonatadeestiomemoriasdelmarquesdebradomi.txt`      |  `train`  |
| Sonata de primavera                  | Ramón del Valle-Inclán        |   `42440`    | `sonatadeprimaveramemoriasdelmarquesdebra.txt`      |  `train`  |
| Sonata de otoño; Sonata de invierno  | Ramón del Valle-Inclán        |   `46182`    | `sonatadeotonosonatadeinviernomemoriasdel.txt`      |  `train`  |
| Tirano Banderas                      | Ramón del Valle-Inclán        |   `68154`    | `tiranobanderasnoveladetierracaliente.txt`          |  `train`  |
| Luces de Bohemia                     | Ramón del Valle-Inclán        |   `68745`    | `lucesdebohemiaesperpento.txt`                      | **`val`** |
| Divinas palabras                     | Ramón del Valle-Inclán        |   `69951`    | `divinaspalabras.txt`                               |  `train`  |
| Romance de lobos                     | Ramón del Valle-Inclán        |   `10506`    | `romancedeloboscomediabarbara.txt`                  |  `train`  |
| El Marqués de Bradomín               | Ramón del Valle-Inclán        |   `58049`    | `elmarquesdebradomincoloquiosromanticos.txt`        |  `train`  |
| Azul...                              | Rubén Darío                   |   `52894`    | `azul.txt`                                          | **`val`** |
| Prosas Profanas                      | Rubén Darío                   |   `47650`    | `prosasprofanas.txt`                                |  `train`  |
| Cantos de Vida y Esperanza           | Rubén Darío                   |   `50341`    | `cantosdevidayesperanzaloscisnesyotrospoe.txt`      |  `train`  |
| Cuentos y crónicas                   | Rubén Darío                   |   `51627`    | `cuentosycronicas.txt`                              |  `train`  |
| Los Raros                            | Rubén Darío                   |   `50365`    | `losraros.txt`                                      |  `train`  |
| El Canto Errante                     | Rubén Darío                   |   `51711`    | `elcantoerrante.txt`                                |  `train`  |

---

## Split Strategy & Leakage Prevention

- **Train Set**: 69 works representing major Spanish literary movements (Siglo de Oro, Romanticismo, Realismo, Naturalismo, Generación del 98, Generación del 27, Modernismo).
- **Validation Set**: 12 held-out works across movements and genres held out in their entirety to prevent data leakage during language model evaluation.

---

## Tokenization (step 5)

Byte-level BPE trained with Hugging Face `tokenizers` on the **training split only**
(`src/train_tokenizer.py`), then applied to both splits (`src/encode.py`).
Regenerate with `sbatch slurm/tokenize.sbatch`.

| Setting           | Value                                                                 |
| :---------------- | :-------------------------------------------------------------------- |
| Model             | Byte-level BPE (GPT-2 style pre-tokenizer, no normalizer, no `<unk>`) |
| Vocabulary size   | 8,192 (incl. 1 special token)                                         |
| Special token     | `<\|endoftext\|>` = id 0, appended after every work                   |
| Trained on        | 69 train works; val works never seen                                  |
| Tokenizer SHA-256 | `2fca7f76fb9d42d0…` (full hash in `meta.json`)                        |

### Output files (`data/tokenized/`)

| File             | Contents                                                                |    Tokens |    Size |
| :--------------- | :---------------------------------------------------------------------- | --------: | ------: |
| `train.bin`      | 69 works + 69 EOT                                                       |         — |       — |
| `val.bin`        | 12 works + 12 EOT                                                       |         — |       — |
| `tiny.bin`       | first 1,500,000 tokens of `train.bin`                                   | 1,500,000 | 2.86 MB |
| `tokenizer.json` | trained tokenizer                                                       |           |         |
| `meta.json`      | vocab size, dtype, EOT id, per-file token counts, SHA-256 of every file |           |         |

Format: flat little-endian **uint16** token ids, no header. Load with
`np.memmap("data/tokenized/train.bin", dtype=np.uint16, mode="r")`.

### Verification

`python src/test_roundtrip.py` reads the `.bin` files back with `np.memmap`, splits on EOT,
decodes every work and checks it equals the clean text byte-for-byte; it also checks
`tiny.bin` is a prefix of `train.bin`, all ids < vocab size, and a set of edge-case strings
(accents, ñ, ¿¡, «», em-dash, emoji). Rerunning the pipeline gives byte-identical files.
