# Spanish Literary Corpus (`data/`)

This directory contains raw, cleaned, and tokenized datasets for training and evaluating the Spanish Mini-GPT model.

_Note: Nothing inside `raw/`, `clean/`, or `tokenized/` is committed to Git. All dataset files are fully reproducible and regenerated on demand via `src/prepare_corpus.py` (see `REPRODUCE.md`)._

---

## Directory Structure

- `data/raw/` — Unmodified, cached plain-text files directly downloaded from Project Gutenberg.
- `data/clean/` — UTF-8 normalized texts with Gutenberg legal headers, footers, transcriber metadata, and footnotes removed.
  - `data/clean/train/` — Training split (69 complete works).
  - `data/clean/val/` — Validation split (12 held-out works across prose, poetry and theatre).
  - `data/clean/catalog.csv` — one row per work: title, author, literary movement, genre, source ID, split, size.
- `data/tokenized/` — BPE tokenizer and uint16 token streams (`train.bin`, `val.bin`, `tiny.bin`, `meta.json`, `tokenizer.json`); see *Tokenization* below.

---

## Sources & Corpus Catalog

| Work Title | Author | Movement | Genre | Gutenberg ID | Clean Filename | Split |
| :-- | :-- | :-- | :-- | :-: | :-- | :-: |
| Don Quijote | Miguel de Cervantes | siglo_de_oro | prosa | `2000` | `donquijote.txt` | `train` |
| Novelas Ejemplares | Miguel de Cervantes | siglo_de_oro | prosa | `61202` | `novelasejemplares.txt` | `train` |
| Obras Escogidas | Gustavo Adolfo Bécquer | romanticismo | mixto | `53552` | `obrasescogidas.txt` | `train` |
| Poesias Completas | Antonio Machado | generacion_98 | poesia | `68525` | `poesiascompletas.txt` | **`val`** |
| Romancero Gitano | Federico García Lorca | generacion_27 | poesia | `72665` | `romancerogitano.txt` | `train` |
| Libro de poemas | Federico García Lorca | generacion_27 | poesia | `75703` | `librodepoemas.txt` | `train` |
| Fortunata y Jacinta | Benito Pérez Galdós | realismo | prosa | `17013` | `fortunatayjacinta.txt` | `train` |
| Misericordia | Benito Pérez Galdós | realismo | prosa | `21831` | `misericordia.txt` | **`val`** |
| Trafalgar | Benito Pérez Galdós | realismo | prosa | `16961` | `trafalgar.txt` | `train` |
| Tormento | Benito Pérez Galdós | realismo | prosa | `31465` | `tormento.txt` | `train` |
| Marianela | Benito Pérez Galdós | realismo | prosa | `17340` | `marianela.txt` | `train` |
| Los pazos de Ulloa | Emilia Pardo Bazán | naturalismo | prosa | `18005` | `lospazosdeulloa.txt` | `train` |
| Insolacion y Morriña | Emilia Pardo Bazán | naturalismo | prosa | `52597` | `insolacionymorrina.txt` | `train` |
| La Tribuna | Emilia Pardo Bazán | naturalismo | prosa | `17491` | `latribuna.txt` | **`val`** |
| Cuentos de Amor | Emilia Pardo Bazán | naturalismo | prosa | `55514` | `cuentosdeamor.txt` | `train` |
| Los cuatro jinetes del apocalipsis | Vicente Blasco Ibáñez | naturalismo | prosa | `24536` | `loscuatrojinetesdelapocalipsis.txt` | `train` |
| Mare Nostrum | Vicente Blasco Ibáñez | naturalismo | prosa | `23236` | `marenostrum.txt` | `train` |
| Los Argonautas | Vicente Blasco Ibáñez | naturalismo | prosa | `25640` | `losargonautas.txt` | `train` |
| Cañas y Barro | Vicente Blasco Ibáñez | naturalismo | prosa | `57781` | `canasybarro.txt` | **`val`** |
| La Regenta | Leopoldo Alas | realismo | prosa | `17073` | `laregenta.txt` | `train` |
| El Señor y los demás son Cuentos | Leopoldo Alas | realismo | prosa | `62361` | `elsenorylosdemassoncuentos.txt` | `train` |
| Su Unico Hijo | Leopoldo Alas | realismo | prosa | `17341` | `suunicohijo.txt` | `train` |
| Pepita Jimenez | Juan Valera | realismo | prosa | `17223` | `pepitajimenez.txt` | `train` |
| Juanita la Larga | Juan Valera | realismo | prosa | `16484` | `juanitalalarga.txt` | `train` |
| Cuentos y Dialogos | Juan Valera | realismo | prosa | `27214` | `cuentosydialogos.txt` | **`val`** |
| En las orillas del Sar | Rosalía de Castro | romanticismo | poesia | `70984` | `enlasorillasdelsar.txt` | `train` |
| Sotileza | José María de Pereda | realismo | prosa | `49388` | `sotileza.txt` | `train` |
| La Puchera | José María de Pereda | realismo | prosa | `55058` | `lapuchera.txt` | `train` |
| La Montalvez | José María de Pereda | realismo | prosa | `25812` | `lamontalvez.txt` | `train` |
| Obras completas de Fígaro, Tomo 1 | Mariano José de Larra | romanticismo | prosa | `67676` | `obrascompletasdefigarotomo1.txt` | `train` |
| Obras completas de Fígaro, Tomo 2 | Mariano José de Larra | romanticismo | prosa | `68444` | `obrascompletasdefigarotomo2.txt` | `train` |
| Don Álvaro, o, La fuerza del Sino | Duque de Rivas | romanticismo | teatro | `59492` | `donalvaroolafuerzadelsino.txt` | **`val`** |
| Recuerdos Del Tiempo Viejo | José Zorrilla | romanticismo | prosa | `53294` | `recuerdosdeltiempoviejo.txt` | `train` |
| Granada, Poema Oriental, precedido de la Leyenda de al-Hamar, Tomo 1 | José Zorrilla | romanticismo | poesia | `55480` | `granadapoemaorientalprecedidodelaleyenda.txt` | `train` |
| Granada, Poema Oriental, precedido de la Leyenda de al-Hamar, Tomo 2 | José Zorrilla | romanticismo | poesia | `58275` | `granadapoemaorientalprecedidodelaleyenda58275.txt` | `train` |
| Traidor, inconfeso y martir | José Zorrilla | romanticismo | teatro | `76073` | `traidorinconfesoymartir.txt` | `train` |
| La gaviota | Fernán Caballero | romanticismo | prosa | `23600` | `lagaviota.txt` | `train` |
| Clemencia: Novela de costumbres | Fernán Caballero | romanticismo | prosa | `60284` | `clemencianoveladecostumbres.txt` | `train` |
| Sab (novela original) | Gertrudis Gómez de Avellaneda | romanticismo | prosa | `75126` | `sabnovelaoriginal.txt` | **`val`** |
| El conde de Candespina (1 de 2) | Patricio de la Escosura | romanticismo | prosa | `75133` | `elcondedecandespina1de2.txt` | `train` |
| El conde de Candespina (2 de 2) | Patricio de la Escosura | romanticismo | prosa | `75134` | `elcondedecandespina2de2.txt` | `train` |
| Viage al Parnaso | Miguel de Cervantes | siglo_de_oro | mixto | `16110` | `viagealparnaso.txt` | `train` |
| Los entremeses | Miguel de Cervantes | siglo_de_oro | teatro | `57955` | `losentremeses.txt` | `train` |
| Historia de la vida del Buscón, llamado Don Pablos, ejemplo de vagabundos y espejo de tacaños | Francisco de Quevedo | siglo_de_oro | prosa | `32315` | `historiadelavidadelbusconllamadodonpablo.txt` | **`val`** |
| Los Sueños, Volume I | Francisco de Quevedo | siglo_de_oro | prosa | `65999` | `lossuenosvolumei.txt` | `train` |
| Los Sueños, Volume II | Francisco de Quevedo | siglo_de_oro | prosa | `66262` | `lossuenosvolumeii.txt` | `train` |
| El Diablo Cojuelo | Luis Vélez de Guevara | siglo_de_oro | prosa | `12457` | `eldiablocojuelo.txt` | `train` |
| Comedias inéditas | Lope de Vega | siglo_de_oro | teatro | `57035` | `comediasineditas.txt` | `train` |
| Fuente Ovejuna | Lope de Vega | siglo_de_oro | teatro | `60198` | `fuenteovejuna.txt` | **`val`** |
| El castigo sin venganza | Lope de Vega | siglo_de_oro | teatro | `78160` | `elcastigosinvenganza.txt` | `train` |
| Teatro selecto, tomo 1 de 4 | Pedro Calderón de la Barca | siglo_de_oro | teatro | `54436` | `teatroselectotomo1de4.txt` | `train` |
| Teatro selecto, tomo 2 de 4 | Pedro Calderón de la Barca | siglo_de_oro | teatro | `55917` | `teatroselectotomo2de4.txt` | `train` |
| Teatro selecto, tomo 3 de 4 | Pedro Calderón de la Barca | siglo_de_oro | teatro | `58643` | `teatroselectotomo3de4.txt` | `train` |
| Teatro selecto, tomo 4 de 4 | Pedro Calderón de la Barca | siglo_de_oro | teatro | `63328` | `teatroselectotomo4de4.txt` | `train` |
| La verdad sospechosa | Juan Ruiz de Alarcón | siglo_de_oro | teatro | `57590` | `laverdadsospechosa.txt` | `train` |
| Los favores del mundo | Juan Ruiz de Alarcón | siglo_de_oro | teatro | `18580` | `losfavoresdelmundo.txt` | `train` |
| Las paredes oyen | Juan Ruiz de Alarcón | siglo_de_oro | teatro | `77445` | `lasparedesoyen.txt` | `train` |
| La perfecta casada | Fray Luis de León | siglo_de_oro | prosa | `61573` | `laperfectacasada.txt` | `train` |
| La Tía Tula (Novela) | Miguel de Unamuno | generacion_98 | prosa | `44358` | `latiatulanovela.txt` | `train` |
| Abel Sánchez: Una Historia de Pasión | Miguel de Unamuno | generacion_98 | prosa | `44512` | `abelsanchezunahistoriadepasion.txt` | `train` |
| Amor y Pedagogía | Miguel de Unamuno | generacion_98 | prosa | `49149` | `amorypedagogia.txt` | `train` |
| Niebla (Nivola) | Miguel de Unamuno | generacion_98 | prosa | `49836` | `nieblanivola.txt` | **`val`** |
| Tres novelas ejemplares y un prólogo | Miguel de Unamuno | generacion_98 | prosa | `55916` | `tresnovelasejemplaresyunprologo.txt` | `train` |
| El espejo de la muerte | Miguel de Unamuno | generacion_98 | prosa | `77759` | `elespejodelamuerte.txt` | `train` |
| Andanzas y visiones españolas | Miguel de Unamuno | generacion_98 | prosa | `78052` | `andanzasyvisionesespanolas.txt` | `train` |
| Del sentimiento trágico de la vida | Miguel de Unamuno | generacion_98 | prosa | `59852` | `delsentimientotragicodelavida.txt` | `train` |
| Vida de Don Quijote y Sancho | Miguel de Unamuno | generacion_98 | prosa | `75472` | `vidadedonquijoteysancho.txt` | `train` |
| Sonata de estío: memorias del marqués de Bradomín | Ramón del Valle-Inclán | generacion_98 | prosa | `42424` | `sonatadeestiomemoriasdelmarquesdebradomi.txt` | `train` |
| Sonata de primavera: memorias del marqués de Bradomín | Ramón del Valle-Inclán | generacion_98 | prosa | `42440` | `sonatadeprimaveramemoriasdelmarquesdebra.txt` | `train` |
| Sonata de otoño; Sonata de invierno: memorias del Marqués de Bradomín | Ramón del Valle-Inclán | generacion_98 | prosa | `46182` | `sonatadeotonosonatadeinviernomemoriasdel.txt` | `train` |
| Tirano Banderas: Novela de tierra caliente | Ramón del Valle-Inclán | generacion_98 | prosa | `68154` | `tiranobanderasnoveladetierracaliente.txt` | `train` |
| Luces de Bohemia: Esperpento | Ramón del Valle-Inclán | generacion_98 | teatro | `68745` | `lucesdebohemiaesperpento.txt` | **`val`** |
| Divinas palabras | Ramón del Valle-Inclán | generacion_98 | teatro | `69951` | `divinaspalabras.txt` | `train` |
| Romance de lobos, comedia barbara | Ramón del Valle-Inclán | generacion_98 | teatro | `10506` | `romancedeloboscomediabarbara.txt` | `train` |
| El Marqués de Bradomín: Coloquios Románticos | Ramón del Valle-Inclán | generacion_98 | teatro | `58049` | `elmarquesdebradomincoloquiosromanticos.txt` | `train` |
| Azul... | Rubén Darío | modernismo | prosa | `52894` | `azul.txt` | **`val`** |
| Prosas Profanas | Rubén Darío | modernismo | poesia | `47650` | `prosasprofanas.txt` | `train` |
| Cantos de Vida y Esperanza, Los Cisnes y otros poemas. | Rubén Darío | modernismo | poesia | `50341` | `cantosdevidayesperanzaloscisnesyotrospoe.txt` | `train` |
| Cuentos y crónicas | Rubén Darío | modernismo | prosa | `51627` | `cuentosycronicas.txt` | `train` |
| Los Raros | Rubén Darío | modernismo | prosa | `50365` | `losraros.txt` | `train` |
| El Canto Errante | Rubén Darío | modernismo | poesia | `51711` | `elcantoerrante.txt` | `train` |

---

## Split Strategy & Leakage Prevention

- **Train Set**: 69 works (~30.13 MB clean text) spanning the Siglo de Oro through the Generación del 27.
- **Validation Set**: 12 works (~3.13 MB clean text) held out in their entirety to prevent data leakage during language model evaluation. Every literary movement except `generacion_27` has at least one held-out work, and the set mixes prose, poetry and theatre.

Movement labels follow the standard periodisation of Spanish literature textbooks and are assigned per author (`MOVEMENT_BY_AUTHOR` in `src/prepare_corpus.py`); a work can override its author's label. Genre (`prosa`, `poesia`, `teatro`, `mixto`) is set per work. Both are written to `data/clean/catalog.csv`.

| Movement | Works |
| :-- | --: |
| `siglo_de_oro` | 19 |
| `romanticismo` | 14 |
| `realismo` | 14 |
| `naturalismo` | 8 |
| `modernismo` | 6 |
| `generacion_98` | 18 |
| `generacion_27` | 2 |

---

## Tokenization (step 5)

Byte-level BPE trained with Hugging Face `tokenizers` on the **training split only**
(`src/train_tokenizer.py`), then applied to both splits (`src/encode.py`).
Regenerate with `sbatch slurm/tokenize.sbatch`.

| Setting | Value |
| :-- | :-- |
| Model | Byte-level BPE (GPT-2 style pre-tokenizer, no normalizer, no `<unk>`) |
| Vocabulary size | 8,192 (incl. 1 special token) |
| Special token | `<\|endoftext\|>` = id 0, appended after every work |
| Trained on | 69 train works (30.13 MB); val works never seen |
| Tokenizer SHA-256 | `33d0978a6ed49fdc…` (full hash in `meta.json`; job 28437) |

### Output files (`data/tokenized/`)

| File | Contents | Tokens | Size |
| :-- | :-- | --: | --: |
| `train.bin` | 69 works + 69 EOT | 9,069,382 | 17.30 MB |
| `val.bin` | 12 works + 12 EOT | 962,497 | 1.84 MB |
| `tiny.bin` | first 1,500,000 tokens of `train.bin` | 1,500,000 | 2.86 MB |
| `tokenizer.json` | trained tokenizer | | |
| `meta.json` | vocab size, dtype, EOT id, per-file token counts, SHA-256 of every file | | |

Format: flat little-endian **uint16** token ids, no header. Load with
`np.memmap("data/tokenized/train.bin", dtype=np.uint16, mode="r")`.

Compression: 3.397 chars/token on train, 3.323 on val (val is
slightly worse since the tokenizer never saw it).

### Token counts per work

| File | Split | Chars | Tokens | Chars/token |
| :-- | :-: | --: | --: | --: |
| `abelsanchezunahistoriadepasion.txt` | train | 162,974 | 49,678 | 3.28 |
| `amorypedagogia.txt` | train | 285,203 | 85,919 | 3.32 |
| `andanzasyvisionesespanolas.txt` | train | 447,359 | 129,977 | 3.44 |
| `cantosdevidayesperanzaloscisnesyotrospoe.txt` | train | 80,156 | 25,160 | 3.19 |
| `clemencianoveladecostumbres.txt` | train | 550,067 | 161,036 | 3.42 |
| `comediasineditas.txt` | train | 450,774 | 157,774 | 2.86 |
| `cuentosdeamor.txt` | train | 375,246 | 108,410 | 3.46 |
| `cuentosycronicas.txt` | train | 142,537 | 43,390 | 3.29 |
| `delsentimientotragicodelavida.txt` | train | 568,646 | 165,361 | 3.44 |
| `divinaspalabras.txt` | train | 116,773 | 42,041 | 2.78 |
| `donquijote.txt` | train | 2,110,726 | 585,443 | 3.61 |
| `elcantoerrante.txt` | train | 89,086 | 27,254 | 3.27 |
| `elcastigosinvenganza.txt` | train | 112,724 | 34,590 | 3.26 |
| `elcondedecandespina1de2.txt` | train | 174,956 | 47,327 | 3.70 |
| `elcondedecandespina2de2.txt` | train | 174,684 | 48,158 | 3.63 |
| `eldiablocojuelo.txt` | train | 496,660 | 154,821 | 3.21 |
| `elespejodelamuerte.txt` | train | 256,861 | 75,215 | 3.42 |
| `elmarquesdebradomincoloquiosromanticos.txt` | train | 135,500 | 42,271 | 3.21 |
| `elsenorylosdemassoncuentos.txt` | train | 298,423 | 85,425 | 3.49 |
| `enlasorillasdelsar.txt` | train | 161,052 | 46,398 | 3.47 |
| `fortunatayjacinta.txt` | train | 2,241,317 | 638,754 | 3.51 |
| `granadapoemaorientalprecedidodelaleyenda.txt` | train | 243,156 | 73,875 | 3.29 |
| `granadapoemaorientalprecedidodelaleyenda58275.txt` | train | 249,988 | 76,384 | 3.27 |
| `insolacionymorrina.txt` | train | 485,450 | 143,921 | 3.37 |
| `juanitalalarga.txt` | train | 413,708 | 117,284 | 3.53 |
| `lagaviota.txt` | train | 530,805 | 152,914 | 3.47 |
| `lamontalvez.txt` | train | 602,297 | 168,830 | 3.57 |
| `laperfectacasada.txt` | train | 231,884 | 67,520 | 3.43 |
| `lapuchera.txt` | train | 693,932 | 199,376 | 3.48 |
| `laregenta.txt` | train | 1,786,173 | 502,850 | 3.55 |
| `lasparedesoyen.txt` | train | 112,803 | 35,815 | 3.15 |
| `latiatulanovela.txt` | train | 172,378 | 52,174 | 3.30 |
| `laverdadsospechosa.txt` | train | 127,722 | 42,076 | 3.04 |
| `librodepoemas.txt` | train | 104,415 | 32,324 | 3.23 |
| `losargonautas.txt` | train | 1,223,629 | 341,877 | 3.58 |
| `loscuatrojinetesdelapocalipsis.txt` | train | 801,678 | 221,432 | 3.62 |
| `losentremeses.txt` | train | 227,823 | 76,978 | 2.96 |
| `losfavoresdelmundo.txt` | train | 142,204 | 45,452 | 3.13 |
| `lospazosdeulloa.txt` | train | 493,595 | 144,231 | 3.42 |
| `losraros.txt` | train | 385,853 | 118,931 | 3.24 |
| `lossuenosvolumei.txt` | train | 422,495 | 135,418 | 3.12 |
| `lossuenosvolumeii.txt` | train | 373,646 | 120,544 | 3.10 |
| `marenostrum.txt` | train | 873,370 | 245,999 | 3.55 |
| `marianela.txt` | train | 293,980 | 83,679 | 3.51 |
| `novelasejemplares.txt` | train | 1,035,130 | 290,629 | 3.56 |
| `obrascompletasdefigarotomo1.txt` | train | 1,019,509 | 279,704 | 3.64 |
| `obrascompletasdefigarotomo2.txt` | train | 815,086 | 228,408 | 3.57 |
| `obrasescogidas.txt` | train | 552,338 | 156,375 | 3.53 |
| `pepitajimenez.txt` | train | 315,000 | 89,749 | 3.51 |
| `prosasprofanas.txt` | train | 96,316 | 30,812 | 3.13 |
| `recuerdosdeltiempoviejo.txt` | train | 426,337 | 123,792 | 3.44 |
| `romancedeloboscomediabarbara.txt` | train | 131,554 | 43,902 | 3.00 |
| `romancerogitano.txt` | train | 36,477 | 12,142 | 3.00 |
| `sonatadeestiomemoriasdelmarquesdebradomi.txt` | train | 134,704 | 39,805 | 3.38 |
| `sonatadeotonosonatadeinviernomemoriasdel.txt` | train | 265,022 | 77,734 | 3.41 |
| `sonatadeprimaveramemoriasdelmarquesdebra.txt` | train | 115,605 | 33,532 | 3.45 |
| `sotileza.txt` | train | 663,874 | 194,719 | 3.41 |
| `suunicohijo.txt` | train | 515,816 | 146,345 | 3.52 |
| `teatroselectotomo1de4.txt` | train | 507,467 | 169,669 | 2.99 |
| `teatroselectotomo2de4.txt` | train | 529,364 | 185,321 | 2.86 |
| `teatroselectotomo3de4.txt` | train | 417,114 | 145,328 | 2.87 |
| `teatroselectotomo4de4.txt` | train | 548,167 | 195,892 | 2.80 |
| `tiranobanderasnoveladetierracaliente.txt` | train | 304,754 | 98,030 | 3.11 |
| `tormento.txt` | train | 477,885 | 136,326 | 3.51 |
| `trafalgar.txt` | train | 300,188 | 84,080 | 3.57 |
| `traidorinconfesoymartir.txt` | train | 155,673 | 47,366 | 3.29 |
| `tresnovelasejemplaresyunprologo.txt` | train | 156,307 | 48,044 | 3.25 |
| `viagealparnaso.txt` | train | 313,278 | 100,940 | 3.10 |
| `vidadedonquijoteysancho.txt` | train | 555,144 | 156,383 | 3.55 |
| `azul.txt` | val | 190,266 | 57,641 | 3.30 |
| `canasybarro.txt` | val | 443,419 | 128,027 | 3.46 |
| `cuentosydialogos.txt` | val | 223,940 | 67,228 | 3.33 |
| `donalvaroolafuerzadelsino.txt` | val | 149,028 | 49,631 | 3.00 |
| `fuenteovejuna.txt` | val | 96,469 | 35,394 | 2.73 |
| `historiadelavidadelbusconllamadodonpablo.txt` | val | 234,726 | 69,671 | 3.37 |
| `latribuna.txt` | val | 382,827 | 113,983 | 3.36 |
| `lucesdebohemiaesperpento.txt` | val | 120,379 | 44,439 | 2.71 |
| `misericordia.txt` | val | 480,278 | 142,070 | 3.38 |
| `nieblanivola.txt` | val | 319,661 | 98,048 | 3.26 |
| `poesiascompletas.txt` | val | 230,628 | 66,531 | 3.47 |
| `sabnovelaoriginal.txt` | val | 326,485 | 89,822 | 3.63 |

### Verification

`python src/test_roundtrip.py` reads the `.bin` files back with `np.memmap`, splits on EOT,
decodes every work and checks it equals the clean text byte-for-byte; it also checks
`tiny.bin` is a prefix of `train.bin`, all ids < vocab size, and a set of edge-case strings
(accents, ñ, ¿¡, «», em-dash, emoji). Rerunning the pipeline gives byte-identical files.
