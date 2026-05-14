# Embroidery AI Shape Lab

Laboratorio per testare una lettura gerarchica di immagini pensata per ricamo:

- separazione sfondo esterno / soggetto principale;
- scelta di un colore base del soggetto;
- estrazione di aree colorate sovrapposte;
- esportazione SVG con livelli leggibili per campiture di filo;
- report JSON con metadati utili per confrontare i test.

Il progetto contiene due percorsi:

- AI-first: un modello vision genera un piano gerarchico di livelli ricamo, poi il codice lo converte in SVG.
- Baseline locale: segmentazione colore senza AI, utile solo come confronto.
- Color trace: vector tracing sperimentale per colori reali, pensato come baseline piu' fedele e controllabile.

## Struttura

```text
input/                 immagini sorgenti
output/                risultati generati
src/ai_embroidery_plan.py piano AI dei livelli
src/ai_plan_to_svg.py SVG guidato dal piano AI
src/color_trace_svg.py trace colore con contorni curvi
src/image_shape_lab.py baseline locale senza AI
docs/test-plan.md      criteri dei test
```

## Uso

### Pipeline AI

Richiede una chiave API OpenAI:

```powershell
notepad .env
```

Inserire:

```text
OPENAI_API_KEY=sk-...
```

Poi lanciare:

```powershell
.\run_ai_pipeline.bat input\pallone.png
```

Output principali:

- `output/<nome_immagine>/ai_plan.json`: lettura semantica AI;
- `output/<nome_immagine>/composition_ai.svg`: SVG a livelli guidato dall'AI;
- `output/<nome_immagine>/ai_svg_report.json`: report del matching tra piano AI e pixel.

La pipeline usa di default `gpt-5.2`, scelto per capacita' multimodali/vision aggiornate.

### Color Trace

Percorso sperimentale fissato come baseline di confronto per i test attuali:

```powershell
.\run_color_trace.bat input\pallone.png
```

Output:

- `output/<nome_immagine>/composition_color_trace.svg`: SVG a livelli colore;
- `output/<nome_immagine>/color_trace_preview.png`: anteprima raster dei colori quantizzati;
- `output/<nome_immagine>/color_trace_report.json`: numero livelli, path e parametri.

Questo ramo non sostituisce `composition_ai.svg`: serve a ottenere un tracciato colore piu' simile ai software di vector tracing classici. La direzione piu' promettente e' usarlo insieme alla lettura AI, dove il tracing costruisce bordi e campiture e l'AI controlla soggetti, sfondo, livelli e perdita di dettagli.

### SVG Semantico AI

Percorso sperimentale in cui l'AI genera anche i poligoni dei livelli, invece di usare solo il colore dei pixel:

```powershell
.\run_semantic_ai_svg.bat input\pallone.png
```

Output:

- `output/<nome_immagine>/semantic_ai_plan.json`: livelli e poligoni normalizzati prodotti dall'AI;
- `output/<nome_immagine>/semantic_ai.svg`: SVG semantico pulito;
- `output/<nome_immagine>/semantic_ai_report.json`: riepilogo layer/poligoni.

Questo percorso e' utile per testare se l'AI riesce a generare maschere intenzionali per ricamo. Non sostituisce ancora una segmentazione pixel-perfect: serve a capire se la direzione semantica produce forme piu' usabili.

### Verifica AI

Controllo di veridicita' dopo la generazione:

```powershell
.\verify_ai_svg.bat input\pallone.png output\pallone\composition_ai.svg output\pallone\ai_plan.json
```

Output:

- `composition_ai_verification.json`: giudizio su fedelta', sfondo/ombra, layer, bordi, perdita dettagli e parametri consigliati.

Questo serve per costruire un ciclo iterativo:

```text
immagine -> lettura AI livelli -> segmentazione/vettoriale -> verifica AI -> manipolatori -> nuovo SVG
```

Runner iterativo automatico:

```powershell
.\run_iterative_ai.bat input\pallone.png
```

Esegue piu' preset di conversione, verifica ogni output con AI e copia nella cartella principale il miglior tentativo. Il riepilogo finisce in `output/<nome>/iteration_summary.json`.

Manipulator disponibili nella conversione SVG:

- `--edge-smooth`: ammorbidisce i bordi prima del vettoriale.
- `--close-pixels`: chiude piccoli buchi o tagli nelle maschere.
- `--overlap-mode`: controlla se dettagli e sfumature possono sovrapporsi (`allow`, `details-only`, `none`).
- `--shadow-mode`: rimuove ombre grigie vicine allo sfondo quando il piano AI le segnala.
- `--min-region-area`: scarta frammenti sotto una soglia.

### Baseline senza AI

Test rapido con immagine campione:

```powershell
.\run_sample.bat
```

Con il Python bundled di Codex:

```powershell
& "C:\Users\l.ercoli\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" src\image_shape_lab.py input\pallone.png
```

Con un Python installato nel sistema:

```powershell
python src\image_shape_lab.py input\pallone.png
```

Oppure:

```powershell
.\process_image.bat input\pallone.png
```

Output generati in `output/<nome_immagine>/`:

- `composition.svg`: tracciati a livelli;
- `subject_mask.png`: maschera soggetto/sfondo;
- `regions_preview.png`: anteprima delle regioni colore;
- `report.json`: colori, aree e parametri usati.

## Parametri utili

```powershell
python src\image_shape_lab.py input\pallone.png --colors 6 --bg-tolerance 38 --min-region-area 80 --max-size 900
```

- `--colors`: numero di cluster colore da cercare nel soggetto.
- `--bg-tolerance`: tolleranza per riconoscere lo sfondo dai bordi.
- `--min-region-area`: elimina frammenti piccoli dalle sovrapposizioni.
- `--max-size`: ridimensiona immagini grandi per test piu' rapidi.
