# Piano Test

## Obiettivo

Capire quanto una pipeline automatica riesce a produrre una composizione vettoriale utile per ricamo, senza trasformare ogni oggetto in un ricalco frammentato.

## Casi consigliati

1. Oggetto semplice su sfondo uniforme.
2. Pallone con colore base chiaro e grafiche colorate sopra.
3. Oggetto con ombra morbida sullo sfondo.
4. Oggetto con colori simili tra base e decorazioni.
5. Oggetto con dettagli piccoli da ignorare o accorpare.

## Criteri di valutazione

- Il contorno principale del soggetto e' unico e riconoscibile.
- Il colore base copre l'intera sagoma del soggetto.
- Le grafiche sopra sono esportate come livelli separati.
- I frammenti piccoli sono pochi e controllabili.
- Le forme sono chiuse e utilizzabili come base per campiture.
- L'ordine dei livelli ha senso per il ricamo: base sotto, dettagli sopra.

## Prossimi innesti AI

- Segmentazione soggetto: SAM/SAM2, rembg o modello vision esterno.
- Classificazione semantica: riconoscimento del tipo soggetto.
- Semplificazione guidata: prompt vision per decidere cosa tenere, fondere o scartare.
- Output ricamo: conversione verso formato intermedio usabile dal software di ricamo.

