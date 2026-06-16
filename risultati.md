# Test di Performance: Problema delle Brocche (12L, 7L, 5L)

## Problema
Tre brocche di capacità 12L, 7L e 5L. Inizialmente la prima è piena (12L), le altre vuote.
Obiettivo: ottenere esattamente 1 litro in una brocca (1,2,3,4 non sono misurabili direttamente).

## Soluzione (BFS)
Percorso trovato in 9 passi:
```
(12,0,0) → (7,0,5) → (7,5,0) → (2,5,5) → (2,7,3) → (9,0,3) → (9,3,0) → (4,3,5) → (4,7,1)
```

## Performance

### Senza Lobster (BFS in Python)
- Media su 1000 esecuzioni: **31.5 µs** (microsecondi)
- Algoritmo: BFS su spazio degli stati (massimo 624 stati)
- Python puro, nessuna dipendenza esterna.

### Con Lobster (motore Datalog Scallop via CLI scli)
- Media su 100 esecuzioni: **12.2 ms** (millisecondi)
- Include overhead di fork+exec (avvio di scli, caricamento del compilatore, parsing, valutazione).
- La valutazione Datalog in sé è sub-millisecond; l'overhead domina.

### Confronto
- Lobster (come strumento CLI) è ~387× più lento del BFS Python.
- Un motore Datalog in-process (es. binding Python `scallopy`) eviterebbe l'overhead di fork e sarebbe competitivo con il BFS Python.
- Per questo piccolo spazio degli stati, un BFS scritto a mano è l'approccio più efficiente.

## File creati
- `test_waterjug.py` – solver BFS Python e benchmark.
- `waterjug.scl` – programma Datalog Scallop per lo stesso problema.
- `test_lobster.c` – programma C che chiama `lobster_tool_exec()` per eseguire il programma Datalog e misura il tempo.
- `test_results.md` – report in inglese.
