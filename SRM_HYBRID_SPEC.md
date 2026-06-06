# Specifica Tecnica: Symbolic Reasoning Module (SRM) per DS4

## AI Ibrida Simbolica + Neurale — Rev. 2 (Tool Esterno)

### 1. Visione

L'AI ibrida simbolica+neurale combina i punti di forza dei due paradigmi:

- **Neurale** (DeepSeek V4 Flash/PRO): ottimo per comprensione del linguaggio, intuito, pattern matching, conoscenza implicita.
- **Simbolico**: ottimo per ragionamento deterministico, manipolazione di regole, vincoli logici, algebra, verifica formale, e conoscenza strutturata che il modello potrebbe non avere o potrebbe allucinare.

L'integrazione permette al modello di **delegare** compiti di ragionamento strutturato a un motore simbolico deterministico, riducendo allucinazioni e migliorando la precisione su problemi che richiedono logica ferrea.

### 2. Architettura: Tool Esterno

A differenza della prima revisione, il SRM **non è un modulo interno a DS4**, ma un **binario C standalone** (`ds4-srm`) che comunica con DS4 tramite:

1. **Tool `srm` nell'agente DS4**: poche righe di dispatch che eseguono `ds4-srm` con argomenti e ne catturano l'output.
2. **Tool `srm` nel server DS4**: tool definition OpenAI-style che invoca il binario esterno.
3. **CLI diretta**: `ds4-srm` può essere usato anche indipendentemente da DS4, per test e scripting.

```
┌────────────────────────────────────────────────────┐
│                    DS4 (DwarfStar)                  │
│                                                     │
│  ┌──────────────────────┐    ┌──────────────────┐  │
│  │   ds4-agent          │    │   ds4-server     │  │
│  │   (tool dispatch)    │    │   (tool defs)     │  │
│  └──────┬───────────────┘    └──────┬───────────┘  │
│         │ exec + stdout            │ exec + stdout │
│         ▼                          ▼               │
│  ┌──────────────────────────────────────────────┐  │
│  │           ds4-srm (binario esterno)          │  │
│  │                                              │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐   │  │
│  │  │  Rule     │  │  Logic   │  │  KB      │   │  │
│  │  │  Engine   │  │  Solver  │  │  Store   │   │  │
│  │  └──────────┘  └──────────┘  └──────────┘   │  │
│  │       │              │              │        │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐   │  │
│  │  │Unification│  │Constraint│  │  Query   │   │  │
│  │  │  & Match  │  │  Solver  │  │  Engine  │   │  │
│  │  └──────────┘  └──────────┘  └──────────┘   │  │
│  └──────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────┘
```

#### Vantaggi dell'approccio tool esterno

- Modifiche minime a DS4: ~20 righe in `ds4_agent.c` per il dispatch, più tool definitions in `ds4_server.c`.
- Il SRM è un progetto autonomo, testabile separatamente senza toccare DS4.
- Può essere sviluppato, compilato e aggiornato indipendentemente.
- Può essere usato anche da altri agenti/script via CLI diretta.
- Lo stato SRM (fatti, regole) può essere salvato su disco indipendentemente dal KV cache.

### 3. Il binario `ds4-srm`

#### 3.1 Modalità di comunicazione

Il binario supporta due modalità:

**A) CLI diretta (argomenti):**
```
ds4-srm --assert "fact: umano(socrate)"
ds4-srm --assert "rule: mortale(X) ← umano(X)"
ds4-srm --query "mortale(socrate)?"
→ VERO
ds4-srm --query "nonno(X, Y)?"
→ X=pluto, Y=paperino
ds4-srm --solve "A+B=10, A>B, A,B∈int 1..9"
→ A=9,B=1 | A=8,B=2 | ...
ds4-srm --prove "mortale(socrate)" --max-depth 10
→ PROVATO: umano(socrate) → mortale(socrate)
```

**B) JSON su stdin/stdout (per tool dispatch):**
```json
// Input
{ "action": "assert", "type": "fact", "content": "umano(socrate)" }

// Output
{ "status": "ok" }
```

```json
// Input
{ "action": "query", "content": "mortale(socrate)?" }

// Output
{ "status": "ok", "result": "VERO", "bindings": [] }
```

La modalità JSON è quella usata dall'agente DS4 e dal server. La CLI diretta è per test e uso manuale.

#### 3.2 Componenti interni

**a) Rule Engine (Motore di regole)**
- Forward chaining: applica regole `IF condizione THEN azione` fino a saturazione.
- Backward chaining: dato un goal, cerca regole che lo dimostrano, con backtracking.
- Supporto per regole con variabili e pattern matching (unificazione).

**b) Logic Solver (Risolutore logico)**
- Risoluzione di clausole Horn (Prolog-like).
- Supporto per logica proposizionale e del prim'ordine.
- Interfaccia per dimostrazione di teoremi su conoscenza di base.

**c) Constraint Solver (Risolutore di vincoli)**
- Vincoli su domini finiti (CSP): variabili con domini, relazioni aritmetiche, alldifferent, ecc.
- Propagazione di vincoli con backtracking limitato.
- Utile per problemi di scheduling, pianificazione, configurazione.

**d) Knowledge Base Store (Archivio di conoscenza)**
- Grafo di conoscenza in memoria: entità, relazioni, attributi.
- Supporto per triplo `(soggetto, predicato, oggetto)`.
- Query pattern matching con variabili.
- Persistenza su disco (formato binario semplice o JSON).

**e) Query Engine (Motore di interrogazione)**
- Linguaggio di query testuale (es. `?x genitore di ?y AND ?y amico di ?z`).
- Traduzione delle query in piani di esecuzione logica.
- Integrazione con la KB e il rule engine per rispondere a domande complesse.

**f) Unification & Pattern Matching**
- Unificazione per espressioni con variabili (es. per matching di regole).
- Supporto per tipi semplici (int, string, symbol, list).

### 4. Integrazione con DS4

#### 4.1 Tool `srm` nell'agente nativo

Viene aggiunto un tool DSML `srm` che l'agente può chiamare. Il dispatch esegue il binario esterno `ds4-srm` con argomenti e cattura l'output.

**Formato DSML:**
```
<|DSML|>tool_calls
<|DSML|>invoke name="srm"
<|DSML|>parameter name="action" string="true">assert fact: umano(socrate)</|DSML|>parameter
</|DSML|>invoke>
</|DSML|>tool_calls>
```

**Azioni supportate dal tool:**
- `assert fact: <testo>` — aggiunge un fatto
- `assert rule: <testo>` — aggiunge una regola
- `query: <testo>?` — interroga la knowledge base
- `solve: <vincoli>` — risolve vincoli
- `prove: <goal>` — dimostra un goal
- `eval: <espressione>` — valuta espressione logica
- `save: <path>` — salva stato su disco
- `load: <path>` — carica stato da disco
- `clear` — resetta tutto

#### 4.2 Tool `srm` nel server API

Il server `ds4-server` espone il tool `srm` come tool definition OpenAI-style su `/v1/chat/completions`, `/v1/responses` e `/v1/messages`. Il server invoca `ds4-srm` in modalità JSON su stdin/stdout.

#### 4.3 Stato persistente

Lo stato del SRM (fatti, regole, KB) può essere:
- Salvato su disco come file JSON dal binario `ds4-srm` stesso (comando `save:` / `load:`).
- Opzionalmente, incluso nel KV cache di DS4 come sezione `SRM` nel file `.kv`.

La persistenza separata è più semplice e non richiede modifiche a `ds4_kvstore.c`.

#### 4.4 Comandi `/srm` nell'agente

L'agente DS4 supporta anche comandi interattivi:
```
ds4> /srm assert fact: umano(socrate)
ds4> /srm query: mortale(socrate)?
→ VERO
ds4> /srm show
→ Fatti: umano(socrate)
   Regole: mortale(X) ← umano(X)
```

### 5. Formato conoscenza e regole

Il linguaggio delle regole è ispirato a Datalog/Prolog semplificato, testuale:

```
# Fatti — predicato con argomenti costanti
umano(socrate).
genitore(pluto, paperino).
genitore(paperino, paperina).

# Regole — clausole Horn con variabili (maiuscole)
mortale(X) ← umano(X).
nonno(X,Y) ← genitore(X,Z) ∧ genitore(Z,Y).

# Vincoli per constraint solving
constrain(A, int, 1, 9).
constrain(B, int, 1, 9).
A + B = 10.
A > B.

# Query
? mortale(socrate).
? nonno(X, Y).
```

### 6. File sorgenti

```
ds4-srm.c       – Implementazione completa del SRM (~3000 righe, stile monolitico)
ds4-srm.h       – Header con API pubblica (opzionale, per future estensioni)
ds4-srm-test.c  – Test suite (~500 righe)
Makefile.srm    – Build per ds4-srm (integrato poi nel Makefile principale)
```

Nessuna modifica a file DS4 esistenti finché il binario non è stabile e testato.

### 7. Build

```makefile
# Makefile.srm
ds4-srm: ds4-srm.c
	$(CC) $(CFLAGS) -o ds4-srm ds4-srm.c -lm

ds4-srm-test: ds4-srm-test.c ds4-srm.c
	$(CC) $(CFLAGS) -o ds4-srm-test ds4-srm-test.c ds4-srm.c -lm

test-srm: ds4-srm-test
	./ds4-srm-test
```

Nessuna dipendenza esterna: tutto C puro, stile DS4.

### 8. Test

La test suite `ds4-srm-test.c` copre:

1. **Fatti**: aggiunta e recupero di fatti semplici
2. **Regole forward**: aggiunta regola, verifica che le conclusioni siano derivate
3. **Regole backward**: query con goal e backtracking
4. **Query con variabili**: binding di variabili nelle query
5. **Unificazione**: pattern matching con variabili
6. **Constraint solving**: problemi semplici con vincoli
7. **Persistenza**: salva e carica stato
8. **Errori**: gestione di input malformati
9. **Regressione**: risultati noti su casi standard

I test sono autogestiti (nessun framework esterno), con output `PASSED`/`FAILED`.

### 9. Roadmap

| Fase | Descrizione | Dipendenze |
|------|-------------|------------|
| 1 | Implementare `ds4-srm.c` (Rule Engine + KB + Query + Unification) | nessuna |
| 2 | Aggiungere Logic Solver e Constraint Solver | fase 1 |
| 3 | Implementare test suite `ds4-srm-test.c` | fase 1-2 |
| 4 | Test approfonditi e validazione | fase 3 |
| 5 | Integrare tool `srm` in `ds4_agent.c` | fase 4 |
| 6 | Integrare tool `srm` in `ds4_server.c` | fase 5 |
| 7 | Documentazione finale e merge | fase 6 |
