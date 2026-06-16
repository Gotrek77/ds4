# Piano di sviluppo SRM — Completezza totale

## Stato attuale (baseline)

- **CSP**: supporta `constrain(Var,int,LO,HI)`, confronti (`>`, `<`, `>=`, `<=`, `=`), aritmetica (`+`, `-`, `*`, `/`) con risultato variabile o numerico. Propagazione solo per aritmetica con variabile risultato. Backtracking naive senza euristiche.
- **Datalog**: forward chaining su fatti, backward chaining su goal (max profondità 20). Nessun tabling. Aritmetica (`add/sub/mul/div/min`) valutabile solo in forward.
- **Tool agent**: `srm_tool.c` espone assert/query/prove/eval/save/load/reset. **Manca** CSP solver e water jug.
- **Water jug**: BFS funzionante, esposto solo nel CLI standalone.

---

## Stato implementazione

### ✅ Priorità 1 — Esporre constraint solver a ds4-agent
- [x] 1.1 Azione `csp` in `srm_tool_exec` — aggiunta
- [x] 1.2 Azione `water` in `srm_tool_exec` — aggiunta
- [ ] 1.3 Esporre funzioni pubbliche nell'header

### ✅ Priorità 2 — Confronti attivi con propagazione
- [x] 2.1 `≠` (not-equal) — parser, check_ex, forward_check
- [x] 2.2 Propagazione comparativa — forward_check (var-var e var-const)
- [x] 2.3 Propagazione uguaglianza — forward_check (implicito)
- [x] Extra: confronti con costante (`A > N`, `A < N`, `A >= N`, `A <= N`, `A != N`)

### ✅ Priorità 3 — Propagazione numerica RHS
- [x] 3.1 A+B=N con A noto → B=N-A
- [x] 3.2 A-B=N con A noto → B=A-N
- [x] 3.3 A*B=N con A noto → B=N/A
- [x] 3.4 A/B=N con A noto → B=A/N
- [x] 3.5 Simmetrico: B noto → A

### ✅ Priorità 6 (parziale) — Miglioramenti parser
- [x] 6.3 `VAR in [LO, HI]` syntax per range variabili
- [x] Auto-creazione variabili con default [0,10] se non definite
- [ ] 6.1 `not` in query
- [ ] 6.2 Rilevamento automatico tipo problema
- [ ] 6.4 All-different constraint
- [ ] 6.5 Ottimizzazione (min/max)

### 🟡 Bug fix
- [x] `A+B=C` ora funziona (era: variabili non create → nessuna soluzione)
- [x] `A+B=C, C>5` ora funziona (era: forward_check non passava rhss)

### ❌ Priorità 4 — Tabling per ricorsione (da fare)
### ❌ Priorità 5 — Aritmetica in backward chaining (da fare)

---

## Ordine di implementazione

```
┌─────────────────────────────────────────────┐
│ 1. Esporre CSP + Water a ds4-agent          │ ← FATTO (manca export header)
├─────────────────────────────────────────────┤
│ 2. Confronti attivi con propagazione        │ ← FATTO
├─────────────────────────────────────────────┤
│ 3. Propagazione numerica RHS                │ ← FATTO
├─────────────────────────────────────────────┤
│ 4. Tabling per ricorsione                   │ ← DA FARE
├─────────────────────────────────────────────┤
│ 5. Aritmetica in backward chaining          │ ← DA FARE
├─────────────────────────────────────────────┤
│ 6. Miglioramenti parser e generali          │ ← PARZIALE
└─────────────────────────────────────────────┘
```
