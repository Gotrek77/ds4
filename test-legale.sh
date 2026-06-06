#!/bin/bash
# Esempio legale: diritto successorio (inheritance law)
# Mostra l'uso del SRM per ragionamento giuridico
# Tutti i comandi sono in una singola invocazione per condividere la KB

SRM=./ds4-srm

echo "=== DIRITTO SUCCESSORIO: esempio di ragionamento legale ==="
echo ""

# Tizio è il de cuius (defunto)
# Caio e Sempronia sono figli di Tizio
# Lucio e Filomena sono nipoti (figli di Caio)
# Merlina è la coniuge di Tizio

$SRM \
  --assert "fact: genitore(tizio, caio)" \
  --assert "fact: genitore(tizio, sempronia)" \
  --assert "fact: genitore(caio, lucio)" \
  --assert "fact: genitore(caio, filomena)" \
  --assert "fact: coniuge(tizio, merlina)" \
  --assert "fact: coniuge(caio, fulvia)" \
  --assert "rule: discendente(X, Y) ← genitore(X, Y)" \
  --assert "rule: discendente(X, Z) ← genitore(X, Y) ∧ discendente(Y, Z)" \
  --assert "rule: erede(X, Y) ← discendente(X, Y)" \
  --assert "rule: erede(X, Y) ← coniuge(X, Y)" \
  --query "?discendente(tizio, X)" \
  --query "?erede(tizio, X)" \
  --prove "?erede(tizio, caio)"

echo ""
echo "=== FINE ESEMPIO ==="
