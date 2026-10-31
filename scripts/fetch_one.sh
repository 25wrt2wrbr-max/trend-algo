#!/bin/zsh
# Fetch daily OHLC history for one ticker via Moli (same-origin Yahoo recipe).
# Usage: fetch_one.sh TICKER OUTFILE PERIOD
TICKER="$1"; OUT="$2"; PERIOD="${3:-5y}"
moli fetch --eval $'
(async () => {
  const r = await fetch("https://query1.finance.yahoo.com/v8/finance/chart/'"$TICKER"'?range='"$PERIOD"'&interval=1d");
  return JSON.stringify({ status: r.status, body: await r.text() });
})()' "https://finance.yahoo.com/quote/$TICKER" > "$OUT" 2>"$OUT.err"
echo "exit=$? $(wc -c < "$OUT") bytes -> $OUT"
