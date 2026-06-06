#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PATH="/Library/TeX/texbin:${PATH:-}"

INPUT="${1:-problem_1_short_answers.md}"
OUTPUT="${2:-problem_1.pdf}"
HEADER="pandoc-pdf-header.tex"
FILTER="pandoc-table-columns.lua"

ARGS=(
  "$INPUT" -o "$OUTPUT"
  --from markdown+tex_math_dollars
  -V geometry:margin=1in
  -V fontsize=11pt
  --pdf-engine=xelatex
  -H "$HEADER"
)

if [[ -f "$FILTER" ]]; then
  ARGS+=(--lua-filter="$FILTER")
fi

pandoc "${ARGS[@]}"
echo "Wrote $OUTPUT"
