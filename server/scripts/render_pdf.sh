#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: render_pdf.sh <input.musicxml> [output.pdf]" >&2
  exit 1
fi

input_path="$1"
output_path="${2:-${input_path%.*}.pdf}"

mscore_bin="${MSCORE_BIN:-${TABSERVER_MSCORE:-mscore}}"

"$mscore_bin" -o "$output_path" "$input_path"
echo "PDF genere: $output_path"
