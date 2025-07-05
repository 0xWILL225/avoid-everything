#!/usr/bin/env bash
#
# Convert all relative mesh/image filenames in a URDF to absolute `file://` URLs.
# Usage:
#   ./absolutify_urdf.bash my_robot.urdf
#
# Result:
#   my_robot_absolute_paths.urdf  (in the same directory)

set -eu

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <urdf-file>" >&2
  exit 1
fi

INPUT="$1"
# absolute directory that contains the URDF
URDF_DIR="$(cd "$(dirname "$INPUT")" && pwd)"
URDF_BASE="$(basename "$INPUT")"
OUTPUT="${URDF_DIR}/${URDF_BASE%.*}_absolute_paths.urdf"

# ------------------------------------------------------------------
#   – scans every line for  filename="…"
#   – if value already begins with "/" or "file:/" it is left intact
#   – otherwise it is rewritten to  file://$URDF_DIR/<value>
# ------------------------------------------------------------------
awk -v root="$URDF_DIR" '
function rewrite(line,    res,off,pre,token,path) {
  res=""; off=1;
  while (match(substr(line,off), /filename="[^"]+"/)) {
     pre   = substr(line, off, RSTART-1);
     token = substr(line, off+RSTART-1, RLENGTH);
     # extract the path value (strip prefix and trailing quote)
     path = token;
     sub(/^filename="/, "", path);
     sub(/"$/, "", path);

     if (path !~ /^\// && path !~ /^file:\//) {
        token = "filename=\"file://" root "/" path "\"";
     }
     res = res pre token;
     off = off + RSTART + RLENGTH - 1;
  }
  res = res substr(line, off);
  return res;
}
{ print rewrite($0) }' "$INPUT" > "$OUTPUT"

echo "⇒  created  $OUTPUT"