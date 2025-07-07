#!/usr/bin/env bash
#
# Spherify collision meshes and convert all relative visual mesh/image filenames 
# in the URDF to absolute `file://` URLs.
# Usage:
#   source prep_urdf.bash my_robot.urdf
#
# Result:
#   my_robot_prepared.urdf  (in the same directory)
# 
# Environment variables set:
#   ROBOT_DESCRIPTION: the original URDF with absolute paths
#   SPHERIZED_ROBOT_DESCRIPTION: the URDF with sphereified collision meshes and absolute paths

set -eu

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <urdf-file>" >&2
  exit 1
fi

INPUT="$1"
# absolute directory that contains the URDF
URDF_DIR="$(cd "$(dirname "$INPUT")" && pwd)"
URDF_BASE="$(basename "$INPUT")"
TEMP_OUTPUT="${URDF_DIR}/${URDF_BASE%.*}_spherized.urdf"
ABSOLUTIFIED_OUTPUT="${URDF_DIR}/${URDF_BASE%.*}_absolute_paths.urdf"
PREPARED_OUTPUT="${URDF_DIR}/${URDF_BASE%.*}_prepared.urdf"

# ------------------------------------------------------------------
# rewrite_mesh_paths():
#   – scans every line for  filename="…"
#   – if value already begins with "/" or "file:/" it is left intact
#   – otherwise it is rewritten to  file://$URDF_DIR/<value>
# ------------------------------------------------------------------
rewrite_mesh_paths() {
  local in="$1" out="$2" root="$3"

  awk -v root="$URDF_DIR" '
  function rewrite(line,    res,off,pre,token,path) {
    res=""; off=1;
    while (match(substr(line,off), /filename="[^"]+"/)) {
       pre   = substr(line, off, RSTART-1);
       token = substr(line, off+RSTART-1, RLENGTH);

       # strip filename="  ...  "
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
  { print rewrite($0) }' "$in" > "$out"
}

# Spherify collision meshes of the robot
python3 /opt/foam/scripts/generate_sphere_urdf.py $INPUT --output $TEMP_OUTPUT --sphere-database $URDF_DIR/sphere_database.json


# Absolutify paths in the URDF
rewrite_mesh_paths "$INPUT" "$ABSOLUTIFIED_OUTPUT" "$URDF_DIR"
rewrite_mesh_paths "$TEMP_OUTPUT" "$PREPARED_OUTPUT" "$URDF_DIR"

# Remove the temporary file
rm $TEMP_OUTPUT 

export ROBOT_DESCRIPTION="$(cat $ABSOLUTIFIED_OUTPUT)"
export SPHERIZED_ROBOT_DESCRIPTION="$(cat $PREPARED_OUTPUT)"

echo -e "\n================================================"
echo -e "\033[36m⇒  created  $ABSOLUTIFIED_OUTPUT\033[0m"
echo -e "\033[36m⇒  created  $PREPARED_OUTPUT\033[0m"
echo -e "\033[36m⇒  set env vars: ROBOT_DESCRIPTION to hold the original URDF and SPHERIZED_ROBOT_DESCRIPTION to hold the URDF with sphereified collision meshes\033[0m"
echo "================================================"