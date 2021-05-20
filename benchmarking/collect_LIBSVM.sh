#!/bin/bash

# Collects the three datasets from LIBSVM used in the experiments.
# It assumes that the environment has curl and bunzip2 installed
# and working.


if [[ "$#" != 2 ]]; then
    echo "Usage: collect_LIBSVM.sh temporary_dir output_dir" 1>&2
    exit 1
fi

TEMP_DIR="$1"
DEST_DIR="$2"

mkdir -p "${TEMP_DIR}" || exit 1
mkdir -p "${DEST_DIR}" || exit 1

# To download the binary problems:
data_source="https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary"
for filename in kdda.t real-sim avazu-app.val; do
    curl "${data_source}/${filename}.bz2" --output "${TEMP_DIR}/${filename}.bz2" || exit 1
    bunzip2 -d "${TEMP_DIR}/${filename}.bz2"
    mv "${TEMP_DIR}/${filename}" "${DEST_DIR}/${filename}"
done
