#!/bin/bash
# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
