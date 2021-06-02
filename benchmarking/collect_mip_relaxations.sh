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
#
# This script illustrates how to collect the instances in the "MIP Relaxations"
# set. Although it can be used directly to collect the benchmark, it is intended
# primarily as a guide to collecting the benchmark - it makes several
# assumptions about the environment. In particular, it assumes wget and unzip
# exist and work as expected.
#
# This collects the MIP relaxations subset of the MIPLIB2017 collection.

if [[ "$#" != 3 ]]; then
  echo "Usage: collect_mip_relaxations.sh temporary_dir" \
      "benchmark_instance_list output_directory" 1>&2
  exit 1
fi

TEMP_DIR="$1"
INSTANCE_FILE="$2"
DEST_DIR="$3"

mkdir -p "${TEMP_DIR}" || exit 1
mkdir -p "${DEST_DIR}" || exit 1

readarray -t instances < <(egrep -v '^#' "${INSTANCE_FILE}")
declare -a filenames
for instance in "${instances[@]}"; do
  filenames+=("${instance}.mps.gz")
done

wget --directory-prefix="${TEMP_DIR}" \
  https://miplib.zib.de/downloads/collection.zip || exit 1

unzip -d "${DEST_DIR}" "${TEMP_DIR}/collection.zip" "${filenames[@]}"
