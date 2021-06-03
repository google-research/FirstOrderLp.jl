#!/bin/bash
# Copyright 2021 The FirstOrderLp Authors
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
# Generates a presolved version of a filtered selection of a benchmark
# collection.

if [[ "$#" -ne 4 ]]; then
    echo "Usage: preprocess.sh path_to_benchmark benchmark_instance_list" \
        "output_directory path_to_papilo_binary"
    exit 1
fi

JULIA="${JULIA:-julia}"

benchmark_path="$1"
benchmark_instance_list="$2"
output_directory="$3"
papilo_binary="$4"

if [[ ! -d "${benchmark_path}" ]]; then
    echo "benchmark path does not exist: ${benchmark_path}"
    exit 1
fi

if [[ ! -f "${benchmark_instance_list}" ]]; then
    echo "Unable to read benchmark instance list: ${benchmark_instance_list}"
    exit 1
fi

if [[ ! -x "${papilo_binary}" ]]; then
    echo "PaPILO binary not found: ${papilo_binary}"
    exit 1
fi

mkdir -p "${output_directory}" || exit 1

gunzip_scratch_dir="$(mktemp -d -p "${output_directory}")"
relaxation_scratch_dir="$(mktemp -d -p "${output_directory}")"

while read instance_name; do
    if [[ "$instance_name" == \#* || -z "$instance_name" ]]; then
        # Skip empty lines and lines starting with "#".
        continue
    fi
    echo "Processing ${instance_name}"
    
    gunzip -c "${benchmark_path}/${instance_name}.mps.gz" \
        >"${gunzip_scratch_dir}/${instance_name}.mps"
    "${JULIA}" --project=. drop_integrality.jl \
        "${gunzip_scratch_dir}/${instance_name}.mps" \
        "${relaxation_scratch_dir}/${instance_name}.mps"
    if (( $? != 0 )); then
        echo "drop_integrality.jl failed"
        exit 1
    fi
    # The "detectlindep" pass is disabled because it uses an LU factorization,
    # which is sometimes excessively slow. For example, it takes about 20
    # minutes for datt256.
    # NOTE: PaPILO appears to take
    # "${relaxation_scratch_dir}/${instance_name}.mps" and use it for the NAME
    # of the instance in the MPS file.
    "${papilo_binary}" presolve \
        -f "${relaxation_scratch_dir}/${instance_name}.mps" \
        -r "${output_directory}/${instance_name}.mps" \
        -v "${output_directory}/${instance_name}.postsolve" \
        --presolve.detectlindep=0
    if (( $? != 0 )); then
        echo "PaPILO failed"
        exit 1
    fi
    gzip "${output_directory}/${instance_name}.mps"
    gzip "${output_directory}/${instance_name}.postsolve"
    rm "${gunzip_scratch_dir}/${instance_name}.mps"
    rm "${relaxation_scratch_dir}/${instance_name}.mps"
done < "${benchmark_instance_list}"

rm -r "${gunzip_scratch_dir}" "${relaxation_scratch_dir}"
