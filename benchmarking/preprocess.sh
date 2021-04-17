#!/bin/bash

set -e  # Exit when any subcommand fails.

if [[ "$#" -ne 3 ]]; then
    echo "Usage: preprocess.sh [path to miplib] [output directory] [path to papilo binary]"
    exit 1
fi

if [[ ! -f miplib2017_instance_list ]]; then
    echo "Unable to find miplib2017_instance_list in the current directory."
    exit 1
fi

# Directory containing Miplib 2017 collection instances.
miplib_path="$1"
output_directory="$2"
papilo_binary="$3"

mkdir -p "${output_directory}"

gunzip_scratch_dir=$(mktemp -d -p "${output_directory}")
relaxation_scratch_dir=$(mktemp -d -p "${output_directory}")

while read instance_name; do
    if [[ "$instance_name" == \#* ]]; then
        # Skip lines starting with "#".
        continue
    fi
    echo "${instance_name}"
    
    cp "${miplib_path}/${instance_name}.mps.gz" "${gunzip_scratch_dir}"
    gunzip "${gunzip_scratch_dir}/${instance_name}.mps.gz"
    julia --project=$(pwd) drop_integrality.jl \
        "${gunzip_scratch_dir}/${instance_name}.mps" \
        "${relaxation_scratch_dir}/${instance_name}.mps"
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
    gzip "${output_directory}/${instance_name}.mps"
    rm "${gunzip_scratch_dir}/${instance_name}.mps"
    rm "${relaxation_scratch_dir}/${instance_name}.mps"
done <miplib2017_instance_list

rm -r "${gunzip_scratch_dir}" "${relaxation_scratch_dir}"
