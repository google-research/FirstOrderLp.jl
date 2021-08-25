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
# This script illustrates how to collect the instances in the "Netlib Benchmark"
# set. Although it can be used directly to collect the benchmark, it is intended
# primarily as a guide to collecting the benchmark - it makes several
# assumptions about the environment. In particular, it assumes wget, gzip,
# gfortran, sed, and cc exist and work as expected.
#
# This collects the linear programming instances from netlib:
# https://www.netlib.org/lp/data

if [[ "$#" != 2 ]]; then
  echo "Usage: collect_netlib_benchmark.sh temporary_directory" \
    "output_directory" 1>&2
  exit 1
fi

TEMP_DIR="$1"
DEST_DIR="$2"

mkdir -p "${TEMP_DIR}" || exit 1
mkdir -p "${DEST_DIR}" || exit 1

# Download and compile the netlib tool for uncompressing files in the
# "compressed MPS" format.
wget --directory-prefix="${TEMP_DIR}" -nv http://www.netlib.org/lp/data/emps.c
cc -O3 -o "${TEMP_DIR}/emps" "${TEMP_DIR}/emps.c"

# Download and convert the main netlib instances.
for f in 25fv47 80bau3b adlittle afiro agg agg2 agg3 bandm beaconfd blend bnl1 \
  bnl2 boeing1 boeing2 bore3d brandy capri cycle czprob d2q06c d6cube degen2 \
  degen3 dfl001 e226 etamacro fffff800 finnis fit1d fit1p fit2d fit2p forplan \
  ganges gfrd-pnc greenbea greenbeb grow15 grow22 grow7 israel kb2 lotfi maros \
  maros-r7 modszk1 nesm perold pilot pilot.ja pilot.we pilot4 pilot87 pilotnov \
  recipe sc105 sc205 sc50a sc50b scagr25 scagr7 scfxm1 scfxm2 scfxm3 scorpion \
  scrs8 scsd1 scsd6 scsd8 sctap1 sctap2 sctap3 seba share1b share2b shell \
  ship04l ship04s ship08l ship08s ship12l ship12s sierra stair standata \
  standgub standmps stocfor1 stocfor2 tuff vtp.base wood1p woodw; do
  wget -nv -O - "http://www.netlib.org/lp/data/${f}" | "${TEMP_DIR}/emps" | \
    gzip > "${DEST_DIR}/${f}.mps.gz"
done

# Download and convert the kennington instances.
for f in cre-a cre-b cre-c cre-d ken-07 ken-11 ken-13 ken-18 osa-07 osa-14 \
  osa-30 osa-60 pds-02 pds-06 pds-10 pds-20; do
  wget -nv -O - "http://www.netlib.org/lp/data/kennington/${f}.gz" \
    | gzip -d | "${TEMP_DIR}/emps" | gzip > "${DEST_DIR}/${f}.mps.gz"
done

# qap is given via a fortran generator and data files.
wget --directory-prefix="${TEMP_DIR}" -nv \
  "http://www.netlib.org/lp/generators/qap/newlp.f"
gfortran -O3 -o "${TEMP_DIR}/newlp" "${TEMP_DIR}/newlp.f"
for n in 8 12 15; do
  wget -nv -O - "http://www.netlib.org/lp/generators/qap/data.${n}" | \
    "${TEMP_DIR}/newlp" | gzip > "${DEST_DIR}/qap${n}.mps.gz"
done

# stocfor3 is given as a shell script containing fortran code and data files
# (using the default fortran unit number->file name mapping).
wget --directory-prefix="${TEMP_DIR}" -nv \
  "http://www.netlib.org/lp/data/stocfor3"
(
  cd "${TEMP_DIR}"
  bash stocfor3
  # Fix a couple of instances of bit-rot (compilers getting pickier).
  sed -i.orig -e 's/ INTEGER\*2 / INTEGER*4 /' common6.for
  gfortran -O3 --std=legacy -o std2mps std2mps.f input.f
  ln -s time7.frs fort.1
  ./emps < core.mpc > fort.2
  ln -s stoch3.frs fort.3
  ./std2mps
  cat fort.11 fort.12 fort.13 fort.14 fort.15 | gzip > stocfor3.mps.gz
)
cp "${TEMP_DIR}/stocfor3.mps.gz" "${DEST_DIR}/stocfor3.mps.gz"

# truss is given as a shell script containing fortran code and data files
# (using hardcoded input and output file names).
wget -nv --directory-prefix="${TEMP_DIR}" "http://www.netlib.org/lp/data/truss"
(
  cd "${TEMP_DIR}"
  bash truss
  gfortran -O3 -o truss.exe truss.f
  ./truss.exe
  gzip < mps > truss.mps.gz
)
cp "${TEMP_DIR}/truss.mps.gz" "${DEST_DIR}/truss.mps.gz"
