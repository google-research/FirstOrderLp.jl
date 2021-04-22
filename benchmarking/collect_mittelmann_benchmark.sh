#!/bin/bash

# This script illustrates how to collect the instances in the "Mittelmann"
# benchmark set. Although it can be used directly to collect the benchmark, it
# is intended primarily as a guide to collecting the benchmark - it makes
# several assumptions about the environment. In particular, it assumes
# wget, tar, bzcat, gzip, zcat, and cc exist and work as expected.

# This collects the union of the instances from:
# * Benchmark of Simplex LP solvers: http://plato.asu.edu/ftp/lpsimp.html
# * Benchmark of Barrier LP solvers: http://plato.asu.edu/ftp/lpbar.html
# * Large Network-LP Benchmark (commercial vs free):
#     http://plato.asu.edu/ftp/network.html

if [[ "$#" != 2 ]]; then
  echo "Usage: collect_mittelmann_benchmark.sh temporary_dir output_dir" 1>&2
  exit 1
fi

TEMP_DIR="$1"
DEST_DIR="$2"

mkdir -p "${TEMP_DIR}" || exit 1
mkdir -p "${DEST_DIR}" || exit 1

cd "${TEMP_DIR}"

# Instances from miplib2010.
wget http://miplib2010.zib.de/download/miplib2010-1.1.3-complete.tgz
tar xzf miplib2010-1.1.3-complete.tgz -C "${DEST_DIR}" --strip-components 3 \
  miplib2010-1.1.3/instances/miplib2010/buildingenergy.mps.gz \
  miplib2010-1.1.3/instances/miplib2010/ds-big.mps.gz \
  miplib2010-1.1.3/instances/miplib2010/rail02.mps.gz \
  miplib2010-1.1.3/instances/miplib2010/shs1023.mps.gz \
  miplib2010-1.1.3/instances/miplib2010/stp3d.mps.gz

# bzipped MPS instances from Mittelmann's lptestset.
for f in chromaticindex1024-7 datt256_lp ex10 fhnw-binschedule0_lp \
    graph40-40_lp irish-electricity neos-3025225_lp neos-5052403-cygnet \
    neos-5251015_lp physiciansched3-3 qap15 rmine15_lp s82_lp s100 s250r10 \
    savsched1 scpm1_lp set-cover-model square41 supportcase10 \
    tpl-tub-ws1617_lp; do
  wget -nv -O - "http://plato.asu.edu/ftp/lptestset/${f}.mps.bz2" | bzcat | \
       gzip > "${DEST_DIR}/${f}.mps.gz"
done

# bzipped MPS instances from Mittelmann's lptestset, removing a trailing "_lp"
# from the name.
for f in datt256 graph40-40 neos-3025225 neos-5251015 rmine15 s82 scpm1; do
  mv "${DEST_DIR}/${f}_lp.mps.gz" "${DEST_DIR}/${f}.mps.gz"
done

# bzipped MPS instances from Mittelmann's lptestset, renaming the file to match
# the testset name.
mv "${DEST_DIR}/chromaticindex1024-7.mps.gz" "${DEST_DIR}/chrom1024-7.mps.gz"
mv "${DEST_DIR}/fhnw-binschedule0_lp.mps.gz" "${DEST_DIR}/fhnw-bin0.mps.gz"
mv "${DEST_DIR}/irish-electricity.mps.gz" "${DEST_DIR}/irish-e.mps.gz"
mv "${DEST_DIR}/neos-5052403-cygnet.mps.gz" "${DEST_DIR}/neos5052403.mps.gz"
mv "${DEST_DIR}/physiciansched3-3.mps.gz" "${DEST_DIR}/psched3-3.mps.gz"
mv "${DEST_DIR}/set-cover-model.mps.gz" "${DEST_DIR}/set-cover.mps.gz"
mv "${DEST_DIR}/supportcase10.mps.gz" "${DEST_DIR}/support10.mps.gz"
mv "${DEST_DIR}/tpl-tub-ws1617_lp.mps.gz" "${DEST_DIR}/tpl-tub-ws16.mps.gz"

# Download and compile the netlib tool for uncompressing files in the
# "compressed MPS" format.
wget -nv http://www.netlib.org/lp/data/emps.c
cc -O3 -o emps emps.c

# bzipped "compressed MPS" instances from Mittelmann's lptestset.
for f in L1_sixm250obs L1_sixm1000obs Linf_520c; do
  wget -nv -O - "http://plato.asu.edu/ftp/lptestset/${f}.bz2" | bzcat | \
       ./emps | gzip > "${DEST_DIR}/${f}.mps.gz"
done

# gzipped "compressed MPS" instances from Mittelmann's lptestset.
for f in fome/fome13 misc/cont1 misc/cont11 misc/neos misc/neos3 \
    misc/ns1687037 misc/ns1688926 misc/stormG2_1000 nug/nug08-3rd pds/pds-100 \
    rail/rail4284; do
  instance="$(basename $f)"
  wget -nv -O - "http://plato.asu.edu/ftp/lptestset/${f}.gz" | zcat | \
       ./emps | gzip > "${DEST_DIR}/${instance}.mps.gz"
done

# gzipped mps network instances from Mittelmann's lptestset.
for f in 16_n14 i_n13 lo10 long15 netlarge1 netlarge2 netlarge3 netlarge6 \
    square15 wide15; do
  wget -nv -O "${DEST_DIR}/${f}.mps.gz" \
      "http://plato.asu.edu/ftp/lptestset/network/${f}.mps.gz"
done

# gzipped "compressed MPS" instances from Meszaros.
for f in infeas/self misc/stat96v1 New/degme New/karted New/tp-6 \
    New/ts-palko; do
  instance="$(basename $f)"
  wget -nv -O - \
      "http://old.sztaki.hu/~meszaros/public_ftp/lptestset/${f}.gz" | zcat | \
    ./emps | gzip > "${DEST_DIR}/${instance}.mps.gz"
done

# The remaining script covers instances that are no longer included in
# Mittelmann's benchmarks, but had been in the somewhat recent past.

if false; then
  # Discontinued instances from miplib2010.
  tar xzf miplib2010-1.1.3-complete.tgz -C "${DEST_DIR}" --strip-components 3 \
    miplib2010-1.1.3/instances/miplib2010/ns1644855.mps.gz

  # Discontinued bzipped MPS instances from Mittelmann's lptestset
  for f in brazil3; do
    wget -nv -O - "http://plato.asu.edu/ftp/lptestset/${f}.mps.bz2" | bzcat | \
         gzip > "${DEST_DIR}/${f}.mps.gz"
  done

  # Discontinued gzipped "compressed MPS" instances from Mittelmann's lptestset.
  for f in misc/neos1 misc/neos2 misc/watson_2 pds/pds-40; do
    instance="$(basename $f)"
    wget -nv -O - "http://plato.asu.edu/ftp/lptestset/${f}.gz" | zcat | \
         ./emps | gzip > "${DEST_DIR}/${instance}.mps.gz"
  done

  # Discontinued gzipped "compressed MPS" instances from Mezaros's website.
  for f in misc/dbic1 misc/nug15 misc/stat96v4; do
    instance="$(basename $f)"
    wget -nv -O - \
        "http://old.sztaki.hu/~meszaros/public_ftp/lptestset/${f}.gz" | zcat | \
      ./emps | gzip > "${DEST_DIR}/${instance}.mps.gz"
  done
fi
