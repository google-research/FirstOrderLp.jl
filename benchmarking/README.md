# Benchmarking

This directory contains scripts for generating datasets used for benchmarking
FirstOrderLp.

## Filtered and preprocessed MIPLIB 2017 collection dataset

1. From the local directory, instantiate the necessary packages by running
   `julia --project=. -e 'import Pkg; Pkg.instantiate()'`.
2. Download the MIPLIB 2017 collection from https://miplib.zib.de/downloads/collection.zip. Unzip it locally.
3. Download and build PaPILO from https://github.com/lgottwald/PaPILO.
4. From the local directory, run `./preprocess.sh`.

`preprocess.sh` has the following argument structure:

```sh
$ ./preprocess.sh path_to_miplib output_directory path_to_papilo_binary
```

For example:

```sh
$ ./preprocess.sh ~/miplib2017_collection ~/preprocessed_dataset ~/PaPILO/build/bin/papilo
```