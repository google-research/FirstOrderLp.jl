# Benchmarking

This directory contains scripts for generating datasets used for benchmarking
FirstOrderLp.

## Filtered and preprocessed MIPLIB 2017 collection dataset

1. From the local directory, instantiate the necessary packages by running
   `julia --project=. -e 'import Pkg; Pkg.instantiate()'`.
2. Download the MIPLIB 2017 collection from
   https://miplib.zib.de/downloads/collection.zip. Unzip it locally.
3. Download and build PaPILO from https://github.com/lgottwald/PaPILO.
4. From the local directory, run `./preprocess.sh`.

`preprocess.sh` has the following argument structure:

```sh
$ ./preprocess.sh path_to_benchmark benchmark_instance_list output_directory \
    path_to_papilo_binary
```

For example:

```sh
$ ./preprocess.sh ~/miplib2017_collection ./miplib2017_instance_list \
    ~/miplib2017_preprocessed ~/PaPILO/build/bin/papilo
```

## Filtered and preprocessed Mittelmann benchmark dataset

This benchmark contains the union of the instances from Hans Mittelmann's 
[Benchmark of Simplex LP Solvers](http://plato.asu.edu/ftp/lpsimp.html),
[Benchmark of Barrier LP Solvers](http://plato.asu.edu/ftp/lpbar.html), and
[Large Network-LP Benchmark (commercial vs
 free)](http://plato.asu.edu/ftp/network.html).
 
1. From the local directory, instantiate the necessary packages by running
   `julia --project=. -e 'import Pkg; Pkg.instantiate()'`.
2. Download the benchmark instances. Either run
   `./collect_mittelmann_benchmark.sh` or follow its steps for downloading the
   instances from the multiple sources and organizing them.
3. Download and build PaPILO from https://github.com/lgottwald/PaPILO.
4. From the local directory, run `./preprocess.sh`.

For example, assuming you have already instantiated the packages and built
PaPILO,

```sh
$ ./collect_mittelmann_benchmark.sh /tmp ~/mittelmann_benchmark
$ ./preprocess.sh ~/mittelmann_benchmark ./mittelmann_instance_list \
    ~/mittelmann_preprocessed ~/PaPILO/build/bin/papilo
```

