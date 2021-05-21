# Benchmarking

This directory contains scripts for generating datasets and processing results
for benchmarking FirstOrderLp.

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

## Preprocessed Mittelmann benchmark dataset

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

## L1 SVM

The L1 SVM instances apply the formulation from equation (5) in "1-norm Support
Vector Machines" by J. Zhu et al. (NIPS, 2003.
https://papers.nips.cc/paper/2003/file/49d4b2faeb4b7b9e745775793141e2b2-Paper.pdf).

1. From the local directory, instantiate the necessary packages by running
   `julia --project=. -e 'import Pkg; Pkg.instantiate()'`.
2. Download the benchmark instances. Run
   `./collect_LIBSVM.sh` .
3. Generate the instances by running `generate_l1_svm_lp.jl`.

For example,

```sh
$ ./collect_LIBSVM.sh /tmp ~/LIBSVM
$ julia --project=. generate_l1_svm_lp.jl --input_filename=~/LIBSVM/duke \
	  --output_filename=~/LIBSVM/duke.mps.gz --regularizer_weight=1.0
```

## Pagerank instances

The Pagerank instances apply the LP formulation from "Subgradient methods for
huge-scale optimization problems" by Y. Nesterov (Mathematical Programming,
2014. https://doi.org/10.1007/s10107-013-0686-4,
http://www.optimization-online.org/DB_FILE/2012/02/3339.pdf (preprint)) to
random Barabasi Albert preferential attachment graphs.

1. From the local directory, instantiate the necessary packages by running
   `julia --project=. -e 'import Pkg; Pkg.instantiate()'`.
2. Generate the instances by running `generate_pagerank.jl`.

For example,

```sh
$ julia --project=. generate_pagerank_lp.jl --num_nodes 10000 \
    --approx_num_edges 30000 --random_seed 1 \
    --output_filename ~/pagerank.10k.mps.gz
```
## Procesing JSON results

`solve_qp.jl` and `solve_lp_external.jl` output JSON files with solve
statistics. Use `process_json_to_csv.jl` to process collections of these files
into a CSV file for analysis.

For example (from the `FirstOrderLp.jl` root directory),

```sh
$ julia --project=scripts scripts/solve_qp.jl \
--instance_path test/trivial_lp_model.mps --method pdhg \
--output_dir /tmp/first_order_lp_solve
$ julia --project=scripts scripts/solve_lp_external.jl \
--instance_path test/trivial_lp_model.mps --solver scs-indirect \
--tolerance 1e-7 --output_dir /tmp/scs_solve
$ echo '{"datasets": [
   {"name": "pdhg", "logs_directory": "/tmp/first_order_lp_solve"},
   {"name": "scs", "logs_directory": "/tmp/scs_solve"}]}' >/tmp/layout.json
$ julia --project=benchmarking benchmarking/process_json_to_csv.jl \
/tmp/layout.json /tmp/dataset.csv
```

This outputs to `/tmp/dataset.csv`.
