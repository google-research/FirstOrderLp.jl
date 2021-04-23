# FirstOrderLp.jl

## Introduction

This repository contains experimental code for solving linear and quadratic
programming problems using first-order methods. It provides basic utilities and
data structures for reading MPS files, rescaling, and implementing saddle-point
methods. Specialized implementations are present for Mirror Prox and
Primal-Dual Hybrid Gradient. It is focused on supporting experiments and
publications and is not a "solver" per se.

## One-time setup

Install Julia 1.6.0 or later. From the root directory of the repository, run:

```shell
$ julia --project=scripts -e 'import Pkg; Pkg.instantiate()'
```

This setup needs to be run again only if the dependencies change.

## Running

Use one of the following two scripts to solve LP instances. All commands below
assume that the current directory is the working directory.

### `solve_qp.jl`

This is the recommended script for using FirstOrderLp. The results are written
to JSON and text files; see the source for the description of the output
formats.

To see the meaning of each argument:

```shell
$ julia --project=scripts scripts/solve_qp.jl --help
```

To solve a test instance with PDHG:

```shell
$ julia --project=scripts scripts/solve_qp.jl \
--instance_path test/trivial_lp_model.mps --iteration_limit 5000 \
--method pdhg --output_dir /tmp/first_order_lp_solve
```

### `solve_lp_external.jl`

This script provides an interface similar to `solve_qp` but for calling
external solvers for baselines. This script does not support quadratic
objectives.

To solve a test instance with SCS's indirect mode:

```shell
$ julia --project=scripts scripts/solve_lp_external.jl \
--instance_path test/trivial_lp_model.mps --iteration_limit 5000 \
--solver scs-indirect --tolerance 1e-7 --output_dir /tmp/scs_solve
```

To solve a test with HiGHS's interior-point mode:

```shell
$ julia --project=scripts scripts/solve_lp_external.jl \
--instance_path test/trivial_lp_model.mps --solver highs-ipm \
--tolerance 1e-7 --output_dir /tmp/highs_solve
```

## Loading the module

Use the following commands to load the FirstOrderLp module from Julia and to
view the docstrings:

```
$ julia --project
…
julia> import FirstOrderLp
julia> ?  # Switch to the help> prompt.
help> FirstOrderLp.optimize
…
help> FirstOrderLp
…
```

## Running the tests

To run the module’s tests run:

```shell
$ julia --color=yes --project -e 'import Pkg; Pkg.test("FirstOrderLp")'
```

## Interpreting the output

When the verbosity option is greater than 2, a table of iteration stats will be
printed with the following headings (split into six groups).

##### runtime

`#iter` = the current iteration number.

`#kkt` = the cumulative number of times the KKT matrix is multiplied.

`seconds` = the cumulative solve time in seconds.

##### residuals

`pr norm` = the euclidean norm of primal residuals (i.e., the constraint
violation).

`du norm` = the euclidean norm of the dual residuals.

`gap` = the gap between the primal and dual objective.

##### solution information

`pr obj` = the primal objective value.

`pr norm` = the euclidean norm of the primal variable vector.

`du norm` = the euclidean norm of the dual variable vector.

##### relative residuals

`rel pr` = the euclidean norm of the primal residuals, relative to the
right-hand-side.

`rel dul` = the euclidean norm of the dual residuals, relative to the primal
linear objective.

`rel gap` = the relative optimality gap.

##### primal ray (verbosity greater than seven only)

`pr norm` = the euclidean norm of the primal residuals for the primal ray
problem.

`linear` = the linear part of the primal ray objective.

`qu norm` = the norm of the quadratic part of the primal ray objective.

##### dual ray (verbosity greater than seven only)

`du norm` = the norm of the dual part of the dual ray.

`dual obj` = the dual ray objective value.

## Auto-formatting Julia code

A one-time step is required to use the auto-formatter:

```shell
$ julia --project=formatter -e 'import Pkg; Pkg.instantiate()'
```

Run the following command to auto-format all Julia code in this directory before
submitting changes:

```shell
$ julia --project=formatter -e 'using JuliaFormatter; format(".")'
```

## Disclaimer

This is not an official Google product.
