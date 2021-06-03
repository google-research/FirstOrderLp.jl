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

module FirstOrderLp

# Built-in Julia packages.
import LinearAlgebra
# We use the Logging package to control the output level of QPSReader, not for
# internal logging.
# TODO: Consider switching internal logging to Logging.
import Logging
import Printf
import SparseArrays
import Random

# Third-party Julia packages.
import GZip
import QPSReader
import Statistics
import StatsBase
import StructTypes

const Diagonal = LinearAlgebra.Diagonal
const diag = LinearAlgebra.diag
const dot = LinearAlgebra.dot
const cholesky = LinearAlgebra.cholesky
const ldlt = LinearAlgebra.ldlt
const lu = LinearAlgebra.lu
const norm = LinearAlgebra.norm
const mean = Statistics.mean
const median = Statistics.median
const nzrange = SparseArrays.nzrange
const nnz = SparseArrays.nnz
const nonzeros = SparseArrays.nonzeros
const rowvals = SparseArrays.rowvals
const sparse = SparseArrays.sparse
const SparseMatrixCSC = SparseArrays.SparseMatrixCSC
const spdiagm = SparseArrays.spdiagm
const spzeros = SparseArrays.spzeros
const quantile = Statistics.quantile
const sample = StatsBase.sample

include("quadratic_programming.jl")
include("trust_region_utils.jl")
include("solve_log.jl")
include("quadratic_programming_io.jl")
include("preprocess.jl")
include("saddle_point.jl")
include("termination.jl")
include("iteration_stats_utils.jl")
include("mirror_prox.jl")
include("primal_dual_hybrid_gradient.jl")

end  # module FirstOrderLp
