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

using FirstOrderLp

# Built-in Julia packages.
import GZip
import LinearAlgebra
import Random
import Test: @test, @testset, @test_throws, @test_logs
import SparseArrays

const diagm = LinearAlgebra.diagm
const norm = LinearAlgebra.norm
const sparse = SparseArrays.sparse
const RestartScheme = FirstOrderLp.RestartScheme

Random.seed!(123)

include("utilities.jl")
include("shared_test_qp_problems.jl")
include("test_saddle_point.jl")
include("test_qp_processing.jl")
include("test_qp_io.jl")
include("test_sparse_linalg.jl")
include("test_mirror_prox.jl")
include("test_primal_dual_hybrid_gradient.jl")
include("test_iteration_stats.jl")
include("test_trust_region_utils.jl")
include("test_termination.jl")
