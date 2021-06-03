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

@testset "row permutation of sparse matrices" begin
  @testset "2x2" begin
    mat = sparse([1.0 0.0; 0.0 1.0])
    FirstOrderLp.row_permute_in_place(mat, [2, 1])
    @test mat == sparse([0.0 1.0; 1.0 0.0])
  end

  @testset "3x2" begin
    mat = sparse([
      1.0 0.0
      0.0 1.0
      2.0 3.0
    ])
    FirstOrderLp.row_permute_in_place(mat, [3, 1, 2])
    @test mat == sparse([
      0.0 1.0
      2.0 3.0
      1.0 0.0
    ])
  end
end
