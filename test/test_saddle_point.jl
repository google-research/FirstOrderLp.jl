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

@testset "test_saddle_point" begin
  # Define problems that we will use for tests
  lp2 = deepcopy(example_lp())
  lp2.objective_vector = [0.0, 0.0, 0.0, 0.0]

  lp3 = deepcopy(example_lp())
  lp3.right_hand_side = [0.0, 0.0, 0.0]

  lp4 = deepcopy(lp3)
  lp4.variable_upper_bound = [Inf, Inf, Inf, Inf]

  lp5 = deepcopy(lp4)
  lp5.variable_lower_bound = [0.0, 0.0, 0.0, 1.0]

  qp = deepcopy(lp2)
  qp.objective_matrix[1, 1] = 1.0

  @testset "test_select_initial_primal_weight" begin
    verbosity = 0
    lp1 = example_lp()
    primal_importance = 1.3
    primal_weight = FirstOrderLp.select_initial_primal_weight(
      lp1,
      [1.0, 1.0, 1.0, 1.0],
      [1.0, 1.0, 1.0],
      primal_importance,
      verbosity,
    )
    @test primal_weight ==
          primal_importance * norm([5.0, 2.0, 1.0, 1.0], 2) /
          norm([12.0, 7.0, 1.0], 2)

    primal_weight = FirstOrderLp.select_initial_primal_weight(
      lp2,
      [1.0, 1.0, 1.0, 1.0],
      [1.0, 1.0, 1.0],
      primal_importance,
      verbosity,
    )
    @test primal_weight == primal_importance

    primal_weight = FirstOrderLp.select_initial_primal_weight(
      lp3,
      [1.0, 1.0, 1.0, 1.0],
      [1.0, 1.0, 1.0],
      primal_importance,
      verbosity,
    )
    @test primal_weight == primal_importance
  end

  @testset "test_compute_langragian_value" begin
    lp = example_lp()
    @test FirstOrderLp.compute_lagrangian_value(lp, zeros(4), zeros(3)) == -14.0
    qp = example_qp()
    @test FirstOrderLp.compute_lagrangian_value(qp, [1.0, 1.0], [0.0]) == 0.5
    @test FirstOrderLp.compute_lagrangian_value(qp, [1.0, 1.0], [1.0]) == 1.5
    @test FirstOrderLp.compute_lagrangian_value(qp, [0.25, 0.0], [0.0]) ==
          -0.125
  end
end
