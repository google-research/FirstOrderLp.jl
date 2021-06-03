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

using DelimitedFiles

@testset "trust_region_utils" begin
  @testset "solve_bound_constrained_trust_region unbounded" begin
    for solve_approximately in [true false]
      # min -x
      # || x || <= 5.0
      variable_lower_bounds = [-Inf]
      variable_upper_bounds = [Inf]
      current_point = [0.0]
      current_gradient = [-1.0]
      target_radius = 5.0
      norm_weights = [1.0]
      result = FirstOrderLp.solve_bound_constrained_trust_region(
        current_point,
        current_gradient,
        variable_lower_bounds,
        variable_upper_bounds,
        norm_weights,
        target_radius,
        solve_approximately,
      )
      @test result.value == -5.0
      @test result.solution == [5.0]

      # min x + y
      # 2 * x^2 + y^2 <= 6.0
      # [x*, y*] = [-1.0, -2.0]
      variable_lower_bounds = [-Inf, -Inf]
      variable_upper_bounds = [Inf, Inf]
      current_point = [0.0, 0.0]
      current_gradient = [1.0, 1.0]
      norm_weights = [2.0, 1.0]
      target_radius = sqrt(6.0)
      result = FirstOrderLp.solve_bound_constrained_trust_region(
        current_point,
        current_gradient,
        variable_lower_bounds,
        variable_upper_bounds,
        norm_weights,
        target_radius,
        solve_approximately,
      )
      @test result.solution ≈ [-1.0, -2.0] atol = 1.0e-8
      @test result.value ≈ -3.0 atol = 1.0e-8
    end
  end

  @testset "solve_bound_constrained_trust_region" begin
    # min -x
    # || x || <= 5.0
    variable_lower_bounds = [-Inf]
    variable_upper_bounds = [Inf]
    current_point = [0.0]
    current_gradient = [-1.0]
    target_radius = 5.0
    norm_weights = [1.0]
    result = FirstOrderLp.solve_bound_constrained_trust_region(
      current_point,
      current_gradient,
      variable_lower_bounds,
      variable_upper_bounds,
      norm_weights,
      target_radius,
      false,
    )
    @test result.solution == [5.0]

    # min -x
    # || x || <= 5.0
    # x <= 0.0
    variable_lower_bounds = [-Inf]
    variable_upper_bounds = [0.0]
    current_point = [0.0]
    current_gradient = [-1.0]
    result = FirstOrderLp.solve_bound_constrained_trust_region(
      current_point,
      current_gradient,
      variable_lower_bounds,
      variable_upper_bounds,
      norm_weights,
      target_radius,
      false,
    )
    @test result.solution == [0.0]

    # min -x
    # || x || <= 5.0
    # x <= 2.0
    variable_lower_bounds = [-Inf]
    variable_upper_bounds = [2.0]
    result = FirstOrderLp.solve_bound_constrained_trust_region(
      current_point,
      current_gradient,
      variable_lower_bounds,
      variable_upper_bounds,
      norm_weights,
      target_radius,
      false,
    )
    @test result.solution == [2.0]

    # min -2 * x - y
    # || [x; y] || <= 5.0
    # x <= 3.0
    # [x*, y*] = [3.0, 4.0]
    variable_lower_bounds = [-Inf, -Inf]
    variable_upper_bounds = [3.0, Inf]
    current_point = [0.0, 0.0]
    current_gradient = [-2.0, -1.0]
    norm_weights = [1.0, 1.0]
    result = FirstOrderLp.solve_bound_constrained_trust_region(
      current_point,
      current_gradient,
      variable_lower_bounds,
      variable_upper_bounds,
      norm_weights,
      target_radius,
      false,
    )
    @test result.solution ≈ [3.0, 4.0] atol = 1.0e-8

    # min -x
    # || [x; y] || <= 5.0
    # x <= 2.0
    # [x*, y*] = [2.0, 0.0]
    variable_lower_bounds = [-Inf, -Inf]
    variable_upper_bounds = [2.0, Inf]
    current_point = [0.0, 0.0]
    current_gradient = [-1.0, 0.0]
    result = FirstOrderLp.solve_bound_constrained_trust_region(
      current_point,
      current_gradient,
      variable_lower_bounds,
      variable_upper_bounds,
      norm_weights,
      target_radius,
      false,
    )
    @test result.solution == [2.0, 0.0]

    # Test norm_weights:
    # min -4.0 * x - 3.0 * y
    # 4.0^2 * x^2 + 3.0^2 * y^2 <= 2.0
    # [x*, y*] = [1.0 / 4.0, 1.0 / 3.0]
    # to see this substitute a = 4.0 x, b = 3.0 y, yielding:
    # min -a - b
    # a^2 + b^2 <= 2.0
    # [a*, b*] = [1.0, 1.0]
    variable_lower_bounds = [-Inf, -Inf]
    variable_upper_bounds = [Inf, Inf]
    current_point = [0.0, 0.0]
    current_gradient = [-4.0, -3.0]
    norm_weights = [4.0^2, 3.0^2]
    target_radius = sqrt(2.0)
    result = FirstOrderLp.solve_bound_constrained_trust_region(
      current_point,
      current_gradient,
      variable_lower_bounds,
      variable_upper_bounds,
      norm_weights,
      target_radius,
      false,
    )
    @test FirstOrderLp.weighted_norm(result.solution, norm_weights) ≈ sqrt(2.0) atol =
      1.0e-8
    @test result.solution ≈ [1.0 / 4.0, 1.0 / 3.0] atol = 1.0e-8

    # Hundred dimensional test
    # min -sum(x[i])
    # || x || <= r
    # x[i] <= i
    #
    # x*[i] = min(i, m)
    # Set r = || x* ||.
    n = 100
    variable_lower_bounds = zeros(n)
    variable_upper_bounds = 1.0 * collect(1:n)
    current_point = zeros(n)
    current_gradient = -ones(n)
    acceptable_relative_error = 1e-6
    norm_weights = ones(length(variable_lower_bounds))
    for m in [10.0, 50.0]
      target_radius = sqrt(sum([min(i, m)^2 for i in 1:n]))
      result = FirstOrderLp.solve_bound_constrained_trust_region(
        current_point,
        current_gradient,
        variable_lower_bounds,
        variable_upper_bounds,
        norm_weights,
        target_radius,
        false,
      )
      @test result.solution ≈ [min(i, m) for i in 1:n] atol = 1.0e-8
    end
  end

  @testset "bound_primal_and_dual_objective" begin
    lp = example_lp()
    primal_norm_weights = ones(4)
    dual_norm_weights = ones(3)
    # Test cases with zero or close to zero duality gap
    for norm in [FirstOrderLp.MAX_NORM, FirstOrderLp.EUCLIDEAN_NORM]
      primal_solution = [1.0, 0.0, 6.0, 2.0]
      dual_solution = [0.5, 4.0, 0.0]
      distance_to_optimality = 10.0
      result = FirstOrderLp.bound_optimal_objective(
        lp,
        primal_solution,
        dual_solution,
        primal_norm_weights,
        dual_norm_weights,
        distance_to_optimality,
        FirstOrderLp.MAX_NORM;
        solve_approximately = false,
      )

      @test result.lower_bound_value == -1.0
      @test result.upper_bound_value == -1.0

      primal_solution = [1.0, 0.0, 5.99999, 2.0]
      dual_solution = [0.50001, 4.0, 0.0]
      result = FirstOrderLp.bound_optimal_objective(
        lp,
        primal_solution,
        dual_solution,
        primal_norm_weights,
        dual_norm_weights,
        distance_to_optimality,
        FirstOrderLp.MAX_NORM;
        solve_approximately = false,
      )
      @test -1.01 < result.lower_bound_value < -1.0
      @test -1.0 < result.upper_bound_value < -0.99
    end

    primal_solution = [1.0, 0.0, 6.0, 1.0]
    dual_solution = [0.0, 4.0, 0.0]
    distance_to_optimality = 2.0
    result = FirstOrderLp.bound_optimal_objective(
      lp,
      primal_solution,
      dual_solution,
      primal_norm_weights,
      dual_norm_weights,
      distance_to_optimality,
      FirstOrderLp.MAX_NORM;
      solve_approximately = false,
    )
    @test result.lower_bound_value == -4.0
    @test result.upper_bound_value == 2.0
    @test result.lower_bound_value ==
          FirstOrderLp.corrected_dual_obj(lp, primal_solution, dual_solution)

    primal_solution = [3.0, 0.0, 6.0, 0.0]
    dual_solution = [0.0, 4.0, 0.0]
    distance_to_optimality = 5.0
    result = FirstOrderLp.bound_optimal_objective(
      lp,
      primal_solution,
      dual_solution,
      primal_norm_weights,
      dual_norm_weights,
      distance_to_optimality,
      FirstOrderLp.EUCLIDEAN_NORM;
      solve_approximately = false,
    )
    @test result.lower_bound_value == -4.0
    @test result.lagrangian_value == -1.0
    @test distance_to_optimality^2 ==
          norm(result.primal_solution - primal_solution, 2)^2 +
          norm(result.dual_solution - dual_solution, 2)^2

    @test result.upper_bound_value == 7.0


    primal_solution = [1.0, 1.0, 4.0, 1.0]
    dual_solution = [0.0, 0.0, 0.0]
    distance_to_optimality = 10.0
    result = FirstOrderLp.bound_optimal_objective(
      example_lp(),
      primal_solution,
      dual_solution,
      primal_norm_weights,
      dual_norm_weights,
      distance_to_optimality,
      FirstOrderLp.MAX_NORM;
      solve_approximately = false,
    )
    corrected_dual_obj =
      FirstOrderLp.corrected_dual_obj(lp, primal_solution, dual_solution)
    @test result.lower_bound_value ==
          FirstOrderLp.corrected_dual_obj(lp, primal_solution, dual_solution)

    primal_solution = [0.5, 0.5, 0.5, 1.0, 1.0, 1.0] # Solution in the interior.
    dual_solution = zeros(3)
    primal_norm_weights = ones(6)
    dual_norm_weights = ones(3)

    result = FirstOrderLp.bound_optimal_objective(
      example_cc_star_lp(),
      primal_solution,
      dual_solution,
      primal_norm_weights,
      dual_norm_weights,
      distance_to_optimality,
      FirstOrderLp.MAX_NORM;
      solve_approximately = false,
    )
    @test result.lagrangian_value == result.upper_bound_value
    @test result.lower_bound_value < result.lagrangian_value
  end
end
