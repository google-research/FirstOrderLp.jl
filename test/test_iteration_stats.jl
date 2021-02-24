# Copyright 2020 Google LLC
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

@testset "IterationStats" begin

  @testset "max_primal_violation" begin
    # min 0
    # s.t. y == 10
    # z >= 11 (as a linear constraint)
    # -1 <= x <= 1
    lp = FirstOrderLp.linear_programming_problem(
      [-1.0, -Inf, -Inf],  # variable_lower_bound
      [1.0, Inf, Inf],  # variable_upper_bound
      zeros(3),  # objective_vector
      0.0,  # objective_constant
      [0.0 1.0 0.0; 0.0 0.0 1.0],  # constraint_matrix
      [10.0, 11.0],  # right_hand_side
      1,  # num_equalities
    )
    @test FirstOrderLp.max_primal_violation(lp, [0.0, 10.0, 11.0]) == 0.0
    @test FirstOrderLp.max_primal_violation(lp, [-2.0, 10.0, 11.0]) ≈ 1.0
    @test FirstOrderLp.max_primal_violation(lp, [3.0, 10.0, 11.0]) ≈ 2.0
    @test FirstOrderLp.max_primal_violation(lp, [0.0, 11.0, 11.0]) ≈ 1.0
    @test FirstOrderLp.max_primal_violation(lp, [0.0, 9.0, 11.0]) ≈ 1.0
    @test FirstOrderLp.max_primal_violation(lp, [0.0, 11.0, 0.0]) ≈ 11.0
  end

  @testset "primal_obj" begin
    qp = example_qp()
    @test FirstOrderLp.primal_obj(qp, [0.0, 0.0]) == 0.0
    @test FirstOrderLp.primal_obj(qp, [1.0, 1.0]) == 0.5
    @test FirstOrderLp.primal_obj(qp, [1.0, 0.0]) == 1.0
    @test FirstOrderLp.primal_obj(qp, [0.0, 1.0]) == -0.5
    @test FirstOrderLp.primal_obj(qp, [0.0, -1.0]) == 1.5
  end

  @testset "dual_stats" begin
    # Primal:
    # min x + 2y
    # x + y >= 1
    # -1 <= x <= 1
    # Dual:
    # max a - b - c
    # s.t. a + b - c = 1
    #      a = 2
    # a, b, c >= 0
    lp = FirstOrderLp.linear_programming_problem(
      [-1.0, -Inf],  # variable_lower_bound
      [1.0, Inf],  # variable_upper_bound
      [1.0, 2.0],  # objective_vector
      0.0,  # objective_constant
      reshape([1.0, 1.0], 1, 2),  # constraint_matrix
      [1.0],  # right_hand_side
      0,  # num_equalities
    )
    # b and c are reduced costs. The dual vector is [a].
    dual_stats0 = FirstOrderLp.compute_dual_stats(lp, [0.0, 0.0], [0.0])
    @test dual_stats0.dual_objective == -1.0
    @test norm(dual_stats0.dual_residual, Inf) == 2.0
    @test dual_stats0.dual_residual == [0.0; 0.0; 2.0]

    dual_stats1 = FirstOrderLp.compute_dual_stats(lp, [0.0, 0.0], [1.0])
    @test dual_stats1.dual_objective == 1.0
    @test norm(dual_stats1.dual_residual, Inf) == 1.0
    @test dual_stats1.dual_residual == [0.0; 0.0; 1.0]

    dual_stats2 = FirstOrderLp.compute_dual_stats(lp, [0.0, 0.0], [2.0])
    @test dual_stats2.dual_objective == 1.0
    @test norm(dual_stats2.dual_residual, Inf) == 0.0

    dual_stats3 = FirstOrderLp.compute_dual_stats(lp, [0.0, 0.0], [3.0])
    @test dual_stats3.dual_objective == 1.0
    @test norm(dual_stats3.dual_residual, Inf) == 1.0

    dual_stats4 = FirstOrderLp.compute_dual_stats(lp, [0.0, 1.0], [-1.0])
    @test dual_stats4.dual_objective == -3.0
    @test dual_stats4.dual_residual == [1.0, 0.0, 3.0]

    lp = FirstOrderLp.linear_programming_problem(
      [Inf, -Inf],  # variable_lower_bound
      [Inf, Inf],  # variable_upper_bound
      [1.0, 2.0],  # objective_vector
      0.0,  # objective_constant
      reshape([1.0, 1.0], 1, 2),  # constraint_matrix
      [1.0],  # right_hand_side
      0,  # num_equalities
    )
    dual_stats5 = FirstOrderLp.compute_dual_stats(lp, [0.0, 1.0], [-1.0])
    @test dual_stats5.dual_objective == -1.0
    @test dual_stats5.dual_residual == [1.0, 2.0, 3.0]


    qp = example_qp()
    dual_stats6 = FirstOrderLp.compute_dual_stats(qp, [0.0, 0.0], [3.0])
    @test dual_stats6.dual_objective == -3.0
    @test norm(dual_stats6.dual_residual, Inf) == 0.0

    dual_stats7 = FirstOrderLp.compute_dual_stats(qp, [0.0, 0.0], [1.0])
    @test dual_stats7.dual_objective == -1.0
    @test norm(dual_stats7.dual_residual, Inf) == 0.0

    dual_stats8 = FirstOrderLp.compute_dual_stats(qp, [0.5, 0.5], [1.0])
    @test dual_stats8.dual_objective == -1.625
    @test norm(dual_stats8.dual_residual, Inf) == 0.0
  end

  @testset "Primal/dual optimal" begin
    # Primal:
    # min x + 2y
    # x + y >= 1
    # -1 <= x <= 1
    # Dual:
    # max a - b - c
    # s.t. a + b - c = 1
    #      a = 2
    # a, b, c >= 0
    lp = FirstOrderLp.linear_programming_problem(
      [-1.0, -Inf],  # variable_lower_bound
      [1.0, Inf],  # variable_upper_bound
      [1.0, 2.0],  # objective_vector
      0.0,  # objective_constant
      reshape([1.0, 1.0], 1, 2),  # constraint_matrix
      [1.0],  # right_hand_side
      0,  # num_equalities
    )
    # b and c are reduced costs. The dual vector is [a].
    stats = FirstOrderLp.compute_iteration_stats(
      lp,
      FirstOrderLp.cached_quadratic_program_info(lp),
      [1.0, 0.0],  # primal_iterate
      [2.0],  # dual_iterate
      [0.0, 0.0],  # primal_ray_estimate
      [0.0],  # dual_ray_estimate
      5,  # iteration_number
      1.5,  # cumulative_kkt_matrix_passses
      5.0,  # cumulative_time_sec
      1e-6,  # eps_optimal_absolute
      1e-6,  # eps_optimal_relative
      1.0,  # step_size
      1.0,  # primal_weight
      FirstOrderLp.POINT_TYPE_CURRENT_ITERATE,
    )
    correct_convergence_info = FirstOrderLp.ConvergenceInformation()
    correct_convergence_info.primal_objective = 1.0
    correct_convergence_info.dual_objective = 1.0
    correct_convergence_info.corrected_dual_objective = 1.0
    correct_convergence_info.l_inf_primal_variable = 1.0
    correct_convergence_info.l2_primal_variable = 1.0
    correct_convergence_info.l_inf_dual_variable = 2.0
    correct_convergence_info.l2_dual_variable = 2.0
    correct_convergence_info.candidate_type =
      FirstOrderLp.POINT_TYPE_CURRENT_ITERATE

    correct_infeas_info = FirstOrderLp.InfeasibilityInformation()
    correct_infeas_info.candidate_type = FirstOrderLp.POINT_TYPE_CURRENT_ITERATE

    correct_stats = FirstOrderLp.IterationStats()
    correct_stats.iteration_number = 5
    correct_stats.cumulative_kkt_matrix_passes = 1.5
    correct_stats.cumulative_time_sec = 5.0
    correct_stats.convergence_information = [correct_convergence_info]
    correct_stats.infeasibility_information = [correct_infeas_info]
    correct_stats.step_size = 1.0
    correct_stats.primal_weight = 1.0

    test_fields_equal(stats, correct_stats)
  end

  @testset "Primal infeasible" begin
    # Primal:
    # min x + 2
    # s.t. x = 10
    # 0 <= x <= 1
    # Dual:
    # max 10a - b + 2
    # s.t. a - b <= 1
    # b >= 0.
    lp = FirstOrderLp.linear_programming_problem(
      [0.0],  # variable_lower_bound
      [1.0],  # variable_upper_bound
      [1.0],  # objective_vector
      2.0,  # objective_constant
      reshape([1.0], 1, 1),  # constraint_matrix
      [10.0],  # right_hand_side
      1,   # num_equalities
    )
    # b is a reduced cost. The dual vector is [a].
    stats = FirstOrderLp.compute_iteration_stats(
      lp,
      FirstOrderLp.cached_quadratic_program_info(lp),
      [2.0],  # primal_iterate
      [1.0],  # dual_iterate
      [0.0],  # primal_ray_estimate
      [1.0],  # dual_ray_estimate
      5,  # iteration_number
      1.5,  # cumulative_kkt_matrix_passses
      5.0,  # cumulative_time_sec
      1e-6,  # eps_optimal_absolute
      1e-6,  # eps_optimal_relative
      1.0,  # step_size
      1.0,  # primal_weight
      FirstOrderLp.POINT_TYPE_CURRENT_ITERATE,
    )
    correct_convergence_info = FirstOrderLp.ConvergenceInformation()
    correct_convergence_info.primal_objective = 4.0
    correct_convergence_info.dual_objective = 10.0 + 2.0
    correct_convergence_info.corrected_dual_objective = 12.0
    correct_convergence_info.l_inf_primal_residual = 8.0
    correct_convergence_info.l2_primal_residual = norm([8.0, 1.0], 2)
    correct_convergence_info.relative_l_inf_primal_residual = 8.0 / (1.0 + 10.0)
    correct_convergence_info.relative_l2_primal_residual =
      norm([8.0, 1.0], 2) / (1.0 + 10.0)
    correct_convergence_info.relative_optimality_gap = 8.0 / (1.0 + 16.0)
    correct_convergence_info.l_inf_primal_variable = 2.0
    correct_convergence_info.l2_primal_variable = 2.0
    correct_convergence_info.l_inf_dual_variable = 1.0
    correct_convergence_info.l2_dual_variable = 1.0
    correct_convergence_info.candidate_type =
      FirstOrderLp.POINT_TYPE_CURRENT_ITERATE

    correct_infeas_info = FirstOrderLp.InfeasibilityInformation()
    correct_infeas_info.dual_ray_objective = 9.0
    correct_infeas_info.candidate_type = FirstOrderLp.POINT_TYPE_CURRENT_ITERATE

    correct_stats = FirstOrderLp.IterationStats()
    correct_stats.iteration_number = 5
    correct_stats.cumulative_kkt_matrix_passes = 1.5
    correct_stats.cumulative_time_sec = 5.0
    correct_stats.convergence_information = [correct_convergence_info]
    correct_stats.infeasibility_information = [correct_infeas_info]
    correct_stats.step_size = 1.0
    correct_stats.primal_weight = 1.0

    test_fields_equal(stats, correct_stats)
  end

  @testset "Dual infeasible" begin
    # Primal:
    # min -x
    # s.t. x >= 10
    # Dual:
    # max 10a
    # s.t. a <= -1
    # a >= 0.
    lp = FirstOrderLp.linear_programming_problem(
      [-Inf],  # variable_lower_bound
      [Inf],  # variable_upper_bound
      [-1.0],  # objective_vector
      0.0,  # objective_constant
      reshape([1.0], 1, 1),  # constraint_matrix
      [10.0],  # right_hand_side
      0,  # num_equalities
    )
    stats = FirstOrderLp.compute_iteration_stats(
      lp,
      FirstOrderLp.cached_quadratic_program_info(lp),
      [10.0],  # primal_iterate
      [0.0],  # dual_iterate
      [1.0],  # primal_ray_estimate
      [0.0],  # dual_ray_estimate
      5,  # iteration_number
      1.5,  # cumulative_kkt_matrix_passses
      5.0,  # cumulative_time_sec
      1e-6,  # eps_optimal_absolute
      1e-6,  # eps_optimal_relative
      1.0,  # step_size
      1.0,  # primal_weight
      FirstOrderLp.POINT_TYPE_CURRENT_ITERATE,
    )
    correct_convergence_info = FirstOrderLp.ConvergenceInformation()
    correct_convergence_info.primal_objective = -10.0
    correct_convergence_info.corrected_dual_objective = -Inf
    correct_convergence_info.l_inf_dual_residual = 1.0
    correct_convergence_info.l2_dual_residual = 1.0
    correct_convergence_info.relative_l_inf_dual_residual = 1.0 / (1.0 + 1.0)
    correct_convergence_info.relative_l2_dual_residual = 1.0 / (1.0 + 1.0)
    correct_convergence_info.relative_optimality_gap = 10.0 / (1.0 + 10.0)
    correct_convergence_info.l_inf_primal_variable = 10.0
    correct_convergence_info.l2_primal_variable = 10.0
    correct_convergence_info.candidate_type =
      FirstOrderLp.POINT_TYPE_CURRENT_ITERATE

    correct_infeas_info = FirstOrderLp.InfeasibilityInformation()
    correct_infeas_info.primal_ray_linear_objective = -1.0
    correct_infeas_info.candidate_type = FirstOrderLp.POINT_TYPE_CURRENT_ITERATE

    correct_stats = FirstOrderLp.IterationStats()
    correct_stats.iteration_number = 5
    correct_stats.cumulative_kkt_matrix_passes = 1.5
    correct_stats.cumulative_time_sec = 5.0
    correct_stats.convergence_information = [correct_convergence_info]
    correct_stats.infeasibility_information = [correct_infeas_info]
    correct_stats.step_size = 1.0
    correct_stats.primal_weight = 1.0

    test_fields_equal(stats, correct_stats)
  end

  @testset "print_to_screen_this_iteration" begin
    termination_evaluation_frequency = convert(Int32, 10)
    @test FirstOrderLp.print_to_screen_this_iteration(
      false, # termination_reason
      1, # iteration,
      2, # verbosity
      termination_evaluation_frequency,
    )
    @test FirstOrderLp.print_to_screen_this_iteration(
      false, # termination_reason
      101, # iteration,
      5, # verbosity
      termination_evaluation_frequency,
    )
    @test !FirstOrderLp.print_to_screen_this_iteration(
      false, # termination_reason
      31, # iteration,
      5, # verbosity
      termination_evaluation_frequency,
    )
    @test !FirstOrderLp.print_to_screen_this_iteration(
      false, # termination_reason
      531, # iteration,
      5, # verbosity
      termination_evaluation_frequency,
    )
    @test FirstOrderLp.print_to_screen_this_iteration(
      true, # termination_reason
      124, # iteration,
      5, # verbosity
      termination_evaluation_frequency,
    )
  end
end
