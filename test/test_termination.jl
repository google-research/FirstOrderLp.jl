# Copyright 2021 Google LLC
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

@testset "termination" begin
  # Different infeasibility information scenarios.
  infeas_info_no_infeasibility1 = FirstOrderLp.InfeasibilityInformation()

  infeas_info_no_infeasibility2 = FirstOrderLp.InfeasibilityInformation()
  infeas_info_no_infeasibility2.primal_ray_linear_objective = -1.0
  infeas_info_no_infeasibility2.primal_ray_quadratic_norm = 1.0
  infeas_info_no_infeasibility2.max_dual_ray_infeasibility = 1.0

  infeas_info_dual_infeasible = deepcopy(infeas_info_no_infeasibility1)
  infeas_info_dual_infeasible.primal_ray_linear_objective = -1.0
  infeas_info_primal_infeasible = deepcopy(infeas_info_no_infeasibility1)
  infeas_info_primal_infeasible.dual_ray_objective = 1.0

  # Test infeasibility criteria.
  eps_primal_infeasible = 1e-6
  eps_dual_infeasible = 1e-6

  @testset "primal infeasibility" begin
    @test !FirstOrderLp.primal_infeasibility_criteria_met(
      eps_primal_infeasible,
      infeas_info_no_infeasibility1,
    )
    @test !FirstOrderLp.primal_infeasibility_criteria_met(
      eps_primal_infeasible,
      infeas_info_no_infeasibility2,
    )
    @test !FirstOrderLp.primal_infeasibility_criteria_met(
      eps_primal_infeasible,
      infeas_info_dual_infeasible,
    )
    @test FirstOrderLp.primal_infeasibility_criteria_met(
      eps_primal_infeasible,
      infeas_info_primal_infeasible,
    )
  end

  @testset "dual infeasibility" begin
    @test !FirstOrderLp.dual_infeasibility_criteria_met(
      eps_dual_infeasible,
      infeas_info_no_infeasibility1,
    )
    @test !FirstOrderLp.dual_infeasibility_criteria_met(
      eps_dual_infeasible,
      infeas_info_no_infeasibility2,
    )
    @test FirstOrderLp.dual_infeasibility_criteria_met(
      eps_dual_infeasible,
      infeas_info_dual_infeasible,
    )
    @test !FirstOrderLp.dual_infeasibility_criteria_met(
      eps_dual_infeasible,
      infeas_info_primal_infeasible,
    )
  end

  # Different convergence information  stats scenarios.
  convergence_information_optimal = FirstOrderLp.ConvergenceInformation()
  convergence_information_optimal.primal_objective = 1.0
  convergence_information_optimal.dual_objective = 1.0
  convergence_information_optimal.l_inf_primal_variable = 1.0
  convergence_information_optimal.l2_primal_variable = 1.0
  convergence_information_optimal.l_inf_dual_variable = 2.0
  convergence_information_optimal.l2_dual_variable = 2.0

  convergence_info_dont_terminate1 = deepcopy(convergence_information_optimal)
  convergence_info_dont_terminate1.primal_objective = 10.0
  convergence_info_dont_terminate2 = deepcopy(convergence_information_optimal)
  convergence_info_dont_terminate2.l_inf_primal_residual = 1.0
  convergence_info_dont_terminate2.l2_primal_residual = 1.0
  convergence_info_dont_terminate3 = deepcopy(convergence_information_optimal)
  convergence_info_dont_terminate3.l_inf_dual_residual = 1.0
  convergence_info_dont_terminate3.l2_dual_residual = 1.0

  iteration_stats_optimal = FirstOrderLp.IterationStats()
  iteration_stats_optimal.iteration_number = 5
  iteration_stats_optimal.cumulative_kkt_matrix_passes = 100.5
  iteration_stats_optimal.cumulative_time_sec = 5.0
  iteration_stats_optimal.convergence_information =
    [convergence_information_optimal]
  iteration_stats_optimal.infeasibility_information =
    [infeas_info_no_infeasibility1]

  iteration_stats_dont_terminate1 = FirstOrderLp.IterationStats()
  iteration_stats_dont_terminate1.iteration_number = 5
  iteration_stats_dont_terminate1.cumulative_kkt_matrix_passes = 100.5
  iteration_stats_dont_terminate1.cumulative_time_sec = 5.0
  iteration_stats_dont_terminate1.convergence_information =
    [convergence_info_dont_terminate1]
  iteration_stats_dont_terminate1.infeasibility_information =
    [infeas_info_no_infeasibility1]

  qp_cache = FirstOrderLp.cached_quadratic_program_info(example_qp())

  for optimality_norm in [FirstOrderLp.L_INF, FirstOrderLp.L2]
    eps_optimal_relative = 1e-4
    eps_optimal_absolute = 1e-4

    @testset "optimality" begin
      # Test optimality criteria.
      @test !FirstOrderLp.optimality_criteria_met(
        optimality_norm,
        eps_optimal_relative,
        eps_optimal_absolute,
        convergence_info_dont_terminate1,
        qp_cache,
      )
      @test !FirstOrderLp.optimality_criteria_met(
        optimality_norm,
        eps_optimal_relative,
        eps_optimal_absolute,
        convergence_info_dont_terminate2,
        qp_cache,
      )
      @test !FirstOrderLp.optimality_criteria_met(
        optimality_norm,
        eps_optimal_relative,
        eps_optimal_absolute,
        convergence_info_dont_terminate3,
        qp_cache,
      )
      @test FirstOrderLp.optimality_criteria_met(
        optimality_norm,
        eps_optimal_relative,
        eps_optimal_absolute,
        convergence_information_optimal,
        qp_cache,
      )
    end

    full_termination_criteria = FirstOrderLp.construct_termination_criteria(
      optimality_norm = optimality_norm,
      eps_optimal_absolute = eps_optimal_absolute,
      eps_optimal_relative = eps_optimal_relative,
      eps_primal_infeasible = eps_primal_infeasible,
      eps_dual_infeasible = eps_dual_infeasible,
      time_sec_limit = 100.0,
      iteration_limit = 10,
      kkt_matrix_pass_limit = 10000.0,
    )

    @testset "check_termination_criteria" begin
      @test FirstOrderLp.check_termination_criteria(
        full_termination_criteria,
        qp_cache,
        iteration_stats_optimal,
      ) == FirstOrderLp.TERMINATION_REASON_OPTIMAL

      @test !FirstOrderLp.check_termination_criteria(
        full_termination_criteria,
        qp_cache,
        iteration_stats_dont_terminate1,
      )

      full_termination_criteria.time_sec_limit = 1.0
      @test FirstOrderLp.check_termination_criteria(
        full_termination_criteria,
        qp_cache,
        iteration_stats_dont_terminate1,
      ) == FirstOrderLp.TERMINATION_REASON_TIME_LIMIT

      full_termination_criteria.time_sec_limit = 10.0
      full_termination_criteria.iteration_limit = 1

      @test FirstOrderLp.check_termination_criteria(
        full_termination_criteria,
        qp_cache,
        iteration_stats_dont_terminate1,
      ) == FirstOrderLp.TERMINATION_REASON_ITERATION_LIMIT

      full_termination_criteria.iteration_limit = 10
      full_termination_criteria.kkt_matrix_pass_limit = 40.0
      @test FirstOrderLp.check_termination_criteria(
        full_termination_criteria,
        qp_cache,
        iteration_stats_dont_terminate1,
      ) == FirstOrderLp.TERMINATION_REASON_KKT_MATRIX_PASS_LIMIT
    end
  end
end
