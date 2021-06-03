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

function generate_mirror_prox_params(;
  l_inf_ruiz_iterations = 0,
  l2_norm_rescaling = false,
  pock_chambolle_alpha = nothing,
  primal_importance,
  scale_invariant_initial_primal_weight = true,
  diagonal_scaling,
  verbosity,
  iteration_limit,
  record_iteration_stats = true,
  restart_scheme = FirstOrderLp.ADAPTIVE_NORMALIZED,
  restart_frequency_if_fixed = 100,
  artificial_restart_threshold = 0.5,
  sufficient_reduction_for_restart = 0.1,
  necessary_reduction_for_restart = 0.8,
  primal_weight_update_smoothing = 0.5,
  termination_evaluation_frequency = 5,
  use_approximate_localized_duality_gap = false,
  restart_to_current_metric = FirstOrderLp.GAP_OVER_DISTANCE_SQUARED,
)
  restart_params = FirstOrderLp.construct_restart_parameters(
    restart_scheme,
    restart_to_current_metric,
    restart_frequency_if_fixed,
    artificial_restart_threshold,
    sufficient_reduction_for_restart,
    necessary_reduction_for_restart,
    primal_weight_update_smoothing,
    use_approximate_localized_duality_gap,
  )
  parameters = FirstOrderLp.MirrorProxParameters(
    l_inf_ruiz_iterations,
    l2_norm_rescaling,
    pock_chambolle_alpha,
    primal_importance,
    scale_invariant_initial_primal_weight,
    diagonal_scaling,
    verbosity,
    record_iteration_stats,
    termination_evaluation_frequency,
    terminate_on_iteration_limit(iteration_limit),
    restart_params,
  )
  return parameters
end

@testset "Saddle point mirror prox" begin
  @testset "Low precision" begin
    parameters = generate_mirror_prox_params(
      primal_importance = 1.0,
      diagonal_scaling = "off",
      verbosity = 0,
      iteration_limit = 400,
      restart_scheme = FirstOrderLp.NO_RESTARTS,
      restart_frequency_if_fixed = 1000,
    )
    problem = example_lp()
    output = FirstOrderLp.optimize(parameters, problem)
    @test output.primal_solution ≈ [1.0; 0.0; 6.0; 2.0] atol = 1.0e-4
    @test output.dual_solution ≈ [0.5; 4.0; 0.0] atol = 1.0e-4
  end
  @testset "Test Verbosity" begin
    parameters = generate_mirror_prox_params(
      primal_importance = 1.0,
      diagonal_scaling = "off",
      verbosity = 10,
      iteration_limit = 400,
      restart_scheme = FirstOrderLp.NO_RESTARTS,
      restart_frequency_if_fixed = 1000,
    )
    problem = example_lp()
    old_stdout = stdout # We don't want to see this output on the screen.
    stdout_rd, stdout_wr = redirect_stdout()
    try
      output = FirstOrderLp.optimize(parameters, problem)
      @test output.primal_solution ≈ [1.0; 0.0; 6.0; 2.0] atol = 1.0e-4
      @test output.dual_solution ≈ [0.5; 4.0; 0.0] atol = 1.0e-4
    finally
      redirect_stdout(old_stdout)
    end
  end
  @testset "record_iteration_stats = false" begin
    parameters = generate_mirror_prox_params(
      primal_importance = 1.0,
      diagonal_scaling = "off",
      verbosity = 0,
      iteration_limit = 400,
      restart_scheme = FirstOrderLp.NO_RESTARTS,
      restart_frequency_if_fixed = 1000,
      record_iteration_stats = false,
    )
    problem = example_lp()
    output = FirstOrderLp.optimize(parameters, problem)
    @test output.primal_solution ≈ [1.0; 0.0; 6.0; 2.0] atol = 1.0e-4
    @test output.dual_solution ≈ [0.5; 4.0; 0.0] atol = 1.0e-4
  end
  @testset "Quadratic Programming 1" begin
    parameters = generate_mirror_prox_params(
      primal_importance = 1.0,
      diagonal_scaling = "l1",
      verbosity = 0,
      iteration_limit = 1000,
      restart_scheme = FirstOrderLp.NO_RESTARTS,
      restart_frequency_if_fixed = 1000,
    )
    problem = example_qp()
    output = FirstOrderLp.optimize(parameters, problem)
    @test output.primal_solution ≈ [0.2; 0.8] atol = 1.0e-4
    @test output.dual_solution ≈ [0.2] atol = 1.0e-4
  end
  @testset "Quadratic Programming 2" begin
    parameters = generate_mirror_prox_params(
      primal_importance = 1.0,
      diagonal_scaling = "l1",
      verbosity = 0,
      iteration_limit = 400,
      restart_scheme = FirstOrderLp.NO_RESTARTS,
      restart_frequency_if_fixed = 1000,
    )
    problem = example_qp2()
    output = FirstOrderLp.optimize(parameters, problem)
    @test output.primal_solution ≈ [0.25; 0.0] atol = 1.0e-4
    @test output.dual_solution ≈ [0.0] atol = 1.0e-4
  end
  @testset "Testing Preprocessing" begin
    @testset "l2 norm rescaling" begin
      parameters = generate_mirror_prox_params(
        l2_norm_rescaling = true,
        primal_importance = 1.0,
        diagonal_scaling = "l1",
        verbosity = 0,
        iteration_limit = 400,
        restart_scheme = FirstOrderLp.NO_RESTARTS,
        restart_frequency_if_fixed = 1000,
      )
      problem = example_qp2()
      output = FirstOrderLp.optimize(parameters, problem)
      @test output.primal_solution ≈ [0.25; 0.0] atol = 1.0e-4
      @test output.dual_solution ≈ [0.0] atol = 1.0e-4
    end
    @testset "ruiz" begin
      parameters = generate_mirror_prox_params(
        l_inf_ruiz_iterations = 10,
        primal_importance = 1.0,
        diagonal_scaling = "l1",
        verbosity = 0,
        iteration_limit = 400,
        restart_scheme = FirstOrderLp.NO_RESTARTS,
        restart_frequency_if_fixed = 1000,
      )
      problem = example_qp2()
      output = FirstOrderLp.optimize(parameters, problem)
      @test output.primal_solution ≈ [0.25; 0.0] atol = 1.0e-4
      @test output.dual_solution ≈ [0.0] atol = 1.0e-4
    end
    @testset "Pock-Chambolle rescaling" begin
      parameters = generate_mirror_prox_params(
        pock_chambolle_alpha = 1.0,
        primal_importance = 1.0,
        diagonal_scaling = "off",
        verbosity = 0,
        iteration_limit = 400,
        restart_scheme = FirstOrderLp.NO_RESTARTS,
        restart_frequency_if_fixed = 1000,
      )
      problem = example_lp()
      output = FirstOrderLp.optimize(parameters, problem)
      @test output.primal_solution ≈ [1.0; 0.0; 6.0; 2.0] atol = 1.0e-4
      @test output.dual_solution ≈ [0.5; 4.0; 0.0] atol = 1.0e-4
    end
  end
  @testset "diagonal_scaling=l2" begin
    parameters = generate_mirror_prox_params(
      primal_importance = 1.0,
      diagonal_scaling = "l2",
      verbosity = 0,
      iteration_limit = 400,
      restart_scheme = FirstOrderLp.NO_RESTARTS,
      restart_frequency_if_fixed = 1000,
    )
    problem = example_lp()
    output = FirstOrderLp.optimize(parameters, problem)
    @test output.primal_solution ≈ [1.0; 0.0; 6.0; 2.0] atol = 1.0e-4
    @test output.dual_solution ≈ [0.5; 4.0; 0.0] atol = 1.0e-4
  end
  @testset "diagonal_scaling=l1" begin
    parameters = generate_mirror_prox_params(
      primal_importance = 1.0,
      diagonal_scaling = "l1",
      verbosity = 0,
      iteration_limit = 400,
      restart_scheme = FirstOrderLp.NO_RESTARTS,
      restart_frequency_if_fixed = 1000,
    )
    problem = example_lp()
    output = FirstOrderLp.optimize(parameters, problem)
    @test output.primal_solution ≈ [1.0; 0.0; 6.0; 2.0] atol = 1.0e-4
    @test output.dual_solution ≈ [0.5; 4.0; 0.0] atol = 1.0e-4
  end
  @testset "restart_scheme=ADAPTIVE_NORMALIZED" begin
    parameters = generate_mirror_prox_params(
      iteration_limit = 700,
      primal_importance = 1.0,
      diagonal_scaling = "off",
      verbosity = 0,
      restart_scheme = FirstOrderLp.ADAPTIVE_NORMALIZED,
    )
    problem = example_lp()
    output = FirstOrderLp.optimize(parameters, problem)
    @test output.primal_solution ≈ [1.0; 0.0; 6.0; 2.0] atol = 1.0e-9
    @test output.dual_solution ≈ [0.5; 4.0; 0.0] atol = 1.0e-9
  end
  @testset "restart_scheme=adaptive_distance" begin
    parameters = generate_mirror_prox_params(
      iteration_limit = 700,
      primal_importance = 1.0,
      diagonal_scaling = "off",
      verbosity = 0,
      restart_scheme = FirstOrderLp.ADAPTIVE_DISTANCE,
    )
    problem = example_lp()
    output = FirstOrderLp.optimize(parameters, problem)
    @test output.primal_solution ≈ [1.0; 0.0; 6.0; 2.0] atol = 1.0e-9
    @test output.dual_solution ≈ [0.5; 4.0; 0.0] atol = 1.0e-9
  end
  @testset "restart_scheme=adaptive_localized" begin
    parameters = generate_mirror_prox_params(
      iteration_limit = 750,
      primal_importance = 1.0,
      diagonal_scaling = "off",
      verbosity = 0,
      restart_scheme = FirstOrderLp.ADAPTIVE_LOCALIZED,
    )
    problem = example_lp()
    output = FirstOrderLp.optimize(parameters, problem)
    @test output.primal_solution ≈ [1.0; 0.0; 6.0; 2.0] atol = 1.0e-9
    @test output.dual_solution ≈ [0.5; 4.0; 0.0] atol = 1.0e-9
  end
  @testset "restart_to_current_metric = no_restart_to_current" begin
    parameters = generate_mirror_prox_params(
      iteration_limit = 700,
      primal_importance = 1.0,
      diagonal_scaling = "off",
      verbosity = 0,
      restart_scheme = FirstOrderLp.ADAPTIVE_NORMALIZED,
      restart_to_current_metric = FirstOrderLp.NO_RESTART_TO_CURRENT,
    )
    problem = example_lp()
    output = FirstOrderLp.optimize(parameters, problem)
    @test output.primal_solution ≈ [1.0; 0.0; 6.0; 2.0] atol = 1.0e-9
    @test output.dual_solution ≈ [0.5; 4.0; 0.0] atol = 1.0e-9
  end
  @testset "use_approximate_localized_duality_gap = true" begin
    parameters = generate_mirror_prox_params(
      iteration_limit = 800,
      primal_importance = 1.0,
      diagonal_scaling = "off",
      verbosity = 0,
      restart_scheme = FirstOrderLp.ADAPTIVE_NORMALIZED,
      restart_frequency_if_fixed = 1000,
      use_approximate_localized_duality_gap = true,
    )
    problem = example_lp()
    output = FirstOrderLp.optimize(parameters, problem)
    @test output.primal_solution ≈ [1.0; 0.0; 6.0; 2.0] atol = 1.0e-9
    @test output.dual_solution ≈ [0.5; 4.0; 0.0] atol = 1.0e-9
  end
  @testset "restart_scheme=fixed_frequency" begin
    parameters = generate_mirror_prox_params(
      iteration_limit = 600,
      primal_importance = 1.0,
      diagonal_scaling = "off",
      verbosity = 0,
      restart_scheme = FirstOrderLp.FIXED_FREQUENCY,
      restart_frequency_if_fixed = 40,
    )
    problem = example_lp()
    output = FirstOrderLp.optimize(parameters, problem)
    @test output.primal_solution ≈ [1.0; 0.0; 6.0; 2.0] atol = 1.0e-8
    @test output.dual_solution ≈ [0.5; 4.0; 0.0] atol = 1.0e-9
  end
  @testset "Quadratic Programming 1 restart_scheme=ADAPTIVE_NORMALIZED" begin
    parameters = generate_mirror_prox_params(
      iteration_limit = 1000,
      primal_importance = 1.0,
      diagonal_scaling = "l1",
      verbosity = 0,
      restart_scheme = FirstOrderLp.ADAPTIVE_NORMALIZED,
      restart_frequency_if_fixed = 100,
    )
    problem = example_qp()
    output = FirstOrderLp.optimize(parameters, problem)
    @test output.primal_solution ≈ [0.2; 0.8] atol = 1.0e-4
    @test output.dual_solution ≈ [0.2] atol = 1.0e-4
  end
  @testset "Quadratic Programming 2 restart_scheme=fixed_frequency" begin
    parameters = generate_mirror_prox_params(
      iteration_limit = 1000,
      primal_importance = 1.0,
      diagonal_scaling = "l1",
      verbosity = 0,
      restart_scheme = FirstOrderLp.FIXED_FREQUENCY,
      restart_frequency_if_fixed = 100,
    )
    problem = example_qp2()
    output = FirstOrderLp.optimize(parameters, problem)
    @test output.primal_solution ≈ [0.25; 0.0] atol = 1.0e-4
    @test output.dual_solution ≈ [0.0] atol = 1.0e-4
  end
  @testset "High precision" begin
    parameters = generate_mirror_prox_params(
      primal_importance = 1.0,
      diagonal_scaling = "off",
      verbosity = 0,
      iteration_limit = 1200,
      restart_scheme = FirstOrderLp.NO_RESTARTS,
      restart_frequency_if_fixed = 1000,
    )
    problem = example_lp()
    output = FirstOrderLp.optimize(parameters, problem)
    @test output.primal_solution ≈ [1.0; 0.0; 6.0; 2.0] atol = 1.0e-9
    @test output.dual_solution ≈ [0.5; 4.0; 0.0] atol = 1.0e-9
  end
  @testset "Primal infeasible instance" begin
    parameters = generate_mirror_prox_params(
      primal_importance = 1.0,
      diagonal_scaling = "off",
      verbosity = 0,
      iteration_limit = 500,
      restart_scheme = FirstOrderLp.NO_RESTARTS,
      restart_frequency_if_fixed = 1000,
    )
    problem = example_lp()
    problem.right_hand_side[3] = 8
    output = FirstOrderLp.optimize(parameters, problem)
    final_stats = output.iteration_stats[end]
    @test output.termination_reason ==
          FirstOrderLp.TERMINATION_REASON_PRIMAL_INFEASIBLE
  end
  @testset "Primal infeasible instance 2" begin
    parameters = generate_mirror_prox_params(
      primal_importance = 1.0,
      diagonal_scaling = "off",
      verbosity = 0,
      iteration_limit = 1100,
      restart_scheme = FirstOrderLp.NO_RESTARTS,
      restart_frequency_if_fixed = 1000,
    )
    problem = example_qp()
    problem.right_hand_side[1] = -5.0
    problem.num_equalities = 1
    parameters.termination_criteria.eps_primal_infeasible = 1e-8
    output = FirstOrderLp.optimize(parameters, problem)
    @test output.termination_reason ==
          FirstOrderLp.TERMINATION_REASON_PRIMAL_INFEASIBLE
  end
  @testset "Dual infeasible instance" begin
    # TODO: Convergence on this problem is really slow, even though it
    # is easy. Find out why.
    parameters = generate_mirror_prox_params(
      primal_importance = 1.0,
      diagonal_scaling = "off",
      verbosity = 0,
      iteration_limit = 10000,
      restart_scheme = FirstOrderLp.NO_RESTARTS,
      restart_frequency_if_fixed = 1000,
    )
    problem = example_qp2()
    problem.variable_lower_bound = [-Inf, -Inf]
    problem.objective_matrix[2, 2] = 0.0
    problem.num_equalities = 0
    parameters.termination_criteria.eps_dual_infeasible = 1e-3
    output = FirstOrderLp.optimize(parameters, problem)
    @test output.termination_reason ==
          FirstOrderLp.TERMINATION_REASON_DUAL_INFEASIBLE
  end
  @testset "LP without bounds" begin
    parameters = generate_mirror_prox_params(
      primal_importance = 1.0,
      diagonal_scaling = "off",
      verbosity = 0,
      iteration_limit = 400,
      restart_scheme = FirstOrderLp.NO_RESTARTS,
      restart_frequency_if_fixed = 1000,
    )
    problem = example_lp_without_bounds()
    output = FirstOrderLp.optimize(parameters, problem)
    @test output.primal_solution ≈ [2.0] atol = 1.0e-9
    @test output.dual_solution ≈ [1.0] atol = 1.0e-9
  end
  # Saddle point mirror prox does very well on this instance. This may be
  # because one can get an optimal solution by putting all primal and dual
  # variables either at a bound or in a wide range.
  @testset "Correlation Clustering: triangle plus" begin
    parameters = generate_mirror_prox_params(
      primal_importance = 1.0,
      diagonal_scaling = "off",
      verbosity = 0,
      iteration_limit = 20,
      restart_scheme = FirstOrderLp.NO_RESTARTS,
      restart_frequency_if_fixed = 1000,
    )
    problem = example_cc_lp()
    output = FirstOrderLp.optimize(parameters, problem)
    # Numerical error occurs because no convergence tolerances are set.
    @test output.termination_reason ==
          FirstOrderLp.TERMINATION_REASON_NUMERICAL_ERROR
    tol = 1e-14
    @test output.primal_solution ≈ [1.0; 1.0; 0.0; 1.0; 0.0; 0.0] atol = tol
    final_stats = output.iteration_stats[end]
    @test final_stats.convergence_information[1].dual_objective ≈ 1.0 atol = tol
    # There are multiple optimal solutions.
    @test output.dual_solution[1] >= 0.0
    @test output.dual_solution[1] <= 1.0 + tol
    @test output.dual_solution[2] >= 0.0
    @test output.dual_solution[2] <= 1.0 + tol
    @test output.dual_solution[3] >= 0.0
    @test output.dual_solution[3] <= 1.0 + tol
    sum = output.dual_solution[1] + output.dual_solution[2]
    @test sum >= 1.0 - tol
  end
  @testset "Correlation Clustering: star" begin
    parameters = generate_mirror_prox_params(
      primal_importance = 1.0,
      diagonal_scaling = "off",
      verbosity = 0,
      iteration_limit = 200,
      restart_scheme = FirstOrderLp.NO_RESTARTS,
      restart_frequency_if_fixed = 1000,
    )
    problem = example_cc_star_lp()
    output = FirstOrderLp.optimize(parameters, problem)
    @test output.primal_solution ≈ [0.5; 0.5; 0.5; 0.0; 0.0; 0.0] atol = 1e-6
    @test output.dual_solution ≈ [0.5; 0.5; 0.5] atol = 1e-6
  end
end
