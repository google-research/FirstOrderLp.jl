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

function generate_primal_dual_hybrid_gradient_params(;
  l_inf_ruiz_iterations = 0,
  l2_norm_rescaling = false,
  pock_chambolle_alpha = nothing,
  iteration_limit = 200,
  primal_importance = 1.0,
  diagonal_scaling = "off",
  verbosity = 0,
  record_iteration_stats = true,
  restart_scheme = FirstOrderLp.NO_RESTARTS,
  restart_frequency_if_fixed = 100,
  artificial_restart_threshold = 0.5,
  use_weighted_average = true,
  sufficient_reduction_for_restart = 0.1,
  necessary_reduction_for_restart = 0.8,
  primal_weight_update_smoothing = 0.5,
  termination_evaluation_frequency = 5,
  use_approximate_localized_duality_gap = false,
  restart_to_current_metric = FirstOrderLp.GAP_OVER_DISTANCE_SQUARED,
  adaptive_step_size = true,
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
  parameters = FirstOrderLp.PdhgParameters(
    l_inf_ruiz_iterations,
    l2_norm_rescaling,
    pock_chambolle_alpha,
    primal_importance,
    diagonal_scaling,
    adaptive_step_size,
    verbosity,
    record_iteration_stats,
    termination_evaluation_frequency,
    terminate_on_iteration_limit(iteration_limit),
    restart_params,
  )
  return parameters
end

@testset "Primal-dual hybrid gradient" begin
  @testset "Low precision" begin
    parameters = generate_primal_dual_hybrid_gradient_params(
      iteration_limit = 300,
      primal_importance = 1.0,
      diagonal_scaling = "off",
      verbosity = 0,
    )
    problem = example_lp()
    output = FirstOrderLp.optimize(parameters, problem)
    @test output.primal_solution ≈ [1.0; 0.0; 6.0; 2.0] atol = 1.0e-4
    @test output.dual_solution ≈ [0.5; 4.0; 0.0] atol = 1.0e-4
  end
  @testset "Terminate with optimal solution" begin
    parameters = generate_primal_dual_hybrid_gradient_params(
      iteration_limit = 1000,
      primal_importance = 1.0,
      diagonal_scaling = "off",
      verbosity = 0,
    )
    problem = example_lp()
    parameters.termination_criteria.eps_optimal_absolute = 1e-8
    output = FirstOrderLp.optimize(parameters, problem)
    @test output.termination_reason == FirstOrderLp.TERMINATION_REASON_OPTIMAL
  end
  @testset "Test Verbosity" begin
    parameters = generate_primal_dual_hybrid_gradient_params(
      iteration_limit = 300,
      primal_importance = 1.0,
      diagonal_scaling = "off",
      verbosity = 10,
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
  @testset "Test Fixed Frequency Restart" begin
    parameters = generate_primal_dual_hybrid_gradient_params(
      iteration_limit = 500,
      primal_importance = 1.0,
      diagonal_scaling = "off",
      verbosity = 0,
      # This test breaks if you set restart_scheme=NO_RESTARTS.
      restart_scheme = FirstOrderLp.FIXED_FREQUENCY,
      restart_frequency_if_fixed = 30,
    )
    problem = example_lp()
    output = FirstOrderLp.optimize(parameters, problem)
    @test output.primal_solution ≈ [1.0; 0.0; 6.0; 2.0] atol = 1.0e-9
    @test output.dual_solution ≈ [0.5; 4.0; 0.0] atol = 1.0e-9
  end
  @testset "Test Adaptive Restart Heuristic" begin
    parameters = generate_primal_dual_hybrid_gradient_params(
      iteration_limit = 600,
      primal_importance = 1.0,
      diagonal_scaling = "off",
      verbosity = 0,
      restart_scheme = FirstOrderLp.ADAPTIVE_NORMALIZED,
    )

    problem = example_lp()
    output = FirstOrderLp.optimize(parameters, problem)
    @test output.primal_solution ≈ [1.0; 0.0; 6.0; 2.0] atol = 1.0e-9
    @test output.dual_solution ≈ [0.5; 4.0; 0.0] atol = 1.0e-9

    @test any(
      it -> it.restart_used == FirstOrderLp.RESTART_CHOICE_RESTART_TO_AVERAGE,
      output.iteration_stats,
    )
  end

  @testset "Test adaptive_step_size=false, no smoothing" begin
    parameters = generate_primal_dual_hybrid_gradient_params(
      iteration_limit = 700,
      primal_importance = 1.0,
      diagonal_scaling = "off",
      verbosity = 0,
      primal_weight_update_smoothing = 0.0, # this test breaks if smoothing=0.5
      restart_scheme = FirstOrderLp.ADAPTIVE_NORMALIZED,
      adaptive_step_size = false,
    )

    problem = example_lp()
    output = FirstOrderLp.optimize(parameters, problem)
    @test output.primal_solution ≈ [1.0; 0.0; 6.0; 2.0] atol = 1.0e-9
    @test output.dual_solution ≈ [0.5; 4.0; 0.0] atol = 1.0e-9

    @test any(
      it -> it.restart_used == FirstOrderLp.RESTART_CHOICE_RESTART_TO_AVERAGE,
      output.iteration_stats,
    )
    step_size = output.iteration_stats[1].step_size
    for i in 2:length(output.iteration_stats)
      @test output.iteration_stats[i].step_size == step_size
    end
  end

  @testset "Test restart_to_current_metric = no_restart_to_current" begin
    parameters = generate_primal_dual_hybrid_gradient_params(
      iteration_limit = 600,
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

    @test any(
      it -> it.restart_used == FirstOrderLp.RESTART_CHOICE_RESTART_TO_AVERAGE,
      output.iteration_stats,
    )
  end

  @testset "Test restart_to_current_metric = gap_over_distance" begin
    parameters = generate_primal_dual_hybrid_gradient_params(
      iteration_limit = 600,
      primal_importance = 1.0,
      diagonal_scaling = "off",
      verbosity = 0,
      restart_scheme = FirstOrderLp.ADAPTIVE_NORMALIZED,
      restart_to_current_metric = FirstOrderLp.GAP_OVER_DISTANCE,
    )

    problem = example_lp()
    output = FirstOrderLp.optimize(parameters, problem)
    @test output.primal_solution ≈ [1.0; 0.0; 6.0; 2.0] atol = 1.0e-9
    @test output.dual_solution ≈ [0.5; 4.0; 0.0] atol = 1.0e-9

    @test any(
      it -> it.restart_used == FirstOrderLp.RESTART_CHOICE_RESTART_TO_AVERAGE,
      output.iteration_stats,
    )
  end

  @testset "Test Adaptive Restart" begin
    parameters = generate_primal_dual_hybrid_gradient_params(
      iteration_limit = 200,
      primal_importance = 1.0,
      diagonal_scaling = "off",
      verbosity = 0,
      restart_scheme = FirstOrderLp.ADAPTIVE_NORMALIZED,
    )

    problem = example_lp()
    problem.objective_vector = [0.0, 0.0, 0.0, 0.0]
    parameters.termination_criteria.eps_optimal_absolute = 1e-8
    output = FirstOrderLp.optimize(parameters, problem)
    @test output.termination_reason == FirstOrderLp.TERMINATION_REASON_OPTIMAL
  end

  @testset "use_approximate_localized_duality_gap = true" begin
    parameters = generate_primal_dual_hybrid_gradient_params(
      iteration_limit = 300,
      primal_importance = 1.0,
      diagonal_scaling = "off",
      verbosity = 0,
      restart_scheme = FirstOrderLp.ADAPTIVE_NORMALIZED,
      use_approximate_localized_duality_gap = true,
    )

    problem = example_lp()
    problem.objective_vector = [0.0, 0.0, 0.0, 0.0]
    parameters.termination_criteria.eps_optimal_absolute = 1e-8
    output = FirstOrderLp.optimize(parameters, problem)
    @test output.termination_reason == FirstOrderLp.TERMINATION_REASON_OPTIMAL
  end

  @testset "Quadratic Programming 1" begin
    parameters = generate_primal_dual_hybrid_gradient_params(
      iteration_limit = 200,
      primal_importance = 1.0,
      diagonal_scaling = "l1",
      verbosity = 0,
    )
    problem = example_qp()
    output = FirstOrderLp.optimize(parameters, problem)
    @test output.primal_solution ≈ [0.2; 0.8] atol = 1.0e-4
    @test output.dual_solution ≈ [0.2] atol = 1.0e-4
  end
  @testset "Quadratic Programming 2" begin
    parameters = generate_primal_dual_hybrid_gradient_params(
      iteration_limit = 200,
      primal_importance = 1.0,
      diagonal_scaling = "l1",
      verbosity = 0,
    )
    problem = example_qp2()
    output = FirstOrderLp.optimize(parameters, problem)
    @test output.primal_solution ≈ [0.25; 0.0] atol = 1.0e-4
    @test output.dual_solution ≈ [0.0] atol = 1.0e-4
  end
  @testset "Test Preprocessing" begin
    @testset "l2 norm rescaling" begin
      parameters = generate_primal_dual_hybrid_gradient_params(
        l2_norm_rescaling = true,
        iteration_limit = 200,
        primal_importance = 1.0,
        diagonal_scaling = "l1",
        verbosity = 0,
      )
      problem = example_qp2()
      output = FirstOrderLp.optimize(parameters, problem)
      @test output.primal_solution ≈ [0.25; 0.0] atol = 1.0e-4
      @test output.dual_solution ≈ [0.0] atol = 1.0e-4
    end
    @testset "ruiz" begin
      parameters = generate_primal_dual_hybrid_gradient_params(
        l_inf_ruiz_iterations = 10,
        iteration_limit = 200,
        primal_importance = 1.0,
        diagonal_scaling = "l1",
        verbosity = 0,
      )
      problem = example_qp2()
      output = FirstOrderLp.optimize(parameters, problem)
      @test output.primal_solution ≈ [0.25; 0.0] atol = 1.0e-4
      @test output.dual_solution ≈ [0.0] atol = 1.0e-4
    end
    @testset "Pock-Chambolle rescaling" begin
      parameters = generate_primal_dual_hybrid_gradient_params(
        pock_chambolle_alpha = 1.0,
        iteration_limit = 200,
        primal_importance = 1.0,
        verbosity = 0,
      )
      problem = example_lp()
      output = FirstOrderLp.optimize(parameters, problem)
      @test output.primal_solution ≈ [1.0; 0.0; 6.0; 2.0] atol = 1.0e-4
      @test output.dual_solution ≈ [0.5; 4.0; 0.0] atol = 1.0e-4
    end
  end
  @testset "diagonal_scaling=l2" begin
    parameters = generate_primal_dual_hybrid_gradient_params(
      iteration_limit = 300,
      primal_importance = 1.0,
      diagonal_scaling = "l2",
      verbosity = 0,
    )
    problem = example_lp()
    output = FirstOrderLp.optimize(parameters, problem)
    @test output.primal_solution ≈ [1.0; 0.0; 6.0; 2.0] atol = 1.0e-4
    @test output.dual_solution ≈ [0.5; 4.0; 0.0] atol = 1.0e-4
  end
  @testset "diagonal_scaling=l1" begin
    parameters = generate_primal_dual_hybrid_gradient_params(
      iteration_limit = 300,
      primal_importance = 1.0,
      diagonal_scaling = "l1",
      verbosity = 0,
    )
    problem = example_lp()
    output = FirstOrderLp.optimize(parameters, problem)
    @test output.primal_solution ≈ [1.0; 0.0; 6.0; 2.0] atol = 1.0e-4
    @test output.dual_solution ≈ [0.5; 4.0; 0.0] atol = 1.0e-4
  end
  @testset "High precision" begin
    parameters = generate_primal_dual_hybrid_gradient_params(
      iteration_limit = 800,
      primal_importance = 1.0,
      diagonal_scaling = "off",
      verbosity = 0,
    )
    problem = example_lp()
    output = FirstOrderLp.optimize(parameters, problem)
    @test output.primal_solution ≈ [1.0; 0.0; 6.0; 2.0] atol = 1.0e-9
    @test output.dual_solution ≈ [0.5; 4.0; 0.0] atol = 1.0e-9
  end
  @testset "Infeasible instance" begin
    parameters = generate_primal_dual_hybrid_gradient_params(
      iteration_limit = 800,
      primal_importance = 1.0,
      diagonal_scaling = "off",
      verbosity = 0,
    )
    problem = example_lp()
    problem.right_hand_side[3] = 8
    output = FirstOrderLp.optimize(parameters, problem)
    final_stats = output.iteration_stats[end]
    @test output.termination_reason ==
          FirstOrderLp.TERMINATION_REASON_PRIMAL_INFEASIBLE
  end
  @testset "LP without bounds" begin
    parameters = generate_primal_dual_hybrid_gradient_params(
      iteration_limit = 400,
      primal_importance = 1.0,
      diagonal_scaling = "off",
      verbosity = 0,
    )
    problem = example_lp_without_bounds()
    output = FirstOrderLp.optimize(parameters, problem)
    @test output.primal_solution ≈ [2.0] atol = 1.0e-9
    @test output.dual_solution ≈ [1.0] atol = 1.0e-9
  end
  @testset "Correlation Clustering: triangle plus" begin
    parameters = generate_primal_dual_hybrid_gradient_params(
      iteration_limit = 15,
      primal_importance = 1.0,
      diagonal_scaling = "off",
      verbosity = 0,
    )
    problem = example_cc_lp()
    output = FirstOrderLp.optimize(parameters, problem)
    tol = 1e-14
    final_stats = output.iteration_stats[end]
    @test output.primal_solution ≈ [1.0; 1.0; 0.0; 1.0; 0.0; 0.0] atol = tol
    @test final_stats.convergence_information[1].dual_objective ≈ 1.0 atol = tol
    # There are multiple optimal solutions.
    @test output.dual_solution[1] >= 0.0
    @test output.dual_solution[2] >= 0.0
    @test output.dual_solution[3] >= 0.0
    sum = output.dual_solution[1] + output.dual_solution[2]
    @test sum >= 1.0 - tol
  end
  @testset "Numerical error" begin
    parameters = generate_primal_dual_hybrid_gradient_params(
      iteration_limit = 150,
      primal_importance = 1.0,
      diagonal_scaling = "off",
      verbosity = 0,
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
    @test output.dual_solution[2] >= 0.0
    @test output.dual_solution[3] >= 0.0
    sum = output.dual_solution[1] + output.dual_solution[2]
    @test sum >= 1.0 - tol
  end
  @testset "Correlation Clustering: star" begin
    parameters = generate_primal_dual_hybrid_gradient_params(
      iteration_limit = 100,
      primal_importance = 1.0,
      diagonal_scaling = "off",
      verbosity = 0,
    )
    problem = example_cc_star_lp()
    output = FirstOrderLp.optimize(parameters, problem)
    @test output.primal_solution ≈ [0.5; 0.5; 0.5; 0.0; 0.0; 0.0] atol = 1e-6
    @test output.dual_solution ≈ [0.5; 0.5; 0.5] atol = 1e-6
  end
end
