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

"""
A PdhgParameters struct specifies the parameters for solving the saddle
point formulation of an problem using primal-dual hybrid gradient.

Quadratic Programming Problem (see quadratic_programming.jl):
minimize 1/2 * x' * objective_matrix * x + objective_vector' * x
         + objective_constant

s.t. constraint_matrix[1:num_equalities, :] * x =
     right_hand_side[1:num_equalities]

     constraint_matrix[(num_equalities + 1):end, :] * x >=
     right_hand_side[(num_equalities + 1):end, :]

     variable_lower_bound <= x <= variable_upper_bound

We use notation from Chambolle and Pock, "On the ergodic convergence rates of a
first-order primal-dual algorithm"
(http://www.optimization-online.org/DB_FILE/2014/09/4532.pdf).
That paper doesn't explicitly use the terminology "primal-dual hybrid gradient"
but their Theorem 1 is analyzing PDHG. In this file "Theorem 1" without further
reference refers to that paper.

Our problem is equivalent to the saddle point problem:
    min_x max_y L(x, y)
where
    L(x, y) = y' K x + f(x) + g(x) - h*(y)
    K = -constraint_matrix
    f(x) = objective_constant + objective_vector' x + 1/2*x' objective_matrix x
    g(x) = 0 if variable_lower_bound <= x <= variable_upper_bound
           otherwise infinity
    h*(y) = -right_hand_side' y if y[(num_equalities + 1):end] >= 0
                                otherwise infinity

Note that the places where g(x) and h*(y) are infinite effectively limits the
domain of the min and max. Therefore there's no infinity in the code.

Here we use Q as the abbreviation of objective_matrix. We use mirror map
1/2 ||x||_X^2 + 1/2 ||y||_Y^2, where X and Y are diagonal matrices.
If `diagonal_scaling`=l1, we set
    X[i,i] = Q[i,i] + (1 / step_size) * (sum_{j!=i}|Q[i,j]| + primal_weight *
    sum_{j}|K[j,i]|)
    Y[j,j] = (1 / step_size) / primal_weight / sum_{i}|K[j,i]|
If `diagonal_scaling`=l2, we set
    X[i,i] = Q[i,i] + (1 / step_size) * sqrt(sum_{j!=i} Q[i,j]^2 +
    primal_weight^2 * sum_{j} K[j,i]^2)
    Y[j,j] = (1 / step_size) / primal_weight / sqrt(sum_{i}K[j,i]^2)
If `diagonal_scaling`=off, we set
    X[i,i] = (1 / step_size) * primal_weight
    Y[j,j] = (1 / step_size) / primal_weight
The step_size and primal_weight are parameters described next.
The parameter primal_weight is adjusted smoothly at each restart; to balance the
primal and dual distances traveled since the last restart; see
compute_new_primal_weight().
In the LP case, using this norm is equivalent to parameterizing the primal and
dual step sizes (tau and sigma in Chambolle and Pock) as:
    primal_step_size = step_size / primal_weight
    dual_step_size = step_size * primal_weight
We adjust step_size to be as large as possible without violating the condition
assumed in Theorem 1. Adjusting the step size unfortunately seems to invalidate
that Theorem (unlike the case of mirror prox) but this step size adjustment
heuristic seems to work fine in practice. See comments in the code for details.

TODO: compare the above step size scheme with the scheme by Goldstein
et al (https://arxiv.org/pdf/1305.0546.pdf).

TODO: explore PDHG variants with tuning parameters, e.g. the
overrelaxed and intertial variants in Chambolle and Pock and the algorithm in
"An Algorithmic Framework of Generalized Primal-Dual Hybrid Gradient Methods for
Saddle Point Problems" by Bingsheng He, Feng Ma, Xiaoming Yuan
(http://www.optimization-online.org/DB_FILE/2016/02/5315.pdf).
"""
struct PdhgParameters
  """
  One of the supported steps: {rescale-columns, ruiz, l2-ruiz, none}.
  "none" means no preprocessing.
  """
  rescaling::String

  """
  Used to bias the computation of the primal/dual balancing parameter
  primal_weight. Must be positive. A value of 1 balances primal and dual
  equally.
  """
  primal_importance::Float64

  """
  Use weighted l2 norm as the Bregman divergence so that it rescales each
  primal and dual variable.
  """
  diagonal_scaling::String

  adaptive_step_size::Bool

  """
  If >= 4 a line of debugging info is printed during some iterations. If >= 2
  some info is printed about the final solution.
  """
  verbosity::Int64

  """
  Whether to record an IterationStats proto.
  """
  record_iteration_stats::Bool

  """
  Check for termination with this frequency (in iterations).
  """
  termination_evaluation_frequency::Int32

  """
  The termination criteria for the algorithm.
  """
  termination_criteria::TerminationCriteria

  """
  Parameters that control when the algorithm restarts and whether it resets to
  the average or the current iterate. Also, controls the primal weight updates.
  """
  restart_params::RestartParameters
end

"""
Defines the primal norm and dual norm using the norms of matrices, step_size
and primal_weight.
"""
function define_norms(
  diagonal_scaling::String,
  diagonal_objective_matrix,
  row_norm_objective_matrix::Vector{Float64},
  row_norm_constraint_matrix::Vector{Float64},
  column_norm_constraint_matrix::Vector{Float64},
  step_size::Float64,
  primal_weight::Float64,
)
  if diagonal_scaling == "l2"
    # If `diagonal_scaling`=l2, we set X[i,i]=||K[:,i]||_2 and
    # Y[j,j]=||K[j,:]||_2
    primal_norm_params =
      diagonal_objective_matrix .+
      1 / step_size *
      sqrt.(
        row_norm_objective_matrix .+
        primal_weight^2 * row_norm_constraint_matrix,
      )
    dual_norm_params =
      1 / step_size / primal_weight * sqrt.(column_norm_constraint_matrix)
  elseif diagonal_scaling == "l1"
    primal_norm_params =
      diagonal_objective_matrix .+
      1 / step_size *
      (row_norm_objective_matrix .+ primal_weight * row_norm_constraint_matrix)
    dual_norm_params =
      1 / step_size / primal_weight * column_norm_constraint_matrix
  else
    primal_norm_params =
      1 / step_size * primal_weight * ones(length(row_norm_constraint_matrix))
    dual_norm_params =
      1 / step_size / primal_weight *
      ones(length(column_norm_constraint_matrix))
  end

  return primal_norm_params, dual_norm_params
end

"""
Logging while the algorithm is running.
"""
function pdhg_specific_log(
  problem::QuadraticProgrammingProblem,
  iteration::Int64,
  current_primal_solution::Vector{Float64},
  current_dual_solution::Vector{Float64},
  original_primal_norm_params::Vector{Float64},
  original_dual_norm_params::Vector{Float64},
  step_size::Float64,
  required_ratio::Float64,
  primal_weight::Float64,
)
  Printf.@printf(
    "   %5d norms=(%9g, %9g) weighted_norm=(%9g, %9g) inv_step_size=%9g req'd=%9g ",
    iteration,
    norm(current_primal_solution),
    norm(current_dual_solution),
    weighted_norm(current_primal_solution, original_primal_norm_params),
    weighted_norm(current_dual_solution, original_dual_norm_params),
    1 / step_size,
    required_ratio
  )
  Printf.@printf(
    "   primal_weight=%18g dual_obj=%18g  inverse_ss=%18g\n",
    primal_weight,
    corrected_dual_obj(problem, current_primal_solution, current_dual_solution),
    required_ratio
  )
end

"""
Logging for when the algorithm terminates.
"""
function pdhg_final_log(
  problem::QuadraticProgrammingProblem,
  avg_primal_solution::Vector{Float64},
  avg_dual_solution::Vector{Float64},
  original_primal_norm_params::Vector{Float64},
  original_dual_norm_params::Vector{Float64},
  verbosity::Int64,
  iteration::Int64,
  termination_reason::TerminationReason,
  last_iteration_stats::IterationStats,
)

  if verbosity >= 2
    infeas = max_primal_violation(problem, avg_primal_solution)
    primal_obj_val = primal_obj(problem, avg_primal_solution)
    dual_stats =
      compute_dual_stats(problem, avg_primal_solution, avg_dual_solution)
    println("Avg solution:")
    Printf.@printf(
      "  pr_infeas=%12g pr_obj=%15.10g dual_infeas=%12g dual_obj=%15.10g\n",
      infeas,
      primal_obj_val,
      norm(dual_stats.dual_residual, Inf),
      dual_stats.dual_objective
    )
    Printf.@printf(
      "  primal norms: L1=%15.10g, L2=%15.10g, weighted L2=%15.10g, Linf=%15.10g\n",
      norm(avg_primal_solution, 1),
      norm(avg_primal_solution),
      weighted_norm(avg_primal_solution, original_primal_norm_params),
      norm(avg_primal_solution, Inf)
    )
    Printf.@printf(
      "  dual norms:   L1=%15.10g, L2=%15.10g, weighted L2=%15.10g, Linf=%15.10g\n",
      norm(avg_dual_solution, 1),
      norm(avg_dual_solution),
      weighted_norm(avg_dual_solution, original_dual_norm_params),
      norm(avg_dual_solution, Inf)
    )
  end

  generic_final_log(
    problem,
    avg_primal_solution,
    avg_dual_solution,
    original_primal_norm_params,
    original_dual_norm_params,
    last_iteration_stats,
    verbosity,
    iteration,
    termination_reason,
  )
end

"""
Estimate the probability that the power method, after k iterations, has relative
error > epsilon.  This is based on Theorem 4.1(a) (on page 13) from
"Estimating the Largest Eigenvalue by the Power and Lanczos Algorithms with a
Random Start"
https://pdfs.semanticscholar.org/2b2e/a941e55e5fa2ee9d8f4ff393c14482051143.pdf
"""
function power_method_failure_probability(
  dimension::Int64,
  epsilon::Float64,
  k::Int64,
)
  if k < 2 || epsilon <= 0.0
    # The theorem requires epsilon > 0 and k >= 2.
    return 1.0
  end
  return min(0.824, 0.354 / (epsilon * (k - 1))) *
         sqrt(dimension) *
         (1.0 - epsilon)^(k - 1 / 2)
end

"""
Estimate the maximum singular value using power method
https://en.wikipedia.org/wiki/Power_iteration, returning a result with
desired_relative_error with probability at least 1 - probability_of_failure.

Note that this will take approximately log(n / delta^2)/(2 * epsilon) iterations
as per the discussion at the bottom of page 15 of

"Estimating the Largest Eigenvalue by the Power and Lanczos Algorithms with a
Random Start"
https://pdfs.semanticscholar.org/2b2e/a941e55e5fa2ee9d8f4ff393c14482051143.pdf

For lighter reading on this topic see
https://courses.cs.washington.edu/courses/cse521/16sp/521-lecture-13.pdf
which does not include the failure probability.

# Output
A tuple containing:
- estimate of the maximum singular value
- the number of power iterations required to compute it
"""
function estimate_maximum_singular_value(
  matrix::SparseMatrixCSC{Float64,Int64};
  probability_of_failure = 0.01::Float64,
  desired_relative_error = 0.1::Float64,
  seed::Int64 = 1,
)
  # Epsilon is the relative error on the eigenvalue of matrix' * matrix.
  epsilon = 1.0 - (1.0 - desired_relative_error)^2
  # Use the power method on matrix' * matrix
  x = randn(Random.MersenneTwister(seed), size(matrix, 2))

  number_of_power_iterations = 0
  while power_method_failure_probability(
    size(matrix, 2),
    epsilon,
    number_of_power_iterations,
  ) > probability_of_failure
    x = x / norm(x, 2)
    x = matrix' * (matrix * x)
    number_of_power_iterations += 1
  end

  # The singular value is the square root of the maximum eigenvalue of
  # matrix' * matrix
  return sqrt(dot(x, matrix' * (matrix * x)) / norm(x, 2)^2),
  number_of_power_iterations
end
"""
`optimize(params::PdhgParameters,
          original_problem::QuadraticProgrammingProblem)`

Solves a quadratic program using primal-dual hybrid gradient.

# Arguments
- `params::PdhgParameters`: parameters.
- `original_problem::QuadraticProgrammingProblem`: the QP to solve.

# Returns
A SaddlePointOutput struct containing the solution found.
"""
function optimize(
  params::PdhgParameters,
  original_problem::QuadraticProgrammingProblem,
)
  validate(original_problem)
  qp_cache = cached_quadratic_program_info(original_problem)
  scaled_problem =
    rescale_problem(params.rescaling, params.verbosity, original_problem)
  problem = scaled_problem.scaled_qp

  primal_size = length(problem.variable_lower_bound)
  dual_size = length(problem.right_hand_side)
  if params.primal_importance <= 0 || !isfinite(params.primal_importance)
    error("primal_importance must be positive and finite")
  end

  # TODO: Correctly account for the number of kkt passes in
  # initialization
  cumulative_kkt_passes = 0.0
  if params.adaptive_step_size
    cumulative_kkt_passes += 0.5
    step_size = 1.0 / norm(problem.constraint_matrix, Inf)
  else
    desired_relative_error = 0.2
    maximum_singular_value, number_of_power_iterations =
      estimate_maximum_singular_value(
        problem.constraint_matrix,
        probability_of_failure = 0.001,
        desired_relative_error = desired_relative_error,
      )
    step_size = (1 - desired_relative_error) / maximum_singular_value
    cumulative_kkt_passes += number_of_power_iterations
  end
  current_primal_solution = zeros(primal_size)
  current_dual_solution = zeros(dual_size)
  solution_weighted_avg = initialize_solution_weighted_average(
    length(current_primal_solution),
    length(current_dual_solution),
  )
  iterations_completed = 0

  # This excludes the pass to compute the interaction term, because it can
  # be removed.
  KKT_PASSES_PER_ITERATION = 1.0
  # Idealized number of KKT passes each time the termination criteria and
  # restart scheme is run. One of these comes from evaluating the gradient at
  # the average solution and evaluating the gradient at the current solution.
  # In practice this number is four.
  KKT_PASSES_PER_TERMINATION_EVALUATION = 2.0

  # With mirror map 1/2 ||x||_X^2 + 1/2 ||y||_Y^2 the theory holds when
  # [X-Q, -A'; -A, Y] is positive semi-definite. This is automatically
  # satisfied when we set
  # X[i,i] = Q[i,i] + step_size * (sum_{j!=i}|Q[i,j]| + 1/primal_weight *
  # sum_{j}|K[j,i]|) and
  # Y[j,j] = 1 / step_size / primal_weight / sum_{i}|K[j,i]|. In practice, using
  # ||.||_2 instead of ||.||_1 to define X and Y often improves the convergence
  # speed for pdhg.
  diagonal_objective_matrix = Vector(diag(problem.objective_matrix))
  if params.diagonal_scaling == "l2"
    row_norm_objective_matrix =
      vec((sum(problem.objective_matrix .^ 2, dims = 1)))
    row_norm_constraint_matrix =
      vec((sum(problem.constraint_matrix .^ 2, dims = 1)))
    column_norm_constraint_matrix =
      vec((sum(problem.constraint_matrix .^ 2, dims = 2)))
  elseif params.diagonal_scaling == "l1"
    row_norm_objective_matrix =
      vec(sum(abs.(problem.objective_matrix), dims = 1))
    row_norm_constraint_matrix =
      vec(sum(abs.(problem.constraint_matrix), dims = 1))
    column_norm_constraint_matrix =
      vec(sum(abs.(problem.constraint_matrix), dims = 2))
  elseif params.diagonal_scaling == "off"
    # diagonal_scaling needs to be "l1" or "l2" for QP.
    @assert iszero(problem.objective_matrix)
    row_norm_objective_matrix = ones(primal_size)
    row_norm_constraint_matrix = ones(primal_size)
    column_norm_constraint_matrix = ones(dual_size)
  end
  original_primal_norm_params, original_dual_norm_params = define_norms(
    params.diagonal_scaling,
    diagonal_objective_matrix,
    row_norm_objective_matrix,
    row_norm_constraint_matrix,
    column_norm_constraint_matrix,
    step_size,
    1.0,
  )
  primal_weight = select_initial_primal_weight(
    problem,
    original_primal_norm_params,
    original_dual_norm_params,
    params.primal_importance,
    params.verbosity,
  )

  primal_weight_update_smoothing =
    params.restart_params.primal_weight_update_smoothing

  primal_norm_params, dual_norm_params = define_norms(
    params.diagonal_scaling,
    diagonal_objective_matrix,
    row_norm_objective_matrix,
    row_norm_constraint_matrix,
    column_norm_constraint_matrix,
    step_size,
    primal_weight,
  )

  iteration_stats = IterationStats[]
  start_time = time()
  # Basic algorithm refers to the primal and dual steps, and excludes restart
  # schemes and termination evaluation.
  time_spent_doing_basic_algorithm = 0.0

  # For restart scheme
  if params.restart_params.restart_scheme == NO_RESTARTS
    # In approximately every 10% of the iterations the solution is averaged.
    # However, this averaging process could get of sync with the number of
    # iterations. This variable makes sure that if we hit the maximum number
    # of iterations the point returned is an average of at least the last 10%
    # of iterations.
    final_avg = false
  end

  # This variable is used in the adaptive restart scheme.
  last_restart_info = create_last_restart_info(
    problem,
    current_primal_solution,
    current_dual_solution,
  )

  # For termination criteria:
  termination_criteria = params.termination_criteria
  iteration_limit = termination_criteria.iteration_limit
  termination_evaluation_frequency = params.termination_evaluation_frequency

  # This flag represents whether a numerical error occurred during the algorithm
  # if it is set to true it will trigger the algorithm to terminate.
  numerical_error = false
  display_iteration_stats_heading(params.verbosity)

  iteration = 0
  while true
    iteration += 1

    # Evaluate the iteration stats at frequency
    # termination_evaluation_frequency, when the iteration_limit is reached,
    # or if a numerical error occurs at the previous iteration.
    if mod(iteration - 1, termination_evaluation_frequency) == 0 ||
       iteration == iteration_limit + 1 ||
       iteration <= 10 ||
       numerical_error
      # TODO: Experiment with evaluating every power of two iterations.
      # This ensures that we do sufficient primal weight updates in the initial
      # stages of the algorithm.
      cumulative_kkt_passes += KKT_PASSES_PER_TERMINATION_EVALUATION
      # Compute the average solution since the last restart point.
      if numerical_error || solution_weighted_avg.sum_solutions_count == 0
        avg_primal_solution = current_primal_solution
        avg_dual_solution = current_dual_solution
      else
        avg_primal_solution, avg_dual_solution =
          compute_average(solution_weighted_avg)
      end

      current_iteration_stats = evaluate_unscaled_iteration_stats(
        scaled_problem,
        qp_cache,
        params.termination_criteria,
        params.record_iteration_stats,
        avg_primal_solution,
        avg_dual_solution,
        iteration,
        time() - start_time,
        cumulative_kkt_passes,
        termination_criteria.eps_optimal_absolute,
        termination_criteria.eps_optimal_relative,
        step_size,
        primal_weight,
        POINT_TYPE_AVERAGE_ITERATE,
      )
      method_specific_stats = current_iteration_stats.method_specific_stats
      method_specific_stats["time_spent_doing_basic_algorithm"] =
        time_spent_doing_basic_algorithm
      update_objective_bound_estimates(
        current_iteration_stats.method_specific_stats,
        problem,
        avg_primal_solution,
        avg_dual_solution,
        primal_norm_params,
        dual_norm_params,
      )
      if params.record_iteration_stats
        push!(iteration_stats, current_iteration_stats)
      end

      # Check the termination criteria.
      termination_reason = check_termination_criteria(
        termination_criteria,
        qp_cache,
        current_iteration_stats,
      )
      if numerical_error && termination_reason == false
        termination_reason = NUMERICAL_ERROR
      end

      # Print table.
      if print_to_screen_this_iteration(
        termination_reason,
        iteration,
        params.verbosity,
        termination_evaluation_frequency,
      )
        display_iteration_stats(current_iteration_stats, params.verbosity)
      end

      if termination_reason != false
        # ** Terminate the algorithm **
        # This is the only place the algorithm can terminate. Please keep it
        # this way.
        pdhg_final_log(
          problem,
          avg_primal_solution,
          avg_dual_solution,
          original_primal_norm_params,
          original_dual_norm_params,
          params.verbosity,
          iteration,
          termination_reason,
          current_iteration_stats,
        )
        return unscaled_saddle_point_output(
          scaled_problem,
          avg_primal_solution,
          avg_dual_solution,
          termination_reason,
          iterations_completed,
          iteration_stats,
        )
      end

      current_iteration_stats.restart_used = run_restart_scheme(
        problem,
        solution_weighted_avg,
        current_primal_solution,
        current_dual_solution,
        last_restart_info,
        iterations_completed,
        primal_norm_params,
        dual_norm_params,
        primal_weight,
        params.verbosity,
        params.restart_params,
      )

      # Update primal_weight
      if current_iteration_stats.restart_used != RESTART_CHOICE_NO_RESTART
        primal_weight = compute_new_primal_weight(
          last_restart_info,
          primal_weight,
          primal_weight_update_smoothing,
          params.verbosity,
        )
      end
    end

    primal_norm_params, dual_norm_params = define_norms(
      params.diagonal_scaling,
      diagonal_objective_matrix,
      row_norm_objective_matrix,
      row_norm_constraint_matrix,
      column_norm_constraint_matrix,
      step_size,
      primal_weight,
    )

    time_spent_doing_basic_algorithm_checkpoint = time()
    # The next lines compute the primal portion of the PDHG algorithm:
    # argmin_x [gradient(f)(current_primal_solution)'x + g(x)
    #          + current_dual_solution' K x
    #          + 0.5*norm_X(x - current_primal_solution)^2]
    # See Sections 2-3 of Chambolle and Pock and the comment above
    # PdhgParameters.
    # This minimization is easy to do in closed form since it can be separated
    # into independent problems for each of the primal variables. The
    # projection onto the primal feasibility set comes from the closed form
    # for the above minimization and the cases where g(x) is infinite - there
    # isn't officially any projection step in the algorithm.

    primal_gradient = compute_primal_gradient(
      problem,
      current_primal_solution,
      current_dual_solution,
    )
    next_primal =
      current_primal_solution .- primal_gradient ./ primal_norm_params
    project_primal!(next_primal, problem)
    # The next two lines compute the dual portion:
    # argmin_y [H*(y) - y' K (2.0*next_primal - current_primal_solution)
    #           + 0.5*norm_Y(y-current_dual_solution)^2]
    dual_gradient = compute_dual_gradient(
      problem,
      2.0 * next_primal - current_primal_solution,
    )
    next_dual = current_dual_solution .+ dual_gradient ./ dual_norm_params
    project_dual!(next_dual, problem)

    delta_primal = next_primal .- current_primal_solution
    delta_dual = next_dual .- current_dual_solution

    movement =
      0.5 *
      weighted_norm(
        delta_primal,
        primal_norm_params - diagonal_objective_matrix,
      )^2 + 0.5 * weighted_norm(delta_dual, dual_norm_params)^2
    if movement == 0.0
      # The algorithm will terminate at the beginning of the next iteration
      numerical_error = true
      continue
    end
    # TODO: compute this using vector ops and cached values for
    # problem.constraint_matrix' * current_dual_solution and
    # problem.constraint_matrix' * next_dual. Thus we only need two
    # matrix-vector products per iteration, not three.
    primal_objective_interaction =
      0.5 * (delta_primal' * problem.objective_matrix * delta_primal) -
      0.5 * weighted_norm(delta_primal, diagonal_objective_matrix)^2
    primal_dual_interaction =
      delta_primal' * problem.constraint_matrix' * delta_dual
    interaction =
      abs(primal_dual_interaction) + abs(primal_objective_interaction)

    # The proof of Theorem 1 requires movement >= interaction.
    required_ratio = interaction / movement
    if params.verbosity >= 6 && print_to_screen_this_iteration(
      false, # termination_reason
      iteration,
      params.verbosity,
      termination_evaluation_frequency,
    )
      pdhg_specific_log(
        problem,
        iteration,
        current_primal_solution,
        current_dual_solution,
        original_primal_norm_params,
        original_dual_norm_params,
        step_size,
        required_ratio,
        primal_weight,
      )
    end
    if required_ratio <= 1
      current_primal_solution = next_primal
      current_dual_solution = next_dual

      weight = step_size
      add_to_solution_weighted_average(
        solution_weighted_avg,
        current_primal_solution,
        current_dual_solution,
        weight,
      )
    end

    if params.adaptive_step_size
      exponent_one = 0.3
      exponent_two = 0.6
      # Our step sizes are a factor
      # 1 - (iteration + 1)^(-exponent_one)/required_ratio
      # smaller than they could be as a margin to reduce rejected steps.
      first_term =
        (1 - (iteration + 1)^(-exponent_one)) / required_ratio * step_size
      second_term = (1 + (iteration + 1)^(-exponent_two)) * step_size
      # From the first term when we have to reject a step, the step_size
      # decreases by a factor of at least 1 - (iteration + 1)^(-exponent_one).
      # From the second term we increase the step_size by a factor of at most
      # 1 + (iteration + 1)^(-exponent_two)
      # Therefore if more than order
      # (iteration + 1)^(exponent_one - exponent_two)
      # fraction of the iterations have a rejected step we overall decrease the
      # step_size. When the step_size is below the inverse of the max singular
      # value we stop having rejected steps.
      step_size = min(first_term, second_term)
    end
    iterations_completed += 1

    time_spent_doing_basic_algorithm +=
      time() - time_spent_doing_basic_algorithm_checkpoint
    cumulative_kkt_passes += KKT_PASSES_PER_ITERATION
  end
end
