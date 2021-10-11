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

"""
A MirrorProxParameters struct specifies the parameters for solving the saddle
point formulation of an problem using mirror prox.

Quadratic Programming Problem (see quadratic_programming.jl):
minimize 1/2 * x' * objective_matrix * x + objective_vector' * x
         + objective_constant

s.t. constraint_matrix[1:num_equalities, :] * x =
     right_hand_side[1:num_equalities]

     constraint_matrix[(num_equalities + 1):end, :] * x >=
     right_hand_side[(num_equalities + 1):end, :]

     variable_lower_bound <= x <= variable_upper_bound

Equivalent saddle point problem:
    min_x max_y phi(x, y)
where
    phi(x, y) = objective_constant + objective_vector' x + right_hand_side' y
                + 1/2 * x' * objective_matrix * x
                - y' constraint_matrix x
    variable_lower_bound <= x <= variable_upper_bound
    y[(num_equalities + 1):end] >= 0

Solve using saddle point mirror prox:
https://blogs.princeton.edu/imabandit/2013/04/23/orf523-mirror-prox/
using mirror map (primal_weight/2) ||x||_X^2 + 1/(2 primal_weight) ||y||_Y^2
where X and Y are diagonal matrices. If `diagonal_scaling`, X[i,i] is set as the
l2 norm of the i-th column of constraint_matrix, and Y[j,j] is set as the l2
norm of the j-th column of constraint_matrix. The intuition behind this setup
is for mirror prox with constant step-size 1, we would like to make sure
[X,-A';-A Y] (A is the constraint_matrix) is positive semi-definite,
which is satisfied automatically for problem with the above X and Y. If not
`diagonal_scaling`, both X and Y are set as identity matrix.

The parameter primal_weight is adjusted dynamically so the two terms contribute
roughly equally.
"""
struct MirrorProxParameters
  """
  Number of L_infinity Ruiz rescaling iterations to apply to the constraint
  matrix. Zero disables this rescaling pass.
  """
  l_inf_ruiz_iterations::Int

  """
  If true, applies L2 norm rescaling after the Ruiz rescaling.
  """
  l2_norm_rescaling::Bool

  """
  If not `nothing`, runs Pock-Chambolle rescaling with the given alpha exponent
  parameter.
  """
  pock_chambolle_alpha::Union{Float64,Nothing}

  """
  Used to bias the initial value of the primal/dual balancing parameter
  primal_weight. Must be positive. See also
  scale_invariant_initial_primal_weight.
  """
  primal_importance::Float64

  """
  If true, computes the initial primal weight with a scale-invariant formula
  biased by primal_importance; see select_initial_primal_weight() for more
  details. If false, primal_importance itself is used as the initial primal
  weight.
  """
  scale_invariant_initial_primal_weight::Bool

  """
  Use weighted norms of rows/columns as the Bregman divergence so that it
  rescales each primal and dual variable. The allowed values are "l1", "l2", and
  "off".
  If `diagonal_scaling`=l1, we set X[i,i]=||K[:,i]||_1 + ||Q[:,i]||_1
  and Y[j,j]=||K[j,:]||_1;
  If `diagonal_scaling`=l2, we set X[i,i]=sqrt(||K[:,i]||_2^2+||Q[:,i]||_2^2)
  and Y[j,j]=||K[j,:]||_2;
  If `diagonal_scaling`=off, we set X[i,i]=1 and Y[j,j]=1;
  """
  diagonal_scaling::String

  """
  If >= 4 a line of debugging info is printed during some iterations. If >= 2
  some info is printed about the final solution.
  """
  verbosity::Int64

  """
  Whether to record an IterationStats object. If false, only iteration stats
  for the final (terminating) iteration are recorded.
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
# TODO: Add validate function to check parameters are correct.

struct MirrorProxVector
  data::Vector{Float64}
  primal_size::Int64
end

mutable struct MirrorProxProblem
  problem::QuadraticProgrammingProblem

  """
  [0 A^T;
   -A 0]
  where A = problem.constraint_matrix
  """
  combo_matrix::SparseMatrixCSC{Float64,Int64}

  params::MirrorProxParameters

  """
  Balances the contribution of primal and dual distances to the mirror map.
  """
  primal_weight::Float64

  """
  The weights of the weighted l2 norm (as the Bregman divergence) used in the
  mirror map.
  """
  mirror_map_scaling::MirrorProxVector
end

# This overloads the "-" operator.
function Base.:-(vec1::MirrorProxVector, vec2::MirrorProxVector)
  @assert vec1.primal_size == vec2.primal_size
  return MirrorProxVector(vec1.data - vec2.data, vec1.primal_size)
end

function Base.:+(vec1::MirrorProxVector, vec2::MirrorProxVector)
  @assert vec1.primal_size == vec2.primal_size
  return MirrorProxVector(vec1.data + vec2.data, vec1.primal_size)
end

function Base.:*(scalar::Float64, vec::MirrorProxVector)
  return MirrorProxVector(scalar * vec.data, vec.primal_size)
end

function Base.:/(vec::MirrorProxVector, scalar::Float64)
  return MirrorProxVector(vec.data / scalar, vec.primal_size)
end

function dot(vec1::MirrorProxVector, vec2::MirrorProxVector)
  @assert vec1.primal_size == vec2.primal_size
  return vec1.data' * vec2.data
end

""" Like .* but that's not entirely trivial to implement for a custom type."""
function pointwise_mult(vec1::MirrorProxVector, vec2::MirrorProxVector)
  @assert vec1.primal_size == vec2.primal_size
  return MirrorProxVector(vec1.data .* vec2.data, vec1.primal_size)
end

""" Like ./ but that's not entirely trivial to implement for a custom type."""
function pointwise_div(vec1::MirrorProxVector, vec2::MirrorProxVector)
  @assert vec1.primal_size == vec2.primal_size
  return MirrorProxVector(vec1.data ./ vec2.data, vec1.primal_size)
end

function make_sp_vec(
  primal::AbstractVector{Float64},
  dual::AbstractVector{Float64},
)
  return MirrorProxVector(vcat(primal, dual), length(primal))
end

function zero_sp_vec(problem::QuadraticProgrammingProblem)
  return make_sp_vec(
    zeros(length(problem.variable_lower_bound)),
    zeros(length(problem.right_hand_side)),
  )
end

function const_sp_vec(
  primal_val::Float64,
  dual_val::Float64,
  problem::QuadraticProgrammingProblem,
)
  return make_sp_vec(
    primal_val * ones(length(problem.variable_lower_bound)),
    dual_val * ones(length(problem.right_hand_side)),
  )
end

function primal_part(vec::MirrorProxVector)
  return view(vec.data, 1:vec.primal_size)
end

function dual_part(vec::MirrorProxVector)
  return view(vec.data, (vec.primal_size+1):length(vec.data))
end

function combo_vec(vec::MirrorProxVector)
  return vec.data
end

function mirror_map(vec::MirrorProxVector, problem::MirrorProxProblem)
  return 0.5 * dot(vec, pointwise_mult(problem.mirror_map_scaling, vec))
end

"""The norm that mirror_map is strongly convex with respect to.
This isn't used but is here for documentation."""
function mirror_map_norm(vec::MirrorProxVector, problem::MirrorProxProblem)
  # This is equivalent to:
  # return sqrt(2 * mirror_map(vec, problem))
  return sqrt(dot(vec, pointwise_mult(problem.mirror_map_scaling, vec)))
end

"""The dual of mirror_map_norm().
This isn't used but is here for documentation."""
function dual_norm(vec::MirrorProxVector, problem::MirrorProxProblem)
  return sqrt(dot(vec, pointwise_div(vec, problem.mirror_map_scaling)))
end

""" Bregman divergence of the mirror map. """
function bregman(
  vec1::MirrorProxVector,
  vec2::MirrorProxVector,
  p::MirrorProxProblem,
)
  # This is specific to this mirror map of course.
  return mirror_map(vec1 - vec2, p)
end

function add_to_solution_weighted_average(
  solution_weighted_average::SolutionWeightedAverage,
  current_solution::MirrorProxVector,
  weight::Float64,
)
  add_to_solution_weighted_average(
    solution_weighted_average,
    primal_part(current_solution),
    dual_part(current_solution),
    weight,
  )
end

""" Returns a 4-element array with entries that sum to the saddle point
objective phi."""
function phi_breakdown(vec::MirrorProxVector, problem::MirrorProxProblem)
  problem = problem.problem
  return [
    problem.objective_constant,
    problem.objective_vector' * primal_part(vec),
    problem.right_hand_side' * dual_part(vec),
    -dual_part(vec)' * problem.constraint_matrix * primal_part(vec),
  ]
end

"""
Returns (gradient of phi with respect to x, -gradient of phi with respect to y).
That's c - A'y, Ax - b.
This is used in place of the subgradient in saddle point mirror prox. It can be
thought of as the concatenation of the loss vectors for the two players of the
zero-sum game.
"""
function pseudo_gradient(solution::MirrorProxVector, p::MirrorProxProblem)
  problem = p.problem
  # TODO: cache this matrix product and reuse it for
  # corrected_dual_obj().
  result = p.combo_matrix' * combo_vec(solution)
  # result is now (-A' y, A x).
  for indx in 1:solution.primal_size
    result[indx] += problem.objective_vector[indx]
  end
  result[1:solution.primal_size] +=
    problem.objective_matrix * primal_part(solution)
  for indx in 1:length(problem.right_hand_side)
    result[solution.primal_size+indx] -= problem.right_hand_side[indx]
  end

  return MirrorProxVector(result, solution.primal_size)
end

function corrected_dual_obj(solution::MirrorProxVector, p::MirrorProxProblem)
  return corrected_dual_obj(
    p.problem,
    primal_part(solution),
    dual_part(solution),
  )
end

"""
Returns the proximal step
argmin_z step_size * gradient' * z + bregman(z, initial_solution)
where z ranges over values satisfying the bounds on the primal and dual
variables.

For the particular mirror map and constraint set we're using this minimization
problem is:
min_z step_size gradient' z + 0.5(z - initial_solution) D (z - initial_solution)
where D is a diagonal matrix with problem.mirror_map_scaling on the diagonal.
This is equivalent (up to additive constant) to
min_z 0.5 (z - initial_solution + step_size D^-1 gradient)' D
(z - initial_solution + step_size D^-1 gradient)
This separates into separate trivial optimization problems for each variable.
"""
function proximal_step(
  p::MirrorProxProblem,
  initial_solution::MirrorProxVector,
  gradient::MirrorProxVector,
  step_size::Float64,
)
  problem = p.problem
  solution =
    initial_solution - step_size * pointwise_div(gradient, p.mirror_map_scaling)
  primal = primal_part(solution)
  dual = dual_part(solution)
  project_primal!(primal, problem)
  project_dual!(dual, problem)
  return solution
end

""" Returns a tuple with stats on how many variables are at the bounds and how
many of the matrix non-zeros would remain after eliminating the variables that
are at their bounds. We say a variable is "active" if it is not at its bounds.
"""
function active_variable_stats(
  solution::MirrorProxVector,
  problem::MirrorProxProblem,
)
  problem = problem.problem
  num_prim_lb = 0
  num_prim_active = 0
  num_prim_ub = 0
  primal = primal_part(solution)
  for idx in 1:length(primal)
    if primal[idx] >= problem.variable_upper_bound[idx]
      num_prim_ub += 1
    else
      if primal[idx] <= problem.variable_lower_bound[idx]
        num_prim_lb += 1
      else
        num_prim_active += 1
      end
    end
  end
  num_dual_lb = 0
  num_dual_active = 0
  dual = dual_part(solution)
  for idx in inequality_range(problem)
    if dual[idx] <= 0
      num_dual_lb += 1
    else
      num_dual_active += 1
    end
  end

  # Number of nonzeros where both the row and column variables are active.
  num_active_nonzeros = 0

  rows = SparseArrays.rowvals(problem.constraint_matrix)
  m, n = size(problem.constraint_matrix)
  for col in 1:n
    if primal[col] == problem.variable_upper_bound[col] ||
       primal[col] == problem.variable_lower_bound[col]
      continue
    end
    for j in SparseArrays.nzrange(problem.constraint_matrix, col)
      row = rows[j]
      if row in equality_range(problem) || dual[row] > 0
        num_active_nonzeros += 1
      end
    end
  end

  return (
    num_prim_lb,
    num_prim_active,
    num_prim_ub,
    num_dual_lb,
    num_dual_active,
    num_active_nonzeros,
  )
end

"""
logs info to STDOUT giving detailed information on the current iteration.
"""
function mirror_prox_specific_log(
  params::MirrorProxParameters,
  p::MirrorProxProblem,
  iteration::Int64,
  accepted_iterations::Int64,
  acceptable_nonlinearity::Bool,
  step_size::Float64,
  inv_stepsize_required::Float64,
  current_solution::MirrorProxVector,
)
  phi_components = phi_breakdown(current_solution, p)
  primal_solution = primal_part(current_solution)
  dual_solution = dual_part(current_solution)
  primal_residual = compute_primal_residual(p.problem, primal_solution)
  Printf.@printf(
    "    (%5d): inv_step_size=%9g %s req'd=%9g ",
    accepted_iterations,
    1.0 / step_size,
    (acceptable_nonlinearity ? ">=" : " <"),
    inv_stepsize_required
  )
  # Provides information on *scaled norms*. Useful for diagnosing poor
  # variable scaling.
  Printf.@printf(
    "l2=(%.2e, %.2e) ",
    norm(primal_solution, 2),
    norm(dual_solution, 2)
  )
  Printf.@printf(
    "l_inf=(%.2e, %.2e)\n",
    norm(primal_solution, Inf),
    norm(dual_solution, Inf)
  )
  Printf.@printf(
    "    primal_weight=%9g phi=%12g=",
    p.primal_weight,
    sum(phi_components)
  )
  println(phi_components)
  if params.verbosity >= 6
    stats = active_variable_stats(current_solution, p)
    Printf.@printf(
      "    primal vars [@lb: %d active: %d @ub: %d] ",
      stats[1],
      stats[2],
      stats[3]
    )
    Printf.@printf(
      "dual eq vars: %d dual ineq vars [@lb: %d active: %d] ",
      length(equality_range(p.problem)),
      stats[4],
      stats[5]
    )
    Printf.@printf(
      "active mat nz: %6.3f%%\n",
      100.0 * stats[6] / SparseArrays.nnz(p.problem.constraint_matrix)
    )
  end
end

function primal_obj(solution::MirrorProxVector, p::MirrorProxProblem)
  problem = p.problem
  return problem.objective_constant +
         problem.objective_vector' * primal_part(solution) +
         0.5 *
         primal_part(solution)' *
         problem.objective_matrix *
         primal_part(solution)
end

"""
Creates a MirrorProxProblem including defining static problem scalings
based on the QuadraticProgrammingProblem.
"""
function initialize_saddle_point_problem(
  problem::QuadraticProgrammingProblem,
  params::MirrorProxParameters,
  combo_mat::SparseMatrixCSC{Float64,Int64},
)
  # With mirror map (primal_weight/2) ||x||_X^2 + 1/(2 primal_weight) ||y||_Y^2
  # the theory holds when
  # [primal_weight X, 0, -Q, -A'
  #  0, Y / primal_weight, A, 0
  #  -Q, A', primal_weight X, 0
  #  -A, 0, 0, Y / primal_weight]
  # is positive semi-definte. This is automatically satisfied when we set
  # X[i,i]=||K[:,i]||_1 + ||Q[:,i]||_1 and Y[j,j]=||K[j,:]||_1. In practice,
  # using ||.||_2 instead of ||.||_1 to define X and Y often improves
  # the convergence speed for mirror prox.
  if params.diagonal_scaling == "l2"
    # If `diagonal_scaling`=l2, we set
    # X[i,i]=sqrt(||K[:,i]||_2^2+||Q[:,i]||_2^2) and Y[j,j]=||K[j,:]||_2.
    primal_scalings =
      sqrt.(
        sum(problem.constraint_matrix .^ 2, dims = 1) +
        sum(problem.objective_matrix .^ 2, dims = 1),
      )
    dual_scalings = sqrt.(sum(problem.constraint_matrix .^ 2, dims = 2))
    p = MirrorProxProblem(
      problem,
      combo_mat,
      params,
      params.primal_importance,
      make_sp_vec(vec(primal_scalings), vec(dual_scalings)),
    )
  elseif params.diagonal_scaling == "l1"
    # If `diagonal_scaling`=l1, we set X[i,i]=||K[:,i]||_1+||Q[:,i]||_1 and
    # Y[j,j]=||K[j,:]||_1.
    primal_scalings = vec(
      sum(abs.(problem.constraint_matrix), dims = 1) +
      sum(abs.(problem.objective_matrix), dims = 1),
    )
    dual_scalings = vec(sum(abs.(problem.constraint_matrix), dims = 2))
    p = MirrorProxProblem(
      problem,
      combo_mat,
      params,
      1.0, # primal_weight
      make_sp_vec(vec(primal_scalings), vec(dual_scalings)),
    )
  else
    # If `diagonal_scaling`=off, we set X[i,i]=1 and Y[j,j]=1.
    p = MirrorProxProblem(
      problem,
      combo_mat,
      params,
      1.0, # primal_weight
      const_sp_vec(
        params.primal_importance,
        1.0 / params.primal_importance,
        problem,
      ),
    )
  end

  return p
end

"""
Change the norms that we use to measure the primal and dual variables to reflect
the change in the primal weight
"""
function update_mirror_prox_norms(p::MirrorProxProblem, primal_weight::Float64)
  if p.primal_weight != primal_weight
    problem = p.problem
    primal_size = length(problem.variable_lower_bound)
    dual_size = length(problem.right_hand_side)
    primal_weight_ratio = primal_weight / p.primal_weight
    p.primal_weight = primal_weight

    for i in 1:primal_size
      primal_part(p.mirror_map_scaling)[i] *= primal_weight_ratio
    end
    for i in 1:dual_size
      dual_part(p.mirror_map_scaling)[i] /= primal_weight_ratio
    end
  end
end

"""
Logging while the algorithm is running.
"""
function mirror_prox_display(
  termination_reason::Union{TerminationReason,Bool},
  iteration::Int64,
  params::MirrorProxParameters,
  current_iteration_stats::IterationStats,
  p::MirrorProxProblem,
  accepted_iterations::Int64,
  acceptable_nonlinearity::Bool,
  step_size::Float64,
  inv_stepsize_required::Float64,
  current_solution::MirrorProxVector,
)
  if print_to_screen_this_iteration(
    termination_reason,
    iteration,
    params.verbosity,
    params.termination_evaluation_frequency,
  )
    display_iteration_stats(current_iteration_stats, params.verbosity)
    if iteration > 1 && params.verbosity >= 6
      mirror_prox_specific_log(
        params,
        p,
        iteration,
        accepted_iterations,
        acceptable_nonlinearity,
        step_size,
        inv_stepsize_required,
        current_solution,
      )
    end
  end
end

"""
`optimize(params::MirrorProxParameters,
          original_problem::QuadraticProgrammingProblem)`

Solves a quadratic program using saddle point mirror prox.

# Arguments
- `params::MirrorProxParameters`: parameters.
- `original_problem::QuadraticProgrammingProblem`: the QP to solve.

# Returns
A SaddlePointOutput struct containing the solution found.
"""
function optimize(
  params::MirrorProxParameters,
  original_problem::QuadraticProgrammingProblem,
)
  validate(original_problem)
  # TODO: Split this code into functions to make easier to read.
  qp_cache = cached_quadratic_program_info(original_problem)
  scaled_problem = rescale_problem(
    params.l_inf_ruiz_iterations,
    params.l2_norm_rescaling,
    params.pock_chambolle_alpha,
    params.verbosity,
    original_problem,
  )
  problem = scaled_problem.scaled_qp

  primal_size = length(problem.variable_lower_bound)
  dual_size = length(problem.right_hand_side)
  combo_mat::SparseMatrixCSC{Float64,Int64} = [
    SparseArrays.spzeros(primal_size, primal_size) problem.constraint_matrix'
    (-problem.constraint_matrix) SparseArrays.spzeros(dual_size, dual_size)
  ]
  validate_termination_criteria(params.termination_criteria)
  current_solution = zero_sp_vec(problem)
  # The theory suggests using the inverse of the max singular value as the step
  # size. We could approximate the max singular value using power iteration but
  # that is not only slower but produces step sizes that are too conservative in
  # practice. The step size adjustment algorithm we use decreases the step
  # size faster than it increases it so it's better to err on the side of a
  # bigger initial step size. Instead we use the max absolute value entry in a
  # matrix, which is a lower bound on the max singular value. This seems to work
  # well, although I usually test with Ruiz rescaling pre-processing so this is
  # effectively just initializing the step size to 1.
  step_size = 1.0 / norm(problem.constraint_matrix, Inf)
  solution_weighted_avg = initialize_solution_weighted_average(
    length(primal_part(current_solution)),
    length(dual_part(current_solution)),
  )
  memory_solution = current_solution
  best_dual = -Inf
  accepted_iterations = 0
  iterations_completed = 0
  acceptable_nonlinearity = false
  inv_stepsize_required = NaN
  KKT_PASSES_PER_ITERATION = 2.0

  # Idealized number of KKT passes each time the termination criteria and
  # restart scheme is run. One of these comes from evaluating the gradient at
  # the average solution and evaluating the gradient at the current solution.
  # The number in the current implementation is four.
  KKT_PASSES_PER_TERMINATION_EVALUATION = 2.0

  p = initialize_saddle_point_problem(problem, params, combo_mat)

  if params.scale_invariant_initial_primal_weight
    primal_weight = select_initial_primal_weight(
      problem,
      primal_part(p.mirror_map_scaling),
      dual_part(p.mirror_map_scaling),
      params.primal_importance,
      params.verbosity,
    )
  else
    primal_weight = params.primal_importance
  end
  update_mirror_prox_norms(p, primal_weight)

  primal_weight_update_smoothing =
    params.restart_params.primal_weight_update_smoothing

  iteration_stats = IterationStats[]
  start_time = time()
  # Basic algorithm refers to the primal and dual steps, and excludes restart
  # schemes and termination evaluation.
  time_spent_doing_basic_algorithm = 0.0

  cumulative_kkt_passes = 0.0

  # This variable is used in the adaptive restart scheme.
  last_restart_info = create_last_restart_info(
    problem,
    primal_part(current_solution),
    dual_part(current_solution),
  ) # TODO: Push running average inside of this, and maybe a
  # counter on number of iterations since a primal weight update (this could be
  # used to give more fined grain control on primal weight update frequency).

  # For termination criteria:
  termination_criteria = params.termination_criteria
  iteration_limit = params.termination_criteria.iteration_limit
  termination_evaluation_frequency = params.termination_evaluation_frequency

  # This flag represents whether a numerical error occurred during the algorithm
  # if it is set to true it will trigger the algorithm to terminate.
  numerical_error = false

  display_iteration_stats_heading(params.verbosity)

  iteration = 0
  while true
    iteration += 1

    current_gradient = pseudo_gradient(current_solution, p)

    # Evaluate the iteration stats at frequency
    # termination_evaluation_frequency, when the iteration_limit is reached,
    # or if a numerical error occurs at the previous iteration.
    if mod(iteration - 1, termination_evaluation_frequency) == 0 ||
       iteration == iteration_limit + 1 ||
       iteration <= 10 ||
       numerical_error
      cumulative_kkt_passes += KKT_PASSES_PER_TERMINATION_EVALUATION

      # Compute the average solution since the last restart point.
      if numerical_error ||
         solution_weighted_avg.sum_primal_solutions_count == 0 ||
         solution_weighted_avg.sum_dual_solutions_count == 0
        avg_primal_solution = primal_part(current_solution)
        avg_dual_solution = dual_part(current_solution)
      else
        avg_primal_solution, avg_dual_solution =
          compute_average(solution_weighted_avg)
      end
      avg_solution = make_sp_vec(avg_primal_solution, avg_dual_solution)

      # Evalute the iteration stats.
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
        p.primal_weight,
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
        primal_part(p.mirror_map_scaling),
        dual_part(p.mirror_map_scaling),
      )

      # Check the termination criteria.
      termination_reason = check_termination_criteria(
        termination_criteria,
        qp_cache,
        current_iteration_stats,
      )
      if numerical_error && termination_reason == false
        termination_reason = TERMINATION_REASON_NUMERICAL_ERROR
      end

      # If we're terminating, record the iteration stats to provide final
      # solution stats.
      if params.record_iteration_stats || termination_reason != false
        push!(iteration_stats, current_iteration_stats)
      end

      mirror_prox_display(
        termination_reason,
        iteration,
        params,
        current_iteration_stats,
        p,
        accepted_iterations,
        acceptable_nonlinearity,
        step_size,
        inv_stepsize_required,
        avg_solution,
      )

      if termination_reason != false
        # ** Terminate the algorithm **
        # This is the only place the algorithm can terminate. Please keep it
        # this way.
        generic_final_log(
          problem,
          primal_part(avg_solution),
          dual_part(avg_solution),
          current_iteration_stats,
          params.verbosity,
          iteration,
          termination_reason,
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
        primal_part(current_solution),
        dual_part(current_solution),
        last_restart_info,
        iterations_completed,
        primal_part(p.mirror_map_scaling),
        dual_part(p.mirror_map_scaling),
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
        update_mirror_prox_norms(p, primal_weight)
      end
    end

    time_spent_doing_basic_algorithm_checkpoint = time()
    current_gradient = pseudo_gradient(current_solution, p)
    # This line can be changed arbitrarily. The analysis will still work except
    # that if the guesses aren't good the steps may be rejected even with a
    # small step size. See Lemma 1 in
    # http://papers.nips.cc/paper/5147-optimization-learning-and-games-with-predictable-sequences.pdf
    guess_gradient = current_gradient
    test_point = proximal_step(p, current_solution, guess_gradient, step_size)
    test_gradient = pseudo_gradient(test_point, p)
    candidate_solution =
      proximal_step(p, current_solution, test_gradient, step_size)

    nonlinearity =
      dot(test_gradient - guess_gradient, test_point - candidate_solution)
    movement =
      bregman(candidate_solution, test_point, p) +
      bregman(test_point, current_solution, p)
    # If we didn't move we must have the problem solved to working precision.
    # Stop now to avoid problems such as divide by zero.
    if movement == 0
      # The algorithm will terminate at the beginning of the next iteration
      numerical_error = true
      continue
    end
    # This must be <= 1/step_size for the mirror prox proof to work.
    # It is at most the max singular value of the constraint matrix.
    inv_stepsize_required = nonlinearity / movement
    acceptable_nonlinearity::Bool = inv_stepsize_required <= 1.0 / step_size

    if acceptable_nonlinearity
      current_solution = candidate_solution

      # Use weighted average, if this was 1.0 it would be standard average.
      weight = step_size
      add_to_solution_weighted_average(
        solution_weighted_avg,
        test_point,
        weight,
      )
      accepted_iterations += 1
    end
    exponent_one = 0.3
    exponent_two = 0.6
    # Our step sizes are a factor 1 - iteration^(-exponent_one) smaller than
    # they could be as margin to reduce rejected steps.
    first_term =
      (1 - (iteration + 1)^(-exponent_one)) / abs(inv_stepsize_required)
    second_term = (1 + (iteration + 1)^(-exponent_two)) * step_size
    # From the first term when we have to reject a step the step_size decreases
    # by a factor of at least 1 - iteration^(-exponent_one).
    # From the second term we increase the step_size by a factor of at most
    # 1 + iteration^(-exponent_two).
    # Therefore if more than order iteration^(exponent_one - exponent_two)
    # fraction of the iterations have a rejected step we overall decrease the
    # step_size. When the step_size is below the inverse of the max singular
    # value we stop having rejected steps.
    step_size = min(first_term, second_term)
    iterations_completed += 1

    time_spent_doing_basic_algorithm +=
      time() - time_spent_doing_basic_algorithm_checkpoint

    cumulative_kkt_passes += KKT_PASSES_PER_ITERATION
  end
end
