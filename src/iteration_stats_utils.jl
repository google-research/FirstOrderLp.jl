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

# Utilities for computing and printing IterationStats.

function max_primal_violation(
  problem::QuadraticProgrammingProblem,
  primal_vec::AbstractVector{Float64},
)
  return norm(compute_primal_residual(problem, primal_vec), Inf)
end

"""
Given quadratic program and a primal solution vector as input this function
computes a vector containing the individual violations of each of the primal
constraints (both equalities and inequalities), variable lower bounds and upper
bounds respectively.
"""
function compute_primal_residual(
  problem::QuadraticProgrammingProblem,
  primal_vec::AbstractVector{Float64},
)
  activities = problem.constraint_matrix * primal_vec

  if isempty(equality_range(problem))
    equality_violation = []
  else
    equality_violation =
      problem.right_hand_side[equality_range(problem)] -
      activities[equality_range(problem)]
  end

  if isempty(inequality_range(problem))
    inequality_violation = []
  else
    inequality_violation =
      max.(
        problem.right_hand_side[inequality_range(problem)] -
        activities[inequality_range(problem)],
        0.0,
      )
  end

  lower_bound_violation = max.(problem.variable_lower_bound - primal_vec, 0.0)
  upper_bound_violation = max.(primal_vec - problem.variable_upper_bound, 0.0)
  return [
    equality_violation
    inequality_violation
    lower_bound_violation
    upper_bound_violation
  ]
end

""" Computes the primal objective.
"""
function primal_obj(
  problem::QuadraticProgrammingProblem,
  primal_solution::AbstractVector{Float64},
)
  return problem.objective_constant +
         problem.objective_vector' * primal_solution +
         0.5 * (primal_solution' * problem.objective_matrix * primal_solution)
end



struct DualStats
  dual_objective::Float64
  dual_residual::Vector{Float64}
  reduced_costs::Vector{Float64}
end

"""
  reduced_costs_dual_objective_contribution(
    variable_lower_bound,
    variable_upper_bound,
    reduced_costs
  )
This function returns the contribution of the
reduced costs to the dual objective value.
"""
function reduced_costs_dual_objective_contribution(
  variable_lower_bound::Vector{Float64},
  variable_upper_bound::Vector{Float64},
  reduced_costs::Vector{Float64},
)
  dual_objective_contribution = 0.0
  for i in 1:length(variable_lower_bound)
    if reduced_costs[i] == 0.0
      continue
    elseif reduced_costs[i] > 0.0
      # A positive reduced cost is associated with a binding lower bound.
      bound_value = variable_lower_bound[i]
    else
      # A negative reduced cost is associated with a binding upper bound.
      bound_value = variable_upper_bound[i]
    end
    if !isfinite(bound_value)
      return -Inf
    else
      dual_objective_contribution += bound_value * reduced_costs[i]
    end
  end

  return dual_objective_contribution
end

"""
  compute_reduced_costs_from_primal_gradient(
    variable_lower_bound,
    variable_upper_bound,
    primal_gradient
  )
where primal_gradient is the gradient of the Lagrangian with respect to the
primal vector. This function returns a vector of reduced costs.
"""
function compute_reduced_costs_from_primal_gradient(
  variable_lower_bound::Vector{Float64},
  variable_upper_bound::Vector{Float64},
  primal_gradient::AbstractVector{Float64},
)
  primal_size = length(primal_gradient)
  reduced_costs = zeros(primal_size)
  for i in 1:primal_size
    if primal_gradient[i] > 0.0
      bound_value = variable_lower_bound[i]
    else
      bound_value = variable_upper_bound[i]
    end
    if isfinite(bound_value)
      reduced_costs[i] = primal_gradient[i]
    end
  end

  return reduced_costs
end

"""
Given a dual vector (i.e., multipliers on the linear constraints) and a primal
vector, returns a DualStats object.
The objective function is linearized about the given
primal solution. Positive values of the dual variables correspond to the lower
bound on the corresponding constraint activities. Reduced costs are computed
from scratch as implied by the primal and dual vectors.
"""
function compute_dual_stats(
  problem::QuadraticProgrammingProblem,
  primal_solution::AbstractVector{Float64},
  dual_solution::AbstractVector{Float64},
)
  objective_product = problem.objective_matrix * primal_solution
  primal_gradient =
    compute_primal_gradient(problem, primal_solution, dual_solution)
  reduced_costs = compute_reduced_costs_from_primal_gradient(
    problem.variable_lower_bound,
    problem.variable_upper_bound,
    primal_gradient,
  )

  # Duals on the inequalities must be nonnegative.
  if !isempty(inequality_range(problem))
    dual_residual = max.(-dual_solution[inequality_range(problem)], 0.0)
  else
    dual_residual = []
  end
  reduced_cost_violations = primal_gradient .- reduced_costs
  dual_residual = [dual_residual; reduced_cost_violations]

  # The dual objective excluding the reduced costs.
  # The quadratic term 0.5 x^T Q x in the objective is linearized as
  #   0.5 x^T Q x >= 0.5 x_0^T Q x_0 + x_0^T Q (x - x_0).
  # The -0.5 in dual_obj's initializer comes from the two x_0^T Q x_0 terms in
  # the above.
  base_dual_objective =
    problem.right_hand_side' * dual_solution + problem.objective_constant -
    0.5 * objective_product' * primal_solution

  dual_objective =
    base_dual_objective + reduced_costs_dual_objective_contribution(
      problem.variable_lower_bound,
      problem.variable_upper_bound,
      reduced_costs,
    )

  return DualStats(dual_objective, dual_residual, reduced_costs)
end

""" Computes the dual objective. It uses the bounds on the primal variables to
correct for any infeasibility in the dual, producing a valid lower bound.
Returns -Inf if not all variables have finite bounds.
"""
function corrected_dual_obj(
  problem::QuadraticProgrammingProblem,
  dual_stats::DualStats,
)
  if norm(dual_stats.dual_residual, Inf) == 0.0
    return dual_stats.dual_objective
  else
    return -Inf
  end
end

function corrected_dual_obj(
  problem::QuadraticProgrammingProblem,
  primal_solution::AbstractVector{Float64},
  dual_solution::AbstractVector{Float64},
)
  dual_stats = compute_dual_stats(problem, primal_solution, dual_solution)
  return corrected_dual_obj(problem, dual_stats)
end


"""
Returns a `ConvergenceInformation` object given a `QuadraticProgrammingProblem`
with given primal and dual vectors.
"""
function compute_convergence_information(
  problem::QuadraticProgrammingProblem,
  qp_cache::CachedQuadraticProgramInfo,
  primal_iterate::Vector{Float64},
  dual_iterate::Vector{Float64},
  eps_ratio::Float64,
  candidate_type::PointType,
)
  num_constraints, num_vars = size(problem.constraint_matrix)
  @assert length(primal_iterate) == num_vars
  @assert length(dual_iterate) == num_constraints

  convergence_info = ConvergenceInformation()

  primal_residual = compute_primal_residual(problem, primal_iterate)
  convergence_info.primal_objective = primal_obj(problem, primal_iterate)
  convergence_info.l_inf_primal_residual = norm(primal_residual, Inf)
  convergence_info.l2_primal_residual = norm(primal_residual, 2)
  convergence_info.relative_l_inf_primal_residual =
    convergence_info.l_inf_primal_residual /
    (eps_ratio + qp_cache.l_inf_norm_primal_right_hand_side)
  convergence_info.relative_l2_primal_residual =
    convergence_info.l2_primal_residual /
    (eps_ratio + qp_cache.l2_norm_primal_right_hand_side)
  convergence_info.l_inf_primal_variable = norm(primal_iterate, Inf)
  convergence_info.l2_primal_variable = norm(primal_iterate, 2)

  dual_stats = compute_dual_stats(problem, primal_iterate, dual_iterate)
  convergence_info.dual_objective = dual_stats.dual_objective
  convergence_info.l_inf_dual_residual = norm(dual_stats.dual_residual, Inf)
  convergence_info.l2_dual_residual = norm(dual_stats.dual_residual, 2)
  convergence_info.relative_l_inf_dual_residual =
    convergence_info.l_inf_dual_residual /
    (eps_ratio + qp_cache.l_inf_norm_primal_linear_objective)
  convergence_info.relative_l2_dual_residual =
    convergence_info.l2_dual_residual /
    (eps_ratio + qp_cache.l2_norm_primal_linear_objective)
  convergence_info.l_inf_dual_variable = norm(dual_iterate, Inf)
  convergence_info.l2_dual_variable = norm(dual_iterate, 2)

  convergence_info.corrected_dual_objective =
    corrected_dual_obj(problem, dual_stats)

  gap = abs(convergence_info.primal_objective - convergence_info.dual_objective)
  abs_obj =
    abs(convergence_info.primal_objective) +
    abs(convergence_info.dual_objective)
  convergence_info.relative_optimality_gap = gap / (eps_ratio + abs_obj)

  convergence_info.candidate_type = candidate_type

  return convergence_info
end

"""
Returns an `InfeasibilityInformation` object given a
`QuadraticProgrammingProblem` and estimates for the primal and dual rays.
The rays do not need to be pre-scaled to have Inf-norm equal to 1.0.
"""
function compute_infeasibility_information(
  problem::QuadraticProgrammingProblem,
  primal_ray_estimate::Vector{Float64},
  dual_ray_estimate::Vector{Float64},
  candidate_type::PointType,
)

  infeas_info = InfeasibilityInformation()

  primal_ray_inf_norm = norm(primal_ray_estimate, Inf)
  if !iszero(primal_ray_inf_norm)
    primal_ray_estimate /= primal_ray_inf_norm
  end

  homogeneous_primal = linear_programming_problem(
    [isfinite(l) ? 0.0 : -Inf for l in problem.variable_lower_bound],
    [isfinite(u) ? 0.0 : Inf for u in problem.variable_upper_bound],
    problem.objective_vector,
    0.0,  # objective_constant
    problem.constraint_matrix,
    zeros(length(problem.right_hand_side)),
    problem.num_equalities,
  )

  homogeneous_residual =
    compute_primal_residual(homogeneous_primal, primal_ray_estimate)
  infeas_info.max_primal_ray_infeasibility = norm(homogeneous_residual, Inf)
  infeas_info.primal_ray_linear_objective =
    problem.objective_vector' * primal_ray_estimate
  infeas_info.primal_ray_quadratic_norm =
    norm(problem.objective_matrix * primal_ray_estimate, Inf)

  homogeneous_dual = linear_programming_problem(
    problem.variable_lower_bound,
    problem.variable_upper_bound,
    zeros(length(problem.objective_vector)),
    0.0,  # objective_constant
    problem.constraint_matrix,
    problem.right_hand_side,
    problem.num_equalities,
  )

  homogeneous_dual_stats =
    compute_dual_stats(homogeneous_dual, primal_ray_estimate, dual_ray_estimate)

  scaling_factor = max(
    norm(dual_ray_estimate, Inf),
    norm(homogeneous_dual_stats.reduced_costs, Inf),
  )
  if !iszero(scaling_factor)
    infeas_info.max_dual_ray_infeasibility =
      norm(homogeneous_dual_stats.dual_residual, Inf) / scaling_factor
    infeas_info.dual_ray_objective =
      homogeneous_dual_stats.dual_objective / scaling_factor
  else
    infeas_info.max_dual_ray_infeasibility = 0.0
    infeas_info.dual_ray_objective = 0.0
  end

  infeas_info.candidate_type = candidate_type

  return infeas_info
end

"""
Returns a `IterationStats` object given a `QuadraticProgrammingProblem`,
primal and dual vectors, estimates for primal and dual rays, and other
iteration statistics corresponding to the fields in the struct.
"""
function compute_iteration_stats(
  problem::QuadraticProgrammingProblem,
  qp_cache::CachedQuadraticProgramInfo,
  primal_iterate::Vector{Float64},
  dual_iterate::Vector{Float64},
  primal_ray_estimate::Vector{Float64},
  dual_ray_estimate::Vector{Float64},
  iteration_number::Integer,
  cumulative_kkt_matrix_passes::Float64,
  cumulative_time_sec::Float64,
  eps_optimal_absolute::Float64,
  eps_optimal_relative::Float64,
  step_size::Float64,
  primal_weight::Float64,
  candidate_type::PointType,
)
  num_constraints, num_vars = size(problem.constraint_matrix)
  @assert length(primal_iterate) == num_vars
  @assert length(primal_ray_estimate) == num_vars
  @assert length(dual_iterate) == num_constraints
  @assert length(dual_ray_estimate) == num_constraints

  stats = IterationStats()
  stats.iteration_number = iteration_number
  stats.cumulative_kkt_matrix_passes = cumulative_kkt_matrix_passes
  stats.cumulative_time_sec = cumulative_time_sec

  stats.convergence_information = [
    compute_convergence_information(
      problem,
      qp_cache,
      primal_iterate,
      dual_iterate,
      eps_optimal_absolute / eps_optimal_relative,
      candidate_type,
    ),
  ]
  stats.infeasibility_information = [
    compute_infeasibility_information(
      problem,
      primal_ray_estimate,
      dual_ray_estimate,
      candidate_type,
    ),
  ]
  stats.step_size = step_size
  stats.primal_weight = primal_weight
  stats.method_specific_stats = Dict{AbstractString,Float64}()

  return stats
end

"""
This code computes the unscaled iteration stats.
The input iterates to this function have been scaled according to
scaled_problem.
"""
function evaluate_unscaled_iteration_stats(
  scaled_problem::ScaledQpProblem,
  qp_cache::CachedQuadraticProgramInfo,
  termination_criteria::TerminationCriteria,
  record_iteration_stats::Bool,
  primal_solution::AbstractVector{Float64},
  dual_solution::AbstractVector{Float64},
  iteration::Int64,
  cumulative_time::Float64,
  cumulative_kkt_passes::Float64,
  eps_optimal_absolute::Float64,
  eps_optimal_relative::Float64,
  step_size::Float64,
  primal_weight::Float64,
  candidate_type::PointType,
)
  # Unscale iterates.
  original_primal_solution::Vector{Float64} =
    primal_solution ./ scaled_problem.variable_rescaling
  original_dual_solution::Vector{Float64} =
    dual_solution ./ scaled_problem.constraint_rescaling

  return compute_iteration_stats(
    scaled_problem.original_qp,
    qp_cache,
    original_primal_solution,
    original_dual_solution,
    original_primal_solution,  # ray estimate
    original_dual_solution,  # ray estimate
    iteration - 1,
    cumulative_kkt_passes,
    cumulative_time,
    eps_optimal_absolute,
    eps_optimal_relative,
    step_size,
    primal_weight,
    candidate_type,
  )
end

"""
   print_to_screen_this_iteration(termination_reason, iteration, verbosity,
      termination_evaluation_frequency)

Decides if we should print iteration stats to screen this iteration.
"""
function print_to_screen_this_iteration(
  termination_reason::Union{TerminationReason,Bool},
  iteration::Int64,
  verbosity::Int64,
  termination_evaluation_frequency::Int32,
)
  if verbosity >= 2
    if termination_reason == false
      num_of_evaluations = (iteration - 1) / termination_evaluation_frequency
      if verbosity >= 9
        display_frequency = 1
      elseif verbosity >= 6
        display_frequency = 3
      elseif verbosity >= 5
        display_frequency = 10
      elseif verbosity >= 4
        display_frequency = 20
      elseif verbosity >= 3
        display_frequency = 50
      else
        return iteration == 1
      end
      # print_to_screen_this_iteration is true every
      # display_frequency * termination_evaluation_frequency iterations.
      return mod(num_of_evaluations, display_frequency) == 0
    else
      return true
    end
  else
    return false
  end
end

"""
  display_iteration_stats_heading(show_infeasibility)

The heading for the iteration stats table. If show_infeasibility is true then
an extended table is printed that includes infeasibility information. See
README.md for documentation on what each heading means.
"""
function display_iteration_stats_heading(show_infeasibility::Bool)
  Printf.@printf(
    "%s | %s | %s | %s |",
    rpad("runtime", 24),
    rpad("residuals", 26),
    rpad(" solution information", 26),
    rpad("relative residuals", 23)
  )
  if show_infeasibility
    Printf.@printf(" %s | %s |", rpad("primal ray", 27), rpad("dual ray", 18))
  end
  println("")
  Printf.@printf(
    "%s %s %s | %s %s  %s | %s %s %s | %s %s %s |",
    rpad("#iter", 7),
    rpad("#kkt", 8),
    rpad("seconds", 7),
    rpad("pr norm", 8),
    rpad("du norm", 8),
    rpad("gap", 7),
    rpad(" pr obj", 9),
    rpad("pr norm", 8),
    rpad("du norm", 7),
    rpad("rel pr", 7),
    rpad("rel du", 7),
    rpad("rel gap", 7)
  )
  if show_infeasibility
    Printf.@printf(
      " %s %s %s | %s %s |",
      rpad("pr norm", 9),
      rpad("linear", 8),
      rpad("qu norm", 8),
      rpad("du norm", 9),
      rpad("dual obj", 8)
    )
  end
  print("\n")
end


function display_iteration_stats_heading(verbosity::Int64)
  if verbosity >= 7
    display_iteration_stats_heading(true)
  elseif verbosity >= 2
    display_iteration_stats_heading(false)
  end
end

"""
Make sure that a float is of a constant length, irrespective if it is negative
or positive.
"""
function lpad_float(number::Float64)
  return lpad(Printf.@sprintf("%.1e", number), 8)
end

"""
Displays a row of the iteration stats table.
"""
function display_iteration_stats(
  stats::IterationStats,
  show_infeasibility::Bool,
)
  # TODO: Decide if we want to show information about all the
  # convergence_information entries.
  if length(stats.convergence_information) > 0
    Printf.@printf(
      "%s  %.1e  %.1e | %.1e  %.1e  %s | %s  %.1e  %.1e | %.1e %.1e %.1e |",
      rpad(string(stats.iteration_number), 6),
      stats.cumulative_kkt_matrix_passes,
      stats.cumulative_time_sec,
      stats.convergence_information[1].l2_primal_residual,
      stats.convergence_information[1].l2_dual_residual,
      lpad_float(
        stats.convergence_information[1].primal_objective -
        stats.convergence_information[1].dual_objective,
      ),
      lpad_float(stats.convergence_information[1].primal_objective),
      stats.convergence_information[1].l2_primal_variable,
      stats.convergence_information[1].l2_dual_variable,
      stats.convergence_information[1].relative_l2_primal_residual,
      stats.convergence_information[1].relative_l2_dual_residual,
      stats.convergence_information[1].relative_optimality_gap
    )
  else
    Printf.@printf(
      "%s  %.1e  %.1e",
      rpad(string(stats.iteration_number), 6),
      stats.cumulative_kkt_matrix_passes,
      stats.cumulative_time_sec
    )
  end

  if show_infeasibility
    # TODO: Decide if we want to show information about all the
    # entries.
    if length(stats.infeasibility_information) > 0
      Printf.@printf(
        " %.1e  %s  %.1e  | %.1e  %s  |",
        stats.infeasibility_information[1].max_primal_ray_infeasibility,
        lpad_float(
          stats.infeasibility_information[1].primal_ray_linear_objective,
        ),
        stats.infeasibility_information[1].primal_ray_quadratic_norm,
        stats.infeasibility_information[1].max_dual_ray_infeasibility,
        lpad_float(stats.infeasibility_information[1].dual_ray_objective)
      )
    end
  end

  print("\n")
end

function display_iteration_stats(stats::IterationStats, verbosity::Int64)
  if verbosity >= 7
    display_iteration_stats(stats, true)
  else
    display_iteration_stats(stats, false)
  end
end

function print_infinity_norms(convergence_info::ConvergenceInformation)
  print("l_inf: ")
  Printf.@printf(
    "primal_res = %.3e, dual_res = %.3e, primal_var = %.3e, dual_var = %.3e",
    convergence_info.l_inf_primal_residual,
    convergence_info.l_inf_dual_residual,
    convergence_info.l_inf_primal_variable,
    convergence_info.l_inf_dual_variable
  )
  println()
end
