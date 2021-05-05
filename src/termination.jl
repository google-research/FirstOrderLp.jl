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

@enum OptimalityNorm L_INF L2

# Relevant readings on infeasibility certificates:
# (1) https://docs.mosek.com/modeling-cookbook/qcqo.html provides references
# explaining why the primal rays imply dual infeasibility and dual rays imply
# primal infeasibility.
# (2) The termination criteria for Mosek's linear programming optimizer
# https://docs.mosek.com/9.0/pythonfusion/solving-linear.html.
# (3) The termination criteria for OSQP is in section 3.3 of
# https://web.stanford.edu/~boyd/papers/pdf/osqp.pdf.
# (4) The termination criteria for SCS is in section 3.5 of
# https://arxiv.org/pdf/1312.3039.pdf.

"A description of solver termination criteria."
mutable struct TerminationCriteria
  "The norm that we are measuring the optimality criteria in."
  optimality_norm::OptimalityNorm

  # Let p correspond to the norm we are using as specified by optimality_norm.
  # If the algorithm terminates with termination_reason =
  # TERMINATION_REASON_OPTIMAL then the following hold:
  # | primal_objective - dual_objective | <= eps_optimal_absolute +
  #  eps_optimal_relative * ( | primal_objective | + | dual_objective | )
  # norm(primal_residual, p) <= eps_optimal_absolute + eps_optimal_relative *
  #  norm(right_hand_side^c, p)
  # norm(dual_residual, p) <= eps_optimal_absolute + eps_optimal_relative *
  #   norm(objective_vector, p)
  # It is possible to prove that a solution satisfying the above conditions
  # also satisfies SCS's optimality conditions (see link above) with ϵ_pri =
  # ϵ_dual = ϵ_gap = eps_optimal_absolute = eps_optimal_relative. (ϵ_pri,
  # ϵ_dual, and ϵ_gap are SCS's parameters).

  """
  Absolute tolerance on the duality gap, primal feasibility, and dual
  feasibility.
  """
  eps_optimal_absolute::Float64

  """
  Relative tolerance on the duality gap, primal feasibility, and dual
  feasibility.
  """
  eps_optimal_relative::Float64

  """
  If the following two conditions hold we say that we have obtained an
  approximate dual ray, which is an approximate certificate of primal
  infeasibility.
    (1) dual_ray_objective > 0.0,
    (2) max_dual_ray_infeasibility / dual_ray_objective <=
        eps_primal_infeasible.
  """
  eps_primal_infeasible::Float64

  """
  If the following three conditions hold we say we have obtained an
  approximate primal ray, which is an approximate certificate of dual
  infeasibility.
    (1) primal_ray_linear_objective < 0.0,
    (2) max_primal_ray_infeasibility / (-primal_ray_linear_objective) <=
        eps_dual_infeasible,
    (3) primal_ray_quadratic_norm / (-primal_ray_linear_objective) <=
        eps_dual_infeasible.
  """
  eps_dual_infeasible::Float64

  """
  If termination_reason = TERMINATION_REASON_TIME_LIMIT then the solver has
  taken at least time_sec_limit time.
  """
  time_sec_limit::Float64

  """
  If termination_reason = TERMINATION_REASON_ITERATION_LIMIT then the solver has
  taken at least iterations_limit iterations.
  """
  iteration_limit::Int32

  """
  If termination_reason = TERMINATION_REASON_KKT_MATRIX_PASS_LIMIT then
  cumulative_kkt_matrix_passes is at least kkt_pass_limit.
  """
  kkt_matrix_pass_limit::Float64
end

function construct_termination_criteria(;
  optimality_norm = L2,
  eps_optimal_absolute = 1.0e-6,
  eps_optimal_relative = 1.0e-6,
  eps_primal_infeasible = 1.0e-8,
  eps_dual_infeasible = 1.0e-8,
  time_sec_limit = Inf,
  iteration_limit = typemax(Int32),
  kkt_matrix_pass_limit = Inf,
)
  return TerminationCriteria(
    optimality_norm,
    eps_optimal_absolute,
    eps_optimal_relative,
    eps_primal_infeasible,
    eps_dual_infeasible,
    time_sec_limit,
    iteration_limit,
    kkt_matrix_pass_limit,
  )
end

function validate_termination_criteria(criteria::TerminationCriteria)
  if criteria.eps_primal_infeasible < 0
    error("eps_primal_infeasible must be nonnegative")
  end
  if criteria.eps_dual_infeasible < 0
    error("eps_dual_infeasible must be nonnegative")
  end
  if criteria.time_sec_limit <= 0
    error("time_sec_limit must be positive")
  end
  if criteria.iteration_limit <= 0
    error("iteration_limit must be positive")
  end
  if criteria.kkt_matrix_pass_limit <= 0
    error("kkt_matrix_pass_limit must be positive")
  end
end

"""
Information about the quadratic program that is used in the termination
criteria. We store it in this struct so we don't have to recompute it.
"""
struct CachedQuadraticProgramInfo
  l_inf_norm_primal_linear_objective::Float64
  l_inf_norm_primal_right_hand_side::Float64
  l2_norm_primal_linear_objective::Float64
  l2_norm_primal_right_hand_side::Float64
end

function cached_quadratic_program_info(qp::QuadraticProgrammingProblem)
  return CachedQuadraticProgramInfo(
    norm(qp.objective_vector, Inf),
    norm(qp.right_hand_side, Inf),
    norm(qp.objective_vector, 2),
    norm(qp.right_hand_side, 2),
  )
end

"""
Check if the algorithm should terminate declaring the optimal solution is found.
"""
function optimality_criteria_met(
  optimality_norm::OptimalityNorm,
  abs_tol::Float64,
  rel_tol::Float64,
  convergence_information::ConvergenceInformation,
  qp_cache::CachedQuadraticProgramInfo,
)

  ci = convergence_information
  abs_obj = abs(ci.primal_objective) + abs(ci.dual_objective)
  gap = abs(ci.primal_objective - ci.dual_objective)

  if optimality_norm == L_INF
    primal_err = ci.l_inf_primal_residual
    primal_err_baseline = qp_cache.l_inf_norm_primal_right_hand_side
    dual_err = ci.l_inf_dual_residual
    dual_err_baseline = qp_cache.l_inf_norm_primal_linear_objective
  elseif optimality_norm == L2
    primal_err = ci.l2_primal_residual
    primal_err_baseline = qp_cache.l2_norm_primal_right_hand_side
    dual_err = ci.l2_dual_residual
    dual_err_baseline = qp_cache.l2_norm_primal_linear_objective
  else
    error("Unknown optimality_norm")
  end

  return dual_err < abs_tol + rel_tol * dual_err_baseline &&
         primal_err < abs_tol + rel_tol * primal_err_baseline &&
         gap < abs_tol + rel_tol * abs_obj

end

# TODO: Allow the termination norm to change for infeasibility
# detection.
"""
Check if the algorithm should terminate declaring the primal is infeasible.
"""
function primal_infeasibility_criteria_met(
  eps_primal_infeasible::Float64,
  infeasibility_information::InfeasibilityInformation,
)
  ii = infeasibility_information
  if ii.dual_ray_objective <= 0.0
    return false
  end
  return ii.max_dual_ray_infeasibility / ii.dual_ray_objective <=
         eps_primal_infeasible
end

"""
Check if the algorithm should terminate declaring the dual is infeasible.
"""
function dual_infeasibility_criteria_met(
  eps_dual_infeasible::Float64,
  infeasibility_information::InfeasibilityInformation,
)
  ii = infeasibility_information
  if ii.primal_ray_linear_objective >= 0.0
    return false
  end
  return ii.max_primal_ray_infeasibility / (-ii.primal_ray_linear_objective) <=
         eps_dual_infeasible &&
         ii.primal_ray_quadratic_norm / (-ii.primal_ray_linear_objective) <=
         eps_dual_infeasible
end

"""
Checks if the given iteration_stats satisfy the termination criteria. Returns
a TerminationReason if so, and false otherwise.
"""
function check_termination_criteria(
  criteria::TerminationCriteria,
  qp_cache::CachedQuadraticProgramInfo,
  iteration_stats::IterationStats,
)
  for convergence_information in iteration_stats.convergence_information
    if optimality_criteria_met(
      criteria.optimality_norm,
      criteria.eps_optimal_absolute,
      criteria.eps_optimal_relative,
      convergence_information,
      qp_cache,
    )
      return TERMINATION_REASON_OPTIMAL
    end
  end
  for infeasibility_information in iteration_stats.infeasibility_information
    if primal_infeasibility_criteria_met(
      criteria.eps_primal_infeasible,
      infeasibility_information,
    )
      return TERMINATION_REASON_PRIMAL_INFEASIBLE
    end
    if dual_infeasibility_criteria_met(
      criteria.eps_dual_infeasible,
      infeasibility_information,
    )
      return TERMINATION_REASON_DUAL_INFEASIBLE
    end
  end
  if iteration_stats.iteration_number >= criteria.iteration_limit
    return TERMINATION_REASON_ITERATION_LIMIT
  elseif iteration_stats.cumulative_kkt_matrix_passes >=
         criteria.kkt_matrix_pass_limit
    return TERMINATION_REASON_KKT_MATRIX_PASS_LIMIT
  elseif iteration_stats.cumulative_time_sec >= criteria.time_sec_limit
    return TERMINATION_REASON_TIME_LIMIT
  else
    return false # Don't terminate.
  end
end

function termination_reason_to_string(termination_reason::TerminationReason)
  # Strip TERMINATION_REASON_ prefix.
  return string(termination_reason)[20:end]
end
