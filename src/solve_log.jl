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

# Defines the data structures used to record solution quality and what happened
# on each iteration.

"""
RestartChoice specifies whether a restart was performed on a given iteration.

# Values

- `RESTART_CHOICE_UNSPECIFIED`: Default value.
- `RESTART_CHOICE_NO_RESTART`: No restart on this iteration.
- `RESTART_CHOICE_WEIGHTED_AVERAGE_RESET`: The weighted average of iterates is
  cleared and reset to the current point. Note that from a mathematical
  perspective this can be equivalently viewed as restarting the algorithm but
  picking the restart point to be the current iterate.
- `RESTART_CHOICE_RESTART_TO_AVERAGE`: The algorithm is restarted at the average
  of iterates since the last restart.
"""
@enum RestartChoice begin
  RESTART_CHOICE_UNSPECIFIED
  RESTART_CHOICE_NO_RESTART
  RESTART_CHOICE_WEIGHTED_AVERAGE_RESET
  RESTART_CHOICE_RESTART_TO_AVERAGE
end

"""
PointType identifies the type of point used to compute the fields in a given
struct; see ConvergenceInformation and InfeasibilityInformation.

# Values

- `POINT_TYPE_UNSPECIFIED`: Default value.
- `POINT_TYPE_CURRENT_ITERATE`: Current iterate (x_k, y_k).
- `POINT_TYPE_ITERATE_DIFFERENCE`: Difference of iterates
  (x_{k+1} - x_k, y_{k+1} - y_k).
- `POINT_TYPE_AVERAGE_ITERATE`: Average of iterates since the last restart.
- `POINT_TYPE_NONE`: There is no corresponding point.
"""
@enum PointType begin
  POINT_TYPE_UNSPECIFIED
  POINT_TYPE_CURRENT_ITERATE
  POINT_TYPE_ITERATE_DIFFERENCE
  POINT_TYPE_AVERAGE_ITERATE
  POINT_TYPE_NONE
end

"""
Information measuring how close a candidate is to establishing feasibility and
optimality; see also TerminationCriteria.
"""
mutable struct ConvergenceInformation
  """
  Type of the candidate point described by this ConvergenceInformation.
  """
  candidate_type::PointType

  """
  The primal objective. The primal need not be feasible.
  """
  primal_objective::Float64

  """
  The dual objective. The dual need not be feasible. The dual objective should
  include the contributions from the reduced costs.
  """
  dual_objective::Float64

  """
  If possible (e.g., when all primal variables have lower and upper bounds), a
  correct dual bound. The field should be set to negative infinity if no
  corrected dual bound is available.
  """
  corrected_dual_objective::Float64

  """
  The maximum violation of any primal constraint, i.e., the l_∞ norm of the
  violations.
  """
  l_inf_primal_residual::Float64

  """
  The l_2 norm of the violations of primal constraints.
  """
  l2_primal_residual::Float64

  """
  The maximum violation of any dual constraint, i.e., the l_∞ norm of the
  violations.
  """
  l_inf_dual_residual::Float64

  """
  The l_2 norm of the violations of dual constraints.
  """
  l2_dual_residual::Float64

  # Relative versions of the residuals, defined as
  #   relative_residual = residual / (eps_ratio + norm),
  # where
  #   eps_ratio = eps_optimal_absolute / eps_optimal_relative
  #   residual = one of the residuals (l{2,_inf}_{primal,dual}_residual)
  #   norm = the relative norm (l{2,_inf} norm of
  #          {constraint_bounds,primal_linear_objective} respectively).
  # If eps_optimal_relative = 0.0, these will all be 0.0.
  #
  # If eps_optimal_relative > 0.0, the absolute and relative termination
  # criteria translate to relative_residual <= eps_optimal_relative.
  # NOTE: The usefulness of these relative residuals is based on their
  # relationship to TerminationCriteria. If the TerminationCriteria change
  # consider adding additional iteration measures here.

  relative_l_inf_primal_residual::Float64

  relative_l2_primal_residual::Float64

  relative_l_inf_dual_residual::Float64

  relative_l2_dual_residual::Float64

  """
  Relative optimality gap:
    |primal_objective - dual_objective| /
    (eps_ratio + |primal_objective| + |dual_objective|)
  """
  relative_optimality_gap::Float64

  """
  The maximum absolute value of the primal variables, i.e., the l_∞ norm. This
  is useful to detect when the primal iterates are diverging. Divergence of the
  primal variables could be an algorithmic issue, or indicate that the dual is
  infeasible.
  """
  l_inf_primal_variable::Float64

  """
  The l_2 norm of the primal variables.
  """
  l2_primal_variable::Float64

  """
  The maximum absolute value of the dual variables, i.e., the l_∞ norm. This is
  useful to detect when the dual iterates are diverging. Divergence of the dual
  variables could be an algorithmic issue, or indicate the primal is infeasible.
  """
  l_inf_dual_variable::Float64

  """
  The l_2 norm of the dual variables.
  """
  l2_dual_variable::Float64
end

function ConvergenceInformation()
  return ConvergenceInformation(POINT_TYPE_UNSPECIFIED, zeros(16)...)
end

"""
Information measuring how close a point is to establishing primal or dual
infeasibility (i.e. has no solution); see also TerminationCriteria.
"""
mutable struct InfeasibilityInformation
  """
  Type of the point used to compute the InfeasibilityInformation.
  """
  candidate_type::PointType

  """
  Let x_ray be the algorithm's estimate of the primal extreme ray where x_ray is
  a vector scaled such that its infinity norm is one. A simple and typical
  choice of x_ray is x_ray = x / | x |_∞ where x is the current primal iterate.
  For this value compute the maximum absolute error in the primal linear program
  with the right hand side set to zero.
  """
  max_primal_ray_infeasibility::Float64

  """
  The value of the linear part of the primal objective (ignoring additive
  constants) evaluated at x_ray, i.e., c' * x_ray where c is the objective
  coefficient vector.
  """
  primal_ray_linear_objective::Float64

  """
  The l_∞ norm of the vector resulting from taking the quadratic matrix from
  primal objective and multiplying it by the primal variables. For linear
  programming problems this is zero.
  """
  primal_ray_quadratic_norm::Float64

  """
  Let (y_ray, r_ray) be the algorithm's estimate of the dual and reduced cost
  extreme ray where (y_ray, r_ray) is a vector scaled such that its infinity
  norm is one. A simple and typical choice of y_ray is
  (y_ray, r_ray) = (y, r) / max(| y |_∞, | r |_∞) where y is the current dual
  iterate and r is the current dual reduced costs. Consider the quadratic
  program we are solving but with the objective (both quadratic and linear
  terms) set to zero. This forms a linear program (label this linear program
  (1)) with no objective. Take the dual of (1) and compute the maximum absolute
  value of the constraint error for (y_ray, r_ray) to obtain the value of
  max_dual_ray_infeasibility.
  """
  max_dual_ray_infeasibility::Float64

  """
  The objective of the linear program labeled (1) in the previous paragraph.
  """
  dual_ray_objective::Float64
end

function InfeasibilityInformation()
  return InfeasibilityInformation(POINT_TYPE_UNSPECIFIED, zeros(5)...)
end

# All values in IterationStats assume that the primal quadratic program is a
# minimization problem and the dual is a maximization problem. Problems should
# be transformed to this form if they are not already in this form. The dual
# vector is defined to be the vector of multipliers on the linear constraints,
# that is, excluding dual multipliers on variable bounds (reduced costs).
mutable struct IterationStats
  """
  The iteration number at which these stats were recorded. By convention,
  iteration counts start at 1, and the stats correspond to the solution *after*
  the iteration. Therefore stats from iteration 0 are the stats at the starting
  point.
  """
  iteration_number::Int32

  """
  A set of statistics measuring how close a point is to establishing primal and
  dual feasibility and optimality. This field is repeated since there might be
  several different points that are considered.
  """
  convergence_information::Vector{ConvergenceInformation}

  """
  A set of statistics measuring how close a point is to establishing primal or
  dual infeasibility (i.e., has no solution). This field is repeated since there
  might be several different points that could establish infeasibility.
  """
  infeasibility_information::Vector{InfeasibilityInformation}

  """
  The cumulative number of passes through the KKT matrix since the start of the
  solve. One pass is a multply by the constraint matrix, its transpose and the
  matrix that defines the quadratic part of the objective.

  For example, each iteration of mirror saddle prox contributes 2.0 to this sum.
  This is a float because it can include fractional passes through the data.
  For example, in an active set method we may only use a submatrix with 20% of
  the nonzeros of the KKT matrix at each iteration in which case 0.2 would be
  added to the total.
  """
  cumulative_kkt_matrix_passes::Float64

  """
  The total number of rejected steps (e.g., within a line search procedure)
  since the start of the solve.
  """
  cumulative_rejected_steps::Int32

  """
  The amount of time passed since we started solving the problem (see solver log
  solve_time_sec) which records total time.
  """
  cumulative_time_sec::Float64

  """
  The kind of restart that occurred at this iteration, or
  RESTART_CHOICE_NO_RESTART if a restart did not occur.
  """
  restart_used::RestartChoice

  """
  Step size used at this iteration. Note that the step size used for the primal
  update is step_size / primal_weight, while the one used for the dual update
  is step_size * primal_weight.
  """
  step_size::Float64

  """
  Primal weight controlling the relation between primal and dual step sizes. See
  field 'step_size' for a detailed description.
  """
  primal_weight::Float64

  method_specific_stats::Dict{String,Float64}
end

function IterationStats()
  return IterationStats(
    0,
    ConvergenceInformation[],
    InfeasibilityInformation[],
    0.0,
    Int32(0),
    0.0,
    RESTART_CHOICE_UNSPECIFIED,
    0.0,
    0.0,
    Dict{String,Float64}(),
  )
end

"""
TerminationReason explains why the solver stopped. See termination.jl for the
precise criteria used to check termination.

# Values

- `TERMINATION_REASON_UNSPECIFIED`: Default value.
- `TERMINATION_REASON_OPTIMAL`
- `TERMINATION_REASON_PRIMAL_INFEASIBLE`: Note in this situation the dual could
  be either unbounded or infeasible.
- `TERMINATION_REASON_DUAL_INFEASIBLE`: Note in this situation the primal could
  either unbounded or infeasible.
- `TERMINATION_REASON_TIME_LIMIT`
- `TERMINATION_REASON_KKT_MATRIX_PASS_LIMIT`
- `TERMINATION_REASON_NUMERICAL_ERROR`
- `TERMINATION_REASON_INVALID_PROBLEM`: Indicates that the solver detected
  invalid problem data, e.g., inconsistent bounds.
- `TERMINATION_REASON_OTHER`
"""
@enum TerminationReason begin
  TERMINATION_REASON_UNSPECIFIED
  TERMINATION_REASON_OPTIMAL
  TERMINATION_REASON_PRIMAL_INFEASIBLE
  TERMINATION_REASON_DUAL_INFEASIBLE
  TERMINATION_REASON_TIME_LIMIT
  TERMINATION_REASON_ITERATION_LIMIT
  TERMINATION_REASON_KKT_MATRIX_PASS_LIMIT
  TERMINATION_REASON_NUMERICAL_ERROR
  TERMINATION_REASON_INVALID_PROBLEM
  TERMINATION_REASON_OTHER
end

mutable struct SolveLog
  """
  The name of the optimization problem.
  """
  instance_name::String

  """
  The command line used to invoke the solve. This is used to record the solve
  parameters. Does not apply if the solve is invoked through an API.
  """
  command_line_invocation::String

  """
  The reason that the solve terminated.
  """
  termination_reason::TerminationReason

  """
  Optional extra information about the termination reason.
  """
  termination_string::String

  """
  The total number of iterations during the solve.
  """
  iteration_count::Int32

  """
  The runtime of the solve. Note: This should not be used for comparing methods
  unless care is taken to control for noise in runtime measurement.
  """
  solve_time_sec::Float64

  """
  The IterationStats corresponding to the solution returned by the solver.
  """
  solution_stats::IterationStats

  """
  The type of the output point. This type specifies the information entry that
  prompted termination. If TerminationReason is TERMINATION_REASON_OPTIMAL, this
  type matches the PointType of the ConvergenceInformation entry (in
  solution_stats) that caused termination. Similarly, if TerminationReason is
  either TERMINATION_REASON_PRIMAL_INFEASIBLE or
  TERMINATION_REASON_DUAL_INFEASIBLE this type identifies the
  InfeasibilityInformation entry that caused termination.
  """
  solution_type::PointType

  """
  A history of iteration stats for the solve. The iteration_number fields should
  be in increasing order. The frequency at which these stats should be recorded
  is not specified. This field is "more" optional than the others because it
  often significantly increases the size of the message, and because the
  information may not be available for third-party solvers.
  """
  iteration_stats::Vector{IterationStats}
end

function SolveLog()
  return SolveLog(
    "",
    "",
    TERMINATION_REASON_UNSPECIFIED,
    "",
    Int32(0),
    0.0,
    IterationStats(),
    POINT_TYPE_UNSPECIFIED,
    IterationStats[],
  )
end

# Needed for JSON serialization.
StructTypes.StructType(::Type{ConvergenceInformation}) = StructTypes.Mutable()
StructTypes.StructType(::Type{InfeasibilityInformation}) = StructTypes.Mutable()
StructTypes.StructType(::Type{IterationStats}) = StructTypes.Mutable()
StructTypes.StructType(::Type{SolveLog}) = StructTypes.Mutable()
