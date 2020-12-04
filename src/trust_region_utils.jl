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
BoundConstrainedTrustRegionResult

Note that we re-use problem (1) as defined in the documentation of the
solve_bound_constrained_trust_region function.
"""
mutable struct BoundConstrainedTrustRegionResult
  "The approximate solution to (1)."
  solution::Vector{Float64}
  "The value of objective_vector' * (solution - center_point)."
  value::Float64
end


"""
Finds a solution to the problem:

argmin_x objective_vector' * x
s.t. variable_lower_bounds <= x <= variable_upper_bounds                    (1)
     || x - center_point || <= target_radius

where || . || is weighted by norm_weights.

for a positive value of target_radius, by solving the related problem

argmin_t objective_vector' * x
s.t. x := min(max(center_point - t * objective_vector, variable_lower_bounds),
              variable_upper_bounds)
     || x - center_point || <= target_radius

Note that the definition of x is just applying the lower/upper bound constraints
to center_point - t * objective_vector.

This problem is solved by computing the breakpoint at which each component of
x switches from varying with t to being fixed at its bounds.  The radius of the
median breakpoint is evaluated, eliminating half of the components.  The process
is iterated until the argmin is identified.

# Inputs:
- `center_point::Vector{AbstractFloat64}`. See (1). It is assumed that
  center_point satisfies the bounds.
- `objective_vector::Vector{Float64}`. See (1).
- `variable_lower_bounds::Vector{Float64}`. See (1).
- `variable_upper_bounds::Vector{Float64}`. See (1).
- `norm_weights::AbstractVector{Float64}`. The weights used for the norm
  || . ||.
- `target_radius::Float64`. See (1).
- `solve_approximately::Bool`. If true, solve the approximate version of (1)
  which only applies the bound constraints to variables already at their bound.

# Output:
BoundConstrainedTrustRegionResult struct.
"""
function solve_bound_constrained_trust_region(
  center_point::AbstractVector{Float64},
  objective_vector::Vector{Float64},
  variable_lower_bounds::Vector{Float64},
  variable_upper_bounds::Vector{Float64},
  norm_weights::AbstractVector{Float64},
  target_radius::Float64,
  solve_approximately::Bool,
)
  if solve_approximately
    return approximately_solve_bound_constrained_trust_region(
      center_point,
      objective_vector,
      variable_lower_bounds,
      variable_upper_bounds,
      norm_weights,
      target_radius,
    )
  end

  @assert 0.0 <= target_radius < Inf
  if target_radius == 0.0 || norm(objective_vector, 2) == 0.0
    return BoundConstrainedTrustRegionResult(copy(center_point), 0.0)
  end

  direction = zeros(length(center_point))
  threshold = zeros(length(center_point))
  for idx in 1:length(center_point)
    if center_point[idx] >= variable_upper_bounds[idx] &&
       objective_vector[idx] <= 0
      continue
    end
    if center_point[idx] <= variable_lower_bounds[idx] &&
       objective_vector[idx] >= 0
      continue
    end
    direction[idx] = -objective_vector[idx] / norm_weights[idx]
    if direction[idx] > 0
      threshold[idx] =
        (variable_upper_bounds[idx] - center_point[idx]) / direction[idx]
    elseif direction[idx] < 0
      threshold[idx] =
        (variable_lower_bounds[idx] - center_point[idx]) / direction[idx]
    else
      # Variable doesn't move.  Rather than an infinite threshold, or a NaN if
      # the corresponding bound was infinite as well, treat it as fixed, which
      # is effectively equivalent.
      threshold[idx] = 0.0
    end
  end

  # The weighted radius squared of the indices discarded because they are below
  # the threshold.
  low_radius_sq = 0.0
  # The weighted norm squared of the objective coefficients of the indices
  # discarded because they are above the threshold.
  high_radius_sq = 0.0

  indices = collect(1:length(center_point))
  # Infinite thresholds can combine with zeros to create NaNs.  To avoid this,
  # handle indices with infinite thresholds separately.
  infinite_indices = filter(i -> isinf(threshold[i]), indices)
  high_radius_sq +=
    weighted_norm(direction[infinite_indices], norm_weights[infinite_indices])^2
  filter!(i -> isfinite(threshold[i]), indices)

  while length(indices) > 0
    test_threshold = median(threshold[indices])
    test_point =
      clamp.(
        center_point[indices] + test_threshold * direction[indices],
        variable_lower_bounds[indices],
        variable_upper_bounds[indices],
      )
    test_radius =
      weighted_norm(test_point - center_point[indices], norm_weights[indices])
    if low_radius_sq + test_radius^2 + test_threshold^2 * high_radius_sq >=
       target_radius^2
      # test_threshold is too high.  Discard indices greater than it.
      discard_indices = filter(i -> threshold[i] >= test_threshold, indices)
      high_radius_sq +=
        weighted_norm(
          direction[discard_indices],
          norm_weights[discard_indices],
        )^2
      filter!(i -> threshold[i] < test_threshold, indices)
    else
      # test_threshold is too low.  Discard indices less than it.
      discard_indices = filter(i -> threshold[i] <= test_threshold, indices)
      discard_point =
        clamp.(
          center_point[discard_indices] +
          test_threshold * direction[discard_indices],
          variable_lower_bounds[discard_indices],
          variable_upper_bounds[discard_indices],
        )
      low_radius_sq +=
        weighted_norm(
          discard_point - center_point[discard_indices],
          norm_weights[discard_indices],
        )^2
      filter!(i -> threshold[i] > test_threshold, indices)
    end
  end

  # target_threshold is the solution of
  # low_radius_sq + target_threshold^2 * high_radius_sq = target_radius^2.
  if high_radius_sq <= 0.0
    # Special case: high_radius_sq = 0.0, means all bounds hit before reaching
    # target radius.
    target_threshold = maximum(threshold)
  else
    target_threshold = sqrt((target_radius^2 - low_radius_sq) / high_radius_sq)
  end
  candidate_point =
    clamp.(
      center_point + target_threshold * direction,
      variable_lower_bounds,
      variable_upper_bounds,
    )
  return BoundConstrainedTrustRegionResult(
    candidate_point,
    dot(objective_vector, candidate_point - center_point),
  )
end

function approximately_solve_bound_constrained_trust_region(
  center_point::AbstractVector{Float64},
  objective_vector::Vector{Float64},
  variable_lower_bounds::Vector{Float64},
  variable_upper_bounds::Vector{Float64},
  norm_weights::AbstractVector{Float64},
  target_radius::Float64,
)
  direction = zeros(length(center_point))
  for idx in 1:length(center_point)
    if center_point[idx] >= variable_upper_bounds[idx] &&
       objective_vector[idx] <= 0
      continue
    end
    if center_point[idx] <= variable_lower_bounds[idx] &&
       objective_vector[idx] >= 0
      continue
    end
    direction[idx] = -objective_vector[idx] / norm_weights[idx]
  end

  direction_norm = weighted_norm(direction, norm_weights)
  if direction_norm > 0.0
    direction *= target_radius / direction_norm
  end

  return BoundConstrainedTrustRegionResult(
    center_point + direction,
    dot(objective_vector, direction),
  )
end

struct OptimalObjectiveBoundResult
  lagrangian_value::Float64
  lower_bound_value::Float64
  upper_bound_value::Float64
  "The primal solution that minimizes the localized duality gap."
  primal_solution::AbstractVector{Float64}
  "The dual solution that minimizes the localized duality gap."
  dual_solution::AbstractVector{Float64}
end

function get_gap(result::OptimalObjectiveBoundResult)
  return result.upper_bound_value - result.lower_bound_value
end

"""
The norm that defines a ball in joint primal-dual space (x,y). If
norm = MAX_NORM then max{|x|_2, |y|_2} where |.|_2 is the Euclidean norm or
the Euclidean norm in the joint space if norm = EUCLIDEAN_NORM.
"""
@enum LocalizedDualityGapNorm MAX_NORM EUCLIDEAN_NORM

"""
Given a maximum distance to optimality this function returns a tuple with
a lower bound, lagrangian value, and a upper bound on the optimal objective.

These are computed by solving
min grad_x L(primal_solution, dual_solution)^T (x - primal_solution)
- grad_y L(primal_solution, dual_solution)^T (y - dual_solution)
s.t. (x,y) in ball of radius distance_to_optimality centered
at (primal_solution,dual_solution).

See the definition of LocalizedDualityGapNorm for information about
how the ball norm is defined. Note if norm=MAX_NORM then this optimization can
be split into a seperate problems in the primal and the dual.

The lower bound is computed via
L(primal_solution, dual_solution) +
grad_x L(primal_solution, dual_solution)^T (x - primal_solution)
where x is the solution to the trust region problem.

The upper bound is computed via
L(primal_solution, dual_solution) +
grad_y L(primal_solution, dual_solution)^T (y - dual_solution)
where y is the solution to the trust region problem.
"""
function bound_optimal_objective(
  problem::QuadraticProgrammingProblem,
  primal_solution::AbstractVector{Float64},
  dual_solution::AbstractVector{Float64},
  primal_norm_weights::AbstractVector{Float64},
  dual_norm_weights::AbstractVector{Float64},
  distance_to_optimality::Float64,
  norm::LocalizedDualityGapNorm;
  solve_approximately::Bool = false,
)
  # TODO: These are also unnecessarily recomputed.
  primal_gradient =
    compute_primal_gradient(problem, primal_solution, dual_solution)

  lagrangian_value =
    compute_lagrangian_value(problem, primal_solution, dual_solution)

  dual_variable_lower_bounds = -Inf * ones(length(dual_solution))
  dual_variable_upper_bounds = Inf * ones(length(dual_solution))
  dual_variable_lower_bounds[inequality_range(problem)] .= 0.0
  dual_gradient = compute_dual_gradient(problem, primal_solution)

  if norm == MAX_NORM
    # We can split the max norm into two cases.
    # Compute the primal part
    primal_result = solve_bound_constrained_trust_region(
      primal_solution,
      primal_gradient,
      problem.variable_lower_bound,
      problem.variable_upper_bound,
      primal_norm_weights,
      distance_to_optimality,
      solve_approximately,
    )

    # Compute the dual part
    dual_result = solve_bound_constrained_trust_region(
      dual_solution,
      -dual_gradient,
      dual_variable_lower_bounds,
      dual_variable_upper_bounds,
      dual_norm_weights,
      distance_to_optimality,
      solve_approximately,
    )
    OptimalObjectiveBoundResult(
      lagrangian_value,
      # lower bound
      lagrangian_value + primal_result.value,
      # upper bound
      lagrangian_value - dual_result.value,
      # primal solution
      primal_result.solution,
      # dual solution
      dual_result.solution,
    )
  elseif norm == EUCLIDEAN_NORM
    z = [primal_solution; dual_solution]
    z_gradient = [primal_gradient; -dual_gradient]
    z_lower_bound = [problem.variable_lower_bound; dual_variable_lower_bounds]
    z_upper_bound = [problem.variable_upper_bound; dual_variable_upper_bounds]
    norm_weights = [primal_norm_weights; dual_norm_weights]

    # Compute the lower bound
    result = solve_bound_constrained_trust_region(
      z,
      z_gradient,
      z_lower_bound,
      z_upper_bound,
      norm_weights,
      distance_to_optimality,
      solve_approximately,
    )
    primal_tr_solution = result.solution[1:length(primal_solution)]
    dual_tr_solution = result.solution[(length(primal_solution)+1):end]

    return OptimalObjectiveBoundResult(
      lagrangian_value,
      # lower bound
      lagrangian_value +
      dot(primal_tr_solution - primal_solution, primal_gradient),
      # upper bound
      lagrangian_value + dot(dual_tr_solution - dual_solution, dual_gradient),
      primal_tr_solution,
      dual_tr_solution,
    )
  else
    error("unknown norm = $norm, value unknown")
  end
end
