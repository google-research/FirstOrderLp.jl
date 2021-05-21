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

# This file is for utility code used by both mirror_prox.jl and
# primal_dual_hybrid_gradient.jl .

"""
A solution computed by saddle point mirror prox or primal-dual hybrid
gradient (PDHG).
"""
struct SaddlePointOutput
  """
  The output primal solution vector.
  """
  primal_solution::Vector{Float64}

  """
  The output dual solution vector.
  """
  dual_solution::Vector{Float64}

  """
  One of the possible values from the TerminationReason enum.
  """
  termination_reason::TerminationReason

  """
  Extra information about the termination reason (may be empty).
  """
  termination_string::String

  """
  The total number of algorithmic iterations for the solve.
  """
  iteration_count::Int32

  """
  Detailed statistics about a subset of the iterations. The collection frequency
  is defined by algorithm parameters.
  """
  iteration_stats::Vector{IterationStats}
end

function unscaled_saddle_point_output(
  scaled_problem::ScaledQpProblem,
  primal_solution::AbstractVector{Float64},
  dual_solution::AbstractVector{Float64},
  termination_reason::TerminationReason,
  iterations_completed::Int64,
  iteration_stats::Vector{IterationStats},
)

  # Unscale iterates.
  original_primal_solution =
    primal_solution ./ scaled_problem.variable_rescaling
  original_dual_solution = dual_solution ./ scaled_problem.constraint_rescaling

  return SaddlePointOutput(
    original_primal_solution,
    original_dual_solution,
    termination_reason,
    termination_reason_to_string(termination_reason),
    iterations_completed,
    iteration_stats,
  )
end

"""
Projects the given point onto a set of bounds.
"""
function projection!(
  primal::AbstractVector{Float64},
  variable_lower_bound::AbstractVector{Float64},
  variable_upper_bound::AbstractVector{Float64},
)
  for idx in 1:length(primal)
    primal[idx] = min(
      variable_upper_bound[idx],
      max(variable_lower_bound[idx], primal[idx]),
    )
  end
end

"""Projects the given primal solution onto the feasible set. That is, all
negative duals for inequality constraints are set to zero."""
function project_primal!(
  primal::AbstractVector{Float64},
  problem::QuadraticProgrammingProblem,
)
  projection!(
    primal,
    problem.variable_lower_bound,
    problem.variable_upper_bound,
  )
end

""" Projects the given dual solution onto the feasible set. That is, all
negative duals for inequality constraints are set to zero."""
function project_dual!(
  dual::AbstractVector{Float64},
  problem::QuadraticProgrammingProblem,
)
  for idx in inequality_range(problem)
    dual[idx] = max(dual[idx], 0.0)
  end
end

"""The weighted l2 norm"""
function weighted_norm(
  vec::AbstractVector{Float64},
  weights::AbstractVector{Float64},
)
  sum = 0.0
  for i in eachindex(vec)
    sum += weights[i] * vec[i] * vec[i]
  end
  return sqrt(sum)
end

"""
Computes the local duality gap.
"""
function compute_localized_duality_gap(
  problem::QuadraticProgrammingProblem,
  primal_solution::AbstractVector{Float64},
  dual_solution::AbstractVector{Float64},
  primal_norm_params::AbstractVector{Float64},
  dual_norm_params::AbstractVector{Float64},
  distance_to_optimality::Float64,
  norm::LocalizedDualityGapNorm,
  use_approximate_localized_duality_gap::Bool,
)
  local_duality_gap = bound_optimal_objective(
    problem,
    primal_solution,
    dual_solution,
    primal_norm_params,
    dual_norm_params,
    distance_to_optimality,
    norm,
    solve_approximately = use_approximate_localized_duality_gap,
  )

  return local_duality_gap
end

mutable struct RestartInfo
  """
  The primal_solution recorded at the last restart point.
  """
  primal_solution::Vector{Float64}

  """
  The dual_solution recorded at the last restart point.
  """
  dual_solution::Vector{Float64}

  """
  Localized duality gap at last restart. This has a value of nothing if no
  restart has occurred.
  """
  last_restart_localized_duality_gap::Union{Nothing,OptimalObjectiveBoundResult}

  """
  The length of the last restart interval.
  """
  last_restart_length::Int64

  """
  The primal distance moved from the restart point two restarts ago and the
  average of the iterates across the last restart.
  """
  primal_distance_moved_last_restart_period::Float64

  """
  The dual distance moved from the restart point two restarts ago and the
  average of the iterates across the last restart.
  """
  dual_distance_moved_last_restart_period::Float64

  """
  Reduction in the potential function that was achieved last time we tried to do
  a restart.
  """
  gap_reduction_ratio_last_trial::Float64
end

function create_last_restart_info(
  problem::QuadraticProgrammingProblem,
  primal_solution::AbstractVector,
  dual_solution::AbstractVector,
)
  return RestartInfo(
    copy(primal_solution),
    copy(dual_solution),
    nothing,
    1,
    0.0,
    0.0,
    1.0,
  )
end

mutable struct SolutionWeightedAverage
  sum_primal_solutions::Vector{Float64}
  sum_dual_solutions::Vector{Float64}
  sum_primal_solutions_count::Int64
  sum_dual_solutions_count::Int64
  sum_primal_solution_weights::Float64
  sum_dual_solution_weights::Float64
end

function initialize_solution_weighted_average(
  primal_size::Int64,
  dual_size::Int64,
)
  return SolutionWeightedAverage(
    zeros(primal_size),
    zeros(dual_size),
    0,
    0,
    0.0,
    0.0,
  )
end

function reset_solution_weighted_average(
  solution_weighted_avg::SolutionWeightedAverage,
)
  solution_weighted_avg.sum_primal_solutions =
    zeros(length(solution_weighted_avg.sum_primal_solutions))
  solution_weighted_avg.sum_dual_solutions =
    zeros(length(solution_weighted_avg.sum_dual_solutions))
  solution_weighted_avg.sum_primal_solutions_count = 0
  solution_weighted_avg.sum_dual_solutions_count = 0
  solution_weighted_avg.sum_primal_solution_weights = 0.0
  solution_weighted_avg.sum_dual_solution_weights = 0.0
  return
end

function add_to_primal_solution_weighted_average(
  solution_weighted_avg::SolutionWeightedAverage,
  current_primal_solution::AbstractVector{Float64},
  weight::Float64,
)
  @assert solution_weighted_avg.sum_primal_solutions_count >= 0
  solution_weighted_avg.sum_primal_solutions .+=
    current_primal_solution * weight
  solution_weighted_avg.sum_primal_solutions_count += 1
  solution_weighted_avg.sum_primal_solution_weights += weight
  return
end

function add_to_dual_solution_weighted_average(
  solution_weighted_avg::SolutionWeightedAverage,
  current_dual_solution::AbstractVector{Float64},
  weight::Float64,
)
  @assert solution_weighted_avg.sum_dual_solutions_count >= 0
  solution_weighted_avg.sum_dual_solutions .+= current_dual_solution * weight
  solution_weighted_avg.sum_dual_solutions_count += 1
  solution_weighted_avg.sum_dual_solution_weights += weight
  return
end

function add_to_solution_weighted_average(
  solution_weighted_avg::SolutionWeightedAverage,
  current_primal_solution::AbstractVector{Float64},
  current_dual_solution::AbstractVector{Float64},
  weight::Float64,
)
  add_to_primal_solution_weighted_average(
    solution_weighted_avg,
    current_primal_solution,
    weight,
  )
  add_to_dual_solution_weighted_average(
    solution_weighted_avg,
    current_dual_solution,
    weight,
  )
  return
end

function compute_average(solution_weighted_avg::SolutionWeightedAverage)
  return solution_weighted_avg.sum_primal_solutions /
         solution_weighted_avg.sum_primal_solution_weights,
  solution_weighted_avg.sum_dual_solutions /
  solution_weighted_avg.sum_dual_solution_weights
end

"""
RestartScheme enum

# Values:

-  `NO_RESTARTS`: No restarts are performed.
-  `FIXED_FREQUENCY`: does a restart every [restart_frequency]
    iterations where [restart_frequency] is a user-specified number.
-  `ADAPTIVE_NORMALIZED`: a heuristic based on
    the normalized duality gap to decide when to restart. The general idea is
    is to demand some minimum decrease in the normalized duality gap if that is
    achieved and if either (i) a large reduction in the normalized duality gap
    occurs, or (ii) the normalized duality gap increases from one iteration to
    the other then we restart.
-  `ADAPTIVE_LOCALIZED`: This option restarts when the potential
    function defined by the localized duality gap divided by the restart length
    falls by a factor of necessary_reduction_for_restart. See RestartParameters
    for documentation for necessary_reduction_for_restart.
-  `ADAPTIVE_DISTANCE`: This option restarts when the potential
    function defined by the distance traveled divided by the restart length
    falls by a factor of necessary_reduction_for_restart.
"""
@enum RestartScheme NO_RESTARTS FIXED_FREQUENCY ADAPTIVE_NORMALIZED ADAPTIVE_LOCALIZED ADAPTIVE_DISTANCE

"""
RestartToCurrentMetric enum

# Values:

- `NO_RESTART_TO_CURRENT`: Always reset to the average.
- `GAP_OVER_DISTANCE`: Decide between the average
  current based on which has a smaller normalized duality gap.
- `GAP_OVER_DISTANCE_SQUARED`: Decide between the
  average current based on which has the smaller normalized duality gap divided
  by distance travelled.
"""

@enum RestartToCurrentMetric NO_RESTART_TO_CURRENT GAP_OVER_DISTANCE GAP_OVER_DISTANCE_SQUARED

mutable struct RestartParameters
  """
  Specifies what type of restart scheme is used.
  """
  restart_scheme::RestartScheme

  """
  Specifies how we decide between restarting to the average or current.
  """
  restart_to_current_metric::RestartToCurrentMetric

  """
  If `restart_scheme` = `FIXED_FREQUENCY` then this number determines the
  frequency that the algorithm is restarted.
  """
  restart_frequency_if_fixed::Int64

  """
  If in the past `artificial_restart_threshold` fraction of iterations no
  restart has occurred then a restart will be artificially triggered. The value
  should be between zero and one. Smaller values will have more frequent
  artificial restarts than larger values.
  """
  artificial_restart_threshold::Float64

  """
  Only applies when `restart_scheme` = `ADAPTIVE`. It is the threshold
  improvement in the quality of the current/average iterate compared with that
  of the last restart that will trigger a restart. The value of this parameter
  should be between zero and one. Smaller values make restarts less frequent,
  larger values make restarts more frequent.
  """
  sufficient_reduction_for_restart::Float64

  """
  Only applies when `restart_scheme` = `ADAPTIVE`. It is the threshold
  improvement in the quality of the current/average iterate compared with that
  of the last restart that is neccessary for a restart to be triggered. If this
  thrshold is met and the quality of the iterates appear to be getting worse
  then a restart is triggered. The value of this parameter should be between
  zero and one, and greater than sufficient_reduction_for_restart. Smaller
  values make restarts less frequent, larger values make restarts more frequent.
  """
  necessary_reduction_for_restart::Float64

  """
  Controls the exponential smoothing of log(primal_weight) when the primal
  weight is updated (i.e., on every restart). Must be between 0.0 and 1.0
  inclusive. At 0.0 the primal weight remains frozen at its initial value.
  """
  primal_weight_update_smoothing::Float64

  use_approximate_localized_duality_gap::Bool
end

function construct_restart_parameters(
  restart_scheme::RestartScheme,
  restart_to_current_metric::RestartToCurrentMetric,
  restart_frequency_if_fixed::Int64,
  artificial_restart_threshold::Float64,
  sufficient_reduction_for_restart::Float64,
  necessary_reduction_for_restart::Float64,
  primal_weight_update_smoothing::Float64,
  use_approximate_localized_duality_gap::Bool,
)
  @assert restart_frequency_if_fixed > 1
  @assert 0.0 < artificial_restart_threshold <= 1.0
  @assert 0.0 <
          sufficient_reduction_for_restart <=
          necessary_reduction_for_restart <=
          1.0
  @assert 0.0 <= primal_weight_update_smoothing <= 1.0

  return RestartParameters(
    restart_scheme,
    restart_to_current_metric,
    restart_frequency_if_fixed,
    artificial_restart_threshold,
    sufficient_reduction_for_restart,
    necessary_reduction_for_restart,
    primal_weight_update_smoothing,
    use_approximate_localized_duality_gap,
  )
end

function compute_localized_duality_gaps(
  problem::QuadraticProgrammingProblem,
  current_primal_solution::AbstractVector{Float64},
  current_dual_solution::AbstractVector{Float64},
  avg_primal_solution::Vector{Float64},
  avg_dual_solution::Vector{Float64},
  primal_norm_params::AbstractVector{Float64},
  dual_norm_params::AbstractVector{Float64},
  last_restart_info::RestartInfo,
  use_approximate_localized_duality_gap::Bool,
)

  # Compute bar{r}_i
  distance_traveled_by_average = sqrt(
    weighted_norm(
      avg_primal_solution - last_restart_info.primal_solution,
      primal_norm_params,
    )^2 +
    weighted_norm(
      avg_dual_solution - last_restart_info.dual_solution,
      dual_norm_params,
    )^2,
  )
  # Compute Delta_{bar{r}_i}(bar{w}_{i+1})
  localized_duality_gap_at_average = compute_localized_duality_gap(
    problem,
    avg_primal_solution,
    avg_dual_solution,
    primal_norm_params,
    dual_norm_params,
    distance_traveled_by_average,
    EUCLIDEAN_NORM,
    use_approximate_localized_duality_gap,
  )

  # Compute hat{r}_i
  distance_traveled_by_current = sqrt(
    weighted_norm(
      current_primal_solution - last_restart_info.primal_solution,
      primal_norm_params,
    )^2 +
    weighted_norm(
      current_dual_solution - last_restart_info.dual_solution,
      dual_norm_params,
    )^2,
  )
  # Compute Delta_{hat{r}_i}(hat{w}_{i+1})
  localized_duality_gap_at_current = compute_localized_duality_gap(
    problem,
    current_primal_solution,
    current_dual_solution,
    primal_norm_params,
    dual_norm_params,
    distance_traveled_by_current,
    EUCLIDEAN_NORM,
    use_approximate_localized_duality_gap,
  )

  return (
    gap_at_average = localized_duality_gap_at_average,
    distance_traveled_by_average = distance_traveled_by_average,
    gap_at_current = localized_duality_gap_at_current,
    distance_traveled_by_current = distance_traveled_by_current,
  )
end

# TODO: This should reference the paper when it is done.

"""
Given that a restart was triggered, this function decides if we should restart
to the average or the current iterate.

The general idea is that mirror-prox/pdhg guarrantee:

Delta_{bar{r}_i}(bar{w}_{i+1}) / bar{r}_i^2 <= C / T_i

where i indicates the restart period, bar{r}_i is the distance traveled by the
average iterates across this current restart period, bar{w}_{i+1} is the
iterates averaged across the restart period, T_i is the number of iterations
across the restart period, C is an unknown constant, and Delta_r(w) is the local
duality gap for radius r at the point w. Therefore, we pick the
current iterate hat{w}_{i+1} if

Delta_{hat{r}_i}(hat{w}_{i+1}) / hat{r}_i^2 <
  Delta_{bar{r}_i}(bar{w}_{i+1}) / bar{r}_i^2.

where hat{r}_i is the distance traveled by the current iterate across the
restart period. This choice guarantees that

Delta_{r_i}(w_{i+1}) / r_i^2 <= Delta_{bar{r}_i}(bar{w}_{i+1}) / bar{r}_i^2
                             <= C / T_i

where w_{i+1} = bar{w}_{i+1} and r_i = bar{r}_i if we choose the average
and w_{i+1} = hat{w}_{i+1} and r_i = hat{r}_i if we choose the current
iterate. In other words, this choice guarantees that you do as least as well as
choosing the average in terms of this metric of progress.

"""
function should_reset_to_average(
  current::OptimalObjectiveBoundResult,
  distance_traveled_by_current::Float64,
  average::OptimalObjectiveBoundResult,
  distance_traveled_by_average::Float64,
  restart_to_current_metric::RestartToCurrentMetric,
)
  current_normalized_gap = get_gap(current) / distance_traveled_by_current
  average_normalized_gap = get_gap(average) / distance_traveled_by_average
  if restart_to_current_metric == GAP_OVER_DISTANCE_SQUARED
    return current_normalized_gap / distance_traveled_by_current >=
           average_normalized_gap / distance_traveled_by_average
  elseif restart_to_current_metric == GAP_OVER_DISTANCE
    return current_normalized_gap >= average_normalized_gap
  else
    return true # reset to average
  end
end

function should_do_adaptive_restart_normalized_duality_gap(
  problem::QuadraticProgrammingProblem,
  primal_norm_params::AbstractVector{Float64},
  dual_norm_params::AbstractVector{Float64},
  candidate_localized_gap::OptimalObjectiveBoundResult,
  candidate_distance_traveled::Float64,
  restart_params::RestartParameters,
  last_restart_info::RestartInfo,
  use_approximate_localized_duality_gap::Bool,
  primal_weight::Float64,
)
  lri = last_restart_info
  distance_traveled_last_restart = sqrt(
    lri.primal_distance_moved_last_restart_period^2 * primal_weight +
    lri.dual_distance_moved_last_restart_period^2 / primal_weight,
  )
  # TODO: Cache this after primal weight is updated
  last_restart = compute_localized_duality_gap(
    problem,
    lri.primal_solution,
    lri.dual_solution,
    primal_norm_params,
    dual_norm_params,
    distance_traveled_last_restart,
    EUCLIDEAN_NORM,
    use_approximate_localized_duality_gap,
  )
  do_restart = false
  normalized_candidate_gap =
    get_gap(candidate_localized_gap) / candidate_distance_traveled
  normalized_last_restart_gap =
    get_gap(last_restart) / distance_traveled_last_restart
  gap_reduction_ratio = normalized_candidate_gap / normalized_last_restart_gap
  if gap_reduction_ratio < restart_params.necessary_reduction_for_restart
    if gap_reduction_ratio < restart_params.sufficient_reduction_for_restart
      do_restart = true
    elseif gap_reduction_ratio > lri.gap_reduction_ratio_last_trial
      # Last time we evaluated the results were better.
      do_restart = true
    end
  end
  lri.gap_reduction_ratio_last_trial = gap_reduction_ratio

  return do_restart
end

# TODO: Consider adding ideas from the heuristic restart strategy
# into the theoretical strategies to make them work better.
function should_do_localized_adaptive_restart(
  candidate_localized_gap::OptimalObjectiveBoundResult,
  candidate_restart_length::Int64,
  restart_params::RestartParameters,
  last_restart_info::RestartInfo,
)
  do_restart = false

  lri = last_restart_info
  if candidate_localized_gap == nothing ||
     lri.last_restart_localized_duality_gap == nothing
    return true
  end

  new_potential = get_gap(candidate_localized_gap) / candidate_restart_length
  old_potential =
    get_gap(lri.last_restart_localized_duality_gap) / lri.last_restart_length
  if new_potential / old_potential <
     restart_params.necessary_reduction_for_restart
    do_restart = true
  end

  return do_restart
end


function should_do_distance_based_adaptive_restart(
  candidate_localized_gap::OptimalObjectiveBoundResult,
  candidate_distance_traveled::Float64,
  candidate_restart_length::Int64,
  restart_params::RestartParameters,
  last_restart_info::RestartInfo,
  primal_weight::Float64,
)
  do_restart = false

  lri = last_restart_info

  distance_traveled = candidate_distance_traveled
  distance_traveled_last_restart = sqrt(
    lri.primal_distance_moved_last_restart_period^2 * primal_weight +
    lri.dual_distance_moved_last_restart_period^2 / primal_weight,
  )
  new_potential = distance_traveled / candidate_restart_length
  old_potential = distance_traveled_last_restart / lri.last_restart_length
  if new_potential / old_potential <
     restart_params.necessary_reduction_for_restart
    do_restart = true
  end

  return do_restart
end


"""
  run_restart_scheme(problem, solution_weighted_avg, current_primal_solution,
    current_dual_solution, last_restart_quality, verbosity, restart_params)

This function decides whether to restart and performs the restart.
If it does restart then it updates the solution_weighted_avg,
current_primal_solution, current_dual_solution, and last_restart_quality
accordingly.

# Inputs
- `problem::QuadraticProgrammingProblem`
- `solution_weighted_avg::SolutionWeightedAverage`: If there is a restart then
   this struct will be reset to the restart point.
- `current_primal_solution::AbstractVector`: If there is a restart then this
   vector might be set to the avg_primal_solution. However, it could remain
   unchanged, if the algorithm thinks the averaged iterate is better than the
   current iterate.
- `current_dual_solution::AbstractVector`: If there is a restart then this
   vector might be set to the avg_dual_solution. However, it could remain
   unchanged, if the algorithm thinks the averaged iterate is better than the
   current iterate.
- `last_restart_info::RestartInfo`: Information stored about the last
   restart point.
- `iterations_completed::Int64`: The number of successful iterations completed
   by the algoirthm (i.e., where the step is accepted).
- `primal_norm_params::AbstractVector{Float64}`: The weights of the weighted l2
   norm used to measure distance in the primal and compute steps.
- `dual_norm_params::AbstractVector{Float64}`: The weights of the weighted l2
   norm used to measure distance in the dual and compute steps.
- `primal_weight::Float64`: The current value of the primal weight.
- `verbosity::Int64`: The output level.
- `restart_params::RestartParameters`: Parameters for making restart decisions.

# Output
A RestartChoice enum which tells us if a restart was performed. See the
the definition of RestartChoice for more details.
"""
function run_restart_scheme(
  problem::QuadraticProgrammingProblem,
  solution_weighted_avg::SolutionWeightedAverage,
  current_primal_solution::AbstractVector,
  current_dual_solution::AbstractVector,
  last_restart_info::RestartInfo,
  iterations_completed::Int64,
  primal_norm_params::AbstractVector{Float64},
  dual_norm_params::AbstractVector{Float64},
  primal_weight::Float64,
  verbosity::Int64,
  restart_params::RestartParameters,
)
  # TODO: Add options for the purposes of benchmarking, e.g., use the
  # last iterate instead of the average, or average across all iterates.
  if solution_weighted_avg.sum_primal_solutions_count > 0 &&
     solution_weighted_avg.sum_dual_solutions_count > 0
    avg_primal_solution, avg_dual_solution =
      compute_average(solution_weighted_avg)
  else
    return RESTART_CHOICE_NO_RESTART
  end

  restart_length = solution_weighted_avg.sum_primal_solutions_count
  artificial_restart = false
  do_restart = false
  # If we have not restarted for a very long time (as a fraction of the
  # iterations completed by the algorithm) then force a restart.
  # A restart is always triggered the first time this function is called,
  # because because restart_params.artificial_restart_threshold is less
  # than or equal to one and iterations_completed == restart_length.
  if restart_length >=
     restart_params.artificial_restart_threshold * iterations_completed
    do_restart = true
    artificial_restart = true
  end
  # Decide if we are going to reset to average.
  if restart_params.restart_scheme == NO_RESTARTS
    reset_to_average = false
    candidate_localized_gap = nothing
  else
    localized_duality_gaps = compute_localized_duality_gaps(
      problem,
      current_primal_solution,
      current_dual_solution,
      avg_primal_solution,
      avg_dual_solution,
      primal_norm_params,
      dual_norm_params,
      last_restart_info,
      restart_params.use_approximate_localized_duality_gap,
    )
    reset_to_average = should_reset_to_average(
      localized_duality_gaps.gap_at_current,
      localized_duality_gaps.distance_traveled_by_current,
      localized_duality_gaps.gap_at_average,
      localized_duality_gaps.distance_traveled_by_average,
      restart_params.restart_to_current_metric,
    )

    if reset_to_average
      candidate_localized_gap = localized_duality_gaps.gap_at_average
      candidate_distance_traveled =
        localized_duality_gaps.distance_traveled_by_average
    else
      candidate_localized_gap = localized_duality_gaps.gap_at_current
      candidate_distance_traveled =
        localized_duality_gaps.distance_traveled_by_current
    end
  end

  if !do_restart
    # Decide if we are going to do a restart.
    if restart_params.restart_scheme == ADAPTIVE_NORMALIZED
      do_restart = should_do_adaptive_restart_normalized_duality_gap(
        problem,
        primal_norm_params,
        dual_norm_params,
        candidate_localized_gap,
        candidate_distance_traveled,
        restart_params,
        last_restart_info,
        restart_params.use_approximate_localized_duality_gap,
        primal_weight,
      )

    elseif (
      restart_params.restart_scheme == ADAPTIVE_LOCALIZED ||
      restart_params.restart_scheme == ADAPTIVE_DISTANCE
    ) && last_restart_info.last_restart_localized_duality_gap == nothing
      do_restart = true # automatically restart if no restarts have occurred
    elseif restart_params.restart_scheme == ADAPTIVE_LOCALIZED
      do_restart = should_do_localized_adaptive_restart(
        candidate_localized_gap,
        restart_length,
        restart_params,
        last_restart_info,
      )
    elseif restart_params.restart_scheme == ADAPTIVE_DISTANCE
      do_restart = should_do_distance_based_adaptive_restart(
        candidate_localized_gap,
        candidate_distance_traveled,
        restart_length,
        restart_params,
        last_restart_info,
        primal_weight,
      )
    elseif restart_params.restart_scheme == FIXED_FREQUENCY &&
           restart_params.restart_frequency_if_fixed <= restart_length
      do_restart = true
    end
  end

  if !do_restart
    return RESTART_CHOICE_NO_RESTART
  else
    if reset_to_average
      if verbosity >= 4
        print("  Restarted to average")
      end
      current_primal_solution .= avg_primal_solution
      current_dual_solution .= avg_dual_solution
    else
      # Current point is much better than average point.
      if verbosity >= 4
        print("  Restarted to current")
      end
    end

    if verbosity >= 4
      print(" after ", rpad(restart_length, 4), " iterations")
      if artificial_restart
        println("*")
      else
        println("")
      end
    end
    reset_solution_weighted_average(solution_weighted_avg)

    update_last_restart_info(
      last_restart_info,
      current_primal_solution,
      current_dual_solution,
      avg_primal_solution,
      avg_dual_solution,
      primal_norm_params,
      dual_norm_params,
      primal_weight,
      candidate_localized_gap,
      restart_length,
    )

    if reset_to_average
      return RESTART_CHOICE_RESTART_TO_AVERAGE
    else
      return RESTART_CHOICE_WEIGHTED_AVERAGE_RESET
    end
  end
end

"""
Given the current primal weight and restart information, returns a recomputed
primal weight.

# Inputs:
- `last_restart_info::RestartInfo`: Information on the last restart point.
- `primal_weight::Float64`: The primal weight is the ratio of step sizes
  between the dual and primal.
- `primal_weight_update_smoothing::Float64`: Parameter that decides how smoothed
  the primal weight updates should be.
- `verbosity::Int64`: Output level.
# Output:
The new primal weight.
"""
function compute_new_primal_weight(
  last_restart_info::RestartInfo,
  primal_weight::Float64,
  primal_weight_update_smoothing::Float64,
  verbosity::Int64,
)
  primal_distance = last_restart_info.primal_distance_moved_last_restart_period
  dual_distance = last_restart_info.dual_distance_moved_last_restart_period
  # Note that the choice of eps() is arbitrary. It could have been eps()*100 or
  # eps()/100. It is just to avoid the case when dual_distance or
  # primal_distance equals zero.
  if primal_distance > eps() && dual_distance > eps()
    new_primal_weight_estimate = dual_distance / primal_distance
    # Exponential moving average.
    # If primal_weight_update_smoothing = 1.0 then there is no smoothing.
    # If primal_weight_update_smoothing = 0.0 then the primal_weight is frozen.
    log_primal_weight =
      primal_weight_update_smoothing * log(new_primal_weight_estimate) +
      (1 - primal_weight_update_smoothing) * log(primal_weight)

    primal_weight = exp(log_primal_weight)
    if verbosity >= 4
      Printf.@printf "  New computed primal weight is %.2e\n" primal_weight
    end

    return primal_weight
  else
    return primal_weight
  end
end

function update_last_restart_info(
  last_restart_info::RestartInfo,
  current_primal_solution::AbstractVector{Float64},
  current_dual_solution::AbstractVector{Float64},
  avg_primal_solution::Vector{Float64},
  avg_dual_solution::Vector{Float64},
  primal_norm_params::AbstractVector{Float64},
  dual_norm_params::AbstractVector{Float64},
  primal_weight::Float64,
  candidate_localized_gap::Union{Nothing,OptimalObjectiveBoundResult},
  restart_length::Int64,
)
  # TODO: Define anchor points and only update these anchor points
  # when a nonartifical restart occurs. This should give better primal weight
  # update performance. The reason I add this comment is that I have noticed
  # that it looks like there is sometimes interaction between the primal
  # weights and restart scheme which causes cycling like behaviour.

  last_restart_info.primal_distance_moved_last_restart_period =
    weighted_norm(
      avg_primal_solution - last_restart_info.primal_solution,
      primal_norm_params,
    ) / sqrt(primal_weight)
  last_restart_info.dual_distance_moved_last_restart_period =
    weighted_norm(
      avg_dual_solution - last_restart_info.dual_solution,
      dual_norm_params,
    ) * sqrt(primal_weight)
  last_restart_info.primal_solution .= current_primal_solution
  last_restart_info.dual_solution .= current_dual_solution

  last_restart_info.last_restart_length = restart_length
  last_restart_info.last_restart_localized_duality_gap = candidate_localized_gap

end

"""
A simple string name for a PointType.
"""
function point_type_label(point_type::PointType)
  if point_type == POINT_TYPE_CURRENT_ITERATE
    return "current"
  elseif point_type == POINT_TYPE_AVERAGE_ITERATE
    return "average"
  elseif point_type == POINT_TYPE_ITERATE_DIFFERENCE
    return "difference"
  else
    return "unknown PointType"
  end
end

"""
Logging for when the algorithm terminates.
"""
function generic_final_log(
  problem::QuadraticProgrammingProblem,
  current_primal_solution::AbstractVector{Float64},
  current_dual_solution::AbstractVector{Float64},
  last_iteration_stats::IterationStats,
  verbosity::Int64,
  iteration::Int64,
  termination_reason::TerminationReason,
)
  if verbosity >= 1
    print("Terminated after $iteration iterations: ")
    println(termination_reason_to_string(termination_reason))
  end

  method_specific_stats = last_iteration_stats.method_specific_stats
  if verbosity >= 3
    for convergence_information in last_iteration_stats.convergence_information
      Printf.@printf(
        "For %s candidate:\n",
        point_type_label(convergence_information.candidate_type)
      )
      # Print more decimal places for the primal and dual objective.
      Printf.@printf(
        "Primal objective: %f, ",
        convergence_information.primal_objective
      )
      Printf.@printf(
        "dual objective: %f, ",
        convergence_information.dual_objective
      )
      Printf.@printf(
        "corrected dual objective: %f \n",
        convergence_information.corrected_dual_objective
      )
    end
    if haskey(method_specific_stats, "estimated_lower_bound") &&
       haskey(method_specific_stats, "estimated_upper_bound")
      Printf.@printf(
        "Estimated optimal objective range: [%f, %f] \n",
        method_specific_stats["estimated_lower_bound"],
        method_specific_stats["estimated_upper_bound"],
      )
    end
    Printf.@printf(
      "Lagrangian value: %f \n",
      method_specific_stats["lagrangian_value"],
    )
  end
  if verbosity >= 4
    Printf.@printf(
      "Time (seconds):\n - Basic algorithm: %.2e\n - Full algorithm:  %.2e\n",
      method_specific_stats["time_spent_doing_basic_algorithm"],
      last_iteration_stats.cumulative_time_sec,
    )
  end

  if verbosity >= 7
    for convergence_information in last_iteration_stats.convergence_information
      print_infinity_norms(convergence_information)
    end
    print_variable_and_constraint_hardness(
      problem,
      current_primal_solution,
      current_dual_solution,
    )
  end
end

function update_objective_bound_estimates(
  method_specific_stats::Dict{String,Float64},
  problem::QuadraticProgrammingProblem,
  current_primal_solution::AbstractVector{Float64},
  current_dual_solution::AbstractVector{Float64},
  primal_norm_weights::AbstractVector{Float64},
  dual_norm_weights::AbstractVector{Float64},
)
  # TODO: Use better estimates of the distance to optimality.
  # TODO: Once these better estimates are developed do a study.
  estimated_primal_distance_to_optimality =
    max(1e-8, weighted_norm(current_primal_solution, primal_norm_weights))
  estimated_dual_distance_to_optimality =
    max(1e-8, weighted_norm(current_dual_solution, dual_norm_weights))

  estimated_local_duality_gap = compute_localized_duality_gap(
    problem,
    current_primal_solution,
    current_dual_solution,
    primal_norm_weights / estimated_primal_distance_to_optimality^2,
    dual_norm_weights / estimated_dual_distance_to_optimality^2,
    1.0,
    MAX_NORM,
    false, # don't compute this approximately
  )

  method_specific_stats["lagrangian_value"] =
    estimated_local_duality_gap.lagrangian_value
  method_specific_stats["estimated_lower_bound"] =
    estimated_local_duality_gap.lower_bound_value
  method_specific_stats["estimated_upper_bound"] =
    estimated_local_duality_gap.upper_bound_value
end

function select_initial_primal_weight(
  problem::QuadraticProgrammingProblem,
  primal_norm_params::AbstractVector{Float64},
  dual_norm_params::AbstractVector{Float64},
  primal_importance::Float64,
  verbosity::Int64,
)
  rhs_vec_norm = weighted_norm(problem.right_hand_side, dual_norm_params)
  obj_vec_norm = weighted_norm(problem.objective_vector, primal_norm_params)
  if obj_vec_norm > 0.0 && rhs_vec_norm > 0.0
    # Note that the units of the objective vector are
    # (objective units / x units) and the units of the right hand side are
    # (objective units / y units). Therefore the units of
    # obj_vec_norm / rhs_vec_norm are (y units / x units).
    # Note that this arguement is somewhat handwavy because each component of x
    # could have different physical units. For example, one objective
    # coefficient could be cost per widget and another could be cost per
    # factory.
    primal_weight = primal_importance * (obj_vec_norm / rhs_vec_norm)
  else
    primal_weight = primal_importance
  end
  if verbosity >= 6
    println("Initial primal weight = $primal_weight")
  end
  return primal_weight
end

# The gradient of the Lagrangian with respect to primal variables. This is
# objective_matrix * primal_solution + objective_vector -
# constraint_matrix' * dual_solution, i.e., the value of the reduced costs
# if we wanted dual feasibilty to hold exactly.
function compute_primal_gradient(
  problem::QuadraticProgrammingProblem,
  primal_solution::AbstractVector{Float64},
  dual_solution::AbstractVector{Float64},
)
  return compute_primal_gradient_from_dual_product(
    problem,
    primal_solution,
    problem.constraint_matrix' * dual_solution,
  )
end

function compute_primal_gradient_from_dual_product(
  problem::QuadraticProgrammingProblem,
  primal_solution::AbstractVector{Float64},
  dual_product::AbstractVector{Float64},
)
  return problem.objective_matrix * primal_solution .+
         problem.objective_vector .- dual_product
end

function compute_dual_gradient(
  problem::QuadraticProgrammingProblem,
  primal_solution::AbstractVector{Float64},
)
  return problem.right_hand_side .- problem.constraint_matrix * primal_solution
end

function compute_lagrangian_value(
  problem::QuadraticProgrammingProblem,
  primal_solution::AbstractVector{Float64},
  dual_solution::AbstractVector{Float64},
)
  return 0.5 *
         dot(primal_solution, problem.objective_matrix * primal_solution) +
         dot(primal_solution, problem.objective_vector) -
         dot(primal_solution, problem.constraint_matrix' * dual_solution) +
         dot(dual_solution, problem.right_hand_side) +
         problem.objective_constant
end
