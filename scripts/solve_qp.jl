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

# This is an interface to FirstOrderLp for solving a single QP or LP and writing
# the solution and solve statistics to a file. Run with --help for a pretty
# description of the arguments. See the comments on solve_instance_and_output
# for a description of the output formats.

# Built-in Julia packages.
import SparseArrays

# Third-party Julia packages.
import ArgParse
import GZip
import JSON3

import FirstOrderLp

function run_with_redirected_stdio(func, stdout_path, stderr_path)
  open(stdout_path, "w") do stdout_io
    open(stderr_path, "w") do stderr_io
      redirect_stdout(stdout_io) do
        redirect_stderr(stderr_io) do
          func()
        end
      end
    end
  end
end

function write_vector_to_file(filename, vector)
  open(filename, "w") do io
    for x in vector
      println(io, x)
    end
  end
end

"""
Solves a linear or quadratic programming problem using one of the methods
implemented in FirstOrderLp. Takes a path to an instance. The instance must have
the extension .mps, .mps.gz, .qps, or .qps.gz.
Creates:
- `instance_full_log.json.gz` with a SolveLog in gzipped JSON format with all
   iteration_stats data returned by the solver
- `instance_summary.json` with a SolveLog in plain text JSON format, identical
   to `instance_full_log.json.gz` but with iteration_stats cleared
- `instance_primal.txt` with the primal solution
- `instance_dual.txt` with the dual solution

If `redirect_stdio` is `True`, then it also writes `instance_stderr.txt`
and `instance_stdout.txt`.
"""
function solve_instance_and_output(
  parameters::Union{
    FirstOrderLp.MirrorProxParameters,
    FirstOrderLp.PdhgParameters,
  },
  output_dir::String,
  instance_path::String,
  redirect_stdio::Bool,
  transform_bounds::Bool,
  fixed_format_input::Bool,
)
  if !isdir(output_dir)
    mkpath(output_dir)
  end

  instance_name =
    replace(basename(instance_path), r"\.(mps|MPS|qps|QPS)(\.gz)?$" => "")

  function inner_solve()
    lower_file_name = lowercase(basename(instance_path))
    if endswith(lower_file_name, ".mps") ||
       endswith(lower_file_name, ".mps.gz") ||
       endswith(lower_file_name, ".qps") ||
       endswith(lower_file_name, ".qps.gz")
      lp = FirstOrderLp.qps_reader_to_standard_form(
        instance_path,
        fixed_format = fixed_format_input,
      )
    else
      error(
        "Instance has unrecognized file extension: ",
        basename(instance_path),
      )
    end

    presolve_info = FirstOrderLp.presolve(
      lp;
      verbosity = parameters.verbosity,
      transform_bounds = transform_bounds,
    )

    if parameters.verbosity >= 1
      println("Instance: ", instance_name)
    end
    running_time = @elapsed begin
      output::FirstOrderLp.SaddlePointOutput =
        FirstOrderLp.optimize(parameters, lp)
    end
    println("Elapsed time: $running_time sec")

    log = FirstOrderLp.SolveLog()
    log.instance_name = instance_name
    log.command_line_invocation = join([PROGRAM_FILE; ARGS...], " ")
    log.termination_reason = output.termination_reason
    log.termination_string = output.termination_string
    log.iteration_count = output.iteration_count
    log.solve_time_sec = running_time
    # This assumes that the last iterate matches the solution returned by the
    # solver.
    log.solution_stats = output.iteration_stats[end]
    # TODO: Update this once we return more than one type.
    log.solution_type = FirstOrderLp.POINT_TYPE_AVERAGE_ITERATE

    summary_output_path = joinpath(output_dir, instance_name * "_summary.json")
    open(summary_output_path, "w") do io
      write(io, JSON3.write(log, allow_inf = true))
    end

    log.iteration_stats = output.iteration_stats
    full_log_output_path =
      joinpath(output_dir, instance_name * "_full_log.json.gz")
    GZip.open(full_log_output_path, "w") do io
      write(io, JSON3.write(log, allow_inf = true))
    end

    primal_solution, dual_solution = FirstOrderLp.undo_presolve(
      presolve_info,
      output.primal_solution,
      output.dual_solution,
    )

    primal_output_path = joinpath(output_dir, instance_name * "_primal.txt")
    write_vector_to_file(primal_output_path, primal_solution)

    dual_output_path = joinpath(output_dir, instance_name * "_dual.txt")
    write_vector_to_file(dual_output_path, dual_solution)
  end

  if redirect_stdio
    stdout_path = joinpath(output_dir, instance_name * "_stdout.txt")
    stderr_path = joinpath(output_dir, instance_name * "_stderr.txt")
    run_with_redirected_stdio(inner_solve, stdout_path, stderr_path)
  else
    inner_solve()
  end

  return
end

"""
Defines parses and args.

# Returns
A dictionary with the values of the command-line arguments.
"""
function parse_command_line()
  arg_parse = ArgParse.ArgParseSettings()

  help_method = "The optimization method to use, must be `mirror-prox` or `pdhg`."

  help_instance_path = "The path to the instance to solve in .mps.gz or .mps format."

  ArgParse.@add_arg_table! arg_parse begin
    "--method"
    help = help_method
    arg_type = String
    required = true

    "--output_dir"
    help = "The directory for output files."
    arg_type = String
    required = true

    "--instance_path"
    help = help_instance_path
    arg_type = String
    required = true

    "--l_inf_ruiz_iterations"
    help =
      "Number of l_infinity Ruiz rescaling iterations to apply to the " *
      "constraint matrix. Zero disables this rescaling pass."
    arg_type = Int
    default = 10

    "--l2_norm_rescaling"
    help = "If true, applies L2 norm rescaling after Ruiz rescaling."
    arg_type = Bool
    default = true

    "--pock_chambolle_alpha"
    help =
      "If specified, enables Pock-Chambolle rescaling using the exponent " *
      "parameter alpha. Alpha must be in the interval [0, 2]. If not " *
      "specified, this rescaling step is skipped."
    arg_type = Float64

    "--primal_importance"
    help =
      "Must be positive. The default of 1 balances primal and dual " *
      "~equally. Somewhat smaller values, e.g. 0.4, seem to yield better " *
      "dual solutions at the cost of increased primal infeasibility. " *
      "This corresponds to the initial value of the primal weight. " *
      "Note that tuning this parameter value to a particular problem may " *
      "significantly improve performance. "
    arg_type = Float64
    default = 1.0

    "--artificial_restart_threshold"
    help =
      "If in the past artificial_restart_threshold fraction of " *
      "iterations no restart has occurred then a restart will be " *
      "artificially triggered. The value should be strictly greater than " *
      "zero and less than or equal to one. " *
      "Smaller values will have more frequent artificial restarts than " *
      "larger values. A value of one means there is exactly one " *
      "artificial restart occurs (on iteration 1)."
    arg_type = Float64
    default = 0.5

    # TODO: Make more detailed description when restart scheme is
    # finalized.
    "--sufficient_reduction_for_restart"
    help =
      "Only supported when restart_scheme=adaptive_normalized. It is " *
      "the threshold improvement in the quality of the current/average " *
      "iterate compared with that of the last restart that will trigger a " *
      "restart. The value of this parameter should be between zero and one. " *
      "Smaller values make restarts less frequent, larger values make " *
      "restarts more frequent."
    arg_type = Float64
    default = 0.1

    "--necessary_reduction_for_restart"
    help =
      "Only supported when restart_scheme is adaptive_normalized, " *
      "adaptive_distance, or adaptive_localized. It is the threshold " *
      "improvement in the quality of the current/average iterate compared " *
      "with that of the last restart that is necessary to trigger a restart. " *
      "If this improvement threshold is met and the quality of the iterates " *
      "appear to be getting worse then a restart is triggered. " *
      "The value of this parameter should be between the value of " *
      "sufficient_reduction_for_restart and one. " *
      "Smaller values make restarts less frequent, larger values make " *
      "restarts more frequent."
    arg_type = Float64
    default = 0.9

    "--primal_weight_update_smoothing"
    help =
      "This parameter controls exponential smoothing of " *
      "log(primal_weight) when updating the primal weight. " *
      "Must be between 0.0 and 1.0 inclusive. At 0.0, the primal " *
      "weight is frozen at the initial value given by the " *
      "`primal_importance` argument. At 1.0, no smoothing is performed. It " *
      "may be preferable to freeze the primal weight if a good estimate is " *
      "available."
    arg_type = Float64
    default = 0.5

    "--verbosity"
    help =
      "The verbosity level for printing. Values between 1 and 4 " *
      "print generic information on the solve process, i.e., a table of the " *
      "iteration statistics. Values greater than 5 provide information " *
      "useful for developers."
    arg_type = Int64
    default = 2

    "--redirect_stdio"
    help = "Redirect stdout and stderr to files (for batch runs)."
    arg_type = Bool
    default = false

    "--diagonal_scaling"
    help =
      "Supported {off, l1, l2}. Use a diagonal matrix to " *
      "define the Bregman distance or, equivalently, rescale the primal " *
      "and dual variables individually."
    arg_type = String
    default = "off"

    "--restart_scheme"
    help =
      "Supported steps: {no_restart, fixed_frequency, " *
      "adaptive_normalized, adaptive_localized, adaptive_distance}. " *
      "See FirstOrderLp.RestartScheme enum for detailed documentation."
    arg_type = String
    default = "adaptive_normalized"

    "--restart_frequency"
    help =
      "Only relevant if --restart_scheme = fixed_frequency. " *
      "Determines the number of iterations until restart."
    arg_type = Int64
    default = 1000

    "--restart_to_current_metric"
    help =
      "Options: {gap_over_distance, gap_over_distance_squared, " *
      "no_restart_to_current}. If this is value is no_restart_to_current " *
      "then always restart to the average. Otherwise, we dynamically decide " *
      "whether we reset to the average or current. There are two options for " *
      "making this decision: gap_over_distance_squared and " *
      "gap_over_distance. gap_over_distance_squared will reset to current " *
      "if the normalized duality gap divided by the distanced travelled for " *
      "the current iterate is better. Alternatively, gap_over_distance" *
      "restarts to the current if it has a better normalized duality gap than" *
      "the average iterate. " *
      "Also, note that this option has no impact if " *
      "restart_scheme=no_restart."
    arg_type = String
    default = "gap_over_distance_squared"

    "--adaptive_step_size"
    help =
      "PDHG only. If this is true use an adaptive step size rule. " *
      "Otherwise, use a constant step size corresponding to an estimate of " *
      "the maximum singular value"
    arg_type = Bool
    default = true

    "--use_approximate_localized_duality_gap"
    help =
      "Whether to use an approximate localized duality gap in the " *
      "restart scheme."
    arg_type = Bool
    default = false

    "--record_iteration_stats"
    help =
      "Whether we record iterations stats. If true then record an " *
      "IterationStats proto with frequency (in iterations) " *
      "equal to termination_evaluation_frequency."
    arg_type = Bool
    default = true

    "--termination_evaluation_frequency"
    help =
      "Frequency (in iterations) that the termination criteron is " *
      "evaluated."
    arg_type = Int64
    default = 40

    # The following parameters specify termination criteria.
    # These correspond to the fields in the TerminationCriteria proto.
    # If the arguments are not provided, the default values are grabbed from the
    # TerminationCriteria proto.
    "--optimality_norm"
    help =
      "The norm for the optimality criteria. Supported options: {l2, " *
      "l_inf}"
    arg_type = String

    "--absolute_optimality_tol"
    help = "The absolute tolerance for the optimality criteria."
    arg_type = Float64

    "--relative_optimality_tol"
    help = "The relative tolerance for the optimality criteria."
    arg_type = Float64

    # The next two parameters specify infeasibility criteria.
    "--eps_primal_infeasible"
    help = "Tolerance for declaring primal infeasibility."
    arg_type = Float64

    "--eps_dual_infeasible"
    help = "Tolerance for declaring dual infeasibility."
    arg_type = Float64

    # The next two parameters are for early termination.
    "--time_sec_limit"
    help = "Time limit in seconds."
    arg_type = Float64

    "--iteration_limit"
    help = "Maximum number of iterations to run."
    arg_type = Int32

    "--kkt_matrix_pass_limit"
    help = "Terminate after this many passes through the KKT matrix."
    arg_type = Int32

    "--transform_bounds_into_linear_constraints"
    help =
      "Transform all bounds into linear constraints. This is for " *
      "ablation study only. It generally makes the problem harder to " *
      "solve."
    arg_type = Bool
    default = false

    "--fixed_format_input"
    help =
      "If true, parse the input MPS/QPS file in fixed format instead of " *
      "free (the default)."
    arg_type = Bool
    default = false
  end

  return ArgParse.parse_args(arg_parse)
end

function string_to_restart_scheme(restart_scheme::String)
  if restart_scheme == "no_restart"
    return FirstOrderLp.NO_RESTARTS
  elseif restart_scheme == "adaptive_normalized"
    return FirstOrderLp.ADAPTIVE_NORMALIZED
  elseif restart_scheme == "adaptive_distance"
    return FirstOrderLp.ADAPTIVE_DISTANCE
  elseif restart_scheme == "adaptive_localized"
    return FirstOrderLp.ADAPTIVE_LOCALIZED
  elseif restart_scheme == "fixed_frequency"
    return FirstOrderLp.FIXED_FREQUENCY
  else
    error("Unknown restart scheme $(restart_scheme)")
  end
end

function string_to_restart_to_current_metric(restart_to_current_metric::String)
  if restart_to_current_metric == "no_restart_to_current"
    return FirstOrderLp.NO_RESTART_TO_CURRENT
  elseif restart_to_current_metric == "gap_over_distance"
    return FirstOrderLp.GAP_OVER_DISTANCE
  elseif restart_to_current_metric == "gap_over_distance_squared"
    return FirstOrderLp.GAP_OVER_DISTANCE_SQUARED
  else
    error(
      "Unknown value for restart_to_current_metric $(restart_to_current_metric)",
    )
  end
end


function main()
  parsed_args = parse_command_line()

  if parsed_args["method"] == "mirror-prox" || parsed_args["method"] == "pdhg"
    restart_params = FirstOrderLp.construct_restart_parameters(
      string_to_restart_scheme(parsed_args["restart_scheme"]),
      string_to_restart_to_current_metric(
        parsed_args["restart_to_current_metric"],
      ),
      parsed_args["restart_frequency"],
      parsed_args["artificial_restart_threshold"],
      parsed_args["sufficient_reduction_for_restart"],
      parsed_args["necessary_reduction_for_restart"],
      parsed_args["primal_weight_update_smoothing"],
      parsed_args["use_approximate_localized_duality_gap"],
    )

    pock_chambolle_alpha = get(parsed_args, "pock_chambolle_alpha", nothing)

    termination_criteria = FirstOrderLp.construct_termination_criteria()
    if parsed_args["optimality_norm"] == "l2"
      termination_criteria.optimality_norm = FirstOrderLp.OptimalityNorm.L2
    elseif parsed_args["optimality_norm"] == "l_inf"
      termination_criteria.optimality_norm = FirstOrderLp.OptimalityNorm.L_INF
    elseif parsed_args["optimality_norm"] !== nothing
      error("Unknown termination norm.")
    end
    for (field_name, arg_name) in [
      (:eps_optimal_absolute, "absolute_optimality_tol"),
      (:eps_optimal_relative, "relative_optimality_tol"),
      (:eps_primal_infeasible, "eps_primal_infeasible"),
      (:eps_dual_infeasible, "eps_dual_infeasible"),
      (:time_sec_limit, "time_sec_limit"),
      (:iteration_limit, "iteration_limit"),
      (:kkt_matrix_pass_limit, "kkt_matrix_pass_limit"),
    ]
      if parsed_args[arg_name] !== nothing
        setproperty!(termination_criteria, field_name, parsed_args[arg_name])
      end
    end

    if parsed_args["method"] == "mirror-prox"
      parameters = FirstOrderLp.MirrorProxParameters(
        parsed_args["l_inf_ruiz_iterations"],
        parsed_args["l2_norm_rescaling"],
        pock_chambolle_alpha,
        parsed_args["primal_importance"],
        parsed_args["diagonal_scaling"],
        parsed_args["verbosity"],
        parsed_args["record_iteration_stats"],
        parsed_args["termination_evaluation_frequency"],
        termination_criteria,
        restart_params,
      )
    elseif parsed_args["method"] == "pdhg"
      parameters = FirstOrderLp.PdhgParameters(
        parsed_args["l_inf_ruiz_iterations"],
        parsed_args["l2_norm_rescaling"],
        pock_chambolle_alpha,
        parsed_args["primal_importance"],
        parsed_args["diagonal_scaling"],
        parsed_args["adaptive_step_size"],
        parsed_args["verbosity"],
        parsed_args["record_iteration_stats"],
        parsed_args["termination_evaluation_frequency"],
        termination_criteria,
        restart_params,
      )
    end
  else
    error("`method` arg must be either `mirror-prox` or `pdhg`.")
  end

  solve_instance_and_output(
    parameters,
    parsed_args["output_dir"],
    parsed_args["instance_path"],
    parsed_args["redirect_stdio"],
    parsed_args["transform_bounds_into_linear_constraints"],
    parsed_args["fixed_format_input"],
  )
end

main()
