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

# A script for calling SCS for benchmark comparisons with FirstOrderLp. Run
# with --help for a pretty description of the arguments. See the comments on
# solve_instance_and_output for a description of the output formats.

# Built-in Julia packages.
import SparseArrays

# Third-party Julia packages.
import ArgParse
import JSON3
import JuMP
import MathOptInterface
import SCS

import FirstOrderLp

const MOI = MathOptInterface

function write_vector_to_file(filename, vector)
  open(filename, "w") do io
    for x in vector
      println(io, x)
    end
  end
end

"""
Creates a JuMP model based on a QuadraticProgrammingProblem. Returns the model,
the decision variables, and the references to the constraints.
"""
function instance_to_model(problem::FirstOrderLp.QuadraticProgrammingProblem)
  if !iszero(problem.objective_matrix)
    error("QP support is not yet implemented.")
  end

  model = JuMP.Model()
  num_variables = length(problem.variable_lower_bound)

  JuMP.@variable(model, x[i = 1:num_variables])
  for i in 1:num_variables
    if isfinite(problem.variable_lower_bound[i])
      JuMP.set_lower_bound(x[i], problem.variable_lower_bound[i])
    end
    if isfinite(problem.variable_upper_bound[i])
      JuMP.set_upper_bound(x[i], problem.variable_upper_bound[i])
    end
  end
  JuMP.@objective(model, Min, problem.objective_vector' * x)

  equality_range = FirstOrderLp.equality_range(problem)
  equalities = JuMP.@constraint(
    model,
    problem.constraint_matrix[equality_range, :] * x .==
    problem.right_hand_side[equality_range]
  )

  inequality_range = FirstOrderLp.inequality_range(problem)
  inequalities = JuMP.@constraint(
    model,
    problem.constraint_matrix[inequality_range, :] * x .>=
    problem.right_hand_side[inequality_range]
  )

  return model, x, [equalities; inequalities]
end

function extract_termination_reason(jump_model)
  termination_status = JuMP.termination_status(jump_model)
  if termination_status == MOI.OPTIMAL
    return FirstOrderLp.OPTIMAL
  elseif termination_status == MOI.INFEASIBLE
    return FirstOrderLp.PRIMAL_INFEASIBLE
  elseif termination_status == MOI.DUAL_INFEASIBLE
    return FirstOrderLp.DUAL_INFEASIBLE
  elseif termination_status == MOI.TIME_LIMIT
    return FirstOrderLp.TIME_LIMIT
  elseif termination_status == MOI.ITERATION_LIMIT
    return FirstOrderLp.ITERATION_LIMIT
  elseif termination_status == MOI.NUMERICAL_ERROR
    return FirstOrderLp.NUMERICAL_ERROR
  else
    return FirstOrderLp.OTHER_TERMINATION_REASON
  end
end

"""
Solves a linear programming problem using SCS. Takes a path to an instance. The
instance must have the extension .mps, or .mps.gz. Creates
`instance_summary.json` with a SolveLog serialized in JSON format,
`instance_primal.txt` with the primal solution, `instance_dual.txt` with the
dual solution, and `instance_stderr.txt`, `instance_stdout.txt`.
"""
function solve_instance_and_output(
  parameters::Vector,
  output_dir::String,
  instance_path::String,
  print_stdout::Bool,
)
  if !isdir(output_dir)
    mkpath(output_dir)
  end

  if endswith(instance_path, ".mps") || endswith(instance_path, ".mps.gz")
    lp = FirstOrderLp.qps_reader_to_standard_form(instance_path)
  else
    error("Instance has unrecognized file extension: ", basename(instance_path))
  end

  presolve_info = FirstOrderLp.presolve(lp)

  jump_model, jump_variables, jump_constraints = instance_to_model(lp)

  JuMP.set_optimizer(jump_model, SCS.Optimizer)
  eps = 0.0
  for (param_name, value) in parameters
    MOI.set(jump_model, MOI.RawParameter(param_name), value)
    if param_name == "eps"
      eps = value
    end
  end

  instance_name =
    replace(replace(basename(instance_path), ".mps.gz" => ""), ".mps" => "")

  println("Instance: ", instance_name)

  stdout_path = joinpath(output_dir, instance_name * "_stdout.txt")
  stderr_path = joinpath(output_dir, instance_name * "_stderr.txt")

  open(stderr_path, "w") do stderr_io
    redirect_stderr(stderr_io) do
      # stdout is handled differently from stderr because we need to read the
      # solver output to populate the SolveLog.
      old_stdout = stdout
      stdout_read, stdout_write = redirect_stdout()

      JuMP.optimize!(jump_model)

      Base.Libc.flush_cstdio()
      redirect_stdout(old_stdout)
      close(stdout_write)

      solve_log = read(stdout_read, String)
      write(stdout_path, solve_log)
      if print_stdout
        println(solve_log)
      end

      running_time = JuMP.solve_time(jump_model)
      println("Recorded solve time: $running_time sec")

      log = FirstOrderLp.SolveLog()
      log.instance_name = instance_name
      log.command_line_invocation = join([PROGRAM_FILE; ARGS...], " ")
      log.termination_reason = extract_termination_reason(jump_model)
      log.termination_string = JuMP.raw_status(jump_model)
      cumulative_kkt_matrix_passes = NaN
      log.iteration_count = Int32(MOI.get(jump_model, SCS.ADMMIterations()))
      # We read the log to extract "avg # CG iterations". This is printed only
      # in SCS's indirect mode.
      for line in split(solve_log, '\n')
        if occursin("avg # CG iterations", line)
          avg_cg_iters = parse(Float64, rstrip(split(line)[6], ','))
          # This formula was given by bodonoghue@.
          cumulative_kkt_matrix_passes =
            log.iteration_count + log.iteration_count * avg_cg_iters
          break
        end
      end
      log.solve_time_sec = running_time

      primal_sol = JuMP.value.(jump_variables)
      dual_sol = JuMP.dual.(jump_constraints)

      last_iteration_stats = FirstOrderLp.compute_iteration_stats(
        lp,
        FirstOrderLp.cached_quadratic_program_info(lp),
        primal_sol,
        dual_sol,
        primal_sol,  # primal_ray_estimate
        dual_sol,  # dual_ray_estimate
        log.iteration_count,
        cumulative_kkt_matrix_passes,
        running_time,
        eps,  # eps_optimal_absolute
        eps,  # eps_optimal_relative
        0.0,  # step_size
        0.0,  # primal_weight
        FirstOrderLp.POINT_TYPE_AVERAGE_ITERATE,
      )
      log.solution_stats = last_iteration_stats

      # Complete iteration stats are not available, so we write only the
      # summary.
      summary_output_path =
        joinpath(output_dir, instance_name * "_summary.json")
      open(summary_output_path, "w") do io
        write(io, JSON3.write(log, allow_inf = true))
      end

      primal_sol, dual_sol =
        FirstOrderLp.undo_presolve(presolve_info, primal_sol, dual_sol)

      primal_output_path = joinpath(output_dir, instance_name * "_primal.txt")
      write_vector_to_file(primal_output_path, primal_sol)

      dual_output_path = joinpath(output_dir, instance_name * "_dual.txt")
      write_vector_to_file(dual_output_path, dual_sol)
    end
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

  help_instance_path = "The path to the instance to solve in .mps.gz, or .mps format."

  help_print_stdout =
    "In addition to redirecting stdout, also print it at " *
    " the end for convenience."

  ArgParse.@add_arg_table! arg_parse begin
    "--solver"
    help = "The solver to use. May be 'scs-direct' or 'scs-indirect'."
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

    "--verbose"
    help = "Enable verbose solver output."
    arg_type = Bool
    default = true

    "--print_stdout"
    help = help_print_stdout
    arg_type = Bool
    default = true

    # SCS parameters are defined at
    # https://github.com/cvxgrp/scs/blob/e5bb794ac014b7a86d127ac03651d2c8a12ecba8/include/scs.h#L44
    # with default values at
    # https://github.com/cvxgrp/scs/blob/e5bb794ac014b7a86d127ac03651d2c8a12ecba8/include/glbopts.h#L30.
    "--scs-eps"
    help = "For SCS only, the value to set for the 'eps' tolerance."
    arg_type = Float64
    default = 1e-5

    "--scs-normalize"
    help = "For SCS only, apply SCS's internal rescaling heuristic."
    arg_type = Bool
    default = false

    "--scs-acceleration_lookback"
    help = "For SCS only, the number of iterations used for Anderson Acceleration"
    arg_type = Int64
    default = 10

    "--scs-cg_rate"
    help = "For scs-indirect only, the rate at which the CG convergence tolerance decreases as a function of the iteration number."
    arg_type = Float64
    default = 2.0

    "--scs-scale"
    help = "For SCS only, when normalize is set, the iterates are additionally scaled by this factor."
    arg_type = Float64
    default = 1.0

    "--iteration_limit"
    help = "Maximum number of iterations to run."
    arg_type = Int64
    default = 10000

    # NOTE: This flag is required for compatibility with experiment scripts.
    "--redirect_stdio"
    help = "Redirect stdout and stderr to files (for batch runs). The false value is not supported."
    arg_type = Bool
    default = true

  end

  return ArgParse.parse_args(arg_parse)
end

function main()
  parsed_args = parse_command_line()

  if !parsed_args["redirect_stdio"]
    error("The current implementation doesn't support redirect_stdio=false.")
  end

  if parsed_args["solver"] == "scs-indirect" ||
     parsed_args["solver"] == "scs-direct"
    parameters = []
    push!(
      parameters,
      ("acceleration_lookback", parsed_args["scs-acceleration_lookback"]),
    )
    # This is the over-relaxation parameter. 1.0 is the best value for LP
    # according to SCS's author.
    push!(parameters, ("alpha", 1.0))
    # Disable rescaling to remove confounding factors.
    if !parsed_args["scs-normalize"]
      push!(parameters, ("normalize", 0))
    end
    push!(parameters, ("scale", parsed_args["scs-scale"]))
    push!(parameters, ("eps", parsed_args["scs-eps"]))
    push!(parameters, ("max_iters", parsed_args["iteration_limit"]))
    if parsed_args["verbose"]
      push!(parameters, ("verbose", 1))
    else
      push!(parameters, ("verbose", 0))
    end
    if parsed_args["solver"] == "scs-indirect"
      push!(parameters, ("linear_solver", SCS.IndirectSolver))
      push!(parameters, ("cg_rate", parsed_args["scs-cg_rate"]))
    else
      push!(parameters, ("linear_solver", SCS.DirectSolver))
    end
  else
    error("Unrecognized solver $(parsed_args["solver"]).")
  end

  solve_instance_and_output(
    parameters,
    parsed_args["output_dir"],
    parsed_args["instance_path"],
    parsed_args["print_stdout"],
  )
end

main()
