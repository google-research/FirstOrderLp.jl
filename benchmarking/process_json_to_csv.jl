# Copyright 2021 Google LLC
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
#
# Processes a collection of solve results in JSON format into a CSV file.
#
# The collection of solve results can correspond to multiple parameter settings.
# We assume that results for each parameter setting are in a separate directory
# and are named as *_summary.json, containing serialized SolveLogs.
# The layout of the results is specified by a JSON file of the following format:
#
# {"datasets": [
#   {"name": "name1", "logs_directory": "path1"},
#   {"name": "name2", "logs_directory": "path2"}
#  ] }
#
# For example, suppose /tmp/layout.json contains the following content:
# {"datasets": [
#   {"name": "pdhg,1e-8", "logs_directory": "/tmp/pdhg_1e-8"},
#   {"name": "pdhg,1e-4", "logs_directory": "/tmp/pdhg_1e-4"},
#   {"name": "scs,1e-8", "logs_directory": "/tmp/scs_1e-8"},
#   {"name": "scs,1e-4", "logs_directory": "/tmp/scs_1e-4"}
#  ] }
#
# Then:
# $ julia --project=. process_json_to_csv.jl /tmp/layout.json /tmp/dataset.csv
#
# will generate a result file in /tmp/dataset.csv containing the summary of each
# solve in all of four directories, with columns identifying the corresponding
# configuration.

import CSV
import DataFrames
import Glob
import JSON3
import StructArrays
import StructTypes

import FirstOrderLp

const DataFrame = DataFrames.DataFrame

mutable struct CsvRow
  experiment_name::String
  instance_name::String

  # These fields correpond to those in ConvergenceInformation.
  primal_objective::Float64
  dual_objective::Float64
  relative_optimality_gap::Float64
  l2_primal_residual::Float64
  l_inf_primal_residual::Float64
  l2_dual_residual::Float64
  l_inf_dual_residual::Float64
  relative_l2_primal_residual::Float64
  relative_l_inf_primal_residual::Float64
  relative_l2_dual_residual::Float64
  relative_l_inf_dual_residual::Float64
  l_inf_primal_variable::Float64
  l2_primal_variable::Float64
  l_inf_dual_variable::Float64

  termination_reason::FirstOrderLp.TerminationReason
  iteration_count::Int32
  cumulative_kkt_matrix_passes::Float64
  solve_time_sec::Float64
end

function CsvRow()
  return CsvRow(
    "",  # experiment_name
    "",  # instance_name
    NaN,  # primal_objective
    NaN,  # dual_objective
    NaN,  # relative_optimality_gap
    NaN,  # l2_primal_residual
    NaN,  # l_inf_primal_residual
    NaN,  # l2_dual_residual
    NaN,  # l_inf_dual_residual
    NaN,  # relative_l2_primal_residual
    NaN,  # relative_l_inf_primal_residual
    NaN,  # relative_l2_dual_residual
    NaN,  # relative_l_inf_dual_residual
    NaN,  # l_inf_primal_variable
    NaN,  # l2_primal_variable
    NaN,  # l_inf_dual_variable
    FirstOrderLp.TERMINATION_REASON_UNSPECIFIED,  # termination_reason
    0,  # iteration_count
    NaN,  # cumulative_kkt_matrix_passes
    NaN,  # solve_time_sec
  )
end

function set_matching_fields(to, from)
  for field_name in intersect(fieldnames(typeof(to)), fieldnames(typeof(from)))
    setfield!(to, field_name, getfield(from, field_name))
  end
end

function solve_log_to_csv_row(
  log::FirstOrderLp.SolveLog,
  experiment_name::String,
)::CsvRow
  row = CsvRow()
  set_matching_fields(row, log)
  row.experiment_name = experiment_name
  row.cumulative_kkt_matrix_passes =
    log.solution_stats.cumulative_kkt_matrix_passes

  point_type = log.solution_type
  # TODO: This doesn't properly handle the case of infeasibility certificates,
  # whose stats are in log.solution_stats.infeasibility_information.
  for convergence_information in log.solution_stats.convergence_information
    if convergence_information.candidate_type == point_type
      set_matching_fields(row, convergence_information)
      break
    end
  end
  return row
end

function rows_to_dataframe(rows::Vector{CsvRow})
  return DataFrame(; StructArrays.components(StructArrays.StructArray(rows))...)
end

struct DatasetNameAndLocation
  name::String
  logs_directory::String
end

struct DatasetList
  datasets::Vector{DatasetNameAndLocation}
end

StructTypes.StructType(::Type{DatasetNameAndLocation}) = StructTypes.Struct()
StructTypes.StructType(::Type{DatasetList}) = StructTypes.Struct()

function read_dataset(dataset_list::DatasetList)::DataFrame
  rows = CsvRow[]
  for name_and_location in dataset_list.datasets
    logs_directory = name_and_location.logs_directory

    log_files = Glob.glob("*_summary.json", logs_directory)
    if length(log_files) == 0
      error("No *_summary.json files found in $logs_directory.")
    end
    for filename in log_files
      log = JSON3.read(read(filename, String), FirstOrderLp.SolveLog)
      push!(rows, solve_log_to_csv_row(log, name_and_location.name))
    end
  end
  return rows_to_dataframe(rows)
end

if length(ARGS) != 2
  error("Usage: process_json_to_csv.jl dataset_layout_csv output_csv")
end

dataset_list = JSON3.read(read(ARGS[1], String), DatasetList)
dataset = read_dataset(dataset_list)
CSV.write(ARGS[2], dataset)
