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
# The layout of the results is specified by a JSON file that matches the
# DatasetList struct defined below.
#
# For example, suppose /tmp/layout.json contains the following content:
# {"datasets": [
#   {"config": {"solver": "pdhg", "tolerance": "1e-8"}
#     "logs_directory": "/tmp/pdhg_1e-8"},
#   {"config": {"solver": "pdhg", "tolerance": "1e-4"}
#     "logs_directory": "/tmp/pdhg_1e-4"},
#   {"config": {"solver": "scs", "tolerance": "1e-8"}
#     "logs_directory": "/tmp/scs_1e-8"},
#   {"config": {"solver": "scs", "tolerance": "1e-4"}
#     "logs_directory": "/tmp/scs_1e-4"}
#  ],
#  "config_labels": ["solver", "tolerance"] }
#
# Then:
# $ julia --project=. process_json_to_csv.jl /tmp/layout.json /tmp/dataset.csv
#
# will generate a result file in /tmp/dataset.csv containing the summary of each
# solve in all of four directories, with a column identifying the corresponding
# configuration.

import CSV
import DataFrames
import Glob
import JSON3
import StructArrays
import StructTypes

import FirstOrderLp

const DataFrame = DataFrames.DataFrame

function copy_fields_to_dict(to::Dict, from, fields::Vector{Symbol})
  for fieldname in fields
    to[fieldname] = getfield(from, fieldname)
  end
  return
end

const SOLVE_LOG_FIELDS_TO_COPY =
  [:instance_name, :termination_reason, :iteration_count, :solve_time_sec]

const CONVERGENCE_INFORMATION_FIELDS_TO_COPY = [
  :primal_objective,
  :dual_objective,
  :relative_optimality_gap,
  :l2_primal_residual,
  :l_inf_primal_residual,
  :l2_dual_residual,
  :l_inf_dual_residual,
  :relative_l2_primal_residual,
  :relative_l_inf_primal_residual,
  :relative_l2_dual_residual,
  :relative_l_inf_dual_residual,
  :l_inf_primal_variable,
  :l2_primal_variable,
  :l_inf_dual_variable,
]

function solve_log_to_dict(log::FirstOrderLp.SolveLog)::Dict
  result = Dict()
  copy_fields_to_dict(result, log, SOLVE_LOG_FIELDS_TO_COPY)
  result[:cumulative_kkt_matrix_passes] =
    log.solution_stats.cumulative_kkt_matrix_passes

  point_type = log.solution_type
  # TODO: This doesn't properly handle the case of infeasibility certificates,
  # whose stats are in log.solution_stats.infeasibility_information.
  for convergence_information in log.solution_stats.convergence_information
    if convergence_information.candidate_type == point_type
      copy_fields_to_dict(
        result,
        convergence_information,
        CONVERGENCE_INFORMATION_FIELDS_TO_COPY,
      )
      break
    end
  end
  return result
end

struct DatasetConfigAndLocation
  # Keys must match up with config_labels.
  config::Dict{String,String}
  logs_directory::String
end

struct DatasetList
  datasets::Vector{DatasetConfigAndLocation}
  config_labels::Vector{String}
end

StructTypes.StructType(::Type{DatasetConfigAndLocation}) = StructTypes.Struct()
StructTypes.StructType(::Type{DatasetList}) = StructTypes.Struct()

function read_dataset(dataset_list::DatasetList)::DataFrame
  rows = Dict[]
  for dataset in dataset_list.datasets
    logs_directory = dataset.logs_directory
    experiment_label =
      join([dataset.config[c] for c in dataset_list.config_labels], ",")

    log_files = Glob.glob("*_summary.json", logs_directory)
    if length(log_files) == 0
      error("No *_summary.json files found in $logs_directory.")
    end
    for filename in log_files
      log = JSON3.read(read(filename, String), FirstOrderLp.SolveLog)
      dict = solve_log_to_dict(log)
      dict[:experiment_label] = experiment_label
      for config_label in dataset_list.config_labels
        dict[Symbol(config_label)] = dataset.config[config_label]
      end
      push!(rows, dict)
    end
  end
  return DataFrame(map(NamedTuple, rows))
end

if length(ARGS) != 2
  error("Usage: process_json_to_csv.jl dataset_list_json output_csv")
end

dataset_list = JSON3.read(read(ARGS[1], String), DatasetList)
dataset = read_dataset(dataset_list)
CSV.write(ARGS[2], dataset)
