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

# Generates a Linear Programming model for solving an L1 SVM binary
# classification problem from the LIBSVM repository.

# Sample usage:
# $ julia --project=. generate_l1_svm_lp.jl --input_filename=../data/duke.tr
#     --output_filename=/tmp/duke.mps.gz --regularizer_weight=1.5

# The LP formulation for L1 SVM is taken from equation (5) in:
# Ji Zhu, Saharon Rosset, Trevor Hastie, and Rob Tibshirani. 2003. 1-norm
# support vector machines. In Proceedings of the 16th International Conference
# on Neural Information Processing Systems (NIPS'03). MIT Press, Cambridge, MA,
# USA, 49–56.
# https://papers.nips.cc/paper/2003/file/49d4b2faeb4b7b9e745775793141e2b2-Paper.pdf

import ArgParse
import JuMP
import SparseArrays
import LinearAlgebra

const SparseMatrixCSC = SparseArrays.SparseMatrixCSC
const sparse = SparseArrays.sparse
const norm = LinearAlgebra.norm
const Diagonal = LinearAlgebra.Diagonal
const nzrange = SparseArrays.nzrange

mutable struct SvmTrainingData
  feature_matrix::SparseMatrixCSC{Float64,Int64}
  labels::Vector{Float64}
end

"""
Populates 'model' with the linear program model for solving the L1 SVM
problem with training set (X, y). The model is given by
  min  sum w[i] + regularizer_weight * sum z[i]
  s.t. w >= 1 - diag(y) * [X 1] * beta
       w >= 0
       z >= beta
       z >= -beta
"""
function populate_libsvm_model(
  model::JuMP.Model,
  data::SvmTrainingData,
  regularizer_weight::Float64,
)
  n, d = size(data.feature_matrix)
  println("Generating a model with $n datapoints and $(d - 1) features.")
  JuMP.@variable(model, beta[i = 1:d])
  JuMP.@variable(model, w[i = 1:n], lower_bound = 0.0)
  JuMP.@variable(model, z[i = 1:d])
  JuMP.@objective(model, Min, sum(w) + regularizer_weight * sum(z))
  JuMP.@constraint(model, z .>= beta)
  JuMP.@constraint(model, z .>= -beta)
  JuMP.@constraint(
    model,
    w .>= (1.0 .- data.labels .* (data.feature_matrix * beta))
  )
  return model
end

"""
Defines parses and args.

# Returns
A dictionary with the values of the command-line arguments.
"""
function parse_command_line()
  arg_parse = ArgParse.ArgParseSettings()

  ArgParse.@add_arg_table! arg_parse begin
    "--input_filename"
    help = "The .train file containing the problem data."
    arg_type = String
    required = true

    "--output_filename"
    help = "Filename for the output .mps (or .mps.gz) model."
    arg_type = String
    required = true

    "--regularizer_weight"
    help = "Weight of the L1 regularizer."
    arg_type = Float64
    required = true
  end

  return ArgParse.parse_args(arg_parse)
end

"""
Loads a LIBSVM file and returns the loaded SvmTrainingData struct.
"""
function load_libsvm_file(file_name::String)
  open(file_name, "r") do io
    labels = Vector{Float64}()
    row_indices = Vector{Int64}()
    col_indices = Vector{Int64}()
    matrix_values = Vector{Float64}()

    row_index = 0
    found_label_one = false
    for line in eachline(io)
      row_index += 1
      split_line = split(line)

      label = parse(Float64, split_line[1])
      # This ensures that labels are 1 or -1. Different datasets use {-1, 1}, {0, 1}, and {1, 2}.
      if label == 1.0
        found_label_one = true
      else
        label = -1.0
      end
      push!(labels, label)

      for i in 2:length(split_line)
        push!(row_indices, row_index)
        matrix_coef = split(split_line[i], ":")
        push!(col_indices, parse(Int64, matrix_coef[1]))
        push!(matrix_values, parse(Float64, matrix_coef[2]))
      end
    end
    @assert found_label_one
    feature_matrix = sparse(row_indices, col_indices, matrix_values)
    return SvmTrainingData(feature_matrix, labels)
  end
end

function normalize_columns(feature_matrix::SparseMatrixCSC{Float64,Int64})
  norm_of_columns = vec(sqrt.(sum(t -> t^2, feature_matrix, dims = 1)))
  norm_of_columns[iszero.(norm_of_columns)] .= 1.0
  return feature_matrix * Diagonal(1.0 ./ norm_of_columns)
end

function remove_empty_columns(feature_matrix::SparseMatrixCSC{Float64,Int64})
  keep_cols = Vector{Int64}()
  for j in 1:size(feature_matrix, 2)
    if length(nzrange(feature_matrix, j)) > 0
      push!(keep_cols, j)
    end
  end
  return feature_matrix[:, keep_cols]
end

function add_intercept(feature_matrix::SparseMatrixCSC{Float64,Int64})
  return [sparse(ones(size(feature_matrix, 1))) feature_matrix]
end


function preprocess_training_data(result::SvmTrainingData)
  result.feature_matrix = remove_empty_columns(result.feature_matrix)
  result.feature_matrix = add_intercept(result.feature_matrix)
  result.feature_matrix = normalize_columns(result.feature_matrix)
  return result
end


function main()
  parsed_args = parse_command_line()

  filename = parsed_args["output_filename"]
  model = JuMP.Model()
  regularizer_weight = parsed_args["regularizer_weight"]
  input_filename = parsed_args["input_filename"]
  data = load_libsvm_file(input_filename)
  data = preprocess_training_data(data)
  populate_libsvm_model(model, data, regularizer_weight)
  JuMP.write_to_file(model, parsed_args["output_filename"])
end

main()
