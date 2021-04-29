# Generates a Linear Programming model for an solving L1 SVM problem from the
# LIBSVM repository.

# Sample usage:
# $ julia --project=. generate_l1_lasso_lp.jl --input_filename=../data/E2006.train
#     --output_filename=/tmp/E2006.mps.gz --regularizer_weight=1.5

# The LP formulation for L1 LASSO has the form
# min \|X * beta - y\|_1 + regularizer_weight \|beta\|_1

import ArgParse
import JuMP
using SparseArrays, LinearAlgebra
include("utils.jl")

"""
Populates 'model' with the linear program model for solving the L1 SVM
problem with training set (X, y). The model is given by
  find sum w[i] + regularizer_weight * sum z[i]
  s.t. w >= 1 - diag(y) * [X 1] * beta
       w >= 0
       z >= beta
       z >= - beta
"""
function populate_l1_lasso_model(
  model::JuMP.Model,
  data::LearningData,
  regularizer_weight::Float64,
)
  n, d = size(data.feature_matrix)
  println("Generating a model with " * string(n) * " datapoints and " * string(d-1) * " features.")
  JuMP.@variable(model, beta[i = 1:d],)
  JuMP.@variable(model, w[i = 1:n])
  JuMP.@variable(model, z[i = 1:d])
  JuMP.@objective(model, Min, sum(w) + sum(z))
  JuMP.@constraint(model, z .>= beta)
  JuMP.@constraint(model, z .>= -beta)
  JuMP.@constraint(model, w .>= data.feature_matrix * beta - data.labels)
  JuMP.@constraint(model, w .>= data.labels - data.feature_matrix * beta)
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

function main()
  parsed_args = parse_command_line()

  filename = parsed_args["output_filename"]
  if !endswith(filename, ".gz")
    # We prefer SCIP's more mature writer, but the version packaged with Julia
    # can't write .gz files.
    backend = SCIP.Optimizer()
    model = JuMP.direct_model(backend)
  else
    model = JuMP.Model()
  end
  regularizer_weight = parsed_args["regularizer_weight"]
  input_filename = parsed_args["input_filename"]

  data = load_libsvm_file(input_filename)
  data = preprocess_learning_data(data)
  populate_l1_lasso_model(model, data, regularizer_weight)
  write_model_to_mps(model, parsed_args["output_filename"])
end

main()
