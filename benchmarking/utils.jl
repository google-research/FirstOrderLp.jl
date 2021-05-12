import JuMP
import SCIP

"""
Writes 'model' to 'filename'. Supports writing a .mps file from a SCIP
direct mode backend, and .mps or .mps.gz files from a JuMP caching model.
"""
function write_model_to_mps(model::JuMP.Model, filename::AbstractString)
  backend = JuMP.backend(model)

  if isa(backend, SCIP.Optimizer)
    if endswith(filename, ".gz")
      error("The SCIP backend cannot write .gz files")
    end
    scip_ptr = backend.mscip.scip
    SCIP.@SC SCIP.SCIPwriteOrigProblem(
      scip_ptr[],
      filename,
      C_NULL,
      true, #genericnames
    )
  else
    # Not SCIP. Fall back to JuMPs less mature writer.
    JuMP.write_to_file(model, filename)
  end
end


"""
The following functions are used to process LIBSVM files.
"""
mutable struct LearningData
  feature_matrix::SparseMatrixCSC{Float64,Int64}
  labels::Vector{Float64}
end

"""
Loads a LIBSVM file into a LearningData struct.
"""
function load_libsvm_file(file_name::String; binary::Bool = false)
  open(file_name, "r") do io
    target = Array{Float64,1}()
    row_indicies = Array{Int64,1}()
    col_indicies = Array{Int64,1}()
    matrix_values = Array{Float64,1}()

    row_index = 0
    for line in eachline(io)
      row_index += 1
      split_line = split(line)
      if binary
        label = parse(Float64, split_line[1])
        # This ensures that labels are 1 or -1. Different dataset use {-1, 1}, {0, 1}, and {1, 2}.
        if abs(label - 1.0) < 1e-05
          label = 1.0
        else
          label = -1.0
        end
        push!(target, label)
      else
        push!(target, parse(Float64, split_line[1]))
      end
      for i in 2:length(split_line)
        push!(row_indicies, row_index)
        matrix_coef = split(split_line[i], ":")
        push!(col_indicies, parse(Int64, matrix_coef[1]))
        push!(matrix_values, parse(Float64, matrix_coef[2]))
      end
    end
    feature_matrix = sparse(row_indicies, col_indicies, matrix_values)
    return LearningData(feature_matrix, target)
  end
end

function normalize_columns(feature_matrix::SparseMatrixCSC{Float64,Int64})
  m = size(feature_matrix, 2)
  normalize_columns_by = ones(m)
  for j in 1:m
    col_vals = feature_matrix[:, j].nzval
    if length(col_vals) > 0
      normalize_columns_by[j] = 1.0 / norm(col_vals, 2)
    end
  end
  return feature_matrix * sparse(1:m, 1:m, normalize_columns_by)
end

function remove_empty_columns(feature_matrix::SparseMatrixCSC{Float64,Int64})
  keep_cols = Array{Int64,1}()
  for j in 1:size(feature_matrix, 2)
    if length(feature_matrix[:, j].nzind) > 0
      push!(keep_cols, j)
    end
  end
  return feature_matrix[:, keep_cols]
end

function add_intercept(feature_matrix::SparseMatrixCSC{Float64,Int64})
  return [sparse(ones(size(feature_matrix, 1))) feature_matrix]
end


function preprocess_learning_data(result::LearningData)
  result.feature_matrix = remove_empty_columns(result.feature_matrix)
  result.feature_matrix = add_intercept(result.feature_matrix)
  result.feature_matrix = normalize_columns(result.feature_matrix)
  return result
end
