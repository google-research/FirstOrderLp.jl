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
Check that the QuadraticProgrammingProblem is valid.
"""
function validate(p::QuadraticProgrammingProblem)
  error_found = false
  if length(p.variable_lower_bound) != length(p.variable_upper_bound)
    @info "$(length(p.variable_lower_bound)) == length(p.variable_lower_bound)
      != length(p.variable_upper_bound) = $(length(p.variable_upper_bound))"
    error_found = true
  end
  if length(p.variable_lower_bound) != length(p.objective_vector)
    @info "$(length(p.variable_lower_bound)) == length(p.variable_lower_bound)
      != length(p.objective_vector)= $(length(p.objective_vector))"
    error_found = true
  end
  if size(p.constraint_matrix, 1) != length(p.right_hand_side)
    @info "$(size(p.constraint_matrix,1)) == size(p.constraint_matrix,1)
      != length(p.right_hand_side)= $(length(p.right_hand_side))"
    error_found = true
  end
  if size(p.constraint_matrix, 2) != length(p.objective_vector)
    @info "$(size(p.constraint_matrix,2)) == size(p.constraint_matrix,2)
      != length(p.objective_vector)= $(length(p.objective_vector))"
    error_found = true
  end
  if size(p.objective_matrix) !=
     (length(p.objective_vector), length(p.objective_vector))
    @info "$(size(p.objective_matrix)) == size(p.objective_matrix)
      is not square with length $(length(p.objective_vector))"
    error_found = true
  end
  if sum(p.variable_lower_bound .== Inf) > 0
    @info "sum(p.variable_lower_bound .== Inf) ==
            $(sum(p.variable_lower_bound .== Inf))"
    error_found = true
  end
  if sum(p.variable_upper_bound .== -Inf) > 0
    @info "sum(p.variable_upper_bound .== -Inf) ==
            $(sum(p.variable_lupper_bound .== -Inf))"
    error_found = true
  end
  if any(isnan, p.variable_lower_bound) || any(isnan, p.variable_upper_bound)
    @info "NaN found in variable bounds of QuadraticProgrammingProblem."
    error_found = true
  end
  if any(!isfinite, p.right_hand_side)
    @info "NaN or Inf found in right hand side of QuadraticProgrammingProblem."
    error_found = true
  end
  if any(!isfinite, p.objective_vector)
    @info "NaN or Inf found in objective vector of QuadraticProgrammingProblem."
    error_found = true
  end
  if any(!isfinite, nonzeros(p.constraint_matrix))
    @info "NaN or Inf found in constraint matrix of
            QuadraticProgrammingProblem."
    error_found = true
  end
  if any(!isfinite, nonzeros(p.objective_matrix))
    @info "NaN or Inf found in objective matrix of QuadraticProgrammingProblem."
    error_found = true
  end

  if error_found
    error("Error found when validating QuadraticProgrammingProblem. See @info
    for details.")
  end

  return true
end

"""
Returns the l2 norm of each row or column of a matrix. The method rescales
the sum-of-squares computation by the largest absolute value if nonzero in order
to avoid overflow.

# Arguments
- `matrix::SparseMatrixCSC{Float64, Int64}`: a sparse matrix.
- `dimension::Int64`: the dimension we want to compute the norm over. Must be
  1 or 2.

# Returns
An array with the l2 norm of a matrix over the given dimension.
"""
function l2_norm(matrix::SparseMatrixCSC{Float64,Int64}, dimension::Int64)
  scale_factor = vec(maximum(abs.(matrix), dims = dimension))
  scale_factor[iszero.(scale_factor)] .= 1.0
  if dimension == 1
    scaled_matrix = matrix * Diagonal(1 ./ scale_factor)
    return scale_factor .* vec(sqrt.(sum(scaled_matrix .^ 2, dims = dimension)))
  end

  if dimension == 2
    scaled_matrix = Diagonal(1 ./ scale_factor) * matrix
    return scale_factor .* vec(sqrt.(sum(scaled_matrix .^ 2, dims = dimension)))
  end
end

"""
Removes the empty rows of a quadratic programming problem.

# Arguments
- `problem::QuadraticProgrammingProblem`: The input quadratic programming
  problem. This is modified to store the transformed problem.
"""
function remove_empty_rows(problem::QuadraticProgrammingProblem)
  seen_row = falses(size(problem.constraint_matrix, 1))
  for row in SparseArrays.rowvals(problem.constraint_matrix)
    seen_row[row] = true
  end
  empty_rows = findall(.!seen_row)

  for row in empty_rows
    if row > problem.num_equalities && problem.right_hand_side[row] > 0.0
      error("The problem is infeasible.")
    elseif row <= problem.num_equalities && problem.right_hand_side[row] != 0.0
      error("The problem is infeasible.")
    end
  end

  if !isempty(empty_rows)
    problem.constraint_matrix = problem.constraint_matrix[seen_row, :]
    problem.right_hand_side = problem.right_hand_side[seen_row]
    num_empty_equalities = sum(empty_rows .<= problem.num_equalities)
    problem.num_equalities -= num_empty_equalities
  end
  return empty_rows
end

"""
Removes the empty columns of a quadratic programming problem.

# Arguments
- `problem::QuadraticProgrammingProblem`: The input linear programming problem.
  This is modified to store the transformed problem. `objective_constant` is
  updated assuming that the eliminated variables are assigned to the best
  possible values.
"""
function remove_empty_columns(problem::QuadraticProgrammingProblem)

  # TODO: Adapt the implementation for quadratic objectives.
  @assert iszero(problem.objective_matrix)
  is_empty_column = [
    isempty(nzrange(problem.constraint_matrix, col)) for
    col in 1:size(problem.constraint_matrix, 2)
  ]
  empty_columns = findall(is_empty_column)
  if isempty(empty_columns)
    return empty_columns
  end

  for col in empty_columns
    objective_coef = problem.objective_vector[col]
    if objective_coef >= 0
      problem.objective_constant +=
        problem.variable_lower_bound[col] * objective_coef
    else
      problem.objective_constant +=
        problem.variable_upper_bound[col] * objective_coef
    end
  end
  is_non_empty = .!is_empty_column
  problem.constraint_matrix = problem.constraint_matrix[:, is_non_empty]
  problem.objective_vector = problem.objective_vector[is_non_empty]
  problem.variable_lower_bound = problem.variable_lower_bound[is_non_empty]
  problem.variable_upper_bound = problem.variable_upper_bound[is_non_empty]
  problem.objective_matrix =
    problem.objective_matrix[is_non_empty, is_non_empty]
  return empty_columns
end

"""
Modifies qp by transforming any finite variable bounds into linear constraints.
"""
function transform_bounds_into_linear_constraints(
  qp::FirstOrderLp.QuadraticProgrammingProblem,
)
  finite_lower_bound_indices = findall(isfinite.(qp.variable_lower_bound))
  finite_upper_bound_indices = findall(isfinite.(qp.variable_upper_bound))

  row_indices =
    1:(length(finite_lower_bound_indices)+length(finite_upper_bound_indices))
  column_indices = [finite_lower_bound_indices; finite_upper_bound_indices]
  nonzeros = [
    ones(length(finite_lower_bound_indices))
    -ones(length(finite_upper_bound_indices))
  ]

  identity_block = SparseArrays.sparse(
    row_indices,
    column_indices,
    nonzeros,
    length(row_indices),
    length(qp.variable_lower_bound),
  )
  qp.constraint_matrix = [qp.constraint_matrix; identity_block]
  qp.right_hand_side = [
    qp.right_hand_side
    qp.variable_lower_bound[finite_lower_bound_indices]
    -qp.variable_upper_bound[finite_upper_bound_indices]
  ]
  qp.variable_lower_bound .= -Inf
  qp.variable_upper_bound .= Inf
  return
end

struct PresolveInfo
  original_primal_size::Int64
  original_dual_size::Int64
  empty_rows::Vector{Int64}
  empty_columns::Vector{Int64}
  variable_lower_bound::Vector{Float64}
  variable_upper_bound::Vector{Float64}
end

"""
Presolve of a quadratic program. The quadratic program modified in place.
Returns a data structure with information allowing the presolve to be `undone`.
"""
function presolve(
  qp::QuadraticProgrammingProblem;
  verbosity::Int64 = 1,
  transform_bounds::Bool = false,
)
  saved_variable_lower_bound = copy(qp.variable_lower_bound)
  saved_variable_upper_bound = copy(qp.variable_upper_bound)

  original_dual_size, original_primal_size = size(qp.constraint_matrix)
  empty_rows = FirstOrderLp.remove_empty_rows(qp)
  # TODO: Remove this check after remove_empty_columns supports
  # quadratic objectives.
  if iszero(qp.objective_matrix)
    empty_columns = FirstOrderLp.remove_empty_columns(qp)
  else
    empty_columns = Array{Int64,1}()
  end

  if verbosity >= 1
    FirstOrderLp.check_for_singleton_constraints(qp)
  end
  # TODO: Write function check_for_singleton_variables(lp)

  if transform_bounds
    FirstOrderLp.transform_bounds_into_linear_constraints(qp)
  end

  return PresolveInfo(
    original_primal_size,
    original_dual_size,
    empty_rows,
    empty_columns,
    saved_variable_lower_bound,
    saved_variable_upper_bound,
  )
end

function check_for_singleton_constraints(qp::QuadraticProgrammingProblem)
  num_single = 0
  num_nz_by_row = zeros(Int64, size(qp.constraint_matrix, 1))
  for row_ind in SparseArrays.rowvals(qp.constraint_matrix)
    num_nz_by_row[row_ind] += 1
  end

  num_single = sum(num_nz_by_row .== 1)
  if num_single > 0
    println("$num_single constraints involving exactly a single variable")
    # TODO: Eliminate constraints involving a single variable by
    # replacing with bounds variables or by fixing the value.
  end
end

"""
Given a solution to the preprocessed problem this function recovers a solution
to the original problem.

# Arguments
- `solution::Vector{Float64}`: The solution after preprocessing.
- `empty_indicies::Vector{Int64}`: Indicies corresponding to portions of the
solution that were eliminated in preprocessing. When we recover the original
solution these indicies will have zero values.
- `original_size::Int64`: Size of the solution vector before preprocessing.
"""
function recover_original_solution(
  solution::Vector{Float64},
  empty_indicies::Vector{Int64},
  original_size::Int64,
)
  # construct a list of nonempty indicies
  nonempty_indicies_binary_list = trues(original_size)
  nonempty_indicies_binary_list[empty_indicies] .= false
  nonempty_indicies = findall(nonempty_indicies_binary_list)

  original_solution = zeros(original_size)
  original_solution[nonempty_indicies] = solution[1:length(nonempty_indicies)]

  return original_solution
end

function undo_presolve(
  presolve_info::PresolveInfo,
  primal_solution::Vector{Float64},
  dual_solution::Vector{Float64},
)
  # recover primal solution
  primal_solution = FirstOrderLp.recover_original_solution(
    primal_solution,
    presolve_info.empty_columns,
    presolve_info.original_primal_size,
  )

  projection!(
    primal_solution,
    presolve_info.variable_lower_bound,
    presolve_info.variable_upper_bound,
  )

  # recover dual solution
  dual_solution = FirstOrderLp.recover_original_solution(
    dual_solution,
    presolve_info.empty_rows,
    presolve_info.original_dual_size,
  )
  return primal_solution, dual_solution
end


"""
Rescales a quadratic programming problem by dividing each row and column of the
constraint matrix by the sqrt its respective L2 norm, adjusting the other
problem data accordingly.

# Arguments
- `problem::QuadraticProgrammingProblem`: The input quadratic programming
  problem. This is modified to store the transformed problem.

# Returns

A tuple of vectors `constraint_rescaling`, `variable_rescaling` such that
the original problem is recovered by
`unscale_problem(problem, constraint_rescaling, variable_rescaling)`.
"""
function l2_norm_rescaling(problem::QuadraticProgrammingProblem)
  num_constraints, num_variables = size(problem.constraint_matrix)

  norm_of_rows = vec(l2_norm(problem.constraint_matrix, 2))
  norm_of_columns = vec(l2_norm(problem.constraint_matrix, 1))

  norm_of_rows[iszero.(norm_of_rows)] .= 1.0
  norm_of_columns[iszero.(norm_of_columns)] .= 1.0

  column_rescale_factor = sqrt.(norm_of_columns)
  row_rescale_factor = sqrt.(norm_of_rows)
  scale_problem(problem, row_rescale_factor, column_rescale_factor)

  return row_rescale_factor, column_rescale_factor
end

"""
Uses a modified Ruiz rescaling algorithm to rescale the matrix M=[Q,A';A,0]
where Q is objective_matrix and A is constraint_matrix, and returns the
cumulative scaling vectors. More details of Ruiz rescaling algorithm can be
found at: http://www.numerical.rl.ac.uk/reports/drRAL2001034.pdf.

In the p=Inf case, both matrices approach having all row and column LInf norms
of M equal to 1 as the number of iterations goes to infinity. This convergence
is fast (linear).

In the p=2 case, the goal is all row L2 norms of [Q,A'] equal to 1 and all row
L2 norms of A equal to sqrt(num_variables/(num_constraints+num_variables))
for QP, and all row L2 norms of A equal to
sqrt(num_variables/num_constraints) for LP. Having a different
goal for the row and col norms is required since the sum of squares of the
entries of the A matrix is the same when the sum is grouped by rows or grouped
by columns. In particular, for the LP case, all L2 norms of A must be
sqrt(num_variables/num_constraints) when all row L2 norm of [Q,A'] equal to 1.

The Ruiz rescaling paper (link above) only analyzes convergence in the p < Inf
case when the matrix is square, and it does not preserve the symmetricity of
the matrix, and that is why we need to modify it for p=2 case.

TODO: figure out when this converges.

# Arguments
- `problem::QuadraticProgrammingProblem`: the quadratic programming problem.
  This is modified to store the transformed problem.
- `num_iterations::Int64` the number of iterations to run Ruiz rescaling
  algorithm. Must be positive.
- `p::Float64`: which norm to use. Must be 2 or Inf.

# Returns

A tuple of vectors `constraint_rescaling`, `variable_rescaling` such that
the original problem is recovered by
`unscale_problem(problem, constraint_rescaling, variable_rescaling)`.
"""
function ruiz_rescaling(
  problem::QuadraticProgrammingProblem,
  num_iterations::Int64,
  p::Float64 = Inf,
)
  num_constraints, num_variables = size(problem.constraint_matrix)
  cum_constraint_rescaling = ones(num_constraints)
  cum_variable_rescaling = ones(num_variables)

  for i in 1:num_iterations
    constraint_matrix = problem.constraint_matrix
    objective_matrix = problem.objective_matrix

    if p == Inf
      variable_rescaling = vec(
        sqrt.(
          max.(
            maximum(abs.(constraint_matrix), dims = 1),
            maximum(abs.(objective_matrix), dims = 1),
          ),
        ),
      )
    else
      @assert p == 2
      variable_rescaling = vec(
        sqrt.(
          sqrt.(
            l2_norm(constraint_matrix, 1) .^ 2 +
            l2_norm(objective_matrix, 1) .^ 2,
          ),
        ),
      )
    end
    variable_rescaling[iszero.(variable_rescaling)] .= 1.0

    if num_constraints == 0
      constraint_rescaling = Float64[]
    else
      if p == Inf
        constraint_rescaling =
          vec(sqrt.(maximum(abs.(constraint_matrix), dims = 2)))
      else
        @assert p == 2
        norm_of_rows = vec(l2_norm(problem.constraint_matrix, 2))

        # If the columns all have norm 1 and the row norms are equal they should
        # equal sqrt(num_variables/num_constraints) for LP, and roughly
        # sqrt(num_variables / (num_constraints + num_variables) for QP.
        if iszero(problem.objective_matrix)
          target_row_norm = sqrt(num_variables / num_constraints)
        else
          target_row_norm =
            sqrt(num_variables / (num_constraints + num_variables))
        end
        constraint_rescaling = vec(sqrt.(norm_of_rows / target_row_norm))
      end
      constraint_rescaling[iszero.(constraint_rescaling)] .= 1.0
    end
    scale_problem(problem, constraint_rescaling, variable_rescaling)

    cum_constraint_rescaling .*= constraint_rescaling
    cum_variable_rescaling .*= variable_rescaling
  end

  return cum_constraint_rescaling, cum_variable_rescaling
end

"""
Rescales `problem` in place. If we let `D = diag(cum_variable_rescaling)` and
`E = diag(cum_constraint_rescaling)`, then `problem` is modified such that:

    objective_matrix = D^-1 objective_matrix D^-1
    objective_vector = D^-1 objective_vector
    objective_constant = objective_constant
    variable_lower_bound = D variable_lower_bound
    variable_upper_bound = D variable_upper_bound
    constraint_matrix = E^-1 constraint_matrix D^-1
    right_hand_side = E^-1 right_hand_side

The scaling vectors should not contain zero.
"""
function scale_problem(
  problem::QuadraticProgrammingProblem,
  constraint_rescaling::Vector{Float64},
  variable_rescaling::Vector{Float64},
)
  @assert all(!iszero, constraint_rescaling)
  @assert all(!iszero, variable_rescaling)
  problem.objective_vector ./= variable_rescaling
  problem.objective_matrix =
    Diagonal(1 ./ variable_rescaling) *
    problem.objective_matrix *
    Diagonal(1 ./ variable_rescaling)
  problem.variable_upper_bound .*= variable_rescaling
  problem.variable_lower_bound .*= variable_rescaling
  problem.right_hand_side ./= constraint_rescaling
  problem.constraint_matrix =
    Diagonal(1 ./ constraint_rescaling) *
    problem.constraint_matrix *
    Diagonal(1 ./ variable_rescaling)
  return
end

"""
Recovers the original problem from the scaled problem and the scaling vectors
in place. The inverse of `scale_problem`. This function should be only used for
testing.
"""
function unscale_problem(
  problem::QuadraticProgrammingProblem,
  constraint_rescaling::Vector{Float64},
  variable_rescaling::Vector{Float64},
)
  scale_problem(problem, 1 ./ constraint_rescaling, 1 ./ variable_rescaling)
  return
end


"""
    row_permute_in_place(matrix::SparseMatrixCSC{Float64, Int64},
                         old_row_to_new::Vector{Int64})

Permutes the rows of `matrix` in place (without allocating a new matrix)
according to the map `old_row_to_new`. Assumes and does not verify that
`old_row_to_new` is a permutation.
"""
function row_permute_in_place(
  matrix::SparseMatrixCSC{Float64,Int64},
  old_row_to_new::Vector{Int64},
)
  coefficients = SparseArrays.nonzeros(matrix)
  row_indices = SparseArrays.rowvals(matrix)
  row_coef_tuples = Tuple{Int64,Float64}[]
  for col in 1:size(matrix, 2)
    nonzero_range = nzrange(matrix, col)
    empty!(row_coef_tuples)
    sizehint!(row_coef_tuples, length(nonzero_range))
    for index_in_matrix in nonzero_range
      new_row = old_row_to_new[row_indices[index_in_matrix]]
      push!(row_coef_tuples, (new_row, coefficients[index_in_matrix]))
    end
    # SparseMatrixCSC requires row indices to be sorted within a column. So we
    # do this sort and then replace the terms in the column.
    sort!(row_coef_tuples, by = t -> t[1])
    for (index_in_column, index_in_matrix) in enumerate(nonzero_range)
      new_row, new_coefficient = row_coef_tuples[index_in_column]
      row_indices[index_in_matrix] = new_row
      coefficients[index_in_matrix] = new_coefficient
    end
  end
  return
end


"""Preprocesses the original problem, and returns a ScaledQpProblem struct.
Applies L_inf Ruiz rescaling for `l_inf_ruiz_iterations` iterations. If
`l2_norm_rescaling` is true, applies L2 norm rescaling. `problem` is not
modified.
"""
function rescale_problem(
  l_inf_ruiz_iterations::Int,
  l2_norm_rescaling::Bool,
  verbosity::Int64,
  original_problem::QuadraticProgrammingProblem,
)
  problem = deepcopy(original_problem)
  if verbosity >= 4
    println("Problem before rescaling:")
    print_problem_details(original_problem)
  end

  num_constraints, num_variables = size(problem.constraint_matrix)
  constraint_rescaling = ones(num_constraints)
  variable_rescaling = ones(num_variables)

  if l_inf_ruiz_iterations > 0
    con_rescale, var_rescale =
      FirstOrderLp.ruiz_rescaling(problem, l_inf_ruiz_iterations, Inf)
    constraint_rescaling .*= con_rescale
    variable_rescaling .*= var_rescale
  end

  if l2_norm_rescaling
    con_rescale, var_rescale = FirstOrderLp.l2_norm_rescaling(problem)
    constraint_rescaling .*= con_rescale
    variable_rescaling .*= var_rescale
  end

  scaled_problem = ScaledQpProblem(
    original_problem,
    problem,
    constraint_rescaling,
    variable_rescaling,
  )

  if verbosity >= 3
    if l_inf_ruiz_iterations == 0 && !l2_norm_rescaling
      println("No rescaling.")
    else
      print("Problem after rescaling ")
      print("(Ruiz iterations = $l_inf_ruiz_iterations, ")
      println("l2_norm_rescaling = $l2_norm_rescaling):")
      print_problem_details(scaled_problem.scaled_qp)
    end
  end

  return scaled_problem
end
