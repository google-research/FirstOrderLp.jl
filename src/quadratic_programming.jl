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

"""
A QuadraticProgrammingProblem struct specifies a quadratic programming problem
with the following format:

```
minimize 1/2 x' * objective_matrix * x + objective_vector' * x
          + objective_constant

s.t. constraint_matrix[1:num_equalities, :] * x =
     right_hand_side[1:num_equalities]

     constraint_matrix[(num_equalities + 1):end, :] * x >=
     right_hand_side[(num_equalities + 1):end]

     variable_lower_bound <= x <= variable_upper_bound
```
The variable_lower_bound may contain `-Inf` elements and variable_upper_bound
may contain `Inf` elements when the corresponding variable bound is not present.
"""
mutable struct QuadraticProgrammingProblem
  """
  The vector of variable lower bounds.
  """
  variable_lower_bound::Vector{Float64}

  """
  The vector of variable upper bounds.
  """
  variable_upper_bound::Vector{Float64}

  """
  The symmetric and positive semidefinite matrix that defines the quadratic
  term in the objective.
  """
  objective_matrix::SparseMatrixCSC{Float64,Int64}

  """
  The linear coefficients of the objective function.
  """
  objective_vector::Vector{Float64}

  """
  The constant term of the objective function.
  """
  objective_constant::Float64

  """
  The matrix of coefficients in the linear constraints.
  """
  constraint_matrix::SparseMatrixCSC{Float64,Int64}

  """
  The vector of right-hand side values in the linear constraints.
  """
  right_hand_side::Vector{Float64}

  """
  The number of equalities in the problem. This value splits the rows of the
  constraint_matrix between the equality and inequality parts.
  """
  num_equalities::Int64
end

"""
Estimates the variable and constraint hardness for a first-order method given
a solution vector. By hardness we mean roughly how much does the constraint or
variable contribute to the worst-case runtime bound. This is useful for
debugging situations where the method is performing poorly.
"""
function print_variable_and_constraint_hardness(
  problem::QuadraticProgrammingProblem,
  primal_solution::AbstractVector{Float64},
  dual_solution::AbstractVector{Float64},
)

  row_norms = get_row_l2_norms(problem.constraint_matrix)
  constraint_hardness = row_norms .* abs.(dual_solution)
  print("Constraint hardness: ")
  Printf.@printf(
    "median_hardness=%f, mean_hardness=%f, quantile_99=%f, hardest=%f\n",
    median(constraint_hardness),
    mean(constraint_hardness),
    quantile(constraint_hardness, 0.99),
    maximum(constraint_hardness)
  )

  col_norms = get_col_l2_norms(problem.constraint_matrix)
  variable_hardness = col_norms .* abs.(primal_solution)
  print("Variable hardness: ")
  Printf.@printf(
    "median_hardness=%f, mean_hardness=%f, quantile_99=%f, hardest=%f\n",
    median(variable_hardness),
    mean(variable_hardness),
    quantile(variable_hardness, 0.99),
    maximum(variable_hardness)
  )
end

function get_row_l2_norms(matrix::SparseMatrixCSC{Float64,Int64})
  row_norm_squared = zeros(size(matrix, 1))
  nzval = nonzeros(matrix)
  rowval = rowvals(matrix)
  for i in 1:length(nzval)
    row_norm_squared[rowval[i]] += nzval[i]^2
  end

  return sqrt.(row_norm_squared)
end

function get_col_l2_norms(matrix::SparseMatrixCSC{Float64,Int64})
  col_norms = zeros(size(matrix, 2))
  for j in 1:size(matrix, 2)
    col_norms[j] = norm(nonzeros(matrix[:, j]), 2)
  end
  return col_norms
end


function get_row_l_inf_norms(matrix::SparseMatrixCSC{Float64,Int64})
  row_norm = zeros(size(matrix, 1))
  nzval = nonzeros(matrix)
  rowval = rowvals(matrix)
  for i in 1:length(nzval)
    row_norm[rowval[i]] = max(abs(nzval[i]), row_norm[rowval[i]])
  end

  return row_norm
end

function get_col_l_inf_norms(matrix::SparseMatrixCSC{Float64,Int64})
  col_norms = zeros(size(matrix, 2))
  for j in 1:size(matrix, 2)
    col_norms[j] = norm(nonzeros(matrix[:, j]), Inf)
    typeof(matrix[:, j])
  end
  return col_norms
end

"""
  print_problem_details(qp)

This is primarily useful for detecting when a problem is poorly conditioned and
needs rescaling.
"""
function print_problem_details(qp::QuadraticProgrammingProblem)
  println(
    "  There are ",
    size(qp.constraint_matrix, 2),
    " variables, ",
    size(qp.constraint_matrix, 1),
    " constraints (including ",
    qp.num_equalities,
    " equalities) and ",
    SparseArrays.nnz(qp.constraint_matrix),
    " nonzero coefficients.",
  )

  print("  Absolute value of nonzero constraint matrix elements: ")
  Printf.@printf(
    "largest=%f, smallest=%f, avg=%f\n",
    maximum(abs, nonzeros(qp.constraint_matrix)),
    minimum(abs, nonzeros(qp.constraint_matrix)),
    sum(abs, nonzeros(qp.constraint_matrix)) /
    length(nonzeros(qp.constraint_matrix))
  )

  col_norms = get_col_l_inf_norms(qp.constraint_matrix)
  row_norms = get_row_l_inf_norms(qp.constraint_matrix)

  print("  Constraint matrix, infinity norm: ")
  Printf.@printf(
    "max_col=%f, min_col=%f, max_row=%f, min_row=%f\n",
    maximum(col_norms),
    minimum(col_norms),
    maximum(row_norms),
    minimum(row_norms)
  )

  if length(nonzeros(qp.objective_matrix)) > 0
    print("  Absolute value of objective matrix elements: ")
    Printf.@printf(
      "largest=%f, smallest=%f, avg=%f\n",
      maximum(abs, nonzeros(qp.objective_matrix)),
      minimum(abs, nonzeros(qp.objective_matrix)),
      sum(abs, nonzeros(qp.objective_matrix)) /
      length(nonzeros(qp.constraint_matrix))
    )
  end

  print("  Absolute value of objective vector elements: ")
  Printf.@printf(
    "largest=%f, smallest=%f, avg=%f\n",
    maximum(abs, qp.objective_vector),
    minimum(abs, qp.objective_vector),
    sum(abs, qp.objective_vector) / length(qp.objective_vector)
  )

  print("  Absolute value of rhs vector elements: ")
  Printf.@printf(
    "largest=%f, smallest=%f, avg=%f\n",
    maximum(abs, qp.right_hand_side),
    minimum(abs, qp.right_hand_side),
    sum(abs, qp.right_hand_side) / length(qp.right_hand_side)
  )

  bound_gaps = qp.variable_upper_bound - qp.variable_lower_bound
  finite_bound_gaps = bound_gaps[isfinite.(bound_gaps)]

  print("  Gap between upper and lower bounds: ")
  Printf.@printf(
    "#finite=%i of %i, largest=%f, smallest=%f, avg=%f\n",
    length(finite_bound_gaps),
    length(bound_gaps),
    length(finite_bound_gaps) > 0 ? maximum(finite_bound_gaps) : NaN,
    length(finite_bound_gaps) > 0 ? minimum(finite_bound_gaps) : NaN,
    length(finite_bound_gaps) > 0 ?
    sum(finite_bound_gaps) / length(finite_bound_gaps) : NaN
  )
end

"""
Creates a QuadraticProgrammingProblem with the objective matrix set to zero.

# Arguments
- `variable_lower_bound::Vector{Float64}`: The lower bound of variables.
- `variable_upper_bound::Vector{Float64}`: The upper bound of variables.
- `objective_vector::Vector{Float64}`: The linear coefficients of
  objective function.
- `objective_constant::Float64`: The constant of objective function
- `variable_upper_bound::Vector{Float64}`: The constant of objective
  function.
- `constraint_matrix::Union{SparseMatrixCSC{Float64,Int64},Array{Float64,2}}`:
  The matrix of coefficients in the linear constraints.
- `right_hand_side::Vector{Float64}`: The vector of right-hand side values in
  the linear constraints.
- `num_equalities::Int64`: The number of equalities in the problem. This value
  splits the rows of the constraint_matrix between the equality and inequality
  parts.

"""
function linear_programming_problem(
  variable_lower_bound::Vector{Float64},
  variable_upper_bound::Vector{Float64},
  objective_vector::Vector{Float64},
  objective_constant::Float64,
  constraint_matrix::Union{SparseMatrixCSC{Float64,Int64},Array{Float64,2}},
  right_hand_side::Vector{Float64},
  num_equalities::Int64,
)

  num_variables = length(variable_lower_bound)
  objective_matrix = spzeros(num_variables, num_variables)
  return QuadraticProgrammingProblem(
    variable_lower_bound,
    variable_upper_bound,
    objective_matrix,
    objective_vector,
    objective_constant,
    sparse(constraint_matrix),
    right_hand_side,
    num_equalities,
  )
end

"""
Returns true if and only if the objective matrix is zero.
"""
function is_linear_programming_problem(problem::QuadraticProgrammingProblem)
  return nnz(problem.objective_matrix) == 0
end

"""
A ScaledQpProblem struct specifies a original quadratic programming problem,
a scaled quadratic programming problem, and the scaling vector, which requires
to satisfy the condition that:

orginal_qp = unscale_problem(scaled_qp, constraint_rescaling,variable_rescaling)
"""
mutable struct ScaledQpProblem
  original_qp::QuadraticProgrammingProblem
  scaled_qp::QuadraticProgrammingProblem
  constraint_rescaling::Vector{Float64}
  variable_rescaling::Vector{Float64}
end

equality_range(problem::QuadraticProgrammingProblem) = 1:problem.num_equalities

function inequality_range(problem::QuadraticProgrammingProblem)
  return (problem.num_equalities+1):size(problem.constraint_matrix, 1)
end
