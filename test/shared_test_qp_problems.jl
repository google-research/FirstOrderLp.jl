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

""" Returns a small test LP.
The LP:
  min 5 x_1 + 2 x_2 + x_3 +   x_4 - 14 s.t.
  2 x_1 +   x_2 + x_3 + 2 x_4  = 12
    x_1 +         x_3         >=  7
                  x_3 -   x_4 >=  1
  0 <= x_1 <= 2
  0 <= x_2 <= 4
  0 <= x_3 <= 6
  0 <= x_4 <= 3

Optimum solutions:
  Primal: x_1 = 1, x_2 = 0, x_3 = 6, x_4 = 2. Value: 5 + 0 + 6 + 2 - 14 = -1.
  Dual: [0.5, 4.0, 0.0]  Value: 6 + 28 - 3.5*6 - 14 = -1
"""
function example_lp()
  return FirstOrderLp.linear_programming_problem(
    [0.0, 0.0, 0.0, 0.0],  # variable_lower_bound
    [2.0, 4.0, 6.0, 3.0],  # variable_upper_bound
    [5.0, 2.0, 1.0, 1.0],  # objective_vector
    -14.0,                 # objective_constant
    [
      2.0 1.0 1.0 2.0
      1.0 0.0 1.0 0.0
      0.0 0.0 1.0 -1.0
    ],                     # constraint_matrix
    [12.0, 7.0, 1.0],      # right_hand_side
    1,                     # num_equalities
  )
end

""" Returns a one-variable LP with a no variable bounds.
The LP:
  min -x_1 s.t.
  -x_1 >= -2 (as a linear constraint)

Optimum solution:
  Primal: x_1 = 2, Value: -2.
  Dual: [1.0]  Value: -2.
"""
function example_lp_without_bounds()
  return FirstOrderLp.linear_programming_problem(
    [-Inf],                # variable_lower_bound
    [Inf],                 # variable_upper_bound
    [-1.0],                # objective_vector
    0.0,                   # objective_constant
    reshape([-1.0], 1, 1), # constraint_matrix
    [-2.0],                # right_hand_side
    0,                     # num_equalities
  )
end

""" Returns a small test QP.
The QP:
  min 2 x_1^2 + 0.5 x_2^2 - x_1 - x_2 s.t.
  x_1 + x_2  <= 1

  0 <= x_1 <= 1
  0 <= x_2 <= 1

Optimum solutions:
  Primal: x_1 = 0.2, x_2 = 0.8.
  Dual: [0.2]  Value: 0.04 + 0.32 - 0.2 - 0.8 = -0.6
"""
function example_qp()
  return FirstOrderLp.QuadraticProgrammingProblem(
    [0.0, 0.0],     # variable_lower_bound
    [1.0, 1.0],     # variable_upper_bound
    [
      4.0 0.0
      0.0 1.0
    ],              # objective_matrix
    [-1.0, -1.0],   # objective_vector
    -0.0,           # objective_constant
    [-1.0 -1.0],    # constraint_matrix
    [-1.0],         # right_hand_side
    0,              # num_equalities
  )
end

""" Returns a small test QP.
The QP:
  min 2 x_1^2 + 0.5 x_2^2 - x_1 - x_2 s.t.
  x_1 + x_2  <= 1

  0 <= x_1 <= 1
  0 <= x_2 <= 1

Optimum solutions:
  Primal: x_1 = 0.25, x_2 = 0.0.
  Dual: [0.0]  Value: 0.125 + 0 - 0.25 - 0 = -0.125
"""
function example_qp2()
  return FirstOrderLp.QuadraticProgrammingProblem(
    [0.0, 0.0],     # variable_lower_bound
    [1.0, 1.0],     # variable_upper_bound
    [
      4.0 0.0
      0.0 1.0
    ],              # objective_matrix
    [-1.0, 1.0],    # objective_vector
    -0.0,           # objective_constant
    [-1.0 -1.0],    # constraint_matrix
    [-1.0],         # right_hand_side
    0,              # num_equalities
  )
end


"""Returns a correlation clustering LP.
This is the LP for minimizing disagreements for correlation clustering for the
4-vertex graph
   1 - 3 - 4
   | /
   2
In integer solutions x_ij is 1 if i and j are in the same cluster and 0
otherwise. The 6 variables are in the order
 x_12, x_13, x_14, x_23, x_24, x_34.
For any distinct i,j,k there's a triangle inequality
  (1-x_ik) <= (1-x_ij) + (1-x_jk)
i.e.
  -x_ij - x_jk + x_ik >= -1.
For brevity we only include 3 out of the 12 possible triangle inequalities: two
needed in the optimal solution and 1 other."""
function example_cc_lp()
  return FirstOrderLp.linear_programming_problem(
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],     # variable_lower_bound
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],     # variable_upper_bound
    [-1.0, -1.0, 1.0, -1.0, 1.0, -1.0], # objective_vector
    4.0,                                # objective_constant
    [
      0.0 -1.0 1.0 0.0 0.0 -1.0
      0.0 0.0 0.0 -1.0 1.0 -1.0
      -1.0 -1.0 0.0 1.0 0.0 0.0
    ],                                  # constraint_matrix
    [-1.0, -1.0, -1.0],                 # right_hand_side
    0,                                  # num_equalities
  )
end

"""Returns another 4-vertex correlation clustering LP.

The variables are x_12, x_13, x_14, x_23, x_24, and x_34.
This time the graph is a star centered at vertex 1.
Only the three triangle inequalities that are needed are included."""
function example_cc_star_lp()
  return FirstOrderLp.linear_programming_problem(
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],    # variable_lower_bound
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],    # variable_upper_bound
    [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], # objective_vector
    3.0,                               # objective_constant
    [
      -1.0 -1.0 0.0 1.0 0.0 0.0
      -1.0 0.0 -1.0 0.0 1.0 0.0
      0.0 -1.0 -1.0 0.0 0.0 1.0
    ],      # constraint_matrix
    [-1.0, -1.0, -1.0],                # right_hand_side
    0,                                 # num_equalities
  )
end

""" Returns a small test LP with rows that are linearly dependent.
The LP:
  min x_1 + 2 x_2 + 3 x_3 +  4 x_4 s.t.
    x_1 +     x_2 +   x_3 +   x_4  == 2
    x_1 +     x_2 +   x_3 +   x_4  == 2
    x_1 +                     x_4  == 1
  0 <= x_1
  0 <= x_2
  0 <= x_3
  0 <= x_4

Optimum solution:
  Primal: x_1 = 1, x_2 = 1, x_3 = 0, x_4 = 0.
  Dual solution is not unique.
  Value: 1 + 2 = 3
"""
function example_lp_dependent_rows()
  return FirstOrderLp.linear_programming_problem(
    [0.0, 0.0, 0.0, 0.0],  # variable_lower_bound
    [Inf, Inf, Inf, Inf],  # variable_upper_bound
    [1.0, 2.0, 3.0, 4.0],  # objective_vector
    0.0,                   # objective_constant
    [
      1.0 1.0 1.0 1.0
      1.0 1.0 1.0 1.0
      1.0 0.0 0.0 1.0
    ],                     # constraint_matrix
    [2.0, 2.0, 1.0],       # right_hand_side
    3,                     # num_equalities
  )
end


""" Returns a small test LP that is primal infeasible.
The LP:
  min -x_1 + 0.5 x_2 s.t.
    -x_1 -    x_2   == 1
  0 <= x_1
  0 <= x_2
"""
function example_lp_easy_primal_infeasible()
  return FirstOrderLp.linear_programming_problem(
    [0.0, 0.0],           # variable_lower_bound
    [Inf, Inf],           # variable_upper_bound
    [1.0, 0.5],           # objective_vector
    0.0,                  # objective_constant
    [-1.0 -1.0],          # constraint_matrix
    [1.0],                # right_hand_side
    1,                    # num_equalities
  )
end

""" Returns a small test LP that is primal infeasible. Compared with
example_lp_easy_primal_infeasible() this is a hard problem. Note that
the nonnegativity constraints have no impact on feasibility.
The LP:
  min x_1 + 2 x_2 + 3 x_3 +  4 x_4 s.t.
    x_1 +     x_2                   == 1
              x_2 +   x_3           == 1
                      x_3 +    x_4  == 1
   x_1  +     x_2 +   x_3 +    x_4  == 2 + tol
  0 <= x_1
  0 <= x_2
  0 <= x_3
  0 <= x_4
"""
function example_lp_hard_primal_infeasible(tol::Float64)
  @assert tol > 0.0
  return FirstOrderLp.linear_programming_problem(
    [0.0, 0.0, 0.0, 0.0],      # variable_lower_bound
    [Inf, Inf, Inf, Inf],      # variable_upper_bound
    [1.0, 2.0, 3.0, 4.0],      # objective_vector
    0.0,                       # objective_constant
    [
      1.0 1.0 0.0 0.0
      0.0 1.0 1.0 0.0
      0.0 0.0 1.0 1.0
      1.0 1.0 1.0 1.0
    ],                         # constraint_matrix
    [1.0, 1.0, 1.0, 2 + tol],  # right_hand_side
    4,                         # num_equalities
  )
end

""" Returns a small test LP that is dual infeasible.
The LP:
  min -x_1 + 0.5 x_2 s.t.
    x_1 -     x_2   == 1
  0 <= x_1
  0 <= x_2
"""
function example_lp_dual_infeasible()
  return FirstOrderLp.linear_programming_problem(
    [0.0, 0.0],            # variable_lower_bound
    [Inf, Inf],            # variable_upper_bound
    [-1.0, 0.4],           # objective_vector
    0.0,                   # objective_constant
    [1.0 -2.0],            # constraint_matrix
    [1.0],                 # right_hand_side
    1,                     # num_equalities
  )
end
