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

@testset "Preprocessing of the original LP" begin
  @testset "Computing l2 norm of a matrix over the given dimension." begin
    matrix = sparse([3.0 0.0 -4.0; 4.0 3.0 0.0])
    @test FirstOrderLp.l2_norm(matrix, 1) ≈ [5.0, 3.0, 4.0] atol = 1.0e-10
    @test FirstOrderLp.l2_norm(matrix, 2) ≈ [5.0, 5.0] atol = 1.0e-10
  end

  @testset "remove_empty_rows with empty inequality constraint" begin
    problem = FirstOrderLp.linear_programming_problem(
      [0.0, 0.0],                   # variable_lower_bound
      [1.0, 2.0],                   # variable_upper_bound
      [1.0, 2.0],                   # objective_vector
      0.0,                          # objective_constant
      [
        2.0 0.0
        1.0 0.0
        0.0 0.0
      ],                            # constraint_matrix
      [1.0, 1.0, 0.0],              # right_hand_side
      1,                            # num_equalities
    )
    FirstOrderLp.remove_empty_rows(problem)
    test_fields_equal(
      problem,
      FirstOrderLp.linear_programming_problem(
        [0.0, 0.0],               # variable_lower_bound
        [1.0, 2.0],               # variable_upper_bound
        [1.0, 2.0],               # objective_vector
        0.0,                      # objective_constant
        [
          2.0 0.0
          1.0 0.0
        ],                        # constraint_matrix
        [1.0, 1.0],               # right_hand_side
        1,                        # num_equalities
      ),
    )
  end

  @testset "remove_empty_rows with empty equality constraint" begin
    problem = FirstOrderLp.linear_programming_problem(
      [0.0, 0.0],                   # variable_lower_bound
      [1.0, 2.0],                   # variable_upper_bound
      [1.0, 2.0],                   # objective_vector
      0.0,                          # objective_constant
      [
        0.0 0.0
        1.0 0.0
        1.0 0.0
      ],                            # constraint_matrix
      [0.0, 1.0, 0.0],              # right_hand_side
      1,                            # num_equalities
    )
    FirstOrderLp.remove_empty_rows(problem)
    test_fields_equal(
      problem,
      FirstOrderLp.linear_programming_problem(
        [0.0, 0.0],               # variable_lower_bound
        [1.0, 2.0],               # variable_upper_bound
        [1.0, 2.0],               # objective_vector
        0.0,                      # objective_constant
        [
          1.0 0.0
          1.0 0.0
        ],                        # constraint_matrix
        [1.0, 0.0],               # right_hand_side
        0,                        # num_equalities
      ),
    )
  end

  @testset "remove_empty_rows errors if an empty row in an inequality constraint
            has a positive right-hand side" begin
    problem = FirstOrderLp.linear_programming_problem(
      [0.0, 0.0],                   # variable_lower_bound
      [1.0, 2.0],                   # variable_upper_bound
      [1.0, 2.0],                   # objective_vector
      0.0,                          # objective_constant
      [
        1.0 0.0
        1.0 0.0
        0.0 0.0
      ],                            # constraint_matrix
      [1.0, 1.0, 1.0],              # right_hand_side
      1,                            # num_equalities
    )
    @test_throws ErrorException FirstOrderLp.remove_empty_rows(problem)
  end

  @testset "remove_empty_rows errors if an empty row in an equality constraint
            has a non-zero right-hand side" begin
    problem = FirstOrderLp.linear_programming_problem(
      [0.0, 0.0],                   # variable_lower_bound
      [1.0, 2.0],                   # variable_upper_bound
      [1.0, 2.0],                   # objective_vector
      0.0,                          # objective_constant
      [
        0.0 0.0
        1.0 0.0
        0.0 1.0
      ],                            # constraint_matrix
      [1.0, 1.0, 1.0],              # right_hand_side
      1,                            # num_equalities
    )
    @test_throws ErrorException FirstOrderLp.remove_empty_rows(problem)
  end

  @testset "remove_empty_columns with removed variable at lower bound" begin
    problem = FirstOrderLp.linear_programming_problem(
      [-1.0, -1.0],                 # variable_lower_bound
      [2.0, 2.0],                   # variable_upper_bound
      [3.0, 2.0],                   # objective_vector
      0.0,                          # objective_constant
      [
        0.0 1.0
        0.0 -1.0
      ],                            # constraint_matrix
      [1.0, 1.0],                   # right_hand_side
      0,                            # num_equalities
    )
    FirstOrderLp.remove_empty_columns(problem)
    test_fields_equal(
      problem,
      FirstOrderLp.linear_programming_problem(
        [-1.0],                       # variable_lower_bound
        [2.0],                        # variable_upper_bound
        [2.0],                        # objective_vector
        -3.0,                         # objective_constant
        [[1.0 -1.0]';],               # constraint_matrix
        [1.0, 1.0],                   # right_hand_side
        0,                            # num_equalities
      ),
    )
  end

  @testset "remove_empty_columns with removed variable at upper bound" begin
    problem = FirstOrderLp.linear_programming_problem(
      [-1.0, -1.0],                 # variable_lower_bound
      [2.0, 2.0],                   # variable_upper_bound
      [-3.0, 2.0],                  # objective_vector
      0.0,                          # objective_constant
      [
        0.0 1.0
        0.0 -1.0
      ],                            # constraint_matrix
      [1.0, 1.0],                   # right_hand_side
      0,                            # num_equalities
    )
    FirstOrderLp.remove_empty_columns(problem)
    test_fields_equal(
      problem,
      FirstOrderLp.linear_programming_problem(
        [-1.0],                       # variable_lower_bound
        [2.0],                        # variable_upper_bound
        [2.0],                        # objective_vector
        -6.0,                         # objective_constant
        [[1.0 -1.0]';],               # constraint_matrix
        [1.0, 1.0],                   # right_hand_side
        0,                            # num_equalities
      ),
    )
  end

  @testset "recover_original_solution" begin
    solution = [1.0, 1.0, 1.0, 5.0]
    empty_indicies = [1, 4]
    original_size = 5
    original_solution = FirstOrderLp.recover_original_solution(
      solution,
      empty_indicies,
      original_size,
    )
    @test original_solution == [0.0, 1.0, 1.0, 0.0, 1.0]
  end

  @testset "presolve" begin
    problem = FirstOrderLp.linear_programming_problem(
      [0.0, 0.0, 1.0],              # variable_lower_bound
      [1.0, 2.0, 2.0],              # variable_upper_bound
      [1.0, 2.0, 0.0],              # objective_vector
      0.0,                          # objective_constant
      [
        1.0 1.0 0.0
        1.0 -1.0 0.0
        0.0 0.0 0.0
      ],                            # constraint_matrix
      [1.0, 1.0, 0.0],              # right_hand_side
      1,                            # num_equalities
    )
    presolve_info = FirstOrderLp.presolve(problem)
    primal_solution, dual_solution =
      FirstOrderLp.undo_presolve(presolve_info, [1.0, 0.0], [1.0, 1.0])
    @test primal_solution == [1.0, 0.0, 1.0]
    @test dual_solution == [1.0, 1.0, 0.0]
  end

  @testset "Presolve doesn't delete empty columns for QP" begin
    problem = FirstOrderLp.QuadraticProgrammingProblem(
      [0.0, 0.0, 0.0],              # variable_lower_bound
      [1.0, 2.0, 1.0],              # variable_upper_bound
      [
        4.0 2.0 0.0
        2.0 1.0 0.0
        0.0 0.0 1.0
      ],                            # objective_matrix
      [1.0, 2.0, 1.0],              # objective_vector
      0.0,                          # objective_constant
      [
        1.0 1.0 0.0
        1.0 -1.0 0.0
        1.0 0.0 0.0
      ],                            # constraint matrix
      [1.0, 1.0, 2.0],              # right_hand_side
      1,                            # num_equalities
    )
    presolve_info = FirstOrderLp.presolve(problem)
    @test size(problem.constraint_matrix) == (3, 3)
  end

  @testset "lp_norm_rescaling for LP" begin
    problem = FirstOrderLp.linear_programming_problem(
      [0.0, 0.0],                   # variable_lower_bound
      [1.0, 2.0],                   # variable_upper_bound
      [1.0, 2.0],                   # objective_vector
      0.0,                          # objective_constant
      [
        1.0 1.0
        1.0 -1.0
        1.0 0.0
      ],                            # constraint_matrix
      [1.0, 1.0, 2.0],              # right_hand_side
      1,                            # num_equalities
    )
    FirstOrderLp.l2_norm_rescaling(problem)
    # Columns are rescaled by [3^(-1/4), 2^(-1/4)]
    # Rows are rescaled by [2^(-1/4), 2^(-1/4), 1.0]
    test_fields_approx_equal(
      problem,
      FirstOrderLp.linear_programming_problem(
        [0.0, 0.0],                         # variable_lower_bound
        [1.0 * (3)^(1/4), 2.0 * (2)^(1/4)],                         # variable_upper_bound
        [1.0 / (3)^(1/4), 2.0 / (2)^(1/4)],                         # objective_vector
        0.0,                                # objective_constant
        [
          (2*3)^(-1/4) (2*2)^(-1/4)
          (2 * 3)^(-1/4) -(2*2)^(-1/4)
          (3)^(-1/4) 0.0
        ],                                  # constraint_matrix
        [(2)^(-1/4), (2)^(-1/4), 2.0],    # right_hand_side
        1,                                  # num_equalities
      ),
    )
  end

  @testset "Ruiz-Rescaling for LP" begin
    problem = FirstOrderLp.linear_programming_problem(
      [0.0, 0.0],                   # variable_lower_bound
      [1.0, 2.0],                   # variable_upper_bound
      [1.0, 2.0],                   # objective_vector
      0.0,                          # objective_constant
      [
        1.0 3.0
        1.0 -2.0
        2.0 0.0
      ],                            # constraint_matrix
      [1.0, 1.0, 2.0],              # right_hand_side
      1,                            # num_equalities
    )
    original_problem = deepcopy(problem)
    cum_constraint_rescaling, cum_variable_rescaling =
      FirstOrderLp.ruiz_rescaling(problem, 1)
    test_fields_approx_equal(
      problem,
      FirstOrderLp.linear_programming_problem(
        [0.0, 0.0],                           # variable_lower_bound
        [sqrt(2), 2.0 * sqrt(3)],             # variable_upper_bound
        [1.0 / sqrt(2), 2.0 / sqrt(3)],       # objective_vector
        0.0,                                  # objective_constant
        [
          1/sqrt(6) 1.0
          0.5 -sqrt(2)/sqrt(3)
          1.0 0.0
        ],                                    # constraint_matrix
        [1 / sqrt(3), 1 / sqrt(2), sqrt(2)],  # right_hand_side
        1,                                    # num_equalities
      ),
    )
    @test cum_variable_rescaling ≈ [2^0.5, 3^0.5]
    @test cum_constraint_rescaling ≈ [3^0.5, 2^0.5, 2^0.5]
    FirstOrderLp.unscale_problem(
      problem,
      cum_constraint_rescaling,
      cum_variable_rescaling,
    )
    test_fields_approx_equal(problem, original_problem)
  end

  @testset "Convergence of Ruiz-Rescaling for LP" begin
    problem = FirstOrderLp.linear_programming_problem(
      [0.0, 0.0],                   # variable_lower_bound
      [1.0, 2.0],                   # variable_upper_bound
      [1.0, 2.0],                   # objective_vector
      0.0,                          # objective_constant
      [
        1.0 3.0
        1.0 -2.0
        2.0 0.0
      ],                            # constraint_matrix
      [1.0, 1.0, 3.0],              # right_hand_side
      1,                            # num_equalities
    )
    original_problem = deepcopy(problem)
    cum_constraint_rescaling, cum_variable_rescaling =
      FirstOrderLp.ruiz_rescaling(problem, 30)

    constraint_matrix = problem.constraint_matrix

    inf_norm_of_columns = vec(sqrt.(maximum(abs.(constraint_matrix), dims = 1)))
    @test all(inf_norm_of_columns .≈ 1)
    inf_norm_of_rows = vec(sqrt.(maximum(abs.(constraint_matrix), dims = 2)))
    @test all(inf_norm_of_rows .≈ 1)
    FirstOrderLp.unscale_problem(
      problem,
      cum_constraint_rescaling,
      cum_variable_rescaling,
    )
    test_fields_approx_equal(problem, original_problem)
  end

  @testset "rescale_problem for LP" begin
    problem = FirstOrderLp.linear_programming_problem(
      [0.0, 0.0],                   # variable_lower_bound
      [1.0, 2.0],                   # variable_upper_bound
      [1.0, 2.0],                   # objective_vector
      0.0,                          # objective_constant
      [
        1.0 3.0
        1.0 -2.0
        2.0 0.0
      ],                            # constraint_matrix
      [1.0, 1.0, 3.0],              # right_hand_side
      1,                            # num_equalities
    )
    scaled_problem = FirstOrderLp.rescale_problem(
      10, # l_inf_ruiz_iterations
      true, # l2_norm_rescaling
      0, # verbosity
      problem,
    )
    # The first argument is updated in place.
    FirstOrderLp.unscale_problem(
      scaled_problem.scaled_qp,
      scaled_problem.constraint_rescaling,
      scaled_problem.variable_rescaling,
    )

    test_fields_approx_equal(
      scaled_problem.scaled_qp,
      scaled_problem.original_qp,
    )
  end

  @testset "Ruiz-Rescaling for QP" begin
    problem = FirstOrderLp.QuadraticProgrammingProblem(
      [-Inf, -2.0],                 # variable_lower_bound
      [1.0, 2.0],                   # variable_upper_bound
      [4.0 2.0; 2.0 1.0],           # objective_matrix
      [1.0, 2.0],                   # objective_vector
      0.0,                          # objective_constant
      [
        1.0 3.0
        1.0 -2.0
        2.0 0.0
      ],                            # constraint_matrix
      [1.0, 1.0, 2.0],              # right_hand_side
      1,                            # num_equalities
    )
    original_problem = deepcopy(problem)
    cum_constraint_rescaling, cum_variable_rescaling =
      FirstOrderLp.ruiz_rescaling(problem, 1)
    test_fields_approx_equal(
      problem,
      FirstOrderLp.QuadraticProgrammingProblem(
        [-Inf, -2.0 * sqrt(3)],                    # variable_lower_bound
        [2.0, 2.0 * sqrt(3)],                      # variable_upper_bound
        [1.0 1.0/sqrt(3); 1.0/sqrt(3) 1/3],  # objective_matrix
        [0.5, 2.0 / sqrt(3)],                      # objective_vector
        0.0,                                       # objective_constant
        [
          0.5/sqrt(3) 1.0
          0.5/sqrt(2) -sqrt(2)/sqrt(3)
          1.0/sqrt(2) 0.0
        ],                                         # constraint_matrix
        [1 / sqrt(3), 1 / sqrt(2), sqrt(2)],       # right_hand_side
        1,                                         # num_equalities
      ),
    )
    @test cum_variable_rescaling ≈ [2.0, sqrt(3)]
    @test cum_constraint_rescaling ≈ [sqrt(3), sqrt(2), sqrt(2)]
    FirstOrderLp.unscale_problem(
      problem,
      cum_constraint_rescaling,
      cum_variable_rescaling,
    )
    test_fields_approx_equal(problem, original_problem)
  end

  @testset "Convergence of Ruiz-Rescaling for QP" begin
    problem = FirstOrderLp.QuadraticProgrammingProblem(
      [-1.0, -2.0],                 # variable_lower_bound
      [1.0, 2.0],                   # variable_upper_bound
      [4.0 2.0; 2.0 1.0],           # objective_matrix
      [1.0, 2.0],                   # objective_vector
      0.0,                          # objective_constant
      [
        1.0 3.0
        1.0 -2.0
        2.0 0.0
      ],                            # constraint_matrix
      [1.0, 1.0, 2.0],              # right_hand_side
      1,                            # num_equalities
    )
    original_problem = deepcopy(problem)
    cum_constraint_rescaling, cum_variable_rescaling =
      FirstOrderLp.ruiz_rescaling(problem, 30)

    constraint_matrix = problem.constraint_matrix
    objective_matrix = problem.objective_matrix

    inf_norm_of_columns = vec(
      sqrt.(
        max.(
          maximum(abs.(constraint_matrix), dims = 1),
          maximum(abs.(objective_matrix), dims = 1),
        ),
      ),
    )
    @test all(inf_norm_of_columns .≈ 1)
    inf_norm_of_rows = vec(sqrt.(maximum(abs.(constraint_matrix), dims = 2)))
    @test all(inf_norm_of_rows .≈ 1)
    FirstOrderLp.unscale_problem(
      problem,
      cum_constraint_rescaling,
      cum_variable_rescaling,
    )
    test_fields_approx_equal(problem, original_problem)
  end

  @testset "L2 Ruiz-Rescaling single iteration for LP" begin
    problem = FirstOrderLp.linear_programming_problem(
      [0.0, 0.0],                   # variable_lower_bound
      [1.0, 2.0],                   # variable_upper_bound
      [1.0, 2.0],                   # objective_vector
      0.0,                          # objective_constant
      [
        1.0 3.0
        1.0 -2.0
        2.0 0.0
      ],                            # constraint_matrix
      [1.0, 1.0, 3.0],              # right_hand_side
      1,                            # num_equalities
    )
    cum_constraint_rescaling, cum_variable_rescaling =
      FirstOrderLp.ruiz_rescaling(problem, 1, 2.0)

    test_fields_approx_equal(
      problem,
      FirstOrderLp.linear_programming_problem(
        [0.0, 0.0],                               # variable_lower_bound
        [1 * 6^0.25, 2 * 13^0.25],                # variable_upper_bound
        [1 / 6^0.25, 2 / 13^0.25],                # objective_vector
        0.0,                                      # objective_constant
        [
          1/(6*15)^0.25 3/(13*15)^0.25
          1/(7.5*6)^0.25 -2/(13*7.5)^0.25
          2/(6*6)^0.25 0
        ],                                        # constraint_matrix
        [1 / 15^0.25, 1 / 7.5^0.25, 3 / 6^0.25],  # right_hand_side
        1,                                        # num_equalities
      ),
    )

    @test cum_variable_rescaling ≈ [6^0.25, 13^0.25]
    @test cum_constraint_rescaling ≈ [15^0.25, 7.5^0.25, 6^0.25]
  end
  @testset "Convergence of L2 Ruiz-Rescaling for LP" begin
    problem = FirstOrderLp.linear_programming_problem(
      [0.0, 0.0],                   # variable_lower_bound
      [1.0, 2.0],                   # variable_upper_bound
      [1.0, 2.0],                   # objective_vector
      0.0,                          # objective_constant
      [
        1.0 3.0
        1.0 -2.0
        2.0 0.0
      ],                            # constraint_matrix
      [1.0, 1.0, 3.0],              # right_hand_side
      1,                            # num_equalities
    )
    FirstOrderLp.ruiz_rescaling(problem, 60, 2.0)

    norm_of_columns = FirstOrderLp.l2_norm(problem.constraint_matrix, 1)
    @test norm_of_columns ≈ [1, 1] atol = 1e-5
    norm_of_rows = FirstOrderLp.l2_norm(problem.constraint_matrix, 2)
    @test norm_of_rows ≈ [sqrt(2 / 3), sqrt(2 / 3), sqrt(2 / 3)] atol = 1e-5
  end

  @testset "Convergence of L2 Ruiz-Rescaling for QP" begin
    problem = FirstOrderLp.QuadraticProgrammingProblem(
      [-Inf, -2.0],                 # variable_lower_bound
      [1.0, 2.0],                   # variable_upper_bound
      [4.0 2.0; 2.0 1.0],           # objective_matrix
      [1.0, 2.0],                   # objective_vector
      0.0,                          # objective_constant
      [
        1.0 3.0
        1.0 -2.0
        2.0 0.0
      ],                            # constraint_matrix
      [1.0, 1.0, 2.0],              # right_hand_side
      1,                            # num_equalities
    )
    cum_constraint_rescaling, cum_variable_rescaling =
      FirstOrderLp.ruiz_rescaling(problem, 1, 2.0)
    test_fields_approx_equal(
      problem,
      FirstOrderLp.QuadraticProgrammingProblem(
        [-Inf, -2 * 18^0.25],                       # variable_lower_bound
        [1 * 26^0.25, 2 * 18^0.25],                 # variable_upper_bound
        [
          4/26^0.5 2/(26*18)^0.25
          2/(26*18)^0.25 1/18^0.5
        ],                                          # objective_matrix
        [1 / 26^0.25, 2 / 18^0.25],                 # objective_vector
        0.0,                                        # objective_constant
        [
          1/(25*26)^0.25 3/(18*25)^0.25
          1/(12.5*26)^0.25 -2/(18*12.5)^0.25
          2/(10*26)^0.25 0
        ],                                          # constraint_matrix
        [1 / 25^0.25, 1 / 12.5^0.25, 2 / 10^0.25],  # right_hand_side
        1,                                          # num_equalities
      ),
    )
    @test cum_variable_rescaling ≈ [26^0.25, 18^0.25]
    @test cum_constraint_rescaling ≈ [25^0.25, 12.5^0.25, 10^0.25]
  end

  @testset "Convergence of L2 Ruiz-Rescaling for QP" begin
    problem = FirstOrderLp.QuadraticProgrammingProblem(
      [-1.0, -2.0],                 # variable_lower_bound
      [1.0, 2.0],                   # variable_upper_bound
      [4.0 2.0; 2.0 1.0],           # objective_matrix
      [1.0, 2.0],                   # objective_vector
      0.0,                          # objective_constant
      [
        1.0 3.0
        1.0 -2.0
        2.0 0.0
      ],                            # constraint_matrix
      [1.0, 1.0, 2.0],              # right_hand_side
      1,                            # num_equalities
    )
    FirstOrderLp.ruiz_rescaling(problem, 100, 2.0)

    constraint_matrix = problem.constraint_matrix
    objective_matrix = problem.objective_matrix

    norm_of_columns = vec(
      sqrt.(
        sqrt.(
          FirstOrderLp.l2_norm(constraint_matrix, 1) .^ 2 +
          FirstOrderLp.l2_norm(objective_matrix, 1) .^ 2,
        ),
      ),
    )
    @test norm_of_columns ≈ [1, 1] atol = 1e-5
    norm_of_rows = FirstOrderLp.l2_norm(problem.constraint_matrix, 2)
    @test norm_of_rows ≈ [sqrt(2 / 5), sqrt(2 / 5), sqrt(2 / 5)] atol = 1e-5
  end

  # This instance is simple enough to compute where it converges in
  # closed form. (It actually converges in a single iteration.)
  @testset "L2 Ruiz-Rescaling simple" begin
    problem = FirstOrderLp.linear_programming_problem(
      [0.0, 0.0],                   # variable_lower_bound
      [1.0, 2.0],                   # variable_upper_bound
      [1.0, 2.0],                   # objective_vector
      0.0,                          # objective_constant
      [
        1.0 1.0
        1.0 -1.0
        1.0 1.0
      ],                            # constraint_matrix
      [1.0, 1.0, 3.0],              # right_hand_side
      1,                            # num_equalities
    )
    FirstOrderLp.ruiz_rescaling(problem, 10, 2.0)

    test_fields_approx_equal(
      problem,
      FirstOrderLp.linear_programming_problem(
        [0.0, 0.0],                            # variable_lower_bound
        [1 * 3^0.25, 2 * 3^0.25],              # variable_upper_bound
        [1 / 3^0.25, 2 / 3^0.25],              # objective_vector
        0.0,                                   # objective_constant
        [
          1/sqrt(3) 1/sqrt(3)
          1/sqrt(3) -1/sqrt(3)
          1/sqrt(3) 1/sqrt(3)
        ],                                     # constraint_matrix
        [1 / 3^0.25, 1 / 3^0.25, 3 / 3^0.25],  # right_hand_side
        1,                                     # num_equalities
      ),
    )
  end
end
