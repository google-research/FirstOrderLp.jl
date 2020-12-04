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

const TRIVIAL_LP_MODEL = FirstOrderLp.QuadraticProgrammingProblem(
  [0.0, 1.0], # variable_lower_bound
  [1.0, 2.0], # variable_upper_bound
  zeros(2, 2), # objective_matrix
  [2.0, -1.0], # objective_vector
  0.0, # objective_constant
  [-1.0 -1.0], # constraint_matrix
  [-3.0], # right_hand_side
  0, # num_equalities
)

const TRIVIAL_QP_MODEL = FirstOrderLp.QuadraticProgrammingProblem(
  [0.0, 1.0], # variable_lower_bound
  [1.0, 2.0], # variable_upper_bound
  [2.0 2.0; 2.0 4.0], # objective_matrix
  [2.0, -1.0], # objective_vector
  0.0, # objective_constant
  [-1.0 -1.0], # constraint_matrix
  [-3.0], # right_hand_side
  0, # num_equalities
)

@testset "Read .mps file (LP)" begin
  qp = FirstOrderLp.qps_reader_to_standard_form(joinpath(
    dirname(@__FILE__),
    "trivial_lp_model.mps",
  ))
  test_fields_equal(qp, TRIVIAL_LP_MODEL)
end

@testset "Read .mps file (QP)" begin
  qp = FirstOrderLp.qps_reader_to_standard_form(joinpath(
    dirname(@__FILE__),
    "trivial_qp_model.mps",
  ))
  test_fields_equal(qp, TRIVIAL_QP_MODEL)
end

@testset "Read .mps.gz file (QP)" begin
  mps_contents =
    read(open(joinpath(dirname(@__FILE__), "trivial_qp_model.mps")), String)
  gzfile = tempname() * ".mps.gz"
  try
    fd = GZip.open(gzfile, "w")
    write(fd, mps_contents)
    close(fd)
    qp = FirstOrderLp.qps_reader_to_standard_form(gzfile)
    test_fields_equal(qp, TRIVIAL_QP_MODEL)
  finally
    rm(gzfile, force = true)
  end
end

@testset "Two-sided rows to slacks" begin
  qp = FirstOrderLp.TwoSidedQpProblem(
    [-Inf, -Inf],  # variable_lower_bound
    [Inf, Inf],  # variable_upper_bound
    [-3.0, -2.0],  # constraint_lower_bound
    [1.0, Inf],  # constraint_upper_bound
    [1.0 1.0; 1.0 1.0],  # constraint_matrix
    2.0,  # objective_offset
    [0.0, 1.0],  # objective_vector
    diagm([1.0, 3.0]),  # objective_matrix
  )
  FirstOrderLp.two_sided_rows_to_slacks(qp)
  test_fields_equal(
    qp,
    FirstOrderLp.TwoSidedQpProblem(
      [-Inf, -Inf, -3.0],  # variable_lower_bound
      [Inf, Inf, 1.0],  # variable_upper_bound
      [0.0, -2.0],  # constraint_lower_bound
      [0.0, Inf],  # constraint_upper_bound
      [
        1.0 1.0 -1.0
        1.0 1.0 0.0
      ],  # constraint_matrix
      2.0,  # objective_offset
      [0.0, 1.0, 0.0],  # objective_vector
      diagm([1.0, 3.0, 0.0]),  # objective_matrix
    ),
  )
end
