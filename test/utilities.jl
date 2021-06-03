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

"""
Tests that all fields of these two structs are the same.

# Arguments
- `struct1`: a struct
- `struct2`: a struct of the same type as struct1
"""
function test_fields_equal(
  struct1::TypeName,
  struct2::TypeName,
) where {TypeName}
  @testset "$field_name matches" for field_name in fieldnames(TypeName)
    @test getfield(struct1, field_name) == getfield(struct2, field_name)
  end
end


function test_fields_equal(
  struct1::FirstOrderLp.IterationStats,
  struct2::FirstOrderLp.IterationStats,
)
  @testset "$field_name matches" for field_name in
                                     fieldnames(FirstOrderLp.IterationStats)
    struct1_field = getfield(struct1, field_name)
    struct2_field = getfield(struct2, field_name)
    if field_name == :convergence_information ||
       field_name == :infeasibility_information
      @test length(struct1_field) == length(struct2_field)
      for (left_entry, right_entry) in
          Iterators.zip(struct1_field, struct2_field)
        test_fields_equal(left_entry, right_entry)
      end
    else
      @test struct1_field == struct2_field
    end
  end
end

"""
Tests that all fields of these two structs are approximately the same.

# Arguments
- `struct1`: a struct
- `struct2`: a struct of the same type as struct1
"""
function test_fields_approx_equal(
  struct1::TypeName,
  struct2::TypeName,
) where {TypeName}
  @testset "$field_name matches" for field_name in fieldnames(TypeName)
    @test getfield(struct1, field_name) ≈ getfield(struct2, field_name)
  end
end

"""
Tests all entries of two tuples are approximately same.
"""
function test_tuple_approx_equal(tuple1, tuple2)
  @test length(tuple1) == length(tuple2)
  @testset "tuple index $i" for i in 1:length(tuple2)
    @test tuple1[i] ≈ tuple2[i]
  end
end

"""
Terminate based on iteration limit only.
The termination tolerance are set to zero to stop any tests from
breaking. This is because when the tests were orginally designed when there
was no termination criteria.
"""
function terminate_on_iteration_limit(n::Int64)
  termination_criteria = FirstOrderLp.construct_termination_criteria(
    optimality_norm = FirstOrderLp.L_INF,
    eps_optimal_absolute = 0.0,
    eps_optimal_relative = 0.0,
    eps_primal_infeasible = 0.0,
    eps_dual_infeasible = 0.0,
    time_sec_limit = 100.0,
    iteration_limit = n,
    kkt_matrix_pass_limit = Inf,
  )
  return termination_criteria
end
