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
#
# This is a short script that transforms an MPS file by dropping all
# integrality constraints.

import SCIP

# Note: The SCIP version bundled with Julia can't process .gz files.
function drop_integrality(input_file, output_file)

  scip = Ref{Ptr{SCIP.SCIP_}}(C_NULL)
  SCIP.@SC SCIP.SCIPcreate(scip)

  @assert scip[] != C_NULL
  SCIP.@SC SCIP.SCIPincludeDefaultPlugins(scip[])

  SCIP.@SC SCIP.SCIPreadProb(scip[], input_file, C_NULL)

  num_vars = SCIP.SCIPgetNVars(scip[])

  vars = Vector{Ptr{SCIP.SCIP_VAR}}(undef, num_vars)

  raw_vars = SCIP.SCIPgetVars(scip[])
  # Copy the vars to an array we own.
  # The SCIP documentation warns that SCIPchgVarType() can invalidate the result
  # of SCIPGetVars().
  vars .= unsafe_wrap(Array, raw_vars, num_vars)

  for var in vars
    # This is an output of chgVarType indicating whether the change makes the
    # problem infeasible. We ignore this value.
    infeasible = Ref{SCIP.SCIP_Bool}()
    SCIP.@SC SCIP.SCIPchgVarType(
      scip[],
      var,
      SCIP.SCIP_VARTYPE_CONTINUOUS,
      infeasible,
    )
  end
  @assert SCIP.SCIPgetNBinVars(scip[]) == 0
  @assert SCIP.SCIPgetNIntVars(scip[]) == 0

  SCIP.@SC SCIP.SCIPwriteOrigProblem(scip[], output_file, C_NULL, false)

  SCIP.@SC SCIP.SCIPfree(scip)
end

if length(ARGS) != 2
  @error "Usage: drop_integrality.jl input_file output_file"
end
drop_integrality(ARGS[1], ARGS[2])
