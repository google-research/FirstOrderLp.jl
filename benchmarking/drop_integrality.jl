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
  @error "Usage: drop_integrality.jl [input file] [output file]"
end
drop_integrality(ARGS[1], ARGS[2])
