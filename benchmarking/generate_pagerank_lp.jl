# Generates a Linear Programming model for computing PageRank of a random graph.

# The LP formulation for PageRank is taken from:
# Nesterov, Y., "Subgradient methods for huge-scale optimization problems",
# Mathematical Programming, 2014. https://doi.org/10.1007/s10107-013-0686-4
# http://www.optimization-online.org/DB_FILE/2012/02/3339.pdf (preprint)

# Barabasi Albert preferential attachment graphs are used as a model of a
# random "web" graph.

import JuMP
import LightGraphs
import SCIP

const SimpleGraph = LightGraphs.SimpleGraph

"""
Populates 'model' with the linear program model for computing the pagerank
vector for 'graph', with the given damping factor. The model is:
  find x
  s.t. damping_factor * (A*x)[i] + (1 - damping_factor)/num_nodes <= x[i]
       sum_i x[i] = 1
       x >= 0
"""
function populate_pagerank_model(
  model::JuMP.Model,
  graph::SimpleGraph,
  damping_factor::Float64 = 0.99,
)
  num_nodes = LightGraphs.nv(graph)
  degrees = [length(LightGraphs.neighbors(graph, i)) for i in 1:num_nodes]
  JuMP.@variable(model, x[i = 1:num_nodes], lower_bound = 0.0)
  JuMP.@constraint(
    model,
    flow[i = 1:num_nodes],
    damping_factor *
    sum(x[j] / degrees[j] for j in LightGraphs.neighbors(graph, i)) - x[i] <=
    -(1 - damping_factor) / num_nodes
  )
  JuMP.@constraint(
    model,
    l1norm,
    sqrt(num_nodes) * sum(x[i] for i in 1:num_nodes) == sqrt(num_nodes)
  )
  return model
end

"""
Writes 'model' to 'filename'.
"""
function write_model_to_mps(model::JuMP.Model, filename::AbstractString)
  backend = JuMP.backend(model)
  # Prefer SCIP's native writer. However, the version of SCIP bundled with Julia
  # can't write .gz files.
  if !endswith(filename, ".gz") && isa(backend, SCIP.Optimizer)
    scip_ptr = backend.mscip.scip
    SCIP.@SC SCIP.SCIPwriteOrigProblem(
      scip_ptr[],
      filename,
      C_NULL,
      true, #genericnames
    )

  else
    # Not SCIP (or gzipped output). Fall back to JuMPs less mature writer.
    JuMP.write_to_file(model, filename)
  end
end

"""
Generates a random Barabasi Albert preferential attachment graph, builds the
corresponding pagerank LP, and writes it to 'filename'.
"""
function generate_pagerank_lp(
  num_nodes::Int,
  approx_num_edges::Int,
  damping_factor::Float64,
  filename::AbstractString,
)
  degree = round(Int, approx_num_edges / num_nodes)
  backend = SCIP.Optimizer()
  model = JuMP.direct_model(backend)
  graph = LightGraphs.SimpleGraphs.barabasi_albert(num_nodes, degree)
  populate_pagerank_model(model, graph, damping_factor)
  write_model_to_mps(model, filename)
end

if length(ARGS) != 4
  @error "Usage: generate_pagerank.jl num_nodes approx_num_edges damping_factor output_file"
end
generate_pagerank_lp(
  parse(Int, ARGS[1]),
  parse(Int, ARGS[2]),
  parse(Float64, ARGS[3]),
  ARGS[4],
)
