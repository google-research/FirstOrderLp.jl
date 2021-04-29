# Generates a Linear Programming model for computing PageRank of a random graph.

# Sample usage:
# $ julia --project=. generate_pagerank_lp.jl --num_nodes 10000 \
#    --approx_num_edges 30000 --output_filename /tmp/pagerank.10k.mps.gz

# The LP formulation for PageRank is taken from:
# Nesterov, Y., "Subgradient methods for huge-scale optimization problems",
# Mathematical Programming, 2014. https://doi.org/10.1007/s10107-013-0686-4
# http://www.optimization-online.org/DB_FILE/2012/02/3339.pdf (preprint)

# Barabasi Albert preferential attachment graphs are used as a model of a
# random "web" graph.

import ArgParse
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
  for i in 1:num_nodes
    transition_sum =
      sum(x[j] / degrees[j] for j in LightGraphs.neighbors(graph, i))
    JuMP.@constraint(
      model,
      damping_factor * transition_sum + (1 - damping_factor) / num_nodes <=
      x[i]
    )
  end
  JuMP.@constraint(
    model,
    L1_norm,
    sqrt(num_nodes) * sum(x[i] for i in 1:num_nodes) == sqrt(num_nodes)
  )
  return model
end

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
Defines parses and args.

# Returns
A dictionary with the values of the command-line arguments.
"""
function parse_command_line()
  arg_parse = ArgParse.ArgParseSettings()

  ArgParse.@add_arg_table! arg_parse begin
    "--num_nodes"
    help = "The number of nodes in the generated graph."
    arg_type = Int
    required = true

    "--approx_num_edges"
    help = "The approximate number of edges in the generated graph."
    arg_type = Int
    required = true

    "--output_filename"
    help = "Filename for the output .mps (or .mps.gz) model."
    arg_type = String
    required = true

    "--damping_factor"
    help = "Damping factor for the pagerank model."
    arg_type = Float64
    default = 0.99

    "--random_seed"
    help = "Seed for the random generator."
    arg_type = Int
    default = 0
  end

  return ArgParse.parse_args(arg_parse)
end

function main()
  parsed_args = parse_command_line()

  filename = parsed_args["output_filename"]
  if !endswith(filename, ".gz")
    # We prefer SCIP's more mature writer, but the version packaged with Julia
    # can't write .gz files.
    backend = SCIP.Optimizer()
    model = JuMP.direct_model(backend)
  else
    model = JuMP.Model()
  end
  num_nodes = parsed_args["num_nodes"]
  approx_num_edges = parsed_args["approx_num_edges"]
  degree = round(Int, approx_num_edges / num_nodes)
  graph = LightGraphs.SimpleGraphs.barabasi_albert(
    num_nodes,
    degree,
    seed = parsed_args["random_seed"],
  )
  populate_pagerank_model(model, graph, parsed_args["damping_factor"])
  write_model_to_mps(model, parsed_args["output_filename"])
end

main()
