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

const SimpleGraph = LightGraphs.SimpleGraph

"""
Populates 'model' with the linear program model for computing the pagerank
vector for 'graph', with the given damping factor. Given an adjacency matrix A,
define S (the stochastic transition matrix) by dividing each column by it's
L_1 norm. The linear programming model is then:
  find x
  s.t. damping_factor * (S*x)[i] + (1 - damping_factor)/num_nodes <= x[i]
       sum_i x[i] = 1
       x >= 0
Note: Pagerank is usually computed for directed graphs. However, this example is
using the barabasi_albert generator, which generates an undirected graph
(SimpleGraph) instead of a directed graph (SimpleDiGraph).
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
    transition_sum = JuMP.@expression(
      model,
      sum(x[j] / degrees[j] for j in LightGraphs.neighbors(graph, i))
    )
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
  model = JuMP.Model()
  num_nodes = parsed_args["num_nodes"]
  approx_num_edges = parsed_args["approx_num_edges"]
  degree = round(Int, approx_num_edges / num_nodes)
  graph = LightGraphs.SimpleGraphs.barabasi_albert(
    num_nodes,
    degree,
    seed = parsed_args["random_seed"],
  )
  populate_pagerank_model(model, graph, parsed_args["damping_factor"])
  JuMP.write_to_file(model, parsed_args["output_filename"])
end

main()
