# Road-graph metrics from the reference OpenStreetMapX pipeline, for parity
# comparison with the Python engine. Run with the MAMUT-routing webapp project:
#   julia --project=<MAMUT-routing>/webapp julia_metrics.jl <city.osm> <out.json>
#
# Mirrors webapp/site_api.jl's workbench_get_map_data_cached option fallback.

using OpenStreetMapX
using Graphs
using SparseArrays
using JSON3

function load_reference_map(osm_path::String; only_intersections::Bool=true, trim_to_connected_graph::Bool=true)
    fallbacks = Tuple{Bool,Bool}[
        (only_intersections, trim_to_connected_graph),
        (only_intersections, false),
        (false, trim_to_connected_graph),
        (false, false),
    ]
    seen = Set{Tuple{Bool,Bool}}()
    for (oi, ttcg) in fallbacks
        (oi, ttcg) in seen && continue
        push!(seen, (oi, ttcg))
        try
            md = get_map_data(osm_path; use_cache=false, only_intersections=oi, trim_to_connected_graph=ttcg)
            return md, oi, ttcg
        catch error
            if error isa ArgumentError && occursin("empty collection", string(error))
                continue
            end
            rethrow(error)
        end
    end
    error("no usable graph for $osm_path")
end

function sampled_pairs(md::MapData, count::Int)
    # Synthetic boundary nodes get hash-derived ids on the Julia side and
    # sequential ids on the Python side; sample only real OSM ids so both
    # engines route between the same nodes.
    osm_ids = sort!(filter(id -> 0 < id < 2^48, collect(keys(md.v))))
    length(osm_ids) < 2 && return Tuple{Int,Int}[]
    picks = [osm_ids[1 + div((i - 1) * (length(osm_ids) - 1), max(count - 1, 1))] for i in 1:count]
    return [(picks[i], picks[i + 1]) for i in 1:(length(picks) - 1)]
end

osm_path = ARGS[1]
out_path = ARGS[2]

md, oi, ttcg = load_reference_map(osm_path)

class_counts = Dict{Int,Int}()
for c in md.class
    class_counts[c] = get(class_counts, c, 0) + 1
end
total_length = sum(md.w[md.v[u_osm], md.v[v_osm]] for (u_osm, v_osm) in md.e)

routes = []
for (from_osm, to_osm) in sampled_pairs(md, 12)
    s_nodes, s_dist, _ = shortest_route(md, from_osm, to_osm)
    f_nodes, f_dist, f_time = fastest_route(md, from_osm, to_osm)
    push!(routes, Dict(
        "from" => from_osm,
        "to" => to_osm,
        "shortest_m" => isempty(s_nodes) ? nothing : s_dist,
        "fastest_s" => isempty(f_nodes) ? nothing : f_time,
        "shortest_nodes" => length(s_nodes),
        "fastest_nodes" => length(f_nodes),
    ))
end

ref = OpenStreetMapX.center(md.bounds)
result = Dict(
    "osm_path" => osm_path,
    "resolved_options" => Dict("only_intersections" => oi, "trim_to_connected_graph" => ttcg),
    "vertices" => length(md.v),
    "edges" => length(md.e),
    "total_edge_length_km" => total_length / 1000.0,
    "edge_class_counts" => Dict(string(k) => v for (k, v) in class_counts),
    "ref_lla" => Dict("lat" => ref.lat, "lon" => ref.lon),
    "sampled_routes" => routes,
)
open(out_path, "w") do io
    JSON3.pretty(io, JSON3.read(JSON3.write(result)))
end
println("wrote $out_path")
