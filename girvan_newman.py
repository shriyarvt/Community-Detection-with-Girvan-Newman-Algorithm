import time
import sys
import os
from pyspark import SparkContext
from itertools import combinations
from collections import defaultdict, deque
import copy                                                                               #changed line


def make_nodes(sc, input_filepath):
    lines = sc.textFile(input_filepath)
    header = lines.first()
    lines = lines.filter(lambda x: x != header)
    lines = lines.map(lambda x: x.strip().split(",")).cache()

    # node dictionary format: {user: {business_1, business_2, ...}}
    nodes = lines.map(lambda x: (x[0], x[1])) \
        .groupByKey() \
        .mapValues(set) \
        .collectAsMap()

    return nodes


def build_graph(all_nodes, threshold):
    edge_graph = defaultdict(list)
    actual_nodes = set()
    node_combinations = list(combinations(all_nodes.keys(), 2))

    # check overalapping bus for all comb
    for u1, u2 in node_combinations:
        common_businesses = all_nodes[u1].intersection(all_nodes[u2])
        if len(common_businesses) >= threshold:
            edge_graph[u1].append(u2)
            edge_graph[u2].append(u1)
            actual_nodes.update([u1, u2])

    return edge_graph, list(actual_nodes)


def calculate_betweenness(graph, nodes):
    def bfs(source):
        distances = {source: 0}
        paths = {source: 1}
        levels = defaultdict(set)
        levels[0].add(source)
        queue = deque([source])
        edge_flows = defaultdict(float)

        # get all shortest paths
        while queue:
            node = queue.popleft()
            for neighbor in graph[node]:
                # new node
                if neighbor not in distances:
                    distances[neighbor] = distances[node] + 1
                    queue.append(neighbor)
                    levels[distances[neighbor]].add(neighbor)

                # add to path count
                if distances[neighbor] == distances[node] + 1:
                    paths[neighbor] = paths.get(neighbor, 0) + paths[node]

        # calculate betweenness
        node_credit = defaultdict(float)
        for level in sorted(levels.keys(), reverse=True)[:-1]:  # all except source
            for node in levels[level]:
                credit = (1 + node_credit[node]) / paths[node]
                for neighbor in graph[node]:
                    if distances.get(neighbor, float('inf')) == distances[node] - 1:
                        edge_flows[(min(neighbor, node), max(neighbor, node))] += paths[neighbor] * credit
                        node_credit[neighbor] += paths[neighbor] * credit

        return edge_flows

    # total betweeness from all nodes
    total_flows = defaultdict(float)
    for node in nodes:
        flows = bfs(node)
        for edge, flow in flows.items():
            total_flows[edge] += flow

    # div by 2 (each path counted twice)
    betweenness = {edge: flow / 2 for edge, flow in total_flows.items()}

    # sort by betweenness desc and edge asc
    return sorted(betweenness.items(), key=lambda x: (-x[1], x[0]))


def find_communities(graph, nodes):
    # finds communities where all nodes all connected by at least 1 path

    communities = []
    remaining_nodes = nodes.copy()

    while remaining_nodes:
        current = remaining_nodes.pop(0)
        visited = set([current])
        queue = [current]

        # bfs
        while queue:
            node = queue.pop(0)
            for neighbor in graph[node]:
                if neighbor not in visited:
                    if neighbor in remaining_nodes:
                        remaining_nodes.remove(neighbor)
                    queue.append(neighbor)
                    visited.add(neighbor)

        communities.append(sorted(list(visited)))

    return sorted(communities, key=lambda x: (len(x), x[0]))


def calculate_modularity(communities, original_graph, curr_graph, total_edges):          #changed line
    modularity = 0

    # calc for each community
    for community in communities:
        for i in community:
            for j in community:
                # calc w formula given
                A = 1 if j in curr_graph[i] else 0                                       #changed line
                k_i, k_j = len(original_graph[i]), len(original_graph[j])
                modularity += A - (k_i * k_j) / (2 * total_edges)

    return modularity / (2 * total_edges)


def get_communities(betweenness_results, edge_graph, nodes):
    current_graph = copy.deepcopy(edge_graph)                                            #changed line
    total_edges = sum(len(v) for v in edge_graph.values()) // 2

    best_modularity = float('-inf')
    best_communities = []

    while betweenness_results:
        # get current communities
        current_communities = find_communities(current_graph, nodes)
        current_modularity = calculate_modularity(current_communities, edge_graph, current_graph, total_edges)#changed line

        # update if curr is better
        if current_modularity > best_modularity:
            best_modularity = current_modularity
            best_communities = current_communities

        # prune edges w highest betweeness
        highest_betweenness = betweenness_results[0][1]
        for edge, value in betweenness_results:
            if value >= highest_betweenness:
                current_graph[edge[0]].remove(edge[1])
                current_graph[edge[1]].remove(edge[0])

        # recalc betweenness
        betweenness_results = calculate_betweenness(current_graph, nodes)

    return best_communities


def main():
    start = time.time()

    threshold = int(sys.argv[1])
    input_file = sys.argv[2]
    bet_output = sys.argv[3]
    comm_output = sys.argv[4]

    # set up
    sc = SparkContext('local[*]', 'hw4task2')
    sc.setLogLevel('WARN')

    # process data
    allnodes = make_nodes(sc, input_file)
    graph, nodes = build_graph(allnodes, threshold)

    # betweeness results
    betweenness_results = calculate_betweenness(graph, nodes)

    # write
    with open(bet_output, "w") as f:
        for edge, score in betweenness_results:
            f.write(f"('{edge[0]}', '{edge[1]}'),{round(score, 5)}\n")

    # find communities
    communities = get_communities(betweenness_results, graph, nodes)

    # write
    with open(comm_output, "w") as f:
        for community in communities:
            add_quotes = ["'" + str(val) + "'" for val in community]
            f.write(", ".join(add_quotes) + "\n")

    # duration
    end = time.time()
    duration = str(end - start)
    print("Duration:", duration)

    sc.stop()


if __name__ == "__main__":
    os.environ['PYSPARK_PYTHON'] = sys.executable
    os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

    main()