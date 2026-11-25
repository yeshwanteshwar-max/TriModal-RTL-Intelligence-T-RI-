import networkx as nx


def build_graph(instances: list[dict]) -> nx.DiGraph:
    graph = nx.DiGraph()
    for instance in instances:
        graph.add_node(instance["name"], kind=instance.get("kind", "cell"))
    return graph
