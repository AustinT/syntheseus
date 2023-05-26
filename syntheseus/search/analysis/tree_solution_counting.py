"""Count the number of solutions in a tree."""

from syntheseus.search.graph.and_or import ANDOR_NODE, AndNode, AndOrGraph, OrNode


def num_solutions_update(node: ANDOR_NODE, graph: AndOrGraph) -> bool:
    """
    Updates a node's "reaction number", which is the current minimum cost
    estimate of synthesizing a molecule from everything below it.
    Returns whether the node's reaction number was updated.
    """
    if isinstance(node, AndNode):
        new_num_routes = 1
        for child in graph.successors(node):
            new_num_routes *= child.data["num_routes"]  # type: ignore
    elif isinstance(node, OrNode):
        new_num_routes = sum([c.data["num_routes"] for c in graph.successors(node)])  # type: ignore

        # One extra route if it is purchasable
        if node.mol.metadata["is_purchasable"]:
            new_num_routes += 1
    else:
        raise TypeError(f"Unexpected node type: {type(node)}")

    # Do update and return whether the value changed
    old_num_routes = node.data.get("num_routes", None)
    node.data["num_routes"] = new_num_routes  # type: ignore
    return new_num_routes != old_num_routes
