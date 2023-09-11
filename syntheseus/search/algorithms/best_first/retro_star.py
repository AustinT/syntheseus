from __future__ import annotations

import logging
import math
from collections import deque
from collections.abc import Sequence

# NOTE: Collection imported here instead of from collections.abc
# to make casting work for python <3.9
from typing import (
    Collection,
    Optional,
    cast,
)

from syntheseus.search.algorithms.base import AndOrSearchAlgorithm
from syntheseus.search.algorithms.best_first.base import GeneralBestFirstSearch
from syntheseus.search.algorithms.mixins import ValueFunctionMixin
from syntheseus.search.graph.and_or import ANDOR_NODE, AndNode, AndOrGraph, OrNode
from syntheseus.search.graph.message_passing import run_message_passing
from syntheseus.search.node_evaluation.base import BaseNodeEvaluator, NoCacheNodeEvaluator

logger = logging.getLogger(__name__)


class MolIsPurchasableCost(NoCacheNodeEvaluator[OrNode]):
    def _evaluate_nodes(  # type: ignore[override]
        self,
        nodes: Sequence[OrNode],
        graph: Optional[AndOrGraph] = None,
    ) -> list[float]:
        return [0.0 if node.mol.metadata.get("is_purchasable") else math.inf for node in nodes]


class RetroStarSearch(
    AndOrSearchAlgorithm[int],
    GeneralBestFirstSearch[AndOrGraph],
    ValueFunctionMixin[OrNode],
):
    def __init__(
        self,
        *args,
        and_node_cost_fn: BaseNodeEvaluator[AndNode],
        or_node_cost_fn: Optional[BaseNodeEvaluator[OrNode]] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if or_node_cost_fn is None:
            or_node_cost_fn = MolIsPurchasableCost()
        self.or_node_cost_fn = or_node_cost_fn
        self.and_node_cost_fn = and_node_cost_fn

    @property
    def requires_tree(self) -> bool:
        return False

    @property
    def reaction_number_estimator(self) -> BaseNodeEvaluator[OrNode]:
        """Alias for value function (they use this term in the paper)"""
        return self.value_function

    def priority_function(self, node: ANDOR_NODE, _: AndNode) -> float:  # type: ignore[override]
        return node.data["retro_star_value"]

    def setup(self, graph: AndOrGraph) -> None:
        # If there is only one node, set its reaction number estimate to 0.
        # This saves a call to the value function
        if len(graph) == 1:
            graph.root_node.data.setdefault("reaction_number_estimate", 0.0)

        # Do initial setup
        return super().setup(graph)

    def set_node_values(  # type: ignore[override]
        self, nodes: Collection[ANDOR_NODE], graph: AndOrGraph
    ) -> Collection[ANDOR_NODE]:
        # Call superclass
        output_nodes = super().set_node_values(nodes, graph)
        del nodes  # unused

        # Fill in node costs and reaction number estimates
        self._set_or_node_costs(
            or_nodes=[
                node
                for node in output_nodes
                if isinstance(node, OrNode) and "retro_star_mol_cost" not in node.data
            ],
            graph=graph,
        )
        self._set_and_node_costs(
            and_nodes=[
                node
                for node in output_nodes
                if isinstance(node, AndNode) and "retro_star_rxn_cost" not in node.data
            ],
            graph=graph,
        )
        self._set_reaction_number_estimate(  # only for leaf nodes
            or_nodes=[
                node
                for node in output_nodes
                if isinstance(node, OrNode)
                and "reaction_number_estimate" not in node.data
                and not node.is_expanded
            ],
            graph=graph,
        )

        # Fill in expansion information
        for node in output_nodes:
            if isinstance(node, OrNode):
                node.data["retro_star_can_expand"] = self.can_expand_node(node, graph)

        # Run updates of reaction number and retro-star value
        return self._run_retro_star_updates(output_nodes, graph)

    def _set_or_node_costs(self, or_nodes: Sequence[OrNode], graph: AndOrGraph) -> None:
        costs = self.or_node_cost_fn(or_nodes, graph=graph)
        assert len(costs) == len(or_nodes)
        for node, cost in zip(or_nodes, costs):
            node.data["retro_star_mol_cost"] = cost

    def _set_and_node_costs(self, and_nodes: Sequence[AndNode], graph: AndOrGraph) -> None:
        costs = self.and_node_cost_fn(and_nodes, graph=graph)
        assert len(costs) == len(and_nodes)
        for node, cost in zip(and_nodes, costs):
            node.data["retro_star_rxn_cost"] = cost

    def _set_reaction_number_estimate(self, or_nodes: Sequence[OrNode], graph: AndOrGraph) -> None:
        costs = self.reaction_number_estimator(or_nodes, graph=graph)
        assert len(costs) == len(or_nodes)
        for node, cost in zip(or_nodes, costs):
            node.data["reaction_number_estimate"] = cost

    def _run_retro_star_updates(
        self, nodes: Collection[ANDOR_NODE], graph: AndOrGraph
    ) -> Collection[ANDOR_NODE]:
        # Initialize all reaction numbers and retro star values
        for node in nodes:
            node.data.setdefault("reaction_number", math.inf)
            node.data.setdefault("retro_star_value", math.inf)
        nodes_to_update = set(cast(Collection[ANDOR_NODE], nodes))

        # NOTE: the following updates assume that depth is set correctly.

        # Perform bottom-up update of `reaction number`,
        # sorting by decreasing depth and not updating children for efficiency
        # (reaction number depends only on children)
        nodes_to_update.update(
            cast(  # mypy doesn't know that `run_message_passing` returns a `Collection[ANDOR_NODE]`
                Collection[ANDOR_NODE],
                run_message_passing(
                    graph=graph,
                    nodes=sorted(nodes_to_update, key=lambda node: node.depth, reverse=True),
                    update_fns=[reaction_number_update],  # type: ignore[list-item]  # confusion about AndOrGraph type
                    update_predecessors=True,
                    update_successors=False,
                ),
            )
        )

        # Perform top-down update of retro-star value,
        # sorting by increasing depth and not updating parents for efficiency
        # (retro star value depends only on parents)
        nodes_to_update.update(
            cast(
                Collection[ANDOR_NODE],
                run_message_passing(
                    graph=graph,
                    nodes=sorted(nodes_to_update, key=lambda node: node.depth, reverse=False),
                    update_fns=[retro_star_value_update],  # type: ignore[list-item]  # confusion about AndOrGraph type
                    update_predecessors=False,
                    update_successors=True,
                ),
            )
        )

        return nodes_to_update

    def _descend_tree_and_choose_node(self, graph) -> OrNode:
        """Returns a leaf node on the optimal expansion route."""

        # Descend the tree along the optimal route to find candidate nodes,
        # ensuring not to visit duplicate nodes.
        candidate_nodes: list[OrNode] = []
        nodes_to_descend: deque[OrNode] = deque([graph.root_node])
        visited_nodes: set[OrNode] = set()
        target_retro_star_value = graph.root_node.data["best_retro_star_value"]
        while len(nodes_to_descend) > 0:
            node = nodes_to_descend.popleft()

            # Has the node been visited before?
            # If so skip it (don't process it multiple times).
            # If not, add it to the set of visited nodes.
            if node in visited_nodes:
                continue
            visited_nodes.add(node)

            children = list(graph.successors(node))
            if self.can_expand_node(node, graph):
                # No children, so this is a candidate node
                candidate_nodes.append(node)
            elif len(children) > 0:
                # Find AND children with matching best retro star value
                matching_children = [
                    child
                    for child in children
                    if math.isclose(child.data["best_retro_star_value"], target_retro_star_value)
                ]
                assert len(matching_children) > 0
                for and_node in matching_children:
                    for grandchild in graph.successors(and_node):
                        if math.isclose(
                            grandchild.data["best_retro_star_value"], target_retro_star_value
                        ):
                            assert isinstance(grandchild, OrNode)
                            nodes_to_descend.append(grandchild)

        # Now there should be at least one candidate node
        assert len(candidate_nodes) > 0

        # If there is more than 1 candidate, choose one with the lowest creation time
        return min(candidate_nodes, key=lambda node: node.creation_time)

    def _get_min_cost_ancenstors(
        self, graph: AndOrGraph, nodes: set[ANDOR_NODE]
    ) -> set[ANDOR_NODE]:
        """Retrive ancestor nodes whose minimum cost path depends on a given set of nodes."""
        ancestor_nodes: set[ANDOR_NODE] = set()  # only nodes which have already been processed
        queue = deque(nodes)  # only confirmed ancestors added (not yet processed)
        while len(queue) > 0:
            node = queue.popleft()

            # If the node has already been processed, skip it (don't process it multiple times)
            if node in ancestor_nodes:
                continue
            ancestor_nodes.add(node)

            # Add parents to the queue if they are eligible
            for parent in graph.predecessors(node):
                if isinstance(parent, AndNode):
                    # AndNodes are always eligible because they always require their children
                    queue.append(parent)  # will always get added
                elif isinstance(parent, OrNode):
                    # OrNodes are eligible if their reaction number matches
                    # and there is no alternative path to the same reaction number (e.g. through purchasing or another AND node)
                    if (
                        # reaction number matches
                        math.isclose(parent.data["reaction_number"], node.data["reaction_number"])
                        # cannot be purchased for this cost
                        and not math.isclose(
                            parent.data["reaction_number"], parent.data["retro_star_mol_cost"]
                        )
                        and not any(  # has a sibling with the same reaction number
                            sibling not in ancestor_nodes
                            and math.isclose(
                                sibling.data["reaction_number"], node.data["reaction_number"]
                            )
                            for sibling in graph.successors(parent)
                        )
                    ):
                        queue.append(parent)
        return ancestor_nodes

    def _run_from_graph_after_setup(self, graph) -> int:
        # Logging setup
        log_level = logging.DEBUG - 1
        logger_active = logger.isEnabledFor(log_level)

        # Run search until time limit or queue is empty
        step = 0
        for step in range(self.limit_iterations):
            eligible_nodes = [n for n in graph.nodes() if self.can_expand_node(n, graph)]
            if self.should_stop_search(graph) or len(eligible_nodes) == 0:
                break

            # Choose node with smallest retro_star_value
            # node = self._descend_tree_and_choose_node(graph)
            node = min(
                eligible_nodes,
                key=lambda n: (n.data["retro_star_value"], n.creation_time),
            )

            # Visit node
            new_nodes = list(self.visit_node(node, graph))

            # Perform pre-update initializations to infinity (prevents infinite loops)
            for n in new_nodes + [node]:
                n.data.setdefault("reaction_number", math.inf)  # necessary for check below
            min_cost_ancestors = self._get_min_cost_ancenstors(
                graph, set(new_nodes) | {node}  # type: ignore[arg-type]  # confusion about AndOrGraph type
            )
            for n in min_cost_ancestors:
                n.data["reaction_number"] = math.inf
                n.data["retro_star_value"] = math.inf
            init_str = f" initialized {len(min_cost_ancestors)} ancestors"

            # Update values of expanded node, current node, and ancestors
            nodes_updated = self.set_node_values(
                new_nodes
                + [node]
                + list(
                    min_cost_ancestors
                ),  # type: ignore[arg-type]  # confusion about AndOrGraph type
                graph,
            )

            # Log str
            if logger_active:
                logger.log(
                    log_level,
                    f"Step {step}: node={node},\n{len(new_nodes)} new nodes created, {len(nodes_updated)} nodes updated, "
                    f"{init_str}, graph size = {len(graph)}, calls to rxn model: {self.reaction_model.num_calls()}",
                )

        return step


def reaction_number_update(node: ANDOR_NODE, graph: AndOrGraph) -> bool:
    """
    Updates a node's "reaction number", which is the current minimum cost
    estimate of synthesizing a molecule from everything below it.
    Returns whether the node's reaction number was updated.
    """
    if isinstance(node, AndNode):
        # Reaction number from equation 7 in Retro*
        new_rn = node.data["retro_star_rxn_cost"] + sum(
            c.data["reaction_number"] for c in graph.successors(node)
        )
    elif isinstance(node, OrNode):
        # Reaction number is the minimum the molecule's purchase cost
        # and the cost of all child synthesis paths,
        # and potentially its reaction number estimate
        possible_costs = [node.data["retro_star_mol_cost"]]
        if node.is_expanded:
            # If the node is expanded, the cost of each child is also an option
            possible_costs.extend([c.data["reaction_number"] for c in graph.successors(node)])
        else:
            # Otherwise the cost of the reaction number estimate is an option.
            # This estimate must be present!
            possible_costs.append(node.data["reaction_number_estimate"])
        new_rn = min(possible_costs)
    else:
        raise TypeError(f"Unexpected node type: {type(node)}")

    # Do update and return whether the value changed
    old_rn = node.data["reaction_number"]
    node.data["reaction_number"] = new_rn
    return not math.isclose(new_rn, old_rn)


def retro_star_value_update(node: ANDOR_NODE, graph: AndOrGraph) -> bool:
    """
    Updates a node's "retro_star_value",
    which is the lowest total cost of any minimal AND/OR graph containing this node,
    rooted at the root node, assuming that the cost of any leaf node is its reaction number.
    This is called V(m|T) in the original Retro* paper (Chen et al 2020).

    Returns whether the node's retro_star_value changed.
    """

    parents = list(graph.predecessors(node))
    if isinstance(node, AndNode):
        # Cost is parent's value, - any contributions from other AND branches,
        # + the current reaction number
        assert len(parents) == 1
        parent = parents[0]
        assert isinstance(parent, OrNode)
        new_value = (
            parent.data["retro_star_value"]
            - parent.data["reaction_number"]
            + node.data["reaction_number"]
        )

        # Special cases to prevent NaNs
        # Could happen if things are initialized as infs,
        # or in certain cases with non-purchasable molecules.
        # In both cases the cause is inf-inf = nan
        if math.isnan(new_value):
            new_value = math.inf

    elif isinstance(node, OrNode):
        # r* Value estimate is minimum r* value of its parents,
        # except the root node: it's r* value estimate is just its RN
        if len(parents) == 0:
            # Root node
            new_value = node.data["reaction_number"]
        else:
            new_value = min(p.data["retro_star_value"] for p in parents)

    else:
        raise TypeError("Unexpected node type")

    # Do update and return whether the value changed
    old_value = node.data["retro_star_value"]
    node.data["retro_star_value"] = new_value
    return not math.isclose(new_value, old_value)


def best_retro_star_value_update(node: ANDOR_NODE, graph: AndOrGraph) -> bool:
    """
    Returns the best retro star value below this node.

    For an OrNode, there are 2 cases:
    1. Node can be expanded: therefore its best retro* value is its retro* value (it will have no children)
    2. Node cannot be expanded: therefore its best retro* value is the minimum of its children's best retro* values (or infinity)

    Information from the algorithm on whether a node can be expanded comes from "retro_star_can_expand" attribute.

    For an AND node, it should just be the best minimum of its children's best retro* values.
    """

    children = list(graph.successors(node))
    if isinstance(node, OrNode):
        if node.data.get("retro_star_can_expand", False):
            assert len(children) == 0
            new_value = node.data["retro_star_value"]
        else:
            new_value = min([math.inf] + [c.data["best_retro_star_value"] for c in children])
    elif isinstance(node, AndNode):
        new_value = math.inf  # default
        for child in children:
            new_value = min(new_value, child.data["best_retro_star_value"])
    else:
        raise TypeError("Unexpected node type")

    # Do update and return whether the value changed
    old_value = node.data.get("best_retro_star_value")
    node.data["best_retro_star_value"] = new_value
    return old_value is None or not math.isclose(new_value, old_value)
