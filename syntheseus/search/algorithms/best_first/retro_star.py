from __future__ import annotations

import logging
import math
from collections import defaultdict, deque
from collections.abc import Sequence

# NOTE: Collection imported here instead of from collections.abc
# to make casting work for python <3.9
from typing import (
    Callable,
    Collection,
    Optional,
    cast,
)

from syntheseus.search.algorithms.base import AndOrSearchAlgorithm
from syntheseus.search.algorithms.best_first.base import GeneralBestFirstSearch
from syntheseus.search.algorithms.mixins import ValueFunctionMixin
from syntheseus.search.graph.and_or import ANDOR_NODE, AndNode, AndOrGraph, OrNode
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
                and self.can_expand_node(node, graph)
            ],
            graph=graph,
        )

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
        logger.log(logging.DEBUG - 1, f"Starting with {len(nodes_to_update)} nodes to update")

        # NOTE: the following updates assume that depth is set correctly.

        trigger_n = 1_000
        reset_n = 100

        # Perform bottom-up update of `reaction number`,
        # sorting by decreasing depth and not updating children for efficiency
        # (reaction number depends only on children)
        rn_nodes, rn_iter, _ = robust_message_passing(
            nodes=sorted(nodes_to_update, key=lambda node: node.depth, reverse=True),
            graph=graph,
            update_fn=reaction_number_update,
            update_predecessors=True,
            update_successors=False,
            reset_function=lambda node, graph: node.data.__setitem__("reaction_number", math.inf),
            num_visits_to_trigger_reset=trigger_n,
            reset_visit_threshold=reset_n,
        )
        logger.log(
            logging.DEBUG - 1,
            f"Reaction number updates: {len(rn_nodes)} / {len(graph)} nodes updated in {rn_iter} iterations",
        )
        nodes_to_update.update(rn_nodes)

        # Perform top-down update of retro-star value,
        # sorting by increasing depth and not updating parents for efficiency
        # (retro star value depends only on parents)
        rv_nodes, rv_iter, _ = robust_message_passing(
            nodes=sorted(nodes_to_update, key=lambda node: node.depth, reverse=False),
            graph=graph,
            update_fn=retro_star_value_update,
            update_predecessors=False,
            update_successors=True,
            reset_function=lambda node, graph: node.data.__setitem__("retro_star_value", math.inf),
            num_visits_to_trigger_reset=trigger_n,
            reset_visit_threshold=reset_n,
        )
        logger.log(
            logging.DEBUG - 1,
            f"Retro star value updates: {len(rv_nodes)} / {len(graph)} nodes updated in {rv_iter} iterations",
        )
        nodes_to_update.update(rv_nodes)

        return nodes_to_update


def reaction_number_update(node: ANDOR_NODE, graph: AndOrGraph) -> bool:
    """
    Updates a node's "reaction number", which is the lowest total
    cost of any minimal AND/OR graph containing this node,
    rooted at the root node, assuming that the cost of any leaf
    node is its reaction number.
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
        elif "reaction_number_estimate" in node.data:
            # Otherwise the cost of the reaction number estimate is an option.
            # By design, it will only be present if the node can be expanded
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
    which is the lowest total cost of any tree containing this node,
    rooted at the root node, assuming that the current costs of each node
    are correct (which is probably not the case for unexpanded nodes).
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
        # r* Value estimate is parent's value (this has no double counting)
        # Except the root node: it's r* value estimate is just its RN
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


message_passing_logger = logging.getLogger("retro-star-robust-message-passing")


def robust_message_passing(
    nodes: Collection[ANDOR_NODE],
    graph: AndOrGraph,
    update_fn: Callable[[ANDOR_NODE, AndOrGraph], bool],
    update_predecessors: bool = True,
    update_successors: bool = True,
    reset_function: Optional[Callable[[ANDOR_NODE, AndOrGraph], None]] = None,
    num_visits_to_trigger_reset: int = 10_000,
    reset_visit_threshold: int = 1000,
) -> tuple[set[ANDOR_NODE], int, int]:
    """
    Modified version of message passing which tracks how many times each node has
    been visited and will "reset" nodes which have been visited too many times.
    This reset uperation can be customized using the `reset_function` argument.

    Specifically, once a node has been updated at least `num_visits_to_trigger_reset` times,
    all nodes visited more than `reset_visit_threshold` times are reset using the `reset_function`.
    """

    assert num_visits_to_trigger_reset >= reset_visit_threshold

    # Initialize
    update_queue = deque(nodes)
    node_to_num_update_without_reset: defaultdict[ANDOR_NODE, int] = defaultdict(int)

    def _queue_adding(node):
        if update_predecessors:
            update_queue.extend(graph.predecessors(node))
        if update_successors:
            update_queue.extend(graph.successors(node))

    # Visit nodes
    n_iter = 0
    n_reset = 0
    while len(update_queue) > 0:
        n_iter += 1
        node = update_queue.popleft()

        # Perform standard update
        if update_fn(node, graph):
            node_to_num_update_without_reset[node] += 1
            _queue_adding(node)

            # Possibly add
            if (
                node_to_num_update_without_reset[node] >= num_visits_to_trigger_reset
                and reset_function is not None
            ):
                message_passing_logger.debug("Doing reset")
                n_reset += 1

                # Reset all nodes updated too many times
                for n, num_updates in node_to_num_update_without_reset.items():
                    if num_updates >= reset_visit_threshold:
                        reset_function(n, graph)
                        node_to_num_update_without_reset[n] = 0
                        update_queue.append(
                            n
                        )  # we updated its value so it should go back into the queue
                        if (
                            n is not node
                        ):  # don't add children of the node we're currently updating twice
                            _queue_adding(n)

    return set(node_to_num_update_without_reset.keys()), n_iter, n_reset
