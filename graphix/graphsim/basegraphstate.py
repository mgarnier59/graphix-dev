from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import networkx as nx
import networkx.classes.reportviews as nx_reportviews

from graphix._db import CLIFFORD_HSZ_DECOMPOSITION, CLIFFORD_MUL
from graphix.ops import Ops
from graphix.sim.statevec import Statevec

from .rxgraphviews import EdgeList, NodeList

if TYPE_CHECKING:
    from collections.abc import Iterator

RUSTWORKX_INSTALLED = False
try:
    import rustworkx as rx
    from rustworkx import PyGraph

    RUSTWORKX_INSTALLED = True
except ModuleNotFoundError:
    rx = None
    PyGraph = None

if sys.version_info >= (3, 10):
    NodesObject = nx_reportviews.NodeView | NodeList
    EdgesObject = nx_reportviews.EdgeView | EdgeList
    GraphObject = nx.Graph | PyGraph
else:
    from typing import Union

    NodesObject = Union[nx_reportviews.NodeView, NodeList]
    EdgesObject = Union[nx_reportviews.EdgeView, EdgeList]
    GraphObject = Union[nx.Graph, PyGraph]


class BaseGraphState(ABC):
    """Base class for graph state simulator.

    Performs Pauli measurements on graph states.
    You can choose between networkx and rustworkx as the backend.
    The default is rustworkx if installed, otherwise networkx.

    ref: M. Elliot, B. Eastin & C. Caves, JPhysA 43, 025301 (2010)
    and PRA 77, 042307 (2008)

    Each node has attributes:
        :`hollow`: True if node is hollow (has local H operator)
        :`sign`: True if node has negative sign (local Z operator)
        :`loop`: True if node has loop (local S operator)
    """

    def __init__(self):
        super().__init__()

    @property
    @abstractmethod
    def nodes(self) -> NodesObject:
        raise NotImplementedError

    @property
    @abstractmethod
    def edges(self) -> EdgesObject:
        raise NotImplementedError

    @property
    @abstractmethod
    def graph(self) -> GraphObject:
        raise NotImplementedError

    @abstractmethod
    def degree(self) -> Iterator[tuple[int, int]]:
        """Returns an iterator for (node, degree) tuples,
        where degree is the number of edges adjacent to the node
        """
        raise NotImplementedError

    @abstractmethod
    def neighbors(self, node: int) -> Iterator:
        """Returns an iterator over all neighbors of node n.

        Parameters
        ----------
        node : int
            A node in the graph

        Returns
        ----------
        iter
            An iterator over all neighbors of node n.
        """
        raise NotImplementedError

    @abstractmethod
    def subgraph(self, nodes: list) -> GraphObject:
        """Returns a subgraph of the graph.

        Parameters
        ----------
        nodes : list
            A list of node indices to generate the subgraph from.

        Returns
        ----------
        GraphObject
            A subgraph of the graph.
        """
        raise NotImplementedError

    @abstractmethod
    def number_of_edges(self, u: int | None = None, v: int | None = None) -> int:
        """Returns the number of edges between two nodes.

        Parameters
        ----------
        u : int, optional
            A node in the graph
        v : int, optional
            A node in the graph

        Returns
        ----------
        int
            The number of edges in the graph. If u and v are specified,
            return the number of edges between those nodes.
        """
        raise NotImplementedError

    @abstractmethod
    def adjacency(self) -> Iterator:
        """Returns an iterator over (node, adjacency dict) tuples for all nodes.

        Returns
        ----------
        iter
            An iterator over (node, adjacency dictionary) for all nodes in the graph.
        """
        raise NotImplementedError

    @abstractmethod
    def remove_node(self, node: int) -> None:
        """Remove a node from the graph.

        Parameters
        ----------
        node : int
            A node in the graph

        Returns
        ----------
        None
        """
        raise NotImplementedError

    @abstractmethod
    def remove_nodes_from(self, nodes: list[int]) -> None:
        """Remove all nodes specified in the list.

        Parameters
        ----------
        nodes : list
            A list of nodes to remove from the graph.

        Returns
        ----------
        None
        """
        raise NotImplementedError

    @abstractmethod
    def remove_edge(self, u: int, v: int) -> None:
        """Remove an edge from the graph.

        Parameters
        ----------
        u : int
            A node in the graph
        v : int
            A node in the graph

        Returns
        ----------
        None
        """
        raise NotImplementedError

    @abstractmethod
    def remove_edges_from(self, edges: list[tuple[int, int]]) -> None:
        """Remove all edges specified in the list.

        Parameters
        ----------
        edges : list of tuples
            A list of edges to remove from the graph.

        Returns
        ----------
        None
        """
        raise NotImplementedError

    def apply_vops(self, vops: dict) -> None:
        """Apply local Clifford operators to the graph state from a dictionary

        Parameters
        ----------
        vops : dict
            dict containing node indices as keys and
            local Clifford indices as values (see graphix.clifford.CLIFFORD)

        Returns
        ----------
        None
        """
        for node, vop in vops.items():
            for lc in reversed(CLIFFORD_HSZ_DECOMPOSITION[vop]):
                if lc == 3:
                    self.z(node)
                elif lc == 6:
                    self.h(node)
                elif lc == 4:
                    self.s(node)

    @abstractmethod
    def add_nodes_from(self, nodes: list[int]) -> None:
        """Add nodes and initialize node properties.

        Parameters
        ----------
        nodes : list[int]
            A list of nodes.

        Returns
        ----------
        None
        """
        raise NotImplementedError

    @abstractmethod
    def add_edges_from(self, edges: list[tuple[int, int]]) -> None:
        """Add edges and initialize node properties of newly added nodes.

        Parameters
        ----------
        edges : list[tuple[int, int]]
            must be given as list of 2-tuples (u, v)

        Returns
        ----------
        None
        """
        raise NotImplementedError

    @abstractmethod
    def get_isolates(self) -> list[int]:
        """Returns a list of isolated nodes (nodes with no edges).

        Returns
        ----------
        list[int]
            A list of isolated nodes.
        """
        raise NotImplementedError

    def get_vops(self) -> dict:
        """Apply local Clifford operators to the graph state from a dictionary

        Parameters
        ----------
            vops : dict
                dict containing node indices as keys and
                local Clifford indices as values (see graphix.clifford.CLIFFORD)
        """
        vops = {}
        for i in self.nodes:
            vop = 0
            if self.nodes[i]["sign"]:
                vop = CLIFFORD_MUL[3][vop]
            if self.nodes[i]["loop"]:
                vop = CLIFFORD_MUL[4][vop]
            if self.nodes[i]["hollow"]:
                vop = CLIFFORD_MUL[6][vop]
            vops[i] = vop
        return vops

    def flip_fill(self, node: int) -> None:
        """Flips the fill (local H) of a node.

        Parameters
        ----------
        node : int
            graph node to flip the fill

        Returns
        ----------
        None
        """
        self.nodes[node]["hollow"] = not self.nodes[node]["hollow"]

    def flip_sign(self, node) -> None:
        """Flips the sign (local Z) of a node.
        Note that application of Z gate is different from `flip_sign`
        if there exist an edge from the node.

        Parameters
        ----------
        node : int
            graph node to flip the sign

        Returns
        ----------
        None
        """
        self.nodes[node]["sign"] = not self.nodes[node]["sign"]

    def advance(self, node: int) -> None:
        """Flips the loop (local S) of a node.
        If the loop already exist, sign is also flipped,
        reflecting the relation SS=Z.
        Note that application of S gate is different from `advance`
        if there exist an edge from the node.

        Parameters
        ----------
        node : int
            graph node to advance the loop.

        Returns
        ----------
        None
        """
        if self.nodes[node]["loop"]:
            self.nodes[node]["loop"] = False
            self.flip_sign(node)
        else:
            self.nodes[node]["loop"] = True

    def h(self, node: int) -> None:
        """Apply H gate to a qubit (node).

        Parameters
        ----------
        node : int
            graph node to apply H gate

        Returns
        ----------
        None
        """
        self.flip_fill(node)

    def s(self, node: int) -> None:
        """Apply S gate to a qubit (node).

        Parameters
        ----------
        node : int
            graph node to apply S gate

        Returns
        ----------
        None
        """
        if self.nodes[node]["hollow"]:
            if self.nodes[node]["loop"]:
                self.flip_fill(node)
                self.nodes[node]["loop"] = False
                self.local_complement(node)
                for i in self.neighbors(node):
                    self.advance(i)
            else:
                self.local_complement(node)
                for i in self.neighbors(node):
                    self.advance(i)
                if self.nodes[node]["sign"]:
                    for i in self.neighbors(node):
                        self.flip_sign(i)
        else:  # solid
            self.advance(node)

    def z(self, node: int) -> None:
        """Apply Z gate to a qubit (node).

        Parameters
        ----------
        node : int
            graph node to apply Z gate

        Returns
        ----------
        None
        """
        if self.nodes[node]["hollow"]:
            for i in self.neighbors(node):
                self.flip_sign(i)
            if self.nodes[node]["loop"]:
                self.flip_sign(node)
        else:  # solid
            self.flip_sign(node)

    # add a CZ method to allow arbitrary inputs.
    def cz(self, node1: int, node2: int) -> None:
        """Apply Z gate to a qubit (node).

        Parameters
        ----------
        node1 : int
            First graph node on which to apply CZ gate
        node2 : int
            Second graph node on which to apply CZ gate

        Returns
        ----------
        None
        """
        # pseudocode first!
        # check RX/N subtletities later
        # hollow and solid are inverse

        # TODO convert to equivalent reduced graph first !!!! Implemented?

        # TODO find better format for the conditions

        # rule 8: two nodes are solid
        if not self.nodes[node1]["hollow"] and not self.nodes[node2]["hollow"]:
            # need to complement THE EDGE. Implement methods in nx/rxgraphstates?
            if (node1, node2) in self.edges:  # check types are ok for checking will work nx not rx
                self.remove_edge(node1, node2)
            else:
                self.add_edges_from([(node1, node2)])  # why linter warning?

        # rule 9: one of the two is hollow (mutually exclusive with rule 8)
        # REFACTOR: same code for both branches with different node identification..

        elif self.nodes[node1]["hollow"] and not self.nodes[node2]["hollow"]:
            hollow_node, solid_node = node1, node2

            # this seems to work
            target_nodes = [solid_node, *list(self.neighbors(hollow_node))]
            self.complement(target_nodes)

            if ((node1, node2) in self.edges and not self.nodes[hollow_node]["sign"]) or (
                (node1, node2) not in self.edges and self.nodes[hollow_node]["sign"]
            ):
                self.flip_sign(solid_node)

        elif not self.nodes[node1]["hollow"] and self.nodes[node2]["hollow"]:
            hollow_node, solid_node = node2, node1

            # this seems to work
            target_nodes = [solid_node, *list(self.neighbors(hollow_node))]
            self.complement(target_nodes)

            if ((node1, node2) in self.edges and not self.nodes[hollow_node]["sign"]) or (
                (node1, node2) not in self.edges and self.nodes[hollow_node]["sign"]
            ):
                self.flip_sign(solid_node)

        # rule 10: two nodes are hollow (mutually exclusive with rules 8 and 9)
        elif self.nodes[node1]["hollow"] and self.nodes[node2]["hollow"]:
            assert (node1, node2) not in self.edges
            self.pivot_cz((node1, node2))

        return

    def equivalent_graph_e1(self, node: int) -> None:
        """Tranform a graph state to a different graph state
        representing the same stabilizer state.
        This rule applies only to a node with loop.

        Parameters
        ----------
        node1 : int
            A graph node with a loop to apply rule E1

        Returns
        ----------
        None
        """
        if not self.nodes[node]["loop"]:
            raise ValueError("node must have loop")
        self.flip_fill(node)
        self.local_complement(node)
        for i in self.neighbors(node):
            self.advance(i)
        self.flip_sign(node)
        if self.nodes[node]["sign"]:
            for i in self.neighbors(node):
                self.flip_sign(i)

    def equivalent_graph_e2(self, node1: int, node2: int) -> None:
        """Tranform a graph state to a different graph state
        representing the same stabilizer state.
        This rule applies only to two connected nodes without loop.

        Parameters
        ----------
        node1, node2 : int
            connected graph nodes to apply rule E2

        Returns
        ----------
        None
        """
        if (node1, node2) not in self.edges and (node2, node1) not in self.edges:
            raise ValueError("nodes must be connected by an edge")
        if self.nodes[node1]["loop"] or self.nodes[node2]["loop"]:
            raise ValueError("nodes must not have loop")
        sg1 = self.nodes[node1]["sign"]
        sg2 = self.nodes[node2]["sign"]
        self.flip_fill(node1)
        self.flip_fill(node2)
        # local complement along edge between node1, node2
        self.local_complement(node1)
        self.local_complement(node2)
        self.local_complement(node1)
        for i in iter(set(self.neighbors(node1)) & set(self.neighbors(node2))):
            self.flip_sign(i)
        if sg1:
            self.flip_sign(node1)
            for i in self.neighbors(node1):
                self.flip_sign(i)
        if sg2:
            self.flip_sign(node2)
            for i in self.neighbors(node2):
                self.flip_sign(i)

    @abstractmethod
    def complement(self, nodes: list[int]) -> None:
        """Perform complementation with respect to a set of nodes

        Parameters
        ----------
        node : list[int]
            chosen nodes for the local complementation

        Returns
        ----------
        None
        """
        raise NotImplementedError

    # @abstractmethod: no longer abstract since relies on complement
    def local_complement(self, node: int) -> None:
        """Perform local complementation of a graph on a node

        Parameters
        ----------
        node : int
            chosen node for the local complementation

        Returns
        ----------
        None
        """
        # new local_complement method
        self.complement(list(self.neighbors(node)))

        return

    def pivot(self, edge: tuple[int, int]) -> None:
        """Perform pivoting [1]_ a.k.a local complementation along an edge [2]_ of a graph.
        Pivoting along the edge (u,v) means applying local complementation on u then local complementation on
        v then local complementation on u again. Furthermore, the operation is symmetric in u and v.
        There exist alternative way of performing pivoting [1]_ [2]_

        .. [1] Backens et al., "There and back again: a circuit extraction tale", Quantum 5, 421, 2021: Def 2.25 and Remark 2.26.
        .. [2] Elliott et al., "Graphical description of the action of Clifford operators on stabilizer states", Phys. Rev. A, 77(4), 2008: bottom of left column and top of right column, p. 5.

        Parameters
        ----------
        edge : int
            chosen edge for the pivoting

        Returns
        ----------
        None

        Raises
        ------
        ValueError
            if the parameter edge is not an edge of the graph.

        See Also
        --------
        pivot_alt:
            Alternative implementation.
        """
        # TODO Network use has_edge
        # https://networkx.org/documentation/stable/reference/classes/generated/networkx.Graph.has_edge.html
        # if edge not in self.edges:
        if edge not in self.edges and edge[::-1] not in list(self.edges):
            raise ValueError("Cannot pivot along an edge that is not an edge of the graph.")

        self.local_complement(edge[0])
        self.local_complement(edge[1])
        self.local_complement(edge[0])

        return

    def pivot_alt(self, edge: tuple[int, int]) -> None:
        """Perform pivoting a.k.a local complementation along an edge of a graph using an alternative way [1]_ [2]_

        .. [1] Backens et al., "There and back again: a circuit extraction tale", Quantum 5, 421, 2021: Def 2.25, Remark 2.26 and in between.
        .. [2] Elliott et al., "Graphical description of the action of Clifford operators on stabilizer states", Phys. Rev. A, 77(4), 2008: bottom of left column and top of right column, p. 5.

        Parameters
        ----------
        edge : int
            chosen edge for the pivoting

        Returns
        ----------
        None

        Raises
        ------
        ValueError
            if the parameter edge is not an edge of the graph.

        See Also
        --------
        pivot:
            Alternative implementation.
        """
        # if edge not in self.edges:
        if (edge not in self.edges) and (edge[::-1] not in set(self.edges)):
            raise ValueError("Cannot pivot along an edge that is not an edge of the graph.")

        node1 = edge[0]
        node2 = edge[1]
        # neighborhoods of node1 and node2 without node1 and node 2.
        neighbors_node1 = set(self.neighbors(node1)) - set([node2])
        neighbors_node2 = set(self.neighbors(node2)) - set([node1])

        # neighbors of node1 only
        neighbors_node1_only = neighbors_node1 - neighbors_node2

        # neighbors of node2 only
        neighbors_node2_only = neighbors_node2 - neighbors_node1

        # neighbors of both node1 and node2 (intersection)
        neighbors_both = neighbors_node1 & neighbors_node2

        # exchange neighborhoods

        # sets are iterable
        # NOTE: smarter to use itertools.product? But watch out if same edge removed several time.
        # seems suboptimal
        self.remove_edges_from(list((node1, i) for i in neighbors_node1_only))
        self.add_edges_from(list((node1, i) for i in neighbors_node2_only))

        self.remove_edges_from(list((node2, i) for i in neighbors_node2_only))
        self.add_edges_from(list((node2, i) for i in neighbors_node1_only))

        # complement the ORIGINAL sets, not the new ones!
        # neighbors_node1 and neighbors_node_2
        self.complement(list(neighbors_node1) + list(neighbors_node2))

        # neighbors_node1 and neighbors_both
        self.complement(list(neighbors_node1) + list(neighbors_both))

        # neighbors_node2 and neighbors_both
        self.complement(list(neighbors_node2) + list(neighbors_both))

        return

    # RESTART HERE
    # doesn't need to be abstract since only depend on flips sign/fill and (local) complementation?
    def pivot_cz(self, edge: tuple[int, int]) -> None:
        """Modified pivoting (a.k.a local complementation along an edge) for CZ gate application [1]_. We use the alternative formulation from [1]_ [2]_.
        Should only be applied to two hollow nodes. Note that in reduced graph form, two hollow nodes are _never_ connected [1]_.

        .. [1] Elliott et al., "Graphical description of the action of Clifford operators on stabilizer states", Phys. Rev. A, 77(4), Appendix and top of right column, p. 5.
            For reduced graphs and hollow nodes Section V C 1.
        .. [2] Backens et al., "There and back again: a circuit extraction tale", Quantum 5, 421, 2021, between Def. 2.25 and Remark 2.26.


        Parameters
        ----------
        edge : int
            chosen edge for the pivoting

        Returns
        ----------
        None

        See Also
        --------
        pivot_alt:
            original implementation
        """
        node1 = edge[0]
        node2 = edge[1]
        # neighborhoods of node1 and node2 without node1 and node 2.
        # edge between node1 and node 2 unchanged
        neighbors_node1 = set(self.neighbors(node1)) - set([node2])
        neighbors_node2 = set(self.neighbors(node2)) - set([node1])

        # neighbors of node1 only
        neighbors_node1_only = neighbors_node1 - neighbors_node2

        # neighbors of node2 only
        neighbors_node2_only = neighbors_node2 - neighbors_node1

        # neighbors of both node1 and node2 (intersection)
        neighbors_both = neighbors_node1 & neighbors_node2

        print(f"neigh 1 only {neighbors_node1_only}")
        print(f"neigh 2 only {neighbors_node2_only}")
        print(f"neigh both {neighbors_both}")
        # exchange neighborhoods

        # sets are iterable
        # NOTE: smarter to use itertools.product? But watch out if same edge removed several time.
        # seems suboptimal
        self.remove_edges_from(list((i, node1) for i in neighbors_node1_only))
        self.add_edges_from(list((node1, i) for i in neighbors_node2_only))

        self.remove_edges_from(list((i, node2) for i in neighbors_node2_only))
        self.add_edges_from(list((node2, i) for i in neighbors_node1_only))

        print("new neigh 1", list(self.neighbors(node1)))
        print("new neigh 2", list(self.neighbors(node2)))

        # now complementation à la Backens
        # modify for CZ later
        # complement links for all three sets: neighbors_node1_only, neighbors_node2_only and neighbors_both

        # neighbors_node1 and neighbors_node2
        # this should work as intended
        self.complement(list(neighbors_node1) + list(neighbors_node2))

        # naive first version
        # this is enough for the first couple of sets since does all the nodes that are linked.
        # for i in neighbors_node1:
        #     for j in self.neighbors(i):
        #         if j in neighbors_node2:
        #             self.complement([i, j])

        # neighbors_node1 and neighbors_both
        self.complement(list(neighbors_node1) + list(neighbors_both))

        # neighbors_node2 and neighbors_both
        self.complement(list(neighbors_node2) + list(neighbors_both))

        # this is wrong, the edge can not be there (my use!)
        # if edge not in self.edges:
        #     raise ValueError("Cannot pivot along an edge that is not an edge of the graph.")

        # find relevant subsets: neighbours of edge[1] only, edge[2] only and edge [1] and edge [2]

        # makes sense to do it here since need the info on the neighbors sets
        # text not very clear. Trying...
        # end of rule 10:
        for i in neighbors_both:
            self.flip_sign(i)

        # if one or the other have sign, flip neigh of the other
        # REFACTOR
        if self.nodes[node1]["sign"]:
            for i in neighbors_node2_only:  # exclude node1 I suppose.
                self.flip_sign(i)

        elif self.nodes[node2]["sign"]:
            for i in neighbors_node1_only:  # exclude node1 I suppose.
                self.flip_sign(i)

        return

    def equivalent_fill_node(self, node: int) -> int:
        """Fill the chosen node by graph transformation rules E1 and E2,
        If the selected node is hollow and isolated, it cannot be filled
        and warning is thrown.

        Parameters
        ----------
        node : int
            node to fill.

        Returns
        ----------
        result : int
            if the selected node is hollow and isolated, `result` is 1.
            if filled and isolated, 2.
            otherwise it is 0.
        """
        if self.nodes[node]["hollow"]:
            if self.nodes[node]["loop"]:
                self.equivalent_graph_e1(node)
                return 0
            else:  # node = hollow and loopless
                if len(list(self.neighbors(node))) == 0:
                    return 1
                for i in self.neighbors(node):
                    if not self.nodes[i]["loop"]:
                        self.equivalent_graph_e2(node, i)
                        return 0
                # if all neighbor has loop, pick one and apply E1, then E1 to the node.
                i = next(self.neighbors(node))
                self.equivalent_graph_e1(i)  # this gives loop to node.
                self.equivalent_graph_e1(node)
                return 0
        else:
            if len(list(self.neighbors(node))) == 0:
                return 2
            else:
                return 0

    def measure_x(self, node: int, choice: int = 0) -> int:
        """perform measurement in X basis
        According to original paper, we realise X measurement by
        applying H gate to the measured node before Z measurement.

        Parameters
        ----------
        node : int
            qubit index to be measured
        choice : int, 0 or 1
            choice of measurement outcome. observe (-1)^choice

        Returns
        ----------
        result : int
            measurement outcome. 0 or 1.
        """
        if choice not in [0, 1]:
            raise ValueError("choice must be 0 or 1")
        # check if isolated
        if len(list(self.neighbors(node))) == 0:
            if self.nodes[node]["hollow"] or self.nodes[node]["loop"]:
                choice_ = choice
            elif self.nodes[node]["sign"]:  # isolated and state is |->
                choice_ = 1
            else:  # isolated and state is |+>
                choice_ = 0
            self.remove_node(node)
            return choice_
        else:
            self.h(node)
            return self.measure_z(node, choice=choice)

    def measure_y(self, node: int, choice: int = 0) -> int:
        """perform measurement in Y basis
        According to original paper, we realise Y measurement by
        applying S,Z and H gate to the measured node before Z measurement.

        Parameters
        ----------
        node : int
            qubit index to be measured
        choice : int, 0 or 1
            choice of measurement outcome. observe (-1)^choice

        Returns
        ----------
        result : int
            measurement outcome. 0 or 1.
        """
        if choice not in [0, 1]:
            raise ValueError("choice must be 0 or 1")
        self.s(node)
        self.z(node)
        self.h(node)
        return self.measure_z(node, choice=choice)

    def measure_z(self, node: int, choice: int = 0) -> int:
        """perform measurement in Z basis
        To realize the simple Z measurement on undecorated graph state,
        we first fill the measured node (remove local H gate)

        Parameters
        ----------
        node : int
            qubit index to be measured
        choice : int, 0 or 1
            choice of measurement outcome. observe (-1)^choice

        Returns
        ----------
        result : int
            measurement outcome. 0 or 1.
        """
        if choice not in [0, 1]:
            raise ValueError("choice must be 0 or 1")
        isolated = self.equivalent_fill_node(node)
        if choice:
            for i in self.neighbors(node):
                self.flip_sign(i)
        if not isolated:
            result = choice
        else:
            result = int(self.nodes[node]["sign"])
        self.remove_node(node)
        return result

    def draw(self, fill_color: str = "C0", **kwargs):
        """Draw decorated graph state.
        Negative nodes are indicated by negative sign of node labels.

        Parameters
        ----------
        fill_color : str, optional
            fill color of nodes
        kwargs : keyword arguments, optional
            additional arguments to supply networkx.draw().
        """
        nqubit = len(self.nodes)
        nodes = list(self.nodes)
        edges = list(self.edges)
        labels = {i: i for i in iter(self.nodes)}
        colors = [fill_color for _ in range(nqubit)]
        for i in range(nqubit):
            if self.nodes[nodes[i]]["loop"]:
                edges.append((nodes[i], nodes[i]))
            if self.nodes[nodes[i]]["hollow"]:
                colors[i] = "white"
            if self.nodes[nodes[i]]["sign"]:
                labels[nodes[i]] = -1 * labels[nodes[i]]
        g = nx.Graph()
        g.add_nodes_from(nodes)
        g.add_edges_from(edges)
        nx.draw(g, labels=labels, node_color=colors, edgecolors="k", **kwargs)

    def to_statevector(self) -> Statevec:
        node_list = list(self.nodes)
        nqubit = len(self.nodes)
        gstate = Statevec(nqubit=nqubit)
        # map graph node indices into 0 - (nqubit-1) for qubit indexing in statevec
        imapping = {node_list[i]: i for i in range(nqubit)}
        mapping = [node_list[i] for i in range(nqubit)]
        for i, j in self.edges:
            gstate.entangle((imapping[i], imapping[j]))
        for i in range(nqubit):
            if self.nodes[mapping[i]]["sign"]:
                gstate.evolve_single(Ops.z, i)
        for i in range(nqubit):
            if self.nodes[mapping[i]]["loop"]:
                gstate.evolve_single(Ops.s, i)
        for i in range(nqubit):
            if self.nodes[mapping[i]]["hollow"]:
                gstate.evolve_single(Ops.h, i)
        return gstate
