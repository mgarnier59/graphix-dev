"""An example of how to generate pseudorandomness based on [1]
R. Mezher, J. Ghalbouni, J. Dgheim, and D. Markham, Efficient Quantum Pseudorandomness with Simple Graph States, Physical Review A 97, (2018).
"""

# eq 20 https://arxiv.org/abs/2205.09900 use schmidt decomp without sing value? just basis expansion?
# TODO plot state framepotential from https://arxiv.org/abs/2410.23353v1 Eq 4 as a function of probability and k
from __future__ import annotations

from collections.abc import Generator, Sequence
from itertools import combinations
import math
from typing import TYPE_CHECKING, Any

import networkx as nx
import numpy as np

from graphix.command import CommandKind, M, Node, X, Z
from graphix.fundamentals import Plane
from graphix.measurements import Measurement
from graphix.opengraph import OpenGraph
from graphix.pattern import Pattern

if TYPE_CHECKING:
    from graphix.sim.base_backend import BackendState
    from graphix.sim.statevec import Statevec

# TODO: use gospel brickwork transpiler


def build_brickwork_pattern(meas: dict[int, Measurement]) -> Pattern:
    """Function to buid the pattern corresponding to a single brick.
    The only free parameters are the measurment angles.
    Numbering is as follows

    0 - 1 - 2 - 3 - 4
            |       |
    5 - 6 - 7 - 8 - 9

    Parameters
    ----------
    meas : dict[int, Measurement]
        measurement of each node to be measured (all except outputs)

    Returns
    -------
    Pattern
        Final brickwork pattern
    """

    # brickwork graph numbering

    inside_graph = nx.Graph([(0, 1), (1, 2), (2, 3), (3, 4), (5, 6), (6, 7), (7, 8), (8, 9), (2, 7), (4, 9)])
    inputs = [0, 5]
    outputs = [4, 9]

    return OpenGraph(inside_graph, meas, inputs, outputs).to_pattern()


def remove_corrections(pattern: Pattern, rng: Generator, prob: float | None = None) -> Pattern:
    """_summary_

    Parameters
    ----------
    pattern : Pattern
        _description_
    prob : float | None, optional
        _description_, by default None

    Returns
    -------
    Pattern
        _description_
    """
    # remove correcctions. If no priob, remove all, else draws and remove probabilistically
    # can modify pattern in place?
    # prob is prob to remove

    new_pattern = Pattern(input_nodes=[0, 5])

    for command in pattern:
        # don't modify preparations and entanglements

        if command.kind == CommandKind.M:
            s_domain, t_domain = command.s_domain, command.t_domain

            if prob is None:
                new_command = M(
                    node=command.node, plane=command.plane, angle=command.angle, s_domain=set(), t_domain=set()
                )
            else:
                # print("start prob")
                new_s_domain, new_t_domain = (
                    remove_corr_util(s_domain, prob, rng),
                    remove_corr_util(t_domain, prob, rng),
                )
                new_command = M(
                    node=command.node,
                    plane=command.plane,
                    angle=command.angle,
                    s_domain=new_s_domain,
                    t_domain=new_t_domain,
                )

        elif command.kind in {CommandKind.X, CommandKind.Z}:
            if prob is None:
                if command.kind == CommandKind.X:
                    new_command = X(node=command.node, domain=set())

                elif command.kind == CommandKind.Z:
                    new_command = Z(node=command.node, domain=set())
            else:
                new_domain = remove_corr_util(command.domain, prob, rng)

                if command.kind == CommandKind.X:
                    new_command = X(node=command.node, domain=new_domain)
                elif command.kind == CommandKind.Z:
                    new_command = Z(node=command.node, domain=new_domain)

        else:
            new_command = command  # type issue for some reason

        # print(command, new_pattern.output_nodes)
        new_pattern.add(new_command)  # type issue for some reason

    return new_pattern


def remove_corr_util(
    domain: set[Node], prob: float, rng: Generator
) -> set[Node]:  # TODO fix generator parameters types
    """util to update sets given a probability to REMOVE elements form it

    Parameters
    ----------
    domain : set[Node]
        set to be modified
    prob : float
        probability of REMOVING an element. Same for all elements

    Returns
    -------
    set[Node]
        updated set
    """
    # only keep if drawn number is larger
    return {i for i in domain if rng.random() > prob}  # type: ignore


# cache to avoid recomputing? or too big?


def fidelity(state1: Statevec, state2: Statevec) -> float:
    return np.abs(state1.psi.flatten().dot(state2.psi.flatten().conj())) ** 2


def compute_state_frame_potential(ens: Sequence[Statevec], k: int) -> np.floating[Any]:  # TODO fix type # also DMs later. Check with TM for backend typing
    """Returns the k-th state frame potential computed over the state ensemble 'ens.
    Applies to Statevecs only so far.
    Computed according to Eq. (4) of
    Nakata, Takeuchi, Kliesch and Darmawan, On computational complexity of unitary and state design properties (2024) https://arxiv.org/abs/2410.23353v1

    Parameters
    ----------
    ens : Sequence[BackendState]
        Ensemble over which to compute the frame potentials
    k : int
        integer

    Returns
    -------
    float
        the computed value
    """
    data: list[float] = []
    state1: Statevec
    state2: Statevec
    for state1, state2 in combinations(ens, 2):
        data.append(fidelity(state1, state2) ** k)

    return np.mean(np.array(data, dtype=np.float64))


def run(shots: int, rng: Generator, prob: float | None = None) -> Sequence[BackendState]:
    measurements1 = {i: Measurement(0, Plane.XY) for i in [1, 3, 5, 7]}
    measurements1.update({i: Measurement(1 / 4, Plane.XY) for i in [0, 2, 6]})
    measurements1.update({8: Measurement(1 / 2, Plane.XY)})

    measurements2 = {i: Measurement(0, Plane.XY) for i in [0, 2, 6, 8]}
    measurements2.update({i: Measurement(1 / 4, Plane.XY) for i in [1, 5, 7]})
    measurements2.update({3: Measurement(1 / 2, Plane.XY)})

    p1 = build_brickwork_pattern(measurements1)
    p2 = build_brickwork_pattern(measurements2)

    # input nodes of p2 to input nodes of p1 :
    mapping = {0: 4, 5: 9}
    pc, _ = p1.compose(p2, mapping)  # mapping_c is need to access it
    pc.standardize()

    # reference pattern is deterministic so pick one of its outputs
    ref = pc.simulate_pattern()

    new_patt = remove_corrections(pc, rng, prob)
    # also sample on all possible patterns if exists some probability?
    # For now just focus on prob = None
    # check determinism
    out_data = []
    for _ in range(shots):
        # compute probability by default
        # no need to reinitialise the backend, it's done internally
        out = new_patt.simulate_pattern()
        out_data.append(out)  # TODO solve typing issue cast as Statevec?
        # fid = np.abs(out.psi.flatten().dot(ref.psi.flatten().conj())) ** 2
        # fid_data.append(fid)
    # print("mean", np.mean(fid_data))

    print(compute_state_frame_potential(out_data, 1))

    return out_data


# TODO
# and check multiple shots
# do flow
# remove corrections (with probability) to match Raussendorf and Briegel.
# frame potential [35], [36], [38], [39] of https://arxiv.org/abs/2505.21621 (Eq. 3)
# use loop over all branch selector if done otherwise sample.
# need to find minimal value for design (which it has to be)
# also look at Briegel and Raussendorf papers


if __name__ == "__main__":
    print("in pseudorand")
    d = 4
    t = 1
    rng = np.random.default_rng()
    data = run(shots=100, rng=rng, prob=0.05)  # test prob + None and 1 should be the same
    print(f"analytical lower bound {1/math.comb(d + t - 1, t)}")
    # print(len(data), data[0:2])
    # nx.draw_networkx(graph, with_labels=True) # add layers https://networkx.org/documentation/stable/reference/generated/networkx.drawing.layout.multipartite_layout.html
    # plt.show()
    # print(og1)
    # print(len(measurements1), measurements1)
    # print(p1)
    # # print(p1.to_latex())
    # print(pc, mapping_c)
    # print(pc.output_nodes)

    # for _ in range(5):
    #     out = pc.simulate_pattern()
    #     fid = np.abs(out.psi.flatten().dot(ref.psi.flatten().conj())) ** 2
    #     print(out, fid)

    # print(remove_corrections(patt, rng, 0.5))
