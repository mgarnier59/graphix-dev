"""An example of how to generate pseudorandomness based on [1]
R. Mezher, J. Ghalbouni, J. Dgheim, and D. Markham, Efficient Quantum Pseudorandomness with Simple Graph States, Physical Review A 97, (2018).
"""

# eq 20 https://arxiv.org/abs/2205.09900 use schmidt decomp without sing value? just basis expansion?
# TODO plot state framepotential from https://arxiv.org/abs/2410.23353v1 Eq 4 as a function of probability and k
from __future__ import annotations

import math
from itertools import combinations, starmap
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from graphix.command import Command, CommandKind, M, Node, X, Z
from graphix.fundamentals import Plane
from graphix.measurements import Measurement
from graphix.opengraph import OpenGraph
from graphix.pattern import Pattern
from graphix.sim.statevec import StatevectorBackend

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.random import Generator

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


def remove_corrections(pattern: Pattern, rng: Generator, prob: float) -> Pattern:
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

    new_pattern = Pattern(input_nodes=pattern.input_nodes)

    for command in pattern:
        # don't modify preparations and entanglements

        if command.kind == CommandKind.M:
            s_domain, t_domain = command.s_domain, command.t_domain

            # print("start prob")
            new_s_domain, new_t_domain = (
                remove_corr_util(s_domain, prob, rng),
                remove_corr_util(t_domain, prob, rng),
            )
            new_command: Command = M(
                node=command.node,
                plane=command.plane,
                angle=command.angle,
                s_domain=new_s_domain,
                t_domain=new_t_domain,
            )
        # X and Z empty domains : don't apply anything (sum is 0)
        elif command.kind == CommandKind.X or command.kind == CommandKind.Z:  # noqa: PLR1714
            new_domain = remove_corr_util(command.domain, prob, rng)

            if command.kind == CommandKind.X:
                new_command = X(node=command.node, domain=new_domain)
            else:  # not X then Z
                new_command = Z(node=command.node, domain=new_domain)

        else:
            new_command = command  # type issue for some reason

        # print(command, new_pattern.output_nodes)
        new_pattern.add(new_command)  # type OK: exhaust all cases

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
    return {i for i in domain if rng.random() > prob}


# cache to avoid recomputing? or too big?


def fidelity(state1: Statevec, state2: Statevec) -> float:
    return float(np.abs(state1.psi.flatten().dot(state2.psi.flatten().conj())) ** 2)


def compute_pairwise_fidelities(ens: Sequence[Statevec]) -> npt.NDArray[np.float64]:
    """computes all pairwise fidelities over an ensemble of pure states

    Parameters
    ----------
    ens : Sequence[Statevec]
        pure state ensemble

    Returns
    -------
    npt.NDArray[np.float64]
        array containing the pairwise fidelities. It's length is n choose k = 2 (without repetion, order doesn't matter).
    """
    # starmap is an iterator
    # fromiter avoid converting to list ninse np.array(iterator) doesn't behave as expected
    return np.fromiter(starmap(fidelity, combinations(ens, 2)), dtype=np.float64)


def compute_state_frame_potential(
    fidelities: npt.NDArray[np.float64], k: int
) -> float:  # TODO fix type # also DMs later. Check with TM for backend typing
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

    # use numpy vectorization
    return float(np.mean(fidelities**k))


def compute_state_frame_potential_std(
    fidelities: npt.NDArray[np.float64], k: int
) -> float:  # TODO fix type # also DMs later. Check with TM for backend typing
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

    # use numpy vectorization
    return float(np.std(fidelities**k))


def build_gadget(n: int) -> Pattern:
    measurements1 = {i: Measurement(0, Plane.XY) for i in [1, 3, 5, 7]}
    measurements1.update({i: Measurement(1 / 4, Plane.XY) for i in [0, 2, 6]})
    measurements1.update({8: Measurement(1 / 2, Plane.XY)})

    measurements2 = {i: Measurement(0, Plane.XY) for i in [0, 2, 6, 8]}
    measurements2.update({i: Measurement(1 / 4, Plane.XY) for i in [1, 5, 7]})
    measurements2.update({3: Measurement(1 / 2, Plane.XY)})

    p1 = build_brickwork_pattern(measurements1)
    p2 = build_brickwork_pattern(measurements2)

    # building the "B" block
    # input nodes of p2 to input nodes of p1 :
    pc, mapping_c = p1.compose(p2, {0: 4, 5: 9})  # mapping_c is needed to access it
    patt_B, mapping_B = pc.compose(p1, {0: mapping_c[4], 5: mapping_c[9]})

    if n < 2:
        raise ValueError("Not possible")

    if n == 2:
        return patt_B

    patt_BB, mapping_BB = patt_B.compose(patt_B, {0: mapping_B[9]})

    patt_acc = patt_BB
    # compose renvoie un nouveau patter (fonctionnel)
    # compose: a gauche celui qui a les sorties à brancher sur les entrées
    # starts at n = 5
    for _ in range(0, (n - 1) // 2 - 1):  # start from BB and count number of BB to add
        patt_next, mapping = patt_BB.compose(patt_acc, {mapping_BB[5]: mapping_B[4]})
        # mapping_BB[5] alwayssince always compose with BB
        # One less node since merge them
        # reorder checks that reordering is a permutation so we don't miss nodes
        patt_next.reorder_output_nodes([mapping[i] for i in patt_acc.output_nodes] + patt_BB.output_nodes[1:])
        patt_next.reorder_input_nodes(
            [mapping[i] for i in patt_acc.input_nodes[:-1]] + patt_BB.input_nodes
        )  # last input patt_acc is merged, all inputs patt_BB kept
        patt_acc = patt_next  # Compose create new patterns. Copies anyhere else.
    # patt_acc: patt_BB

    if n % 2 == 0:
        patt_next, mapping = patt_B.compose(patt_acc, {mapping_BB[5]: mapping_B[4]})
        patt_next.reorder_output_nodes([mapping[i] for i in patt_acc.output_nodes] + patt_B.output_nodes[1:])
        patt_next.reorder_input_nodes([mapping[i] for i in patt_acc.input_nodes[:-1]] + patt_B.input_nodes)
        patt_acc = patt_next

    # nothing to do for n = 3

    return patt_acc


def build_E(nqubits: int, nlayers: int) -> Pattern:
    gadget_pattern = build_gadget(nqubits)

    patt_acc = Pattern(input_nodes=range(nqubits))

    for _ in range(nlayers):
        # two ways are OK by symmetry
        # patt_acc output nodes were reordered. They're merged. So only new outputs. Compose keeps order within a pattern.
        patt_acc, _ = patt_acc.compose(gadget_pattern, dict(zip(gadget_pattern.input_nodes, patt_acc.output_nodes)))

    return patt_acc


def run(
    nqubits: int, nlayers: int, shots: int, rng: Generator, prob: float
) -> tuple[list[float], list[float]]:  # Sequence[BackendState]
    # patt = build_gadget(n=nqubits)
    patt = build_E(nqubits=nqubits, nlayers=nlayers)
    new_patt = remove_corrections(patt, rng, prob)
    # also sample on all possible patterns if exists some probability?
    # For now just focus on prob = None
    # check determinism

    out_data: list[Statevec] = []  # Statevec subtype of BackendState but don't know it before simulator is selected
    for _ in range(shots):
        backend = StatevectorBackend()
        # compute probability by default
        # no need to reinitialise the backend, it's done internally
        new_patt.simulate_pattern(backend=backend)
        out_data.append(backend.state)  # TODO solve typing issue cast as Statevec?
        # fid = np.abs(out.psi.flatten().dot(ref.psi.flatten().conj())) ** 2
        # fid_data.append(fid)
    # print("mean", np.mean(fid_data))

    # print(compute_state_frame_potential(out_data, 1))

    fids = compute_pairwise_fidelities(out_data)
    print(f"{fids=}")

    return [compute_state_frame_potential(fids, t) for t in range(1, 4)], [
        compute_state_frame_potential_std(fids, t) for t in range(1, 4)
    ]  # out_data


def plot_result(data: list[list[float]], x: npt.NDArray[np.float64], stds: list[list[float]]) -> None:
    # data is list of 3-tuples

    # Unpack: Each variable now holds all values of one tuple position
    curve1, curve2, curve3 = zip(*data)  # Adjust based on your tuple size
    std1, std2, std3 = zip(*stds)

    # Theoretical reference lines for each curve
    theoretical_values = [1 / math.comb(d + t - 1, t) for t in range(1, 4)]  # Replace with your theoretical values

    # Plot each unpacked sequence
    plt.errorbar(x, curve1, fmt="+r", capsize=2, elinewidth=0.9, yerr=std1, label=r"$t = 1$", ms=10)
    plt.errorbar(x, curve2, fmt="bx", capsize=2, elinewidth=0.9, yerr=std2, label=r"$t = 2$", ms=10)
    plt.errorbar(x, curve3, fmt="g2", capsize=2, elinewidth=0.9, yerr=std3, label=r"$t = 3$", ms=15)
    colors = ["r", "b", "g"]
    # Plot horizontal (theoretical) lines
    for idx, yval in enumerate(theoretical_values):
        plt.axhline(y=yval, linestyle=":", label=rf"Lower bound $t={idx + 1}$", color=colors[idx])

    plt.legend()
    plt.xlabel("Correction removal probability")
    plt.ylabel(r"$F_t$")
    plt.title(f"State frame potential for n = {nqb} qubits, {nlayers} layers and {nshots} shots")
    plt.show()


# TODO
# and check multiple shots
# do flow
# remove corrections (with probability) to match Raussendorf and Briegel.
# frame potential [35], [36], [38], [39] of https://arxiv.org/abs/2505.21621 (Eq. 3)
# use loop over all branch selector if done otherwise sample.
# need to find minimal value for design (which it has to be)
# also look at Briegel and Raussendorf papers

# import logging

# logger = logging.getLogger(__name__)


if __name__ == "__main__":
    # logging.basicConfig(filename='myapp.log', level=logging.INFO)
    # logger.info('Started')
    # this is global
    # put everything in function! main
    nqb = 4
    nlayers = 2
    d = 2**nqb
    nshots = 100
    # t = 3
    rng = np.random.default_rng()
    res: list[list[float]] = []
    res_std: list[list[float]] = []
    p_vals = np.linspace(1, 1, num=1, endpoint=True)
    for p in tqdm(p_vals):
        print(f"{p=}")
        data, stds = run(
            nqubits=nqb, nlayers=nlayers, shots=nshots, rng=rng, prob=p
        )  # test prob + None and 1 should be the same
        res.append(data)
        res_std.append(stds)
    print(res)
    print(f"analytical lower bounds {[1 / math.comb(d + t - 1, t) for t in range(1, 4)]}")
    plot_result(data=res, x=p_vals, stds=res_std)
    # logger.info('Finished')
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
