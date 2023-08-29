import numpy as np
from graphix.ops import Ops


class Channel:
    """(Noise) Channel class in the Kraus representation

    Attributes
    ----------
    nqubit : int
        number of qubits acted on by the Kraus operators
    size : int
        number of Kraus operators
    kraus_ops : array_like(dict())
        the data in format
        array_like(dict): [{parameter: scalar, operator: array_like}, {parameter: scalar, operator: array_like}, ...]


    Methods
    -------
    says(sound=None)
        Prints the animals name and what sound it makes

    Returns
    -------
    Channel object
        containing the corresponding Kraus operators

    """

    # TODO json compatibility and allow to import channels from file?
    # TODO ? or *data and build from several (parameter, operator) couples?
    def __init__(self, kraus_data):
        """
        Parameters
        ----------

        kraus_data : array_like
            array of Kraus operator data.
            array_like(dict): [{parameter: scalar, operator: array_like}, {parameter: scalar, operator: array_like}, ...]

        Raises
        ------
        ValueError
            If empty array_like is provided.
        ....
        """

        # check there is data
        if not kraus_data:
            raise ValueError("Cannot instantiate the channel with empty data.")

        if not isinstance(kraus_data, (list, np.ndarray, tuple)):
            raise TypeError(f"The data must be a list, a numpy.ndarray or a tuple not a {type(kraus_data)}.")

        # check that data is correctly formatted before assigning it to the object.
        assert check_data_values_type(kraus_data)
        assert check_data_dims(kraus_data)

        # check that the channel is properly normalized i.e
        # \sum_K_i^\dagger K_i = Identity
        assert check_data_normalization(kraus_data)

        self.nqubit = int(np.log2(kraus_data[0]["operator"].shape[0]))
        self.kraus_ops = kraus_data

        # np.asarray(data, dtype=np.complex128)
        # number of Kraus operators in the Channel
        self.size = len(kraus_data)

    def __repr__(self):
        return f"Channel object with {self.size} Kraus operators of dimension {self.nqubit}."

    def is_normalized(self):
        return check_data_normalization(self.kraus_ops)


def check_data_dims(data) -> bool:

    # convert to set to remove duplicates
    dims = list(set([i["operator"].shape for i in data]))
    # or list({[i["operator"].shape for i in data]}) using set comprehension

    # check all the same dimensions and that they are square matrices
    if len(dims) != 1:
        raise ValueError(f"All provided Kraus operators don't have the same dimension {dims}!")

    if dims[0][0] != dims[0][1]:
        raise ValueError(f"All provided Kraus operators have the same shape {dims[0]} but are not square matrices!")

    # check consistency with tensor of qubit local Hilbert spaces
    data_dim = np.log2(dims[0][0])
    if not np.isclose(data_dim, int(data_dim)):
        raise ValueError(f"Incorrect data dimension {data_dim}: not consistent with qubits.")
    # data_dim = int(data_dim)

    return True


def check_data_values_type(data) -> bool:
    # convert to set to remove duplicates

    # TODO put error raising here instead. And deaggregate this mess to raise useful errors.

    # also check the values in the arrays !!!
    value_types = list(set([isinstance(i, dict) for i in data]))

    if value_types == [True]:

        key0_values = list(set([list(i.keys())[0] == "parameter" for i in data]))
        key1_values = list(set([list(i.keys())[1] == "operator" for i in data]))

        if key0_values == [True] and key1_values == [True]:
            operator_types = list(set([isinstance(i["operator"], np.ndarray) for i in data]))

            if operator_types == [True]:
                operator_dtypes = list(
                    set([i["operator"].dtype in [float, complex, np.float64, np.complex128] for i in data])
                )

                if operator_dtypes == [True]:
                    par_types = list(
                        set([isinstance(i["parameter"], (float, complex, np.float64, np.complex128)) for i in data])
                    )

                    if par_types == [True]:
                        pass
                    else:
                        raise TypeError("All parameters are not scalars")

                else:
                    raise TypeError(
                        f"All operators  {list([i['operator'].dtype == (float or complex or np.float64 or np.complex128) for i in data])}."
                    )
            # do not have the same dtype    {[i['operator'].dtype  for i in data]},
            else:
                raise TypeError("All operators don't have the same type and must be np.ndarray.")
        else:
            raise KeyError("The keys of the indivudal Kraus operators must be parameter and operator.")
    else:
        raise TypeError("All values are not dictionaries.")

    return True


def check_data_normalization(data) -> bool:

    opsu = np.array([i["parameter"] * i["parameter"].conj() * i["operator"].conj().T @ i["operator"] for i in data])

    if not np.allclose(np.sum(opsu, axis=0), np.eye(2 ** int(np.log2(len(data[0]["operator"]))))):
        raise ValueError(f"The specified channel is not normalized. {np.sum(opsu, axis=0)}")
    return True


def create_dephasing_channel(prob: float) -> Channel:
    """single-qubit dephasing channel
    .. math::
        (1-p) \rho + p Z  \rho Z

    Parameters
    ----------
    prob : float
        The probability associated to the channel

    Returns
    -------
    Channel object
        containing the corresponding Kraus operators
    """
    return Channel(
        [{"parameter": np.sqrt(1 - prob), "operator": np.eye(2)}, {"parameter": np.sqrt(prob), "operator": Ops.z}]
    )


def create_depolarising_channel(prob: float) -> Channel:
    """single-qubit depolarizing channel
    .. math::
        (1-p) \rho + \frac{p}{3} (X * \rho * X + Y * rho * Y + Z * rho * Z) = (1 - 4\frac{p}{3}) \rho + 4 \frac{p}{3} Id
    but my format is better with X, Y Z
    """
    return Channel(
        [
            {"parameter": np.sqrt(1 - prob), "operator": np.eye(2)},
            {"parameter": np.sqrt(prob / 3.0), "operator": Ops.x},
            {"parameter": np.sqrt(prob / 3.0), "operator": Ops.y},
            {"parameter": np.sqrt(prob / 3.0), "operator": Ops.z},
        ]
    )


def create_2_qubit_dephasing_channel(prob: float) -> Channel:
    """two-qubit dephasing channel
    .. math::
        (1-p) \rho + \frac{p}{3} (Z_1  \rho  Z_1 + Z_2  \rho  Z_2 + Z_1 Z_2  \rho  Z_1 Z_2)

    Parameters
    ----------
    prob : float
        The probability associated to the channel

    Returns
    -------
    Channel object
        containing the corresponding Kraus operators
    """
    # NOTE kinf of useless since just the tensor product of single-qubit channels.
    # There for testing purposes only.
    return Channel(
        [
            {"parameter": np.sqrt(1 - prob), "operator": np.kron(np.eye(2), np.eye(2))},
            {"parameter": np.sqrt(prob / 3.0), "operator": np.kron(Ops.z, np.eye(2))},
            {"parameter": np.sqrt(prob / 3.0), "operator": np.kron(np.eye(2), Ops.z)},
            {"parameter": np.sqrt(prob / 3.0), "operator": np.kron(Ops.z, Ops.z)},
        ]
    )


# # maybe later if not dict of dict but array_like of array_like
# def to_kraus(data):
#     r"""Convert input data into Kraus operator set [KrausOp, KrausOp, ...].
#     Each KrausOp has unitary matrix and target qubit index info.

#     Parameters
#     ----------
#         data : array_like
#             Data to convert into Kraus operator set.
#             Input data must be either (i)single Operator or (ii)Kraus set.
#             Relation among quantum channel, input data and returned Kraus operator set is as follwos:
#                 (i) quantum channel: :math:`E(\rho) = A \rho A^\dagger`
#                     input data: [A (2d-array-like), qarg(int)]
#                     returns: [KrausOp]
#                 (ii) quantum channel: :math:`E(\rho) = \sum_i A_i \rho A_i^\dagger`
#                     input data: [(A_1, int), (A_2, int), ...]
#                     returns: [KrausOp, KrausOp, ...]
#     Returns
#     -------
#         kraus : list. [KrausOp, ...]
#             KrausOp set.
#     """
#     if isinstance(data, (list, tuple, np.ndarray)):
#         if len(data) <= 1:
#             raise ValueError(
#                 "Input data must be either single Kraus Operator, single Kraus set or generalized Kraus set"
#                 " with target qubit indices."
#             )
#         # (i) If input is [2d-array-like, int], the first data is a single unitary matrix A for channel:
#         # E(rho) = A * rho * A^\dagger
#         # and the second data is target qubit index.
#         if _is_kraus_op(data):
#             return [KrausOp(data=data[0], qarg=data[1])]

#         # (ii) If input is list of [2d-array-likes, int], it is a single Kraus set for channel:
#         # E(rho) = \sum_i A_i * rho * A_i^\dagger
#         # with target qubit indices.
#         elif isinstance(data, (list, tuple, np.ndarray)) and _is_kraus_op(data[0]):
#             if isinstance(data, np.ndarray):
#                 data = data.tolist()
#             kraus = [KrausOp(data=data[0][0], qarg=data[0][1])]
#             for A_i in data[1:]:
#                 A_i = KrausOp(data=A_i[0], qarg=A_i[1])
#                 if _is_kraus_op(A_i):
#                     raise ValueError("All Kraus operators must have same shape.")
#                 kraus.append(A_i)
#             return kraus
#         else:
#             raise ValueError(
#                 "Input data must be either (i)single Operator (2d-array-like)"
#                 " or (ii)single Kraus set (list of 2d-array-likes)"
#                 " with qubit indices for each Operator."
#             )
#     else:
#         raise TypeError("Input data must be list, tupple, or array_like.")


# def generate_dephasing_kraus(p, qarg):
#     """Return Kraus operators for a dephasing channel.

#     Parameters
#     ----------
#         p : float
#             Probability of dephasing error.
#         qarg : int
#             Target qubit index.
#     """
#     assert isinstance(qarg, int)
#     assert 0 <= p <= 1
#     return to_kraus([[np.sqrt(1 - p) * np.eye(2), qarg], [np.sqrt(p) * np.diag([1, -1]), qarg]])


# def generate_depolarizing_kraus(p, nqubits):
#     """Return Kraus operators for a depolarizing channel."""
#     pass


# def generate_amplitude_damping_kraus(p, nqubits):
#     """Return Kraus operators for an amplitude damping channel."""
#     pass


# def _is_kraus_op(data):
#     """Check if data is a Kraus operator.
#     Currently, Kraus operator is defined as a list of [2d-array-like, int].
#     This might be changed in the future to support Kraus operator of the form [2^k dim array-like, int] (k <= n).
#     """
#     if not isinstance(data, (list, tuple, np.ndarray)):
#         return False
#     if len(data) != 2:
#         return False
#     if not isinstance(data[1], int):
#         return False
#     if not isinstance(data[0], np.ndarray):
#         return np.array(data[0]).shape == (2, 2)
#     return data[0].shape == (2, 2)
