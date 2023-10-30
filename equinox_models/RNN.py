import abc

import equinox as eqx
import jax
import jax.numpy as jnp
from equinox import AbstractVar


class AbstractRNNCell(eqx.Module):
    """Abstract RNN Cell class."""

    cell: AbstractVar[eqx.Module]
    hidden_size: AbstractVar[int]

    @abc.abstractmethod
    def __init__(self, data_dim, hidden_dim, *, key):
        """Initialize RNN cell."""
        raise NotImplementedError

    @abc.abstractmethod
    def __call__(self, state, input):
        """Call method for RNN cell."""
        raise NotImplementedError


class LinearCell(AbstractRNNCell):
    cell: eqx.nn.Linear
    hidden_size: int

    def __init__(self, data_dim, hidden_dim, *, key):
        self.cell = eqx.nn.Linear(data_dim + hidden_dim, hidden_dim, key=key)
        self.hidden_size = hidden_dim

    def __call__(self, state, input):
        return self.cell(jnp.concatenate([state, input]))


class GRUCell(AbstractRNNCell):
    cell: eqx.nn.GRUCell
    hidden_size: int

    def __init__(self, data_dim, hidden_dim, *, key):
        self.cell = eqx.nn.GRUCell(data_dim, hidden_dim, key=key)
        self.hidden_size = hidden_dim

    def __call__(self, state, input):
        return self.cell(input, state)


class LSTMCell(AbstractRNNCell):
    cell: eqx.nn.LSTMCell
    hidden_size: int

    def __init__(self, data_dim, hidden_dim, *, key):
        self.cell = eqx.nn.LSTMCell(data_dim, hidden_dim, key=key)
        self.hidden_size = hidden_dim

    def __call__(self, state, input):
        state = (state,) * 2
        return self.cell(input, state)[0]


class MLPCell(AbstractRNNCell):
    cell: eqx.nn.MLP
    hidden_size: int

    def __init__(self, data_dim, hidden_dim, depth, width, *, key):
        self.cell = eqx.nn.MLP(data_dim + hidden_dim, hidden_dim, width, depth, key=key)
        self.hidden_size = hidden_dim

    def __call__(self, state, input):
        return self.cell(jnp.concatenate([state, input]))


class RNN(eqx.Module):
    cell: AbstractRNNCell
    output_layer: eqx.nn.Linear
    hidden_dim: int
    classification: bool

    def __init__(self, cell, label_dim, classification=True, *, key):
        self.cell = cell
        self.output_layer = eqx.nn.Linear(
            self.cell.hidden_size, label_dim, use_bias=False, key=key
        )
        self.hidden_dim = self.cell.hidden_size
        self.classification = classification

    def __call__(self, x):
        hidden = jnp.zeros((self.hidden_dim,))

        def scan_fn(state, input):
            out = self.cell(state, input)
            return out, out

        final_state, all_states = jax.lax.scan(scan_fn, hidden, x)

        if self.classification:
            return jax.nn.softmax(self.output_layer(final_state), axis=0)
        else:
            return jax.vmap(self.output_layer)(all_states)
