from collections.abc import Callable
from typing_extensions import TypeAlias

import diffrax as dfx
import equinox as eqx
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, PRNGKeyArray, PyTree


# These type aliases are unnecessary, but I (packquickly) like them :)
Linear: TypeAlias = eqx.nn.Linear
MLP: TypeAlias = eqx.nn.MLP
Interpolator: TypeAlias = dfx.AbstractGlobalInterpolation
Solver: TypeAlias = dfx.AbstractSolver


class VectorField(eqx.Module):
    """The vector field of the neural CDE, a basic feedforward network."""

    mlp: MLP
    data_size: int
    model_size: int

    def __init__(
        self,
        data_size: int,
        model_size: int,
        hidden_width: int,
        depth: int,
        activation: Callable,
        final_activation: Callable,
        *,
        key: PRNGKeyArray,
    ):
        """**Arguments:**

        - `data_size`: The dimension of the input features.
        - `model_size`: The hidden state size of the model, the dimension of the
            feedforward network output.
        - `hidden_width`: The hidden size of the feedforward network, including the
            output layer.
        - `depth`: The depth of the feedforward network.
        - `activation`: The activation between each hidden layer.
        - `final_activation`: The activation function after the output layer.
        - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
            initialisation.
        """
        self.data_size = data_size
        self.model_size = model_size
        eqx.nn.MLP(
            in_size=self.model_size,
            out_size=self.model_size * data_size,
            width_size=hidden_width,
            depth=depth,
            activation=activation,
            final_activation=final_activation,
            key=key,
        )

    def __call__(self, t: Array, y: PyTree[Array], args: PyTree) -> PyTree:
        return self.mlp(y).reshape(self.model_size, self.data_size)


class NCDEClassifier(eqx.Module):
    """A simple classifier using a neural controlled differential equation
    (neural CDE.)
    """

    encoder: MLP
    vector_field: VectorField
    decoder: Linear
    interpolator: type[Interpolator]
    solver: Solver
    use_meanpool: bool

    def __init__(
        self,
        data_size: int,
        model_size: int,
        mlp_hidden_width: int,
        mlp_depth: int,
        encoder_width: int,
        encoder_depth: int,
        activation: Callable = jnn.gelu,
        final_activation: Callable = jnn.tanh,
        n_classes: int = 2,
        interpolator: type[Interpolator] = dfx.LinearInterpolation,
        solver: Solver = dfx.Tsit5(),
        use_meanpool: bool = False,
        *,
        key: PRNGKeyArray,
    ):
        """**Arguments:**

        - `data_size`: The dimension of the input features.
        - `model_size`: The hidden state size of the model, the dimension of the
            feedforward network output.
        - `mlp_hidden_width`: The hidden size of the vector field feedforward network,
            including the output layer.
        - `mlp_depth`: The depth of the vector field feedforward network.
        - `encoder_width`: The hidden size of the input feedforward network,
            including the output layer.
        - `encoder_depth`: The depth of the input feedforward network.
        - `activation`: The activation between each hidden layer. Defaults to GeLU.
        - `final_activation`: The activation function after the output layer. Defaults
            to tanh, this is a good choice for neural CDEs.
        - `n_classes`: The number of output classes.
        - `interpolator`: The Diffrax interpolator class to use for interpolation.
            Defaults to `diffrax.LinearInterpolation.`
        - `solver`: The Diffrax solver to use to solve the neural CDE. Defaults to
            `diffrax.Tsit5()`.
        - `use_meanpool`: Determines how to go from an output of shape (times, feats) to
            (feats) for use in classification. Uses the mean over time if `True`, and
            the value at the final time if `False`.
        - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
            initialisation.
        """
        encoder_key, mlp_key, decoder_key = jr.split(key, 3)
        self.encoder = eqx.nn.MLP(
            data_size, model_size, encoder_width, encoder_depth, key=encoder_key
        )
        self.vector_field = VectorField(
            data_size,
            model_size,
            mlp_hidden_width,
            mlp_depth,
            activation,
            final_activation,
            key=mlp_key,
        )
        # If there are 2 output classes, we only need a 1-dimensional output for
        # classification.
        if n_classes == 2:
            out_dim = 1
        else:
            out_dim = n_classes

        self.decoder = eqx.nn.Linear(model_size, out_dim, key=decoder_key)
        self.interpolator = interpolator
        self.solver = solver
        self.use_meanpool = use_meanpool

    def __call__(self, times: Float[Array, " times"], xs: Float[Array, "times feats"]):
        if self.interpolator == dfx.CubicInterpolation:
            ys = dfx.backward_hermite_coefficients(times, xs)
        else:
            ys = xs
        control = self.interpolator(times, ys)
        cde_term = dfx.ControlTerm(self.vector_field, control)
        ode_term = cde_term.to_ode()
        dt0 = None
        y0 = self.encoder(control.evaluate(times[0]))
        saveat = dfx.SaveAt(t1=True)
        solution = dfx.diffeqsolve(
            ode_term,
            self.solver,
            times[0],
            times[-1],
            dt0,
            y0,
            stepsize_controller=dfx.PIDController(rtol=1e-3, atol=1e-6),
            saveat=saveat,  # pyright: ignore
        )

        if self.use_meanpool:
            out = jnp.mean(solution.ys, axis=0)
        else:
            out = solution.ys[-1]

        logits = self.decoder(out)
        if out.size == 1:
            classify = jnn.sigmoid
        else:
            classify = jnn.softmax
        return classify(logits)
