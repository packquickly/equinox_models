# Recurrent Neural Networks

Recurrent neural networks (RNNs) are a family of discrete-time sequence to 
sequence models which take the form

$$ h_{t+1} = f_\theta(h_t, x_t), $$

where $h_t$ is the hidden state at time $t$, $x_t$ is the input at time $t$,
and $f_\theta$ is the *recurrent cell* with trainable parameters $\theta$. A 
RNN's output is typically a linear map of the hidden state, 
$y_t = l_{\phi}(h_t).$

## Implementation

The RNN class takes a recurrent cell as an argument. The recurrent cell should 
be a class which inherits from AbstractRNNCell. This class should implement 
`__init__` and `__call__` methods. The `__init__` method should initialise 
the trainable parameters of the cell and take the hidden state dimension as an 
argument. The `__call__` method should take the current state and input, and 
return the next state. 
