import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple

N_GATES = 5
GateSpans = namedtuple('GateSpans', ['I', 'F', 'G', 'O', 'R'])

ACTIVATIONS = {
    'sigmoid':  nn.Sigmoid(),
    'tanh':     nn.Tanh(),
    'hard_tanh':nn.Hardtanh(),
    'relu':     nn.ReLU(),
}

class EpLSTMCell(nn.Module):
    """
    Episodic LSTM cell, with external memory injection via r gate.
      - Two linear layers: input_kernel & recurrent_kernel
      - 5 gates: i, f, g, o, r (input, forget, memory, output, read)
      - External memory injected via r gate
      - Forget gate bias = 1
      - Activations selectable via constructor
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        recurrent_activation: str = 'sigmoid'
    ):
        """
        Parameters
        ----------
        input_size : int
            Dimension of x_t
        hidden_size : int
            Dimension of h_t, c_t
        recurrent_activation : str
            Gate activation for i, f, o, r 
        """
        super().__init__()
        self.Dx = input_size
        self.Dh = hidden_size

        # two linear layers, input_kernel & recurrent_kernel
        # mapping to 5 gates: i, f, g, o, r
        self.input_kernel = nn.Linear(self.Dx, self.Dh * N_GATES)
        self.recurrent_kernel = nn.Linear(self.Dh, self.Dh * N_GATES)

        # choose activation functions, sigmoid by default
        if isinstance(recurrent_activation, str):
            self.fun_rec = ACTIVATIONS[recurrent_activation]
        else:
            self.fun_rec = recurrent_activation

        # initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize parameters, setting forget gate bias to 1.
        """

        nn.init.zeros_(self.input_kernel.bias)
        nn.init.zeros_(self.recurrent_kernel.bias)

        # xavier/orthogonal
        nn.init.xavier_uniform_(self.input_kernel.weight)
        nn.init.orthogonal_(self.recurrent_kernel.weight)

    def forward(
        self,
        x_t: torch.Tensor, 
        m_t: torch.Tensor, 
        state: tuple[torch.Tensor, torch.Tensor]
    ):
        """
        Parameters
        ----------
        x_t : [B, Dx]
            Input at current time step
        m_t : [B, Dh]
            External memory to inject
        state : (h_{t-1}, c_{t-1})
            Previous hidden and cell states, each [B, Dh]

        Returns
        -------
        h_t : [B, Dh]
        (h_t, c_t) : next hidden and cell states
        """
        (h_tm1, c_tm1) = state

        x_proj = self.input_kernel(x_t)
        h_proj = self.recurrent_kernel(h_tm1)

        # chunk into 5 gates
        Xi, Xf, Xg, Xo, Xr = x_proj.chunk(N_GATES, dim=-1)
        Hi, Hf, Hg, Ho, Hr = h_proj.chunk(N_GATES, dim=-1)

        # i, f, o, r -> self.fun_rec
        ft = self.fun_rec(Xf + Hf)
        ot = self.fun_rec(Xo + Ho)
        it = self.fun_rec(Xi + Hi)
        rt = self.fun_rec(Xr + Hr)

        # g -> tanh
        gt = torch.tanh(Xg + Hg)

        # new cell state: c_t -> f * c_tm1 + i * g + r * m_t
        c_t = (ft * c_tm1) + (it * gt) + (rt * torch.tanh(m_t))

        # new hidden state: h_t -> o * tanh(c_t)
        h_t = ot * torch.tanh(c_t)

        return h_t, (h_t, c_t)