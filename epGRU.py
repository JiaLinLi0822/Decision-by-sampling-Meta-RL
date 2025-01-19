import torch
import torch.nn as nn
from collections import namedtuple

N_GATES = 5
GateSpans = namedtuple('GateSpans', ['Z', 'R', 'N', 'O', 'READ'])

ACTIVATIONS = {
    'sigmoid':   nn.Sigmoid(),
    'tanh':      nn.Tanh(),
    'hard_tanh': nn.Hardtanh(),
    'relu':      nn.ReLU(),
}

class EpGRUCell(nn.Module):
    """
    Episodic GRU cell with external memory injection.
    - Two linear layers: input_kernel & recurrent_kernel
    - 5 gates: z, r, n, o, read
    - External memory is injected via the read gate
    - Gate activation (z, r, o, read) uses `recurrent_activation`, candidate n uses tanh
    - c_{t-1} is unused by GRU, but included for interface compatibility.
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
            Dimensionality of x_t
        hidden_size : int
            Dimensionality of h_t
        recurrent_activation : str
            Activation function for gates z, r, o, read (default 'sigmoid')
        """

        super().__init__()
        self.Dx = input_size
        self.Dh = hidden_size

        # Two linear maps for input x_t and hidden h_{t-1}, each output 5*Dh
        self.input_kernel = nn.Linear(self.Dx, self.Dh * N_GATES)
        self.recurrent_kernel = nn.Linear(self.Dh, self.Dh * N_GATES)

        # Select activation function for gates
        if isinstance(recurrent_activation, str):
            self.fun_rec = ACTIVATIONS[recurrent_activation]
        else:
            self.fun_rec = recurrent_activation

        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize weights and biases.
        Consistent with the style from EpLSTMCell:
        - Bias set to zero
        - input_kernel.weight uses xavier_uniform_
        - recurrent_kernel.weight uses orthogonal_
        """
        nn.init.zeros_(self.input_kernel.bias)
        nn.init.zeros_(self.recurrent_kernel.bias)

        nn.init.xavier_uniform_(self.input_kernel.weight)
        nn.init.orthogonal_(self.recurrent_kernel.weight)

    def forward(
        self,
        x_t: torch.Tensor,
        m_t: torch.Tensor,
        state: torch.Tensor
        ) -> torch.Tensor:
        
        '''
        Parameters
        ----------
        x_t : [B, Dx]
            Input at current time step
        m_t : [B, Dh]
            External memory, to be injected via the read gate
        state : h_{t-1}

        Returns
        -------
        h_t : [B, Dh]
            Hidden state at current time step
        '''

        h_tm1 = state

        x_proj = self.input_kernel(x_t)     # => [B, 5*Dh]
        h_proj = self.recurrent_kernel(h_tm1)  # => [B, 5*Dh]

        # Split into 5 gates: z, r, n, o, read
        Xz, Xr, Xn, Xo, Xm = x_proj.chunk(N_GATES, dim=-1)
        Hz, Hr, Hn, Ho, Hm = h_proj.chunk(N_GATES, dim=-1)

        # z, r, o, read => recurrent_activation (default: sigmoid)
        # n => tanh
        z_t = self.fun_rec(Xz + Hz)        # update gate
        r_t = self.fun_rec(Xr + Hr)        # reset gate
        o_t = self.fun_rec(Xo + Ho)        # output modulation gate
        read_t = self.fun_rec(Xm + Hm)     # read gate for external memory
        n_t = torch.tanh(Xn + (Hn * r_t))  # candidate hidden

        # 4) Combine the gates to get new hidden state
        #    h'_t = n_t * o_t
        #    h_t = z_t * h_{t-1} + (1 - z_t)*h'_t + read_t * tanh(m_t)
        h_prime = n_t * o_t
        h_t = z_t * h_tm1 + (1.0 - z_t) * h_prime + read_t * torch.tanh(m_t)

        return h_t