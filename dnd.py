import torch as T
import torch.nn.functional as F

ALL_KERNELS = ['cosine', 'l1', 'l2']

class DND:
    """
    A DND (Differentiable Neural Dictionary) that supports batch training:
      - You can pass [batch_size, memory_dim] or [memory_dim] into save_memory / get_memory.
      - Memory is stored as a list of row vectors (keys, vals).
      - We do a 1-NN search to retrieve the best match for each query.
    """

    def __init__(self, dict_len, memory_dim, kernel='l2'):
        """
        Parameters
        ----------
        dict_len : int
            The maximum capacity of this dictionary
        memory_dim : int
            Dimensionality of each key / value vector
        kernel : str
            Metric for memory search: 'l2', 'l1', or 'cosine'
        """
        self.dict_len = dict_len
        self.kernel = kernel
        self.memory_dim = memory_dim

        self.encoding_off = False
        self.retrieval_off = False

        self.reset_memory()
        self.check_config()

    def reset_memory(self):
        """Clear the dictionary."""
        self.keys = []  # each element shape = [memory_dim]
        self.vals = []  # each element shape = [memory_dim]

    def check_config(self):
        assert self.dict_len > 0
        assert self.kernel in ALL_KERNELS, f"Kernel {self.kernel} not supported. Must be one of {ALL_KERNELS}."

    def inject_memories(self, input_keys, input_vals):
        """
        Inject a list of pre-defined keys and values.
        input_keys, input_vals: list of Tensors, each shape = [memory_dim].
        """
        assert len(input_keys) == len(input_vals)
        for k, v in zip(input_keys, input_vals):
            self.save_memory(k, v)

    def save_memory(self, memory_key, memory_val):
        """
        Save memory(s) into DND.

        memory_key : torch.Tensor
            shape [memory_dim] or [batch_size, memory_dim]
        memory_val : torch.Tensor
            shape [memory_dim] or [batch_size, memory_dim]

        We'll store each row into self.keys/self.vals as [memory_dim].
        If encoding_off == True, we skip writing.
        """
        if self.encoding_off:
            return

        # Ensure shapes are at least 2D: [B, memory_dim].
        if memory_key.dim() == 1:
            # single entry => add batch dimension
            memory_key = memory_key.unsqueeze(0)
            memory_val = memory_val.unsqueeze(0)

        batch_size = memory_key.size(0)
        for i in range(batch_size):
            k_i = memory_key[i].detach().cpu().clone()  # shape [memory_dim]
            v_i = memory_val[i].detach().cpu().clone()
            self.keys.append(k_i)
            self.vals.append(v_i)
            # check overflow
            if len(self.keys) > self.dict_len:
                # remove oldest
                self.keys.pop(0)
                self.vals.pop(0)

    def get_memory(self, query_key):
        """
        1-NN retrieval for one or multiple queries.

        query_key : torch.Tensor
            shape [memory_dim] or [batch_size, memory_dim]

        returns:
            shape [1, memory_dim] if single input,
            shape [B, memory_dim] if batch input
        """
        # If retrieval_off or empty memory => return zero vector
        if len(self.keys) == 0 or self.retrieval_off:
            return self._empty_batch_return(query_key)

        single_query = (query_key.dim() == 1)
        if single_query:
            # => [1, memory_dim]
            query_key = query_key.unsqueeze(0)

        batch_size = query_key.size(0)
        out = []
        for i in range(batch_size):
            # shape [memory_dim]
            q_i = query_key[i]
            similarities = compute_similarities(q_i, self.keys, self.kernel)
            best_val = self._get_memory(similarities)  # shape [memory_dim]
            out.append(best_val)

        # stack => [B, memory_dim]
        out = T.stack(out, dim=0)
        if single_query:
            # keep shape [1, memory_dim] for single
            return out
        else:
            # [B, memory_dim] for batch
            return out

    def _get_memory(self, similarities, policy='1NN'):
        """
        Return the best matched memory_val by '1NN' policy.
        similarities shape = [n_memories]
        returns shape = [memory_dim]
        """
        if policy != '1NN':
            raise ValueError(f"unrecognized recall policy: {policy}")
        best_id = T.argmax(similarities)
        best_val = self.vals[best_id]
        return best_val

    def _empty_batch_return(self, query_key):
        """
        If memory is empty or retrieval_off => return zero vector in the same batch shape.
        """
        if query_key.dim() == 1:
            return T.zeros((1, self.memory_dim), device=query_key.device)
        else:
            bsz = query_key.size(0)
            return T.zeros((bsz, self.memory_dim), device=query_key.device)


#### Helpers
def compute_similarities(query_key, key_list, metric):
    """
    Compute similarity between query_key vs. each item in key_list.
    query_key: shape [memory_dim]
    key_list: list of Tensors, each shape=[memory_dim]
    metric: 'cosine'|'l1'|'l2'

    returns shape=[len(key_list)] similarity (or negative distance)
    """
    # [1, memory_dim]
    q = query_key.view(1, -1).float()
    # [n_memories, memory_dim]
    M = T.stack(key_list).float()

    if metric == 'cosine':
        similarities = F.cosine_similarity(q, M)  # => [n_memories]
    elif metric == 'l1':
        # negative L1 distance => shape [n_memories]
        similarities = -F.pairwise_distance(q, M, p=1)
    elif metric == 'l2':
        # negative L2 distance
        similarities = -F.pairwise_distance(q, M, p=2)
    else:
        raise ValueError(f"unrecog metric: {metric}")
    return similarities