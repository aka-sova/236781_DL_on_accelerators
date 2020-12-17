import re
import torch
import torch.nn as nn
import torch.utils.data
from torch import Tensor
from typing import Iterator


def char_maps(text: str):
    """
    Create mapping from the unique chars in a text to integers and
    vice-versa.
    :param text: Some text.
    :return: Two maps.
        - char_to_idx, a mapping from a character to a unique
        integer from zero to the number of unique chars in the text.
        - idx_to_char, a mapping from an index to the character
        represented by it. The reverse of the above map.

    """
    # TODO:
    #  Create two maps as described in the docstring above.
    #  It's best if you also sort the chars before assigning indices, so that
    #  they're in lexical order.
    # ====== YOUR CODE: ======

    # unique chars sorted
    sorted_chars = sorted(list(set(text)))

    char_to_idx = { sorted_char : idx for idx, sorted_char in enumerate(sorted_chars)}
    idx_to_char = { idx:  sorted_char for idx, sorted_char in enumerate(sorted_chars)}

    # ========================
    return char_to_idx, idx_to_char


def remove_chars(text: str, chars_to_remove):
    """
    Removes all occurrences of the given chars from a text sequence.
    :param text: The text sequence.
    :param chars_to_remove: A list of characters that should be removed.
    :return:
        - text_clean: the text after removing the chars.
        - n_removed: Number of chars removed.
    """
    # TODO: Implement according to the docstring.
    # ====== YOUR CODE: ======

    # this is the most elegant way to implement a counter that i found
    class Null_replacer(object):
        def __init__(self):
            self.counter = 0

        def __call__(self, match):
            self.counter += 1
            return ""

    # a function being called when the match happens
    null_replacer = Null_replacer()

    chars_to_remove_str = "".join(chars_to_remove)
    chars_to_remove_str = f"[{chars_to_remove_str}]"

    text_clean = re.sub(chars_to_remove_str, null_replacer, text)
    n_removed = null_replacer.counter

    # ========================
    return text_clean, n_removed


def chars_to_onehot(text: str, char_to_idx: dict) -> Tensor:
    """
    Embed a sequence of chars as a a tensor containing the one-hot encoding
    of each char. A one-hot encoding means that each char is represented as
    a tensor of zeros with a single '1' element at the index in the tensor
    corresponding to the index of that char.
    :param text: The text to embed.
    :param char_to_idx: Mapping from each char in the sequence to it's
    unique index.
    :return: Tensor of shape (N, D) where N is the length of the sequence
    and D is the number of unique chars in the sequence. The dtype of the
    returned tensor will be torch.int8.
    """
    # TODO: Implement the embedding.
    # ====== YOUR CODE: ======

    # use the broadcasting

    text_chars = list(text)

    N = len(text_chars)
    D = len(char_to_idx.keys())
    result = torch.zeros(size=(N, D), dtype=torch.int8)

    text_idxs = [char_to_idx[char] for char in text_chars]

    idxs = torch.arange(N)
    result[idxs, text_idxs] = 1

    # ========================
    return result


def onehot_to_chars(embedded_text: Tensor, idx_to_char: dict) -> str:
    """
    Reverses the embedding of a text sequence, producing back the original
    sequence as a string.
    :param embedded_text: Text sequence represented as a tensor of shape
    (N, D) where each row is the one-hot encoding of a character.
    :param idx_to_char: Mapping from indices to characters.
    :return: A string containing the text sequence represented by the
    embedding.
    """
    # TODO: Implement the reverse-embedding.
    # ====== YOUR CODE: ======

    idx_num_vec = torch.argmax(embedded_text, axis=1).tolist()

    chars_list = [idx_to_char[idx] for idx in idx_num_vec]
    result = "".join(chars_list)

    # ========================
    return result


def chars_to_labelled_samples(text: str, char_to_idx: dict, seq_len: int, device="cpu"):
    """
    Splits a char sequence into smaller sequences of labelled samples.
    A sample here is a sequence of seq_len embedded chars.
    Each sample has a corresponding label, which is also a sequence of
    seq_len chars represented as indices. The label is constructed such that
    the label of each char is the next char in the original sequence.
    :param text: The char sequence to split.
    :param char_to_idx: The mapping to create and embedding with.
    :param seq_len: The sequence length of each sample and label.
    :param device: The device on which to create the result tensors.
    :return: A tuple containing two tensors:
    samples, of shape (N, S, V) and labels of shape (N, S) where N is
    the number of created samples, S is the seq_len and V is the embedding
    dimension.
    """
    # TODO:
    #  Implement the labelled samples creation.
    #  1. Embed the given text.
    #  2. Create the samples tensor by splitting to groups of seq_len.
    #     Notice that the last char has no label, so don't use it.
    #  3. Create the labels tensor in a similar way and convert to indices.
    #  Note that no explicit loops are required to implement this function.
    # ====== YOUR CODE: ======

    embedded_text = chars_to_onehot(text = text, char_to_idx=char_to_idx)

    N = (embedded_text.shape[0] - 1) // seq_len
    S = seq_len
    V = len(char_to_idx)

    # -- samples

    N_idx_tensor = torch.arange(start = 0, end = N*S*V, step = S*V).view(N, 1, 1)
    S_idx_tensor = torch.arange(start = 0, end = V*S, step = V).view(1, S, 1)
    V_idx_tensor = torch.arange(V).view(1, 1, V)

    NSV_tensor = N_idx_tensor + S_idx_tensor + V_idx_tensor # [N x S x V]
    NSV_flat = NSV_tensor.view(-1)

    embdedded_text_flat = embedded_text.view(-1)

    fixed_indixes = embdedded_text_flat[NSV_flat]
    samples = fixed_indixes.view(N, S, V)


    # -- labels

    text_list = [char_to_idx[char] for char in list(text)]
    text_chars_tensor = torch.tensor(text_list)

    N_idx_tensor_single_step = torch.arange(start=0, end=N * S, step=S).view(N, 1, 1)
    S_idx_tensor_single_step = torch.arange(start=1, end=S+1,   step=1).view(1, S, 1)
    NS_tensor_single_char = N_idx_tensor_single_step + S_idx_tensor_single_step

    fixed_char_indixes = text_chars_tensor[NS_tensor_single_char[:, :, 0].view(-1)]
    labels = fixed_char_indixes.view(N, S)

    samples.to(device)
    labels.to(device)

    # ========================
    return samples, labels


def hot_softmax(y, dim=0, temperature=1.0):
    """
    A softmax which first scales the input by 1/temperature and
    then computes softmax along the given dimension.
    :param y: Input tensor.
    :param dim: Dimension to apply softmax on.
    :param temperature: Temperature.
    :return: Softmax computed with the temperature parameter.
    """
    # TODO: Implement based on the above.
    # ====== YOUR CODE: ======

    y_exp = torch.exp(y/temperature)

    softmax_sum = torch.sum(y_exp, dim=dim)
    softmax_sum = torch.unsqueeze(softmax_sum, dim=dim)
    result = y_exp / softmax_sum


    # ========================
    return result


def generate_from_model(model, start_sequence, n_chars, char_maps, T):
    """
    Generates a sequence of chars based on a given model and a start sequence.
    :param model: An RNN model. forward should accept (x,h0) and return (y,
    h_s) where x is an embedded input sequence, h0 is an initial hidden state,
    y is an embedded output sequence and h_s is the final hidden state.
    :param start_sequence: The initial sequence to feed the model.
    :param n_chars: The total number of chars to generate (including the
    initial sequence).
    :param char_maps: A tuple as returned by char_maps(text).
    :param T: Temperature for sampling with softmax-based distribution.
    :return: A string starting with the start_sequence and continuing for
    with chars predicted by the model, with a total length of n_chars.
    """
    assert len(start_sequence) < n_chars
    device = next(model.parameters()).device
    char_to_idx, idx_to_char = char_maps
    out_text = start_sequence

    # TODO:
    #  Implement char-by-char text generation.
    #  1. Feed the start_sequence into the model.
    #  2. Sample a new char from the output distribution of the last output
    #     char. Convert output to probabilities first.
    #     See torch.multinomial() for the sampling part.
    #  3. Feed the new char into the model.
    #  4. Rinse and Repeat.
    #  Note that tracking tensor operations for gradient calculation is not
    #  necessary for this. Best to disable tracking for speed.
    #  See torch.no_grad().
    # ====== YOUR CODE: ======

    sampling_amount = 100 # what's the right number?..
    new_sequence = []

    # encode the start sequence
    sequence = chars_to_onehot(start_sequence, char_to_idx) # num chars x embedding size
    sequence = sequence.to(dtype=torch.float)
    sequence_batch = torch.unsqueeze(sequence, dim = 0) # B x S x V

    # initial
    hidden_state = None


    with torch.no_grad():
        for char_num in range(n_chars - len(start_sequence)):

            # 1. feed the inputs
            y, h = model.forward(input = sequence_batch, hidden_state = hidden_state)

            # 2. take the last output, turn into probabilities
            scores = y[:, -1, :] # B x S x O   , we take last char
            output_probs = hot_softmax(scores, temperature=T) # B x O

            # 3. sample the output char from the probabilities
            sampling_indixes = torch.multinomial(output_probs, sampling_amount, replacement=True)

            # find the most sampled char from the probabilities
            indixes, counts = sampling_indixes.unique(return_counts=True)
            chosen_char_embedded = indixes[torch.argmax(counts)]
            chosen_char_str = idx_to_char[chosen_char_embedded.item()]
            new_sequence.append(chosen_char_str)

            # print(f"Chosen char: {chosen_char_str}")

            # propagate the hidden state, change the sequence
            hidden_state = h
            # remove first char, place last char
            sequence_batch = torch.roll(sequence_batch, -1, 1) # do -1 shift along dim of S.
            sequence_batch[0, -1, :] =chosen_char_embedded

    out_text = start_sequence + "".join(new_sequence)

    # ========================

    return out_text


class SequenceBatchSampler(torch.utils.data.Sampler):
    """
    Samples indices from a dataset containing consecutive sequences.
    This sample ensures that samples in the same index of adjacent
    batches are also adjacent in the dataset.
    """

    def __init__(self, dataset: torch.utils.data.Dataset, batch_size):
        """
        :param dataset: The dataset for which to create indices.
        :param batch_size: Number of indices in each batch.
        """
        super().__init__(dataset)
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[int]:
        # TODO:
        #  Return an iterator of indices, i.e. numbers in range(len(dataset)).
        #  dataset and represents one  batch.
        #  The indices must be generated in a way that ensures
        #  that when a batch of size self.batch_size of indices is taken, samples in
        #  the same index of adjacent batches are also adjacent in the dataset.
        #  In the case when the last batch can't have batch_size samples,
        #  you can drop it.
        idx = None  # idx should be a 1-d list of indices.
        # ====== YOUR CODE: ======

        dataset_size = len(self.dataset)
        batches_num = dataset_size // self.batch_size
        dataset_size_cut = batches_num * self.batch_size


        # move simple (but costly) way

        # idx_list = [list(range(batch_num, dataset_size_cut + batches_num - (batches_num - batch_num + 1), batches_num))
        #             for batch_num in range(batches_num)]
        #
        # # hope it's okay. It has better performance than sum
        # from itertools import chain
        #
        # idx = list(chain(*idx_list))

        # with matrixes

        hor_tensor = torch.arange(batches_num) # horizontal - 0, 1, 2
        ver_tensor = torch.arange(0, dataset_size_cut - batches_num + 1, batches_num) # vertical - 0, 3, 6

        hor_tensor = hor_tensor.view(1,batches_num)
        ver_tensor = ver_tensor.view(self.batch_size, 1)

        idxs_tensor = torch.reshape((hor_tensor + ver_tensor).T,  [1, -1])

        idx = idxs_tensor[0, :].tolist()

        # ========================
        return iter(idx)

    def __len__(self):
        return len(self.dataset)


class MultilayerGRU(nn.Module):
    """
    Represents a multi-layer GRU (gated recurrent unit) model.
    """

    def __init__(self, in_dim, h_dim, out_dim, n_layers, dropout=0):
        """
        :param in_dim: Number of input dimensions (at each timestep).
        :param h_dim: Number of hidden state dimensions.
        :param out_dim: Number of input dimensions (at each timestep).
        :param n_layers: Number of layer in the model.
        :param dropout: Level of dropout to apply between layers. Zero
        disables.
        """
        super().__init__()
        assert in_dim > 0 and h_dim > 0 and out_dim > 0 and n_layers > 0

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.h_dim = h_dim
        self.n_layers = n_layers
        self.layer_params = []

        # TODO: Create the parameters of the model for all layers.
        #  To implement the affine transforms you can use either nn.Linear
        #  modules (recommended) or create W and b tensor pairs directly.
        #  Create these modules or tensors and save them per-layer in
        #  the layer_params list.
        #  Important note: You must register the created parameters so
        #  they are returned from our module's parameters() function.
        #  Usually this happens automatically when we assign a
        #  module/tensor as an attribute in our module, but now we need
        #  to do it manually since we're not assigning attributes. So:
        #    - If you use nn.Linear modules, call self.add_module() on them
        #      to register each of their parameters as part of your model.
        #    - If you use tensors directly, wrap them in nn.Parameter() and
        #      then call self.register_parameter() on them. Also make
        #      sure to initialize them. See functions in torch.nn.init.
        # ====== YOUR CODE: ======

        # each layer creates new RNN block
        # Each RNN block has:
        #   for z_t : W_xz, W_hz, b_z
        #   for r_t : W_xr, W_hr, b_r
        #   for g_t : W_xg, W_hg, b_g

        # following RNN implementation in pytorch (many code will appear similar to theirs):
        #   I stack W_xz, W_xr, W_xg together since all
        #   all them multiply the same input matrix. Same for the W_hz, W_hr, W_hg
        #   for the first layer the input is the actual input, for next layers the hidden state is the input

        # init the parameters with normal distribution, mean = 0, std = 0.1
        mean = 0
        std = 0.1

        for layer in range(n_layers+1):

            # for layers > 1, the input is hidden state from previous layer
            layer_input_size  = in_dim  if layer == 0           else h_dim

            if layer < n_layers:
                self.layer_params.append(layer)
                self.layer_params[layer] = {}
                self.layer_params[layer]["W_xz"] = nn.Parameter(torch.Tensor(layer_input_size, h_dim))
                self.layer_params[layer]["W_xr"] = nn.Parameter(torch.Tensor(layer_input_size, h_dim))
                self.layer_params[layer]["W_xg"] = nn.Parameter(torch.Tensor(layer_input_size, h_dim))
                self.layer_params[layer]["W_hz"] = nn.Parameter(torch.Tensor(h_dim, h_dim))
                self.layer_params[layer]["W_hr"] = nn.Parameter(torch.Tensor(h_dim, h_dim))
                self.layer_params[layer]["W_hg"] = nn.Parameter(torch.Tensor(h_dim, h_dim))
                self.layer_params[layer]["b_z"] = nn.Parameter(torch.Tensor(h_dim))
                self.layer_params[layer]["b_r"] = nn.Parameter(torch.Tensor(h_dim))
                self.layer_params[layer]["b_g"] = nn.Parameter(torch.Tensor(h_dim))
            else:

                # register the output params
                self.layer_params.append(layer)
                self.layer_params[layer] = {}
                self.layer_params[layer]["W_hy"] = nn.Parameter(torch.Tensor(h_dim, out_dim))
                self.layer_params[layer]["b_y"] =  nn.Parameter(torch.Tensor(out_dim))

            # init and register

            for param in self.layer_params[layer].keys():
                torch.nn.init.normal_(self.layer_params[layer][param], mean=mean, std=std)
                self.register_parameter(f"{param}_l_{layer}", self.layer_params[layer][param])




        if dropout > 0:
            self.layer_params.append(n_layers+1)
            self.layer_params[n_layers+1] = nn.Dropout(p=dropout)

        # ========================

    def forward(self, input: Tensor, hidden_state: Tensor = None):
        """
        :param input: Batch of sequences. Shape should be (B, S, I) where B is
        the batch size, S is the length of each sequence and I is the
        input dimension (number of chars in the case of a char RNN).
        :param hidden_state: Initial hidden state per layer (for the first
        char). Shape should be (B, L, H) where B is the batch size, L is the
        number of layers, and H is the number of hidden dimensions.
        :return: A tuple of (layer_output, hidden_state).
        The layer_output tensor is the output of the last RNN layer,
        of shape (B, S, O) where B,S are as above and O is the output
        dimension.
        The hidden_state tensor is the final hidden state, per layer, of shape
        (B, L, H) as above.
        """
        batch_size, seq_len, _ = input.shape

        layer_states = []
        for i in range(self.n_layers):
            if hidden_state is None:
                layer_states.append(
                    torch.zeros(batch_size, self.h_dim, device=input.device)
                )
            else:
                layer_states.append(hidden_state[:, i, :])

        layer_input = input
        layer_output = None

        # TODO:
        #  Implement the model's forward pass.
        #  You'll need to go layer-by-layer from bottom to top (see diagram).
        #  Tip: You can use torch.stack() to combine multiple tensors into a
        #  single tensor in a differentiable manner.
        # ====== YOUR CODE: ======


        dropout = False
        if self.n_layers+2 == len(self.layer_params):
            dropout = True

        W_xz =  []
        W_hz =  []
        b_z =   []
        W_xr =  []
        W_hr =  []
        b_r =   []
        W_xg =  []
        W_hg =  []
        b_g =   []

        # for layer in range(self.n_layers):
        #
        #     # next hidden state of each previous time block
        #     # is the input to the next time block as h_(t-1)
        #
        #     ### stack the weights
        #     ### ----------------
        #
        #     W_xz.append(torch.stack([self.layer_params[layer]["W_xz"]] * batch_size))
        #     W_hz.append(torch.stack([self.layer_params[layer]["W_hz"]] * batch_size))
        #     b_z.append(torch.stack([self.layer_params[layer]["b_z"]] * batch_size))
        #
        #     W_xr.append(torch.stack([self.layer_params[layer]["W_xr"]] * batch_size))
        #     W_hr.append(torch.stack([self.layer_params[layer]["W_hr"]] * batch_size))
        #     b_r.append(torch.stack([self.layer_params[layer]["b_r"]] * batch_size))
        #
        #     W_xg.append(torch.stack([self.layer_params[layer]["W_xg"]] * batch_size))
        #     W_hg.append(torch.stack([self.layer_params[layer]["W_hg"]] * batch_size))
        #     b_g.append(torch.stack([self.layer_params[layer]["b_g"]] * batch_size))
        #
        # W_hy_stacked = torch.stack([self.layer_params[self.n_layers]["W_hy"]] * batch_size) # B x H x O
        # b_y_stacked = torch.stack([self.layer_params[self.n_layers]["b_y"]] * batch_size)

        for layer in range(self.n_layers):

            # next hidden state of each previous time block
            # is the input to the next time block as h_(t-1)

            ### stack the weights
            ### ----------------

            W_xz.append(self.layer_params[layer]["W_xz"])
            W_hz.append(self.layer_params[layer]["W_hz"])
            b_z.append(self.layer_params[layer]["b_z"])

            W_xr.append(self.layer_params[layer]["W_xr"])
            W_hr.append(self.layer_params[layer]["W_hr"])
            b_r.append(self.layer_params[layer]["b_r"])

            W_xg.append(self.layer_params[layer]["W_xg"])
            W_hg.append(self.layer_params[layer]["W_hg"])
            b_g.append(self.layer_params[layer]["b_g"])

        W_hy = self.layer_params[self.n_layers]["W_hy"] # H x O
        b_y = self.layer_params[self.n_layers]["b_y"]


        out_dim = W_hy.shape[1]
        hidden_dim = W_hy.shape[0]
        layer_output = torch.zeros(batch_size, seq_len, out_dim) # B x S x O
        hidden_state = torch.zeros(batch_size, self.n_layers, hidden_dim)

        # first deal with all inputs at time 0, going from bottom to top
        # then time 1, etc..


        for s in range(seq_len):

            for layer in range(self.n_layers):


                ### input x
                ### ----------------
                # for first layer the input is the real input, for next layers it is the output of the previous layer

                if layer == 0:
                    x_t = layer_input[:, s, :] # B x I
                else:
                    x_t = layer_states[layer - 1 ] # B x H
                ### ----------------


                ### input h
                ### ----------------
                if dropout:
                    dropout_layer = self.layer_params[-1]
                    h_t_m1 = dropout_layer(layer_states[layer]) # B x H
                else:
                    h_t_m1 = layer_states[layer] # B x H
                ### ----------------

                ### calculations
                #
                # print(x_t.shape)
                # print(W_xz[layer].shape)
                # print(torch.matmul(x_t, W_xz[layer]).shape)

                # update gate
                z_t = torch.matmul(x_t, W_xz[layer]) + torch.matmul(h_t_m1, W_hz[layer]) + b_z[layer] # B x H
                torch.sigmoid_(z_t)

                # reset gate
                r_t = torch.matmul(x_t, W_xr[layer]) + torch.matmul(h_t_m1, W_hr[layer]) + b_r[layer]
                torch.sigmoid_(r_t) # B x H

                g_t = torch.matmul(x_t, W_xg[layer]) + torch.matmul(torch.mul(r_t, h_t_m1), W_hg[layer]) + b_g[layer]
                torch.tanh_(g_t)

                layer_states[layer] = torch.mul(z_t, h_t_m1) + torch.mul((1-z_t), g_t) # B x H
                #
                # print(z_t.shape)
                # print(r_t.shape)
                # print(g_t.shape)

                #  the outputs
                if layer == self.n_layers - 1:
                    # print(layer_states[layer].shape)
                    # print(W_hy_stacked.shape)
                    layer_output[:, s, :] = torch.matmul(layer_states[layer], W_hy) + b_y

            # the final hidden states
            if s == seq_len - 1:
                for layer in range(self.n_layers):
                    hidden_state[:, layer, :] = layer_states[layer]

        # layer_states (L x B x H)
        # layer_output (B, S, O)
        # hidden_state (B, L, H)

        # ========================
        return layer_output, hidden_state
