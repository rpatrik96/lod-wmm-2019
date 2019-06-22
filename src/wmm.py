import math

import numpy as np
import torch

"""-------------------------------------------------------------------------"""
"""---------------------------------WMM base--------------------------------"""
"""-------------------------------------------------------------------------"""

class WMMBase(object):
    """WMMBase
        Class for implementing common operations for WMM.

        Args:
            p (float): Probability of executing the wmm operation
            coverage (float): specifies the maximum  affected area of a weight matrix,
                                given in (0.0, 1.0]
            layer(str or :object `list` of :obj:`str`): affected layer or list of affected layers
            mode(str): mode of operation
            layer_modifier(str or :object `list` of :obj:`str`, optional): selects the affected gate(s),
                        used only for LSTM/LSTMCell and GRU/GRUCell
    """

    def __init__(self, env, p=0.1, coverage=0.1, layer=None, mode="rect", layer_modifier=None):
        self.env = env
        self.p = p
        self.coverage = coverage
        self.layer = layer if type(layer) is list else [layer]
        self.mode = mode

        # ['input', 'forget', 'cell', 'output']   for LSTM and ['reset', 'update', 'new'] for GRU
        self.layer_modifier = layer_modifier if type(layer_modifier) is list else [layer_modifier]

        # entropy log
        self.entropy_dict = {}

        # weight log
        self.weight_dict = {}

    def _create_tensor_dict(self, weight_tensor, name, layer_type):
        tensor_dict = {}
        if layer_type == torch.nn.LSTM or layer_type == torch.nn.LSTMCell:
            dim_per_gate = weight_tensor.shape[0] // 4

            if 'input' or 'all' in self.layer_modifier:
                tensor_dict[name + '_input'] = weight_tensor[0:dim_per_gate - 1, :]

            if 'forget' or 'all' in self.layer_modifier:
                tensor_dict[name + '_forget'] = weight_tensor[dim_per_gate:2 * dim_per_gate - 1, :]

            if 'cell' or 'all' in self.layer_modifier:
                tensor_dict[name + '_cell'] = weight_tensor[2 * dim_per_gate:3 * dim_per_gate - 1, :]

            if 'output' or 'all' in self.layer_modifier:
                tensor_dict[name + '_output'] = weight_tensor[3 * dim_per_gate:4 * dim_per_gate - 1, :]

        elif layer_type == torch.nn.GRU or layer_type == torch.nn.GRUCell:
            dim_per_gate = weight_tensor.shape[0] / 3

            if 'reset' in self.layer_modifier:
                tensor_dict[name + '_reset'] = weight_tensor[0:dim_per_gate - 1, :]

            if 'update' in self.layer_modifier:
                tensor_dict[name + '_update'] = weight_tensor[dim_per_gate:2 * dim_per_gate - 1, :]

            if 'new' in self.layer_modifier:
                tensor_dict[name + '_new'] = weight_tensor[2 * dim_per_gate:3 * dim_per_gate - 1, :]

        else:
            tensor_dict[name] = weight_tensor

        return tensor_dict

    def _tensor_entropy(self, tensor):
        flattened_tensor = tensor.view(1, -1).cpu().detach().numpy()
        num_bins = max(5, flattened_tensor.shape[-1] // 20)  # equidistant edges
        hist, bin_edges = np.histogram(flattened_tensor, num_bins, density=True)

        # approximate the probability density function
        pdf_appr = hist * (bin_edges[1] - bin_edges[0])
        pdf_appr = pdf_appr[pdf_appr > 10e-6]

        return np.sum(-np.multiply(pdf_appr, np.log2(pdf_appr)))

    def _log_hist_and_entropy(self, tensor, name):
        # init entropy dict
        if name not in self.entropy_dict.keys():
            self.entropy_dict[name] = []
            self.weight_dict[name] = []

        # append new element
        self.entropy_dict[name].append(self._tensor_entropy(tensor))
        self.weight_dict[name].append(tensor)

    def calculate_entropy(self, model):
        cumulative_entropy = 0.0

        try:

            # iterate through all matrices and sum the partial entropies
            for layer in model._modules.values():
                for tensor in layer._parameters.values():
                    cumulative_entropy += self._tensor_entropy(tensor)

            return cumulative_entropy
        except:
            return -1

"""-------------------------------------------------------------------------"""
"""--------------------------Weight reinitialization------------------------"""
"""-------------------------------------------------------------------------"""

class WeightReinit(WMMBase):
    def __init__(self, p=0.1, coverage=0.1, layer=None, mode="rect", layer_modifier=None):
        super().__init__(env="Reinit", p=p, coverage=coverage, layer=layer, mode=mode, layer_modifier=layer_modifier)

        # Todo: future
        self.interval = None
        self.min_flag = None
        self.alpha = None

    def _rect_mode(self, tensor, weight_mask, temp_weights):
        num_dim = min(2, len(tensor.shape))

        # Todo: maybe a dict would be better...
        rand_list = []
        len_list = []

        # generate begin indices and lengths for the masks
        for i in range(num_dim):
            rand_list.append(np.random.randint(tensor.shape[-(i + 1)]))
            len_list.append(np.random.randint(1, max(2, int(
                    self.coverage * tensor.shape[-(i + 1)]))))

        # create the rectangular submask
        mask = torch.zeros_like(tensor)

        # fill the mask with ones (+1 added because tensor slicing is exclusive at the end)
        if num_dim is 2:
            mask[rand_list[1]: rand_list[1] + len_list[1] + 1,
            rand_list[0]: rand_list[0] + len_list[0] + 1] = 1
        else:
            mask[rand_list[0]: rand_list[0] + len_list[0] + 1] = 1

        weight_mask *= mask

        # masked_scatter_ copies data from temp_weights into layer.weight.data
        # where the mask has a value of one
        tensor.masked_scatter_(weight_mask.byte(), temp_weights)

    def run(self, model):
        if model.training:
            for name, layer in model._modules.items():

                if True in {n in name or n is 'all' for n in self.layer}:
                    # omit layers such as maxpool

                    # get layers which have a oarameter containing 'weight'
                    weight_name_set = {(w if 'weight' in w else None) for w in layer._parameters.keys()} - {None}
                    for weight in weight_name_set:

                        tensor_dict = self._create_tensor_dict(layer._parameters[weight].data, name, type(layer))
                        for tensor_name, tensor in tensor_dict.items():
                            if 'grad' not in name and 'reload' not in name:  # Todo: futurefr
                                num_columns = tensor.shape[-1]  # n parameter for Xavier

                                # Todo: make available the choice of initializer
                                # Xavier weights generation
                                temp_weights = torch.zeros_like(tensor, device=tensor.device) \
                                    .uniform_(-1 / math.sqrt(num_columns), 1 / math.sqrt(num_columns))

                                # carry out only if p!=0
                                if self.p:
                                    weight_mask = torch.rand_like(tensor).lt_(self.p)  # mask for weights

                                    if self.mode == 'rect':
                                        self._rect_mode(tensor, weight_mask, temp_weights)

                                    # self._log_hist_and_entropy(tensor, tensor_name)

"""-------------------------------------------------------------------------"""
"""------------------------------Weight shuffle-----------------------------"""
"""-------------------------------------------------------------------------"""

class WeightShuffle(WMMBase):
    def __init__(self, p=0.1, coverage=0.1, layer=None, mode="rect", layer_modifier=None):
        super().__init__(env="Shuffle", p=p, coverage=coverage, layer=layer, mode=mode, layer_modifier=layer_modifier)

        # Constants
        self.affected_dim = 2

    def run(self, model):
        if model.training:
            for name, layer in model._modules.items():
                if True in {n in name or n is 'all' for n in self.layer}:
                    # omit layers such as maxpool

                    # get layers which have a oarameter containing 'weight'
                    weight_name_set = {(w if 'weight' in w else None) for w in layer._parameters.keys()} - {None}
                    for weight in weight_name_set:

                        tensor_dict = self._create_tensor_dict(layer._parameters[weight].data, name, type(layer))
                        for tensor_name, tensor in tensor_dict.items():

                            # calculate shape and number of dimensions
                            tensor_shape = tensor.shape
                            tensor_dim = len(tensor_shape)

                            # enable maximum 2d elements being shuffled
                            dim2shuffle = min(self.affected_dim, tensor_dim)

                            # count number of elements (number in the last dim2shuffle dimensions)
                            num_elem = 1
                            for i in range(tensor_dim - dim2shuffle):
                                num_elem *= tensor_shape[i]

                            # create view according to this number
                            tensor_view = tensor.view(num_elem, -1)
                            elem_len = tensor_view.shape[-1]

                            for i in range(num_elem):
                                if self.p > np.random.random():
                                    if self.mode == 'rect':
                                        # generate values for index and length
                                        rand_index = np.random.randint(0, elem_len)
                                        rand_len = np.random.randint(1, max(2, int(self.coverage * elem_len)))

                                        # index and shuffle
                                        np.random.shuffle(tensor_view[i][rand_index: rand_index + rand_len])

                            # self._log_hist_and_entropy(tensor, tensor_name)
