'''
This opcounter is adapted from https://github.com/sovrasov/macs-counter.pytorch

Copyright (C) 2021 Sovrasov V. - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the MIT license.
 * You should have received a copy of the MIT license with
 * this file. If not visit https://opensource.org/licenses/MIT
'''

import numpy as np
import torch.nn as nn
import torch
from diffusers.models.activations import GELU as DiffusersGELU
from diffusers.models.normalization import AdaLayerNormZeroSingle, AdaLayerNormContinuous, AdaLayerNormZero, RMSNorm
from diffusers.models.transformers.transformer_flux import EmbedND as FluxEmbedND
from diffusers.models.embeddings import (Timesteps, CombinedTimestepTextProjEmbeddings,
                                         CombinedTimestepGuidanceTextProjEmbeddings, TimestepEmbedding,
                                         PixArtAlphaTextProjection, CombinedTimestepLabelEmbeddings)
from diffusers.models.attention_processor import Attention, FluxAttnProcessor2_0, FluxSingleAttnProcessor2_0


@torch.no_grad()
def count_ops_and_params(model, example_inputs):
    global CUSTOM_MODULES_MAPPING
    model = copy.deepcopy(model)
    macs_model = add_macs_counting_methods(model)
    macs_model.eval()
    macs_model.start_macs_count(ost=sys.stdout, verbose=False,
                                ignore_list=[])
    if isinstance(example_inputs, dict):
        _ = macs_model(**example_inputs)
    elif isinstance(example_inputs, (tuple, list)):
        _ = macs_model(*example_inputs)
    else:
        _ = macs_model(example_inputs)
    macs_count, params_count = macs_model.compute_average_macs_cost()
    macs_model.stop_macs_count()
    CUSTOM_MODULES_MAPPING = {}
    return macs_count, params_count


def empty_macs_counter_hook(module, input, output):
    module.__macs__ += 0


def upsample_macs_counter_hook(module, input, output):
    output_size = output[0]
    batch_size = output_size.shape[0]
    output_elements_count = batch_size
    for val in output_size.shape[1:]:
        output_elements_count *= val
    module.__macs__ += int(output_elements_count)


def relu_macs_counter_hook(module, input, output):
    active_elements_count = output.numel()
    module.__macs__ += int(active_elements_count)


def diffusers_gelu_macs_counter_hook(module, input, output):
    active_elements_count = output.numel()
    module.__macs__ += int(active_elements_count)
    module.__macs__ += module.proj.__macs__


def linear_macs_counter_hook(module, input, output):
    input = input[0]
    # pytorch checks dimensions, so here we don't care much
    output_last_dim = output.shape[-1]
    bias_macs = output_last_dim if module.bias is not None else 0
    module.__macs__ += int(np.prod(input.shape) * output_last_dim + bias_macs)


def pool_macs_counter_hook(module, input, output):
    input = input[0]
    module.__macs__ += int(np.prod(input.shape))


def bn_macs_counter_hook(module, input, output):
    input = input[0]

    batch_macs = np.prod(input.shape)
    if module.affine:
        batch_macs *= 2
    module.__macs__ += int(batch_macs)


def layer_norm_macs_counter_hook(module, input, output):
    input = input[0]

    batch_macs = 4 * np.prod(input.shape)
    if module.elementwise_affine:
        batch_macs *= 1.5
    module.__macs__ += int(batch_macs)


def conv_macs_counter_hook(conv_module, input, output):
    # Can have multiple inputs, getting the first one
    input = input[0]

    batch_size = input.shape[0]
    output_dims = list(output.shape[2:])

    kernel_dims = list(conv_module.kernel_size)
    in_channels = conv_module.in_channels
    out_channels = conv_module.out_channels
    groups = conv_module.groups

    filters_per_channel = out_channels // groups
    conv_per_position_macs = int(np.prod(kernel_dims)) * \
                             in_channels * filters_per_channel

    active_elements_count = batch_size * int(np.prod(output_dims))

    overall_conv_macs = conv_per_position_macs * active_elements_count

    bias_macs = 0

    if conv_module.bias is not None:
        bias_macs = out_channels * active_elements_count

    overall_macs = overall_conv_macs + bias_macs

    conv_module.__macs__ += int(overall_macs)


def rnn_macs(macs, rnn_module, w_ih, w_hh, input_size):
    # matrix matrix mult ih state and internal state
    macs += w_ih.shape[0] * w_ih.shape[1]
    # matrix matrix mult hh state and internal state
    macs += w_hh.shape[0] * w_hh.shape[1]
    if isinstance(rnn_module, (nn.RNN, nn.RNNCell)):
        # add both operations
        macs += rnn_module.hidden_size
    elif isinstance(rnn_module, (nn.GRU, nn.GRUCell)):
        # hadamard of r
        macs += rnn_module.hidden_size
        # adding operations from both states
        macs += rnn_module.hidden_size * 3
        # last two hadamard product and add
        macs += rnn_module.hidden_size * 3
    elif isinstance(rnn_module, (nn.LSTM, nn.LSTMCell)):
        # adding operations from both states
        macs += rnn_module.hidden_size * 4
        # two hadamard product and add for C state
        macs += rnn_module.hidden_size + rnn_module.hidden_size + rnn_module.hidden_size
        # final hadamard
        macs += rnn_module.hidden_size + rnn_module.hidden_size + rnn_module.hidden_size
    return macs


def rnn_macs_counter_hook(rnn_module, input, output):
    """
    Takes into account batch goes at first position, contrary
    to pytorch common rule (but actually it doesn't matter).
    If sigmoid and tanh are hard, only a comparison macs should be accurate
    """
    macs = 0
    # input is a tuple containing a sequence to process and (optionally) hidden state
    inp = input[0]
    batch_size = inp[0].shape[0]
    seq_length = inp[0].shape[1]
    num_layers = rnn_module.num_layers

    for i in range(num_layers):
        w_ih = rnn_module.__getattr__('weight_ih_l' + str(i))
        w_hh = rnn_module.__getattr__('weight_hh_l' + str(i))
        if i == 0:
            input_size = rnn_module.input_size
        else:
            input_size = rnn_module.hidden_size
        macs = rnn_macs(macs, rnn_module, w_ih, w_hh, input_size)
        if rnn_module.bias:
            b_ih = rnn_module.__getattr__('bias_ih_l' + str(i))
            b_hh = rnn_module.__getattr__('bias_hh_l' + str(i))
            macs += b_ih.shape[0] + b_hh.shape[0]

    macs *= batch_size
    macs *= seq_length
    if rnn_module.bidirectional:
        macs *= 2
    rnn_module.__macs__ += int(macs)


def rnn_cell_macs_counter_hook(rnn_cell_module, input, output):
    macs = 0
    inp = input[0]
    batch_size = inp.shape[0]
    w_ih = rnn_cell_module.__getattr__('weight_ih')
    w_hh = rnn_cell_module.__getattr__('weight_hh')
    input_size = inp.shape[1]
    macs = rnn_macs(macs, rnn_cell_module, w_ih, w_hh, input_size)
    if rnn_cell_module.bias:
        b_ih = rnn_cell_module.__getattr__('bias_ih')
        b_hh = rnn_cell_module.__getattr__('bias_hh')
        macs += b_ih.shape[0] + b_hh.shape[0]

    macs *= batch_size
    rnn_cell_module.__macs__ += int(macs)


def multihead_attention_counter_hook(multihead_attention_module, input, output):
    macs = 0
    q, k, v = input

    batch_first = multihead_attention_module.batch_first \
        if hasattr(multihead_attention_module, 'batch_first') else False
    if batch_first:
        batch_size = q.shape[0]
        len_idx = 1
    else:
        batch_size = q.shape[1]
        len_idx = 0

    dim_idx = 2

    qdim = q.shape[dim_idx]
    kdim = k.shape[dim_idx]
    vdim = v.shape[dim_idx]

    qlen = q.shape[len_idx]
    klen = k.shape[len_idx]
    vlen = v.shape[len_idx]

    num_heads = multihead_attention_module.num_heads
    assert qdim == multihead_attention_module.embed_dim

    if multihead_attention_module.kdim is None:
        assert kdim == qdim
    if multihead_attention_module.vdim is None:
        assert vdim == qdim

    macs = 0

    # Q scaling
    macs += qlen * qdim

    # Initial projections
    macs += (
            (qlen * qdim * qdim)  # QW
            + (klen * kdim * kdim)  # KW
            + (vlen * vdim * vdim)  # VW
    )

    if multihead_attention_module.in_proj_bias is not None:
        macs += (qlen + klen + vlen) * qdim

    # attention heads: scale, matmul, softmax, matmul
    qk_head_dim = qdim // num_heads
    v_head_dim = vdim // num_heads

    head_macs = (
            (qlen * klen * qk_head_dim)  # QK^T
            + (qlen * klen)  # softmax
            + (qlen * klen * v_head_dim)  # AV
    )

    macs += num_heads * head_macs

    # final projection, bias is always enabled
    macs += qlen * vdim * (vdim + 1)

    macs *= batch_size
    multihead_attention_module.__macs__ += int(macs)


def flux_embed_macs_counter_hook(module, input, output):
    ids = input[0]
    n_axes = ids.shape[-1]
    batch_size, seq_length = ids.shape[:-1]

    total_macs = 0

    # Loop over the axes
    for i in range(n_axes):
        dim = module.axes_dim[i]

        # MACs in the rope function
        macs_scale_and_omega = dim  # MACs for scale and omega computation
        macs_einsum = batch_size * seq_length * (dim // 2)  # MACs for einsum
        macs_trig = 2 * macs_einsum  # MACs for cosine and sine

        total_macs += macs_scale_and_omega + macs_einsum + macs_trig

    module.__macs__ += total_macs


def diffusers_timesteps_counter_hook(module, input, output):
    timesteps = input[0]
    N = timesteps.shape[0]
    embedding_dim = module.num_channels
    half_dim = embedding_dim // 2

    # Exponent calculation
    macs_exponent = half_dim

    # Embedding computation
    macs_embedding = N * half_dim

    # Scaling the embedding
    macs_scaling = N * half_dim

    # Sine and cosine computations
    macs_trig = 2 * N * half_dim

    # Total MACs
    total_macs = macs_exponent + macs_embedding + macs_scaling + macs_trig

    module.__macs__ += total_macs


def generic_module_counter_hook(module, input, output):
    for m in module.children():
        module.__macs__ += m.__macs__


def flux_combined_timestep_text_proj_embed_macs_counter_hook(module, input, output):
    output_elements_count = output.numel()
    total_macs = output_elements_count  # MACs for the sum
    for m in module.children():
        total_macs += m.__macs__
    module.__macs__ += total_macs


def flux_combined_timestep_guidance_text_proj_embed_macs_counter_hook(module, input, output):
    output_elements_count = output.numel()
    total_macs = 2 * output_elements_count  # MACs for the summations
    for m in module.children():
        total_macs += m.__macs__
    module.__macs__ += total_macs


def rms_norm_macs_counter_hook(module, input, output):
    input = input[0]
    batch_macs = np.prod(input.shape)
    module.__macs__ += (int(batch_macs) * 3)
    if module.weight is not None:
        module.__macs__ += int(batch_macs)


def ada_layer_norm_continuous_macs_counter_hook(module, input, output):
    input = input[0]
    batch_macs = np.prod(input.shape)
    module.__macs__ += (int(batch_macs) * 3)
    for m in module.children():
        module.__macs__ += m.__macs__


def ada_layer_norm_zero_single_macs_counter_hook(module, input, output):
    input = input[0]
    batch_macs = np.prod(input.shape)
    module.__macs__ += (int(batch_macs) * 4)
    for m in module.children():
        module.__macs__ += m.__macs__


def ada_layer_norm_zero_macs_counter_hook(module, input, output):
    input = input[0]
    batch_macs = np.prod(input.shape)
    module.__macs__ += (int(batch_macs) * 8)
    for m in module.children():
        module.__macs__ += m.__macs__


def flux_attention_counter_hook(module, input, output):

    for m in module.children():
        if isinstance(m, nn.ModuleList):
            # to_out has a linear and a dropout, only the linear is counted
            module.__macs__ += m[0].__macs__
        else:
            module.__macs__ += m.__macs__

    if not isinstance(module.processor, (FluxAttnProcessor2_0, FluxSingleAttnProcessor2_0)):
        raise ValueError("Unknown Attention Processor for Flux Transformer")

    if isinstance(module.processor, FluxSingleAttnProcessor2_0):
        hidden_states = output
        seq_length = hidden_states.shape[1]
    else:
        hidden_states = output[0]
        encoder_hidden_states = output[1]
        seq_length = hidden_states.shape[1] + encoder_hidden_states.shape[1]

    batch_size, _, hidden_size = hidden_states.shape
    num_heads = module.heads
    head_dim = hidden_size // num_heads

    # q * k^T
    macs_qk = batch_size * num_heads * seq_length * seq_length * head_dim

    # scaling
    macs_scale = batch_size * num_heads * seq_length * seq_length

    # softmax
    macs_softmax = batch_size * num_heads * seq_length * seq_length

    # attention * v
    macs_att_v = batch_size * num_heads * seq_length * seq_length * head_dim

    module.__macs__ += macs_qk + macs_scale + macs_softmax + macs_att_v


CUSTOM_MODULES_MAPPING = {
}

MODULES_MAPPING = {
    # convolutions
    nn.Conv1d: conv_macs_counter_hook,
    nn.Conv2d: conv_macs_counter_hook,
    nn.Conv3d: conv_macs_counter_hook,
    # activations
    nn.ReLU: relu_macs_counter_hook,
    nn.PReLU: relu_macs_counter_hook,
    nn.ELU: relu_macs_counter_hook,
    nn.LeakyReLU: relu_macs_counter_hook,
    nn.ReLU6: relu_macs_counter_hook,
    # Poolings
    nn.MaxPool1d: pool_macs_counter_hook,
    nn.AvgPool1d: pool_macs_counter_hook,
    nn.AvgPool2d: pool_macs_counter_hook,
    nn.MaxPool2d: pool_macs_counter_hook,
    nn.MaxPool3d: pool_macs_counter_hook,
    nn.AvgPool3d: pool_macs_counter_hook,
    nn.AdaptiveMaxPool1d: pool_macs_counter_hook,
    nn.AdaptiveAvgPool1d: pool_macs_counter_hook,
    nn.AdaptiveMaxPool2d: pool_macs_counter_hook,
    nn.AdaptiveAvgPool2d: pool_macs_counter_hook,
    nn.AdaptiveMaxPool3d: pool_macs_counter_hook,
    nn.AdaptiveAvgPool3d: pool_macs_counter_hook,
    # BNs
    nn.BatchNorm1d: bn_macs_counter_hook,
    nn.BatchNorm2d: bn_macs_counter_hook,
    nn.BatchNorm3d: bn_macs_counter_hook,

    nn.InstanceNorm1d: bn_macs_counter_hook,
    nn.InstanceNorm2d: bn_macs_counter_hook,
    nn.InstanceNorm3d: bn_macs_counter_hook,
    nn.GroupNorm: bn_macs_counter_hook,
    # FC
    nn.Linear: linear_macs_counter_hook,
    # Upscale
    nn.Upsample: upsample_macs_counter_hook,
    # Deconvolution
    nn.ConvTranspose1d: conv_macs_counter_hook,
    nn.ConvTranspose2d: conv_macs_counter_hook,
    nn.ConvTranspose3d: conv_macs_counter_hook,
    # RNN
    nn.RNN: rnn_macs_counter_hook,
    nn.GRU: rnn_macs_counter_hook,
    nn.LSTM: rnn_macs_counter_hook,
    nn.RNNCell: rnn_cell_macs_counter_hook,
    nn.LSTMCell: rnn_cell_macs_counter_hook,
    nn.GRUCell: rnn_cell_macs_counter_hook,
    nn.MultiheadAttention: multihead_attention_counter_hook,


    # Activation
    DiffusersGELU: diffusers_gelu_macs_counter_hook,
    nn.SiLU: relu_macs_counter_hook,
    # Embedding
    FluxEmbedND: flux_embed_macs_counter_hook,
    Timesteps: diffusers_timesteps_counter_hook,
    CombinedTimestepTextProjEmbeddings: flux_combined_timestep_text_proj_embed_macs_counter_hook,
    CombinedTimestepGuidanceTextProjEmbeddings: flux_combined_timestep_guidance_text_proj_embed_macs_counter_hook,
    PixArtAlphaTextProjection: generic_module_counter_hook,
    TimestepEmbedding: generic_module_counter_hook,
    CombinedTimestepLabelEmbeddings: flux_combined_timestep_text_proj_embed_macs_counter_hook,
    # Normalization
    nn.LayerNorm: layer_norm_macs_counter_hook,
    AdaLayerNormContinuous: ada_layer_norm_continuous_macs_counter_hook,
    AdaLayerNormZeroSingle: ada_layer_norm_zero_single_macs_counter_hook,
    AdaLayerNormZero: ada_layer_norm_zero_macs_counter_hook,
    RMSNorm: rms_norm_macs_counter_hook,
    # Diffusers Attention
    Attention: flux_attention_counter_hook
}

if hasattr(nn, 'GELU'):
    MODULES_MAPPING[nn.GELU] = relu_macs_counter_hook

import sys
from functools import partial
import torch.nn as nn
import copy


def accumulate_macs(self):
    if is_supported_instance(self):
        return self.__macs__
    else:
        sum = 0
        for m in self.children():
            sum += m.accumulate_macs()
        return sum


def get_model_parameters_number(model):
    params_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params_num


def add_macs_counting_methods(net_main_module):
    # adding additional methods to the existing module object,
    # this is done this way so that each function has access to self object
    net_main_module.start_macs_count = start_macs_count.__get__(net_main_module)
    net_main_module.stop_macs_count = stop_macs_count.__get__(net_main_module)
    net_main_module.reset_macs_count = reset_macs_count.__get__(net_main_module)
    net_main_module.compute_average_macs_cost = compute_average_macs_cost.__get__(
        net_main_module)

    net_main_module.reset_macs_count()

    return net_main_module


def compute_average_macs_cost(self):
    """
    A method that will be available after add_macs_counting_methods() is called
    on a desired net object.
    Returns current mean macs consumption per image.
    """

    for m in self.modules():
        m.accumulate_macs = accumulate_macs.__get__(m)

    macs_sum = self.accumulate_macs()

    for m in self.modules():
        if hasattr(m, 'accumulate_macs'):
            del m.accumulate_macs

    params_sum = get_model_parameters_number(self)
    return macs_sum / self.__batch_counter__, params_sum


def start_macs_count(self, **kwargs):
    """
    A method that will be available after add_macs_counting_methods() is called
    on a desired net object.
    Activates the computation of mean macs consumption per image.
    Call it before you run the network.
    """
    add_batch_counter_hook_function(self)

    seen_types = set()

    def add_macs_counter_hook_function(module, ost, verbose, ignore_list):
        if type(module) in ignore_list:
            seen_types.add(type(module))
            if is_supported_instance(module):
                module.__params__ = 0
        elif is_supported_instance(module):
            if hasattr(module, '__macs_handle__'):
                return
            if type(module) in CUSTOM_MODULES_MAPPING:
                handle = module.register_forward_hook(
                    CUSTOM_MODULES_MAPPING[type(module)])
            else:
                handle = module.register_forward_hook(MODULES_MAPPING[type(module)])
            module.__macs_handle__ = handle
            seen_types.add(type(module))
        else:
            if verbose and not type(module) in (nn.Sequential, nn.ModuleList) and \
                    not type(module) in seen_types:
                print('Warning: module ' + type(module).__name__ +
                      ' is treated as a zero-op.', file=ost)
            seen_types.add(type(module))

    self.apply(partial(add_macs_counter_hook_function, **kwargs))


def stop_macs_count(self):
    """
    A method that will be available after add_macs_counting_methods() is called
    on a desired net object.
    Stops computing the mean macs consumption per image.
    Call whenever you want to pause the computation.
    """
    remove_batch_counter_hook_function(self)
    self.apply(remove_macs_counter_hook_function)
    self.apply(remove_macs_counter_variables)


def reset_macs_count(self):
    """
    A method that will be available after add_macs_counting_methods() is called
    on a desired net object.
    Resets statistics computed so far.
    """
    add_batch_counter_variables_or_reset(self)
    self.apply(add_macs_counter_variable_or_reset)


# ---- Internal functions
def batch_counter_hook(module, input, output):
    batch_size = 1
    if len(input) > 0:
        # Can have multiple inputs, getting the first one
        input = input[0]
        batch_size = len(input)
    else:
        pass
        print('Warning! No positional inputs found for a module,'
              ' assuming batch size is 1.')
    module.__batch_counter__ += batch_size


def add_batch_counter_variables_or_reset(module):
    module.__batch_counter__ = 0


def add_batch_counter_hook_function(module):
    if hasattr(module, '__batch_counter_handle__'):
        return

    handle = module.register_forward_hook(batch_counter_hook)
    module.__batch_counter_handle__ = handle


def remove_batch_counter_hook_function(module):
    if hasattr(module, '__batch_counter_handle__'):
        module.__batch_counter_handle__.remove()
        del module.__batch_counter_handle__


def add_macs_counter_variable_or_reset(module):
    if is_supported_instance(module):
        if hasattr(module, '__macs__') or hasattr(module, '__params__'):
            print('Warning: variables __macs__ or __params__ are already '
                  'defined for the module' + type(module).__name__ +
                  ' ptmacs can affect your code!')
            module.__ptmacs_backup_macs__ = module.__macs__
            module.__ptmacs_backup_params__ = module.__params__
        module.__macs__ = 0
        module.__params__ = get_model_parameters_number(module)


def is_supported_instance(module):
    if type(module) in MODULES_MAPPING or type(module) in CUSTOM_MODULES_MAPPING:
        return True
    return False


def remove_macs_counter_hook_function(module):
    if is_supported_instance(module):
        if hasattr(module, '__macs_handle__'):
            module.__macs_handle__.remove()
            del module.__macs_handle__


def remove_macs_counter_variables(module):
    if is_supported_instance(module):
        if hasattr(module, '__macs__'):
            del module.__macs__
            if hasattr(module, '__ptmacs_backup_macs__'):
                module.__macs__ = module.__ptmacs_backup_macs__
        if hasattr(module, '__params__'):
            del module.__params__
            if hasattr(module, '__ptmacs_backup_params__'):
                module.__params__ = module.__ptmacs_backup_params__
