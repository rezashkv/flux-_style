import torch
from diffusers import FluxPipeline
import matplotlib.pyplot as plt

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)

lora_name_or_paths = ['multimodalart/flux-tarot-v1', 'SebastianBodza/Flux_Aquarell_Watercolor_v2',
                      'alvdansen/frosting_lane_flux', 'davisbro/half_illustration', 'veryVANYA/ps1-style-flux']
lora_names = ['tarot', 'aquarelle', 'frosting_lane', 'half_illustration', 'ps1']
weight_names = ["flux_tarot_v1_lora.safetensors", "lora.safetensors", "flux_dev_frostinglane_araminta_k.safetensors",
                "flux_train_replicate.safetensors", "ps1_style_flux_v1.safetensors"]

for lora_name_or_path, lora_name, weight_name in zip(lora_name_or_paths, lora_names, weight_names):
    state_dict = pipe.lora_state_dict(lora_name_or_path, weight_name=weight_name)
    single_transformer_blocks = sorted(list(
        set([k[:-len(".lora_A.weight")] for k in state_dict if ".single_transformer_blocks" in k])))
    transformer_blocks = sorted(list(
        set([k[:-len(".lora_A.weight")] for k in state_dict if ".transformer_blocks" in k])))

    single_transformer_blocks_norms = {}
    for key in single_transformer_blocks:
        lora_A = state_dict[key + ".lora_A.weight"]
        lora_B = state_dict[key + ".lora_B.weight"]
        # lora weight is the product of the two lora weights
        lora_delta = lora_B @ lora_A

        # find norm of lora_delta
        norm = torch.linalg.norm(lora_delta)
        single_transformer_blocks_norms[key.replace("transformer.single_transformer_blocks.", "")] = norm

    transformer_blocks_norms = {}
    for key in transformer_blocks:
        lora_A = state_dict[key + ".lora_A.weight"]
        lora_B = state_dict[key + ".lora_B.weight"]
        # lora weight is the product of the two lora weights
        lora_delta = lora_B @ lora_A

        # find norm of lora_delta
        norm = torch.linalg.norm(lora_delta)
        transformer_blocks_norms[key.replace("transformer.transformer_blocks.", "")] = norm

    # plot the norms in a bar chart. both in the same plot but different colors
    fig, ax = plt.subplots(figsize=(80, 30))
    s_keys = sorted(single_transformer_blocks_norms.keys(), key=lambda x: int(x.split('.')[0]))
    s_values = [single_transformer_blocks_norms[k] for k in s_keys]
    s_keys = [f"single.{k}" for k in s_keys]
    keys = sorted(transformer_blocks_norms.keys(), key=lambda x: int(x.split('.')[0]))
    values = [transformer_blocks_norms[k] for k in keys]
    ax.bar(keys, values, color='r', label='transformer_blocks')
    ax.bar(s_keys, s_values, color='b', label='single_transformer_blocks')

    # change the direction of x labels
    ax.set_xticklabels(keys + s_keys, rotation=90)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Norm')
    ax.set_title(f'{lora_name} Lora Weight Analysis')

    plt.legend()

    plt.show()
