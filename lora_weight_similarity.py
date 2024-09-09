import torch
from diffusers import FluxPipeline, FluxTransformer2DModel, SD3Transformer2DModel
import matplotlib.pyplot as plt

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)

lora_name_or_paths = ['multimodalart/flux-tarot-v1', 'SebastianBodza/Flux_Aquarell_Watercolor_v2',
                      # 'alvdansen/frosting_lane_flux', 'davisbro/half_illustration', 'veryVANYA/ps1-style-flux'
                      ]
lora_names = ['tarot', 'aquarelle',
              # 'frosting_lane', 'half_illustration', 'ps1'
              ]
weight_names = ["flux_tarot_v1_lora.safetensors", "lora.safetensors",
                # "flux_dev_frostinglane_araminta_k.safetensors",
                # "flux_train_replicate.safetensors", "ps1_style_flux_v1.safetensors"
                ]

weights_dict = {k: dict() for k in lora_names}
for lora_name_or_path, lora_name, weight_name in zip(lora_name_or_paths, lora_names, weight_names):
    state_dict = pipe.lora_state_dict(lora_name_or_path, weight_name=weight_name)
    single_transformer_blocks = sorted(list(
        set([k[:-len(".lora_A.weight")] for k in state_dict if ".single_transformer_blocks" in k])))
    transformer_blocks = sorted(list(
        set([k[:-len(".lora_A.weight")] for k in state_dict if ".transformer_blocks" in k])))

    for key in single_transformer_blocks:
        lora_A = state_dict[key + ".lora_A.weight"]
        lora_B = state_dict[key + ".lora_B.weight"]
        # lora weight is the product of the two lora weights
        lora_delta = lora_B @ lora_A
        weights_dict[lora_name][key.replace("transformer.single_transformer_blocks.", "single.")] = lora_delta

        # find norm of lora_delta
    for key in transformer_blocks:
        lora_A = state_dict[key + ".lora_A.weight"]
        lora_B = state_dict[key + ".lora_B.weight"]
        # lora weight is the product of the two lora weights
        lora_delta = lora_B @ lora_A
        weights_dict[lora_name][key.replace("transformer.transformer_blocks.", ".")] = lora_delta

# pairwise_sims = dict()
# # calculate average matrix similarity between lora weights across all lora names
#
# for i in range(len(lora_names)):
#     for j in range(i + 1, len(lora_names)):
#         pairwise_sims[(lora_names[i], lora_names[j])] = dict()
#         for key in weights_dict[lora_names[0]]:
#             similarity = torch.nn.functional.cosine_similarity(weights_dict[lora_names[i]][key].flatten(),
#                                                                weights_dict[lora_names[j]][key].flatten(),
#                                                                dim=0).item()
#             pairwise_sims[(lora_names[i], lora_names[j])][key] = similarity

# # plot the pairwise similarities
# for key in pairwise_sims:
#     fig, ax = plt.subplots(figsize=(80, 30))
#     s_keys = sorted([k for k in weights_dict[lora_names[0]].keys() if "single" in k], key=lambda x: int(x.split('.')[1]))
#     s_values = [pairwise_sims[key][k] for k in s_keys]
#     b_keys = sorted([k for k in weights_dict[lora_names[0]].keys() if "single" not in k], key=lambda x: int(x.split('.')[1]))
#     b_values = [pairwise_sims[key][k] for k in b_keys]
#     ax.bar(b_keys, b_values, color='r', label='transformer_blocks')
#     ax.bar(s_keys, s_values, color='b', label='single_transformer_blocks')
#     # change the direction of x labels
#     ax.set_xticklabels(b_keys + s_keys, rotation=90)
#     ax.set_xlabel('Layer')
#     ax.set_ylabel('similarity')
#     plt.title(f"Pairwise similarities between {key[0]} and {key[1]}")
#     plt.legend()
#     # save the plot
#     plt.savefig(f"./lora_analysis/{key[0]}_{key[1]}_pairwise_similarity.png")


groups = {"t.attn": [k for k in weights_dict[lora_names[0]].keys() if "attn" in k and "single" not in k],
          "t.ff": [k for k in weights_dict[lora_names[0]].keys() if "ff." in k and "proj" in k and "single" not in k],
          "t.ff_context": [k for k in weights_dict[lora_names[0]].keys() if "ff_context" in k and "proj" not in k and
                           "single" not in k],
          "s.attn": [k for k in weights_dict[lora_names[0]].keys() if "attn" in k and "single" in k],
          "s.ff": [k for k in weights_dict[lora_names[0]].keys() if "proj_mlp" in k and "single" in k],
          }

# calculate average similarity within each group for each pair of lora weights
group_sims = dict()
for i in range(len(lora_names)):
    for j in range(i + 1, len(lora_names)):
        group_sims[(lora_names[i], lora_names[j])] = dict()
        for group in groups:
            group_sims[(lora_names[i], lora_names[j])][group] = dict()
            for key in groups[group]:
                similarity = torch.nn.functional.cosine_similarity(weights_dict[lora_names[i]][key],
                                                                   weights_dict[lora_names[j]][key],
                                                                   dim=0).mean().item()
                group_sims[(lora_names[i], lora_names[j])][group][key] = similarity

# for each pair of lora weights, plot the average similarity within each group

for key in group_sims:
    # get the average similarity within each group
    group_avg_sims = {group: sum([group_sims[key][group][k] for k in groups[group]]) / len(groups[group]) for group in groups}
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.bar(group_avg_sims.keys(), group_avg_sims.values())
    ax.set_xlabel('Group')
    ax.set_ylabel('similarity')
    plt.title(f"Average similarity within each group between {key[0]} and {key[1]}")
    # save the plot
    plt.savefig(f"./lora_analysis/{key[0]}_{key[1]}_group_similarity-2.png")

