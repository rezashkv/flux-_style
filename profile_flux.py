import torch
from diffusers import FluxTransformer2DModel
from op_counter import count_ops_and_params


model = FluxTransformer2DModel.from_pretrained("black-forest-labs/FLUX.1-dev", subfolder="transformer")

example_inputs = {
    "hidden_states": torch.rand(1, 1024, 64),
    "timestep": torch.tensor([10]) / 1000,
    "guidance": torch.tensor([3.5]),
    "pooled_projections": torch.randn(1, 768),
    "encoder_hidden_states": torch.randn(1, 512, 4096),
    "txt_ids": torch.zeros(1, 512, 3),
    "img_ids": torch.zeros(1, 1024, 3),
}

transformer_macs, transformer_params = count_ops_and_params(model, example_inputs)
print("Number of Flux transformer parameters: {}", transformer_params/1e9, "G")
print("Number of Flux transformer MACs: {}", transformer_macs/1e12, "T")

hidden_states = model.x_embedder(example_inputs["hidden_states"])

timestep = example_inputs["timestep"].to(example_inputs["hidden_states"].dtype) * 1000
if example_inputs["guidance"] is not None:
    guidance = example_inputs["guidance"].to(example_inputs["hidden_states"].dtype) * 1000
else:
    guidance = None

temb = (
    model.time_text_embed(timestep, example_inputs["pooled_projections"])
    if example_inputs["guidance"] is None
    else model.time_text_embed(timestep,
                               guidance,
                               example_inputs["pooled_projections"])
)
encoder_hidden_states = model.context_embedder(example_inputs["encoder_hidden_states"])

txt_ids = example_inputs["txt_ids"].expand(example_inputs["img_ids"].size(0), -1, -1)
ids = torch.cat((txt_ids, example_inputs["img_ids"]), dim=1)
image_rotary_emb = model.pos_embed(ids)


block_inputs = {
    "hidden_states": hidden_states,
    "encoder_hidden_states": encoder_hidden_states,
    "temb": temb,
    "image_rotary_emb": image_rotary_emb,
}

block_macs, block_params = count_ops_and_params(model.transformer_blocks[17], block_inputs)
print("Number of FLux block parameters: {}", block_params/1e6, "M")
print("Flux block MACs: {}", block_macs/1e9, "G")

for name, module in model.transformer_blocks[17].named_modules():
    params = sum(p.numel() for p in module.parameters())
    print(f"{name} has {params} parameters")


hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
single_block_inputs = {
    "hidden_states": hidden_states,
    "temb": temb,
    "image_rotary_emb": image_rotary_emb,
}

single_block_macs, single_block_params = count_ops_and_params(model.single_transformer_blocks[0], single_block_inputs)
print("Number of single Flux block parameters: {}", single_block_params/1e6, "M")
print("Single Flux block MACs: {}", single_block_macs/1e9, "G")

for name, module in model.single_transformer_blocks[0].named_modules():
    params = sum(p.numel() for p in module.parameters())
    print(f"{name} has {params} parameters")


