import os
import argparse

import torch

from datasets import load_dataset
from diffusers import FluxPipeline
from accelerate import Accelerator
from accelerate.utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--lora_name_or_path", type=str, default="XLabs-AI/flux-lora-collection")
    parser.add_argument("--lora_scale", type=float, default=0.95)
    parser.add_argument("--torch_dtype", type=str, default="torch.bfloat16")
    parser.add_argument("--weight_name", type=str, default="disney_lora.safetensors")
    parser.add_argument("--adapter", type=str, default="disney")
    parser.add_argument("--trigger_phrase", type=str, default="disney style")
    parser.add_argument("--dataset", type=str, default="phiyodr/coco2017")
    parser.add_argument("--caption_column", type=str, default="captions")
    parser.add_argument("--id_column", type=str, default="image_id")
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num_inference_steps", type=int, default=40)
    parser.add_argument("--max_sequence_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def get_dataset(dataset_name, num_samples=10000, split="train", caption_column="captions", id_column="image_id",
                trigger_phrase="disney style"):
    dataset = load_dataset(dataset_name)[split]
    dataset = dataset.select_columns([caption_column, id_column])
    dataset = dataset.shuffle()
    dataset = dataset.select(range(num_samples))
    dataset = dataset.map(lambda x: {caption_column: max(x[caption_column], key=len)}, num_proc=4)
    dataset = dataset.map(lambda x: {caption_column: x[caption_column] + f", {trigger_phrase}"}, num_proc=4)
    return dataset


if __name__ == "__main__":
    args = parse_args()
    accelerator = Accelerator()
    if args.seed is not None:
        set_seed(args.seed)

    args.output_dir = f"{args.output_dir}/{args.model_name_or_path.split('/')[-1]}-{args.adapter}"
    os.makedirs(args.output_dir, exist_ok=True)
    captions = get_dataset(args.dataset, num_samples=args.num_samples, caption_column=args.caption_column,
                           id_column=args.id_column, trigger_phrase=args.trigger_phrase)
    dataloader = torch.utils.data.DataLoader(captions, batch_size=args.batch_size * accelerator.num_processes,
                                             shuffle=False)

    dataloader = accelerator.prepare(dataloader)
    pipe = FluxPipeline.from_pretrained(args.model_name_or_path,
                                        torch_dtype=torch.bfloat16 if args.torch_dtype == "torch.bfloat16" else torch.float32)
    pipe.load_lora_weights(args.lora_name_or_path, weight_name=args.weight_name, adapter_name=args.adapter)

    num_params = sum(p.numel() for p in pipe.transformer.parameters())
    print(f"Number of parameters in Transformer: {num_params}")

    pipe = pipe.to(accelerator.device)

    for batch in dataloader:
        if args.seed is None:
            generator = None
        else:
            generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

        images = pipe(batch[args.caption_column], num_inference_steps=args.num_inference_steps,
                      guidance_scale=args.guidance_scale, generator=generator,
                      height=args.height, width=args.width, max_sequence_length=args.max_sequence_length).images

        for i, image in enumerate(images):
            image.save(f"{args.output_dir}/{batch[args.id_column][i]}.png")
