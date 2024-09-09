import json
import os
from datasets import load_dataset

ids = [img[:-4] for img in os.listdir('./aquarelle')]
dataset = load_dataset("phiyodr/coco2017")['train'].select_columns(['image_id', 'captions'])
dataset = dataset.filter(lambda x: str(x['image_id']) in ids)
dataset = dataset.map(lambda x: {"captions": max(x['captions'], key=len)}, num_proc=4)
dataset = dataset.rename_column("captions", "caption")
roots = ["aquarelle", "frosting_lane", "half_illustration", "ps1", "tarot", "yarn"]
triggers = ["AQUACOLTOK style", "frstingln style", "half illustration style", "PS1 style", "trtcrd tarot style",
            "yarn style"]

with open("metadata.jsonl", "w") as f:
    for i, root in enumerate(roots):
        trigger = triggers[i]
        style_dataset = dataset.map(lambda x: {"caption": x['caption'] + f", {triggers[i]}"}, num_proc=4)
        for item in style_dataset:
            item["style"] = root
            item["file_name"] = f"./{root}/{item['image_id']}.png"
            del item["image_id"]
            f.write(json.dumps(item) + '\n')