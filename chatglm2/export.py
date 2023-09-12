import os
from transformers import AutoTokenizer, AutoModel
import torch
from pathlib import Path
import argparse

model_path = Path('onnx_model')
if model_path.exists() == False:
    os.mkdir(model_path)

from typing import List, Tuple

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('-h',
                    '--help',
                    action='help',
                    help='Show this help message and exit.')
parser.add_argument('-m',
                    '--model_id',
                    required=True,
                    type=str,
                    help='Required. orignal model path')
parser.add_argument('-o',
                    '--onnx_path',
                    required=True,
                    type=str,
                    help='Required. onnx model path')
args = parser.parse_args()

query = "想要出国留学，应该怎么办？"
history = [(
    "你好",
    "你好👋!我是人工智能助手 ChatGLM2-6B,很高兴见到你,欢迎问我任何问题。",
)]


def build_inputs(device,
                 tokenizer,
                 query: str,
                 history: List[Tuple[str, str]] = None):
    prompt = ""
    for i, (old_query, response) in enumerate(history):
        prompt += "[Round {}]\n\n问：{}\n\n答：{}\n\n".format(
            i + 1, old_query, response)
    prompt += "[Round {}]\n\n问：{}\n\n答：".format(len(history) + 1, query)
    inputs = tokenizer([prompt], return_tensors="pt")
    inputs = inputs.to(device)
    return inputs


tokenizer = AutoTokenizer.from_pretrained(args.model_id,
                                          trust_remote_code=True)
model = AutoModel.from_pretrained(args.model_id,
                                  trust_remote_code=True).float()

device = 'cpu'
model.eval()
# input_tensors
input_tensors = build_inputs(device, tokenizer, query, history)

print(" --- forward first --- ")
outputs = model.forward(**input_tensors)

print("--- second forward ---")
attention_mask = input_tensors["attention_mask"]
position_ids = input_tensors["position_ids"]
past_key_values = outputs["past_key_values"]
# copy from forward in second time
input_ids = torch.tensor([[30910]]).to(device)

# copy from _update_model_kwargs_for_generation in modeling_chatglm.py
attention_mask = torch.cat(
    [attention_mask,
     attention_mask.new_ones((attention_mask.shape[0], 1))],
    dim=-1)
new_position_id = position_ids[..., -1:].clone()
new_position_id += 1
position_ids = torch.cat([position_ids, new_position_id], dim=-1)
# copy from prepare_inputs_for_generation in modeling_chatglm.py
position_ids = position_ids[..., -1:]
# print shape
print("input_ids shape:", input_ids.shape, "; type:", input_ids.dtype)
print("position_ids shape:", position_ids.shape, "; type: ", input_ids.dtype)
print("attention_mask shape:", attention_mask.shape, "; type: ",
      attention_mask.dtype)
print("one past_key_value shape: ", past_key_values[0][0].shape, "; type:",
      past_key_values[0][0].dtype)
print("logits shape: ", outputs["logits"].shape)
outputs2 = model.forward(input_ids=input_ids,
                         attention_mask=attention_mask,
                         position_ids=position_ids,
                         past_key_values=past_key_values)
print("--- export onnx ---")
# ---prepare for onnx export ---
input_names = ["input_ids", 'position_ids', "attention_mask"]
output_names = ["logits"]
dynamic_axes = {
    'input_ids': {
        0: "batch_size",
        1: "sequence"
    },
    'position_ids': {
        0: "batch_size",
        1: "sequence"
    },
    "attention_mask": {
        0: "batch_size",
        1: "past_sequence + sequence"
    },
    "logits": {
        0: "batch_size",
        1: "sequence"
    }
}
for layer_idx in range(model.config.num_layers):
    # --- input key and value ---
    past_key_name = f"past_key_values.{layer_idx}.key"
    past_value_name = f"past_key_values.{layer_idx}.value"
    input_names += [past_key_name, past_value_name]
    # --- output key and value ---
    present_key_name = f"present_key_values.{layer_idx}.key"
    present_value_name = f"present_key_values.{layer_idx}.value"
    output_names += [present_key_name, present_value_name]
    dynamic_axes.update({
        past_key_name: {
            0: "past_sequence",
            1: "batch_size",
        },
        past_value_name: {
            0: "past_sequence",
            1: "batch_size",
        },
        present_key_name: {
            0: "past_sequence + 1",
            1: "batch_size"
        },
        present_value_name: {
            0: "past_sequence + 1",
            1: "batch_size"
        }
    })

with torch.no_grad():
    torch.onnx.export(
        model,
        args=(input_ids, position_ids, attention_mask, past_key_values),
        f=args.onnx_path,
        opset_version=15,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )