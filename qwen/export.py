import os
import openvino as ov
from openvino.runtime import serialize
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
from pathlib import Path
import argparse

onnx_model_path = Path('onnx_model')
ir_model_path = Path('ir_model')
if onnx_model_path.exists() == False:
    os.mkdir(onnx_model_path)
if ir_model_path.exists() == False:
    os.mkdir(ir_model_path)

onnx_model = Path('onnx_model') / "qwen.onnx"
ir_model = Path('ir_model') / "qwen.xml"

from typing import List, Tuple

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('-h',
                    '--help',
                    action='help',
                    help='Show this help message and exit.')
parser.add_argument('-m',
                    '--model_id',
                    default='Qwen/Qwen-7B-Chat',
                    required=False,
                    type=str,
                    help='orignal model path')
parser.add_argument('-cw',
                    '--compress_weight',
                    default=False,
                    required=False,
                    type=bool,
                    help='Weights Compression')
args = parser.parse_args()

query = "想要出国留学，应该怎么办？"
history = [(
    "你好",
    "你好👋!我是人工智能助手，很高兴见到你,欢迎问我任何问题。",
)]


def build_inputs(
    tokenizer,
    query: str,
    history: List[Tuple[str, str]] = None,
    system: str = "",
    max_window_size: int = 6144,
    chat_format: str = "chatml",
):
    if history is None:
        history = []

    if chat_format == "chatml":
        im_start, im_end = "<|im_start|>", "<|im_end|>"
        im_start_tokens = [tokenizer.im_start_id]
        im_end_tokens = [tokenizer.im_end_id]
        nl_tokens = tokenizer.encode("\n")

        def _tokenize_str(role, content):
            return f"{role}\n{content}", tokenizer.encode(
                role, allowed_special=set()
            ) + nl_tokens + tokenizer.encode(content, allowed_special=set())

        system_text, system_tokens_part = _tokenize_str("system", system)
        system_tokens = im_start_tokens + system_tokens_part + im_end_tokens

        raw_text = ""
        context_tokens = []

        for turn_query, turn_response in reversed(history):
            query_text, query_tokens_part = _tokenize_str("user", turn_query)
            query_tokens = im_start_tokens + query_tokens_part + im_end_tokens
            response_text, response_tokens_part = _tokenize_str(
                "assistant", turn_response
            )
            response_tokens = im_start_tokens + response_tokens_part + im_end_tokens

            next_context_tokens = nl_tokens + query_tokens + nl_tokens + response_tokens
            prev_chat = (
                f"\n{im_start}{query_text}{im_end}\n{im_start}{response_text}{im_end}"
            )

            current_context_size = (
                len(system_tokens) + len(next_context_tokens) + len(context_tokens)
            )
            if current_context_size < max_window_size:
                context_tokens = next_context_tokens + context_tokens
                raw_text = prev_chat + raw_text
            else:
                break

        context_tokens = system_tokens + context_tokens
        raw_text = f"{im_start}{system_text}{im_end}" + raw_text
        context_tokens += (
            nl_tokens
            + im_start_tokens
            + _tokenize_str("user", query)[1]
            + im_end_tokens
            + nl_tokens
            + im_start_tokens
            + tokenizer.encode("assistant")
            + nl_tokens
        )
        raw_text += f"\n{im_start}user\n{query}{im_end}\n{im_start}assistant\n"

    elif chat_format == "raw":
        raw_text = query
        context_tokens = tokenizer.encode(raw_text)
    else:
        raise NotImplementedError(f"Unknown chat format {chat_format!r}")

    return raw_text, context_tokens


tokenizer = AutoTokenizer.from_pretrained(args.model_id,
                                          trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(args.model_id, device_map="auto", trust_remote_code=True).eval()

# Specify hyperparameters for generation
model.generation_config = GenerationConfig.from_pretrained(args.model_id, trust_remote_code=True)
device = 'cpu'
# input_tensors
text, tensors = build_inputs(tokenizer, query, history)
input_tensors = tokenizer([text], return_tensors="pt")
input_tensors = input_tensors.to(device)

print(" --- forward first --- ")
outputs = model.forward(**input_tensors)

print("--- second forward ---")
attention_mask = input_tensors["attention_mask"]
position_ids = input_tensors["attention_mask"]
past_key_values = outputs["past_key_values"]
# copy from forward in second time
input_ids = torch.tensor([[30910]]).to(device)
token_type_ids = torch.tensor([[0]]).to(device)
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
print("token_type_ids shape:", token_type_ids.shape, "; type:", token_type_ids.dtype)
print("position_ids shape:", position_ids.shape, "; type: ", input_ids.dtype)
print("attention_mask shape:", attention_mask.shape, "; type: ",
      attention_mask.dtype)
print("one past_key_value shape: ", past_key_values[0][0].shape, "; type:",
      past_key_values[0][0].dtype)
print("logits shape: ", outputs["logits"].shape)
outputs2 = model.forward(input_ids=input_ids,
                         token_type_ids=token_type_ids,
                         attention_mask=attention_mask,
                         position_ids=position_ids,
                         past_key_values=past_key_values)
# ---prepare for onnx export ---
input_names = ["input_ids"]
output_names = ["logits"]
dynamic_axes = {
    'input_ids': {
        0: "batch_size",
        1: "sequence"
    },
    'token_type_ids': {
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
for layer_idx in range(model.config.num_hidden_layers):
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
input_names += ["attention_mask", 'position_ids', 'token_type_ids']

if args.compress_weight == True:
    print("--- compress weight ---")
    from nncf import compress_weights
    model = compress_weights(model)
    

with torch.no_grad():
    torch.onnx.export(
        model,
        args=(input_ids, past_key_values, attention_mask, position_ids, token_type_ids),
        f="onnx_model/qwen.onnx",
        opset_version=15,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=False,
    )
if args.compress_weight == True:
    ov_model = ov.convert_model(onnx_model)
else:
    ov_model = ov.convert_model(onnx_model, compress_to_fp16=True)
serialize(ov_model, str(ir_model))