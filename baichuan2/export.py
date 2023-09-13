import os
import openvino as ov
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

onnx_model = Path('onnx_model') / "baichuan2.onnx"
ir_model = Path('ir_model') / "baichuan2.xml"

from typing import List, Tuple

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('-h',
                    '--help',
                    action='help',
                    help='Show this help message and exit.')
parser.add_argument('-m',
                    '--model_id',
                    default='baichuan-inc/Baichuan2-7B-Chat',
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

query_history = [{"role": "user", "content": "你好"}, {"role": "assistant", "content": "你好👋!我是人工智能助手, 很高兴见到你,欢迎问我任何问题。"},{"role": "user", "content": "想要出国留学，应该怎么办？"}]

def flattenize_inputs(inputs):
    """
    Helper function for making nested inputs flattens
    """
    flatten_inputs = []
    for input_data in inputs:
        if input_data is None:
            continue
        if isinstance(input_data, (list, tuple)):
            flatten_inputs.extend(flattenize_inputs(input_data))
        else:
            flatten_inputs.append(input_data)
    return flatten_inputs

def build_inputs(device, model, tokenizer, messages: List[dict], max_new_tokens: int=0):
    def _parse_messages(messages, split_role="user"):
        system, rounds = "", []
        round = []
        for i, message in enumerate(messages):
            if message["role"] == "system":
                assert i == 0
                system = message["content"]
                continue
            if message["role"] == split_role and round:
                rounds.append(round)
                round = []
            round.append(message)
        if round:
            rounds.append(round)
        return system, rounds

    max_new_tokens = max_new_tokens or model.generation_config.max_new_tokens
    max_input_tokens = model.config.model_max_length - max_new_tokens
    system, rounds = _parse_messages(messages, split_role="user")
    system_tokens = tokenizer.encode(system)
    max_history_tokens = max_input_tokens - len(system_tokens)

    history_tokens = []
    for round in rounds[::-1]:
        round_tokens = []
        for message in round:
            if message["role"] == "user":
                round_tokens.append(model.generation_config.user_token_id)
            else:
                round_tokens.append(model.generation_config.assistant_token_id)
            round_tokens.extend(tokenizer.encode(message["content"]))
        if len(history_tokens) == 0 or len(history_tokens) + len(round_tokens) <= max_history_tokens:
            history_tokens = round_tokens + history_tokens  # concat left
            if len(history_tokens) < max_history_tokens:
                continue
        break

    input_tokens = system_tokens + history_tokens
    if messages[-1]["role"] != "assistant":
        input_tokens.append(model.generation_config.assistant_token_id)
    input_tokens = input_tokens[-max_input_tokens:]  # truncate left
    return torch.LongTensor([input_tokens]).to(device)

tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=False,
                                          trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(args.model_id, device_map="auto", trust_remote_code=True).eval()

# Specify hyperparameters for generation
model.generation_config = GenerationConfig.from_pretrained(args.model_id, trust_remote_code=True)

device = 'cpu'
# input_tensors
input_ids = build_inputs(device, model, tokenizer, query_history)
attention_mask = torch.ones(input_ids.shape[0], input_ids.shape[1]).to(device)
position_ids = torch.arange(0,input_ids.shape[1]).unsqueeze(0)
print("input_ids shape:", input_ids.shape, "; type:", input_ids.dtype)
print("position_ids shape:", position_ids.shape, "; type: ", input_ids.dtype)
print("attention_mask shape:", attention_mask.shape, "; type: ",
      attention_mask.dtype)
print(" --- forward first --- ")
outputs = model.forward(input_ids=input_ids,
                        position_ids=position_ids,
                         attention_mask=attention_mask
                         )

print("--- second forward ---")
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
input_names = ["input_ids", "attention_mask", 'position_ids']
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
            0: "batch_size",
            2: "past_sequence",
        },
        past_value_name: {
            0: "batch_size",
            2: "past_sequence",
        },
        present_key_name: {
            0: "batch_size",
            2: "past_sequence + 1"
        },
        present_value_name: {
            0: "batch_size",
            2: "past_sequence + 1"
        }
    })

if args.compress_weight == True:
    print("--- compress weight ---")
    from nncf import compress_weights
    model = compress_weights(model)
    
dummy_inputs = {"input_ids": input_ids, "attention_mask": attention_mask, "position_ids": position_ids, "past_key_values": past_key_values}
model.config.torchscript = True
ov_model = ov.convert_model(model, example_input=dummy_inputs)
for inp_name, m_input, input_data in zip(input_names, ov_model.inputs, flattenize_inputs(dummy_inputs.values())):
    input_node = m_input.get_node()
    if input_node.element_type == ov.Type.dynamic:
        m_input.get_node().set_element_type(ov.Type.f32)
    shape = list(input_data.shape)
    if inp_name in dynamic_axes:
        for k in dynamic_axes[inp_name]:
            shape[k] = -1
    input_node.set_partial_shape(ov.PartialShape(shape))
    m_input.get_tensor().set_names({inp_name})
    
for out, out_name in zip(ov_model.outputs, output_names):
    out.get_tensor().set_names({out_name})     

ov_model.validate_nodes_and_infer_types()
ov.save_model(ov_model, ir_model)