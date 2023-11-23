from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import sys
import openvino as ov
import torch
import types
from pathlib import Path
import argparse

utils_file_path = Path('.')
sys.path.append(str(utils_file_path))
from utils import flattenize_inputs

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('-h',
                    '--help',
                    action='help',
                    help='Show this help message and exit.')
parser.add_argument('-m',
                    '--model_id',
                    default='THUDM/chatglm3-6b',
                    required=False,
                    type=str,
                    help='orignal model path')
args = parser.parse_args()


@torch.jit.script_if_tracing
def _chatglm2_get_context_layer(query_layer: torch.Tensor, key_layer: torch.Tensor, value_layer: torch.Tensor):
    mask = torch.zeros((query_layer.shape[-2], key_layer.shape[-2]), dtype=query_layer.dtype)
    if query_layer.shape[2] == key_layer.shape[2]:
        tmp_mask = torch.ones((query_layer.shape[-2], key_layer.shape[-2]), dtype=torch.bool).triu(diagonal=1)
        mask.masked_fill_(tmp_mask, float("-inf"))

    context_layer = torch.nn.functional.scaled_dot_product_attention(query_layer, key_layer, value_layer, attn_mask=mask)
    return context_layer


def _core_attention_forward(self, query_layer, key_layer, value_layer, attention_mask):
    query_layer, key_layer, value_layer = [k.permute(1, 2, 0, 3) for k in [query_layer, key_layer, value_layer]]
    if attention_mask is None:
        context_layer = _chatglm2_get_context_layer(query_layer, key_layer, value_layer)
    else:
        attention_mask = ~attention_mask
        context_layer = torch.nn.functional.scaled_dot_product_attention(
            query_layer, key_layer, value_layer, attention_mask
        )
    context_layer = context_layer.permute(2, 0, 1, 3)
    new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size_per_partition,)
    context_layer = context_layer.reshape(*new_context_layer_shape)

    return context_layer


def _patch_chatglm_core_attention_forward(model: "PreTrainedModel"):
    for block in model.transformer.encoder.layers:
        block.self_attention.core_attention.forward = types.MethodType(
            _core_attention_forward, block.self_attention.core_attention
        )


if 'chatglm2' in args.model_id:
    ir_model_path = Path('chatglm') / Path('chatglm2')
    if ir_model_path.exists() == False:
        os.mkdir(ir_model_path)
elif 'chatglm3' in args.model_id:
    ir_model_path = Path('chatglm') / Path('chatglm3')
    if ir_model_path.exists() == False:
        os.mkdir(ir_model_path)
else:
    raise NotImplementedError(f"Unsupported model id {args.model_id!r}")

ir_model = ir_model_path / "openvino_model.xml"
model = AutoModelForCausalLM.from_pretrained(args.model_id,                    
                                            torch_dtype=torch.float32,
                                            trust_remote_code=True,)
# _patch_chatglm_core_attention_forward(model)
model.config.save_pretrained(ir_model_path)
model.config.use_cache = True

outs = model(input_ids=torch.ones((1, 10), dtype=torch.long),
             position_ids=torch.arange(0, 10, dtype=torch.long))
inputs = ["input_ids"]
outputs = ["logits"]

dynamic_shapes = {"input_ids": {1: "seq_len"}, "position_ids": {1: "seq_len"}, "attention_mask": {1: "seq_len"}}
inputs += ["position_ids", "attention_mask"]
for idx in range(len(outs.past_key_values)):
    inputs.extend(
        [f"past_key_values.{idx}.key", f"past_key_values.{idx}.value"])
    dynamic_shapes[inputs[-1]] = {0: "past_sequence + 1"}
    dynamic_shapes[inputs[-2]] = {0: "past_sequence + 1"}
    outputs.extend([f"present.{idx}.key", f"present.{idx}.value"])

dummy_inputs = {
    "input_ids": torch.ones((1, 1), dtype=torch.long),
    "position_ids": torch.tensor([[10]], dtype=torch.long),
    "attention_mask": torch.ones((1, 11), dtype=torch.long),
    "past_key_values": outs.past_key_values
}

model.config.torchscript = True

print("====Exporting IR=====")
ov_model = ov.convert_model(model, example_input=dummy_inputs)
for inp_name, m_input, input_data in zip(
        inputs, ov_model.inputs, flattenize_inputs(dummy_inputs.values())):
    input_node = m_input.get_node()
    if input_node.element_type == ov.Type.dynamic:
        m_input.get_node().set_element_type(ov.Type.f32)
    shape = list(input_data.shape)
    if inp_name in dynamic_shapes:
        for k in dynamic_shapes[inp_name]:
            shape[k] = -1
    input_node.set_partial_shape(ov.PartialShape(shape))
    m_input.get_tensor().set_names({inp_name})

for out, out_name in zip(ov_model.outputs, outputs):
    out.get_tensor().set_names({out_name})

ov_model.validate_nodes_and_infer_types()
ov.save_model(ov_model, ir_model)

print("====Exporting tokenizer=====")
tokenizer = AutoTokenizer.from_pretrained(
    args.model_id, trust_remote_code=True)
tokenizer.save_pretrained(ir_model_path)
