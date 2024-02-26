from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import sys
import openvino as ov
import torch
import types
from typing import Tuple, Optional
from pathlib import Path
import argparse
from transformers.modeling_outputs import BaseModelOutputWithPast

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
        context_layer = torch.nn.functional.scaled_dot_product_attention(
            query_layer, key_layer, value_layer, attention_mask
        )
    context_layer = context_layer.permute(2, 0, 1, 3)
    new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size_per_partition,)
    context_layer = context_layer.reshape(*new_context_layer_shape)

    return context_layer


def _chatglm_transformer_forward(
        self,
        input_ids,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.BoolTensor] = None,
        full_attention_mask: Optional[torch.BoolTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None
):
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    batch_size, seq_length = input_ids.shape

    if inputs_embeds is None:
        inputs_embeds = self.embedding(input_ids)

    if self.pre_seq_len is not None:
        if past_key_values is None:
            past_key_values = self.get_prompt(batch_size=batch_size, device=input_ids.device,
                                              dtype=inputs_embeds.dtype)
        if attention_mask is not None:
            attention_mask = torch.cat([attention_mask.new_ones((batch_size, self.pre_seq_len)), attention_mask], dim=-1)

    if full_attention_mask is None:
        if (attention_mask is not None and not attention_mask.all()) or (past_key_values and seq_length != 1):
            full_attention_mask = self.get_masks(input_ids, past_key_values, padding_mask=attention_mask)
        elif past_key_values is not None:
            full_attention_mask = torch.ones(batch_size, seq_length, seq_length,
                                             device=input_ids.device,
                                             dtype=torch.float) * float("-inf")
            full_attention_mask.triu_(diagonal=1)
            past_length = 0
            if past_key_values:
                past_length = past_key_values[0][0].shape[0]
            if past_length:
                full_attention_mask = torch.cat((torch.zeros(batch_size, seq_length, past_length,
                                                             device=input_ids.device), full_attention_mask), dim=-1)
            full_attention_mask.unsqueeze_(1)

    # Rotary positional embeddings
    rotary_pos_emb = self.rotary_pos_emb(self.seq_length)
    if position_ids is not None:
        rotary_pos_emb = rotary_pos_emb[position_ids]
    else:
        rotary_pos_emb = rotary_pos_emb[None, :seq_length]
    rotary_pos_emb = rotary_pos_emb.transpose(0, 1).contiguous()

    # Run encoder.
    hidden_states, presents, all_hidden_states, all_self_attentions = self.encoder(
        inputs_embeds, full_attention_mask, rotary_pos_emb=rotary_pos_emb,
        kv_caches=past_key_values, use_cache=use_cache, output_hidden_states=output_hidden_states
    )

    if not return_dict:
        return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)

    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=presents,
        hidden_states=all_hidden_states,
        attentions=all_self_attentions,
    )


def _patch_chatglm_forward(model: "PreTrainedModel"):
    model.transformer.forward = types.MethodType(_chatglm_transformer_forward, model.transformer)
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
_patch_chatglm_forward(model)
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
