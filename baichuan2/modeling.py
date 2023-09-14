import sys
import numpy as np
from transformers import AutoTokenizer
from openvino.runtime import Core, Tensor
from pathlib import Path

utils_file_path = Path('../utils.py')
sys.path.append(str(utils_file_path))
from utils import process_response, sample_next_token

# input & output names
past_names = [
    f"past_key_values.{i}.{name}" for i in range(32)
    for name in ["key", "value"]
]
present_names = [
    f"present.{i}.{name}" for i in range(32)
    for name in ["key", "value"]
]


class BaichuanModel():

    def __init__(self,
                 tokenizer_path,
                 device='CPU',
                 model_path='/home/ethan/intel/ChineseLLM.openvino/baichuan2/ir_model/baichuan2.xml') -> None:
        self.tokenizer =  AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False,
                                          trust_remote_code=True)
        core = Core()

        print(" --- reading model --- ")
        # read the model and corresponding weights from file
        self.model = core.read_model(model_path)

        print(" --- model compiling --- ")
        # compile the model for CPU devices
        self.request = core.compile_model(
            model=self.model, device_name=device).create_infer_request()
        self.eos_token_id = self.tokenizer.eos_token_id

    def generate_sequence(self,
                          prompt: str,
                          max_generated_tokens=100,
                          top_k=50,
                          top_p=0.85,
                          temperature=1):
        tokens = self.tokenizer([prompt], return_tensors="np")
        input_ids = tokens['input_ids']
        input_ids = np.concatenate(
                ([[195]], input_ids, [[196]]), axis=-1)
        attention_mask = np.ones((input_ids.shape[0], input_ids.shape[1]), dtype=np.int64)
        past_key_values = None
        num_iteration = 0
        output_tokens = []
        while True:
            inputs = {"input_ids": input_ids}
            if past_key_values is not None:
                inputs.update(past_key_values)
            else:
                shape_input_ids = input_ids.shape
                for input_name in past_names:
                    model_inputs = self.model.input(input_name)
                    shape = model_inputs.get_partial_shape()
                    if shape[0].is_dynamic:
                        shape[0] = shape_input_ids[0]
                    if shape[2].is_dynamic:
                        shape[2] = 0
                    inputs[input_name] = Tensor(
                        model_inputs.get_element_type(), shape.get_shape())
            if attention_mask is not None:
                inputs["attention_mask"] = attention_mask
            print(attention_mask)
            self.request.start_async(inputs, share_inputs=True)
            self.request.wait()
            num_iteration += 1
            logits = self.request.get_tensor("logits").data
            past_key_values = tuple(
                self.request.get_tensor(key).data for key in present_names)
            past_key_values = {
                k: v
                for k, v in zip(past_names, past_key_values)
            }
            print(logits[0, -1].shape)
            next_token = sample_next_token(logits[0, -1],
                                           top_k=top_k,
                                           top_p=top_p,
                                           temperature=temperature)
            print(next_token)
            output_tokens += [next_token]

            if next_token == self.eos_token_id or len(
                    output_tokens) > max_generated_tokens:
                break
            attention_mask = np.concatenate(
                (attention_mask, [[1]]), axis=-1)
            input_ids = np.array([[next_token]], dtype=np.longlong)
        return output_tokens, num_iteration

    def generate_iterate(self,
                         prompt: str,
                         max_generated_tokens,
                         top_k=20,
                         top_p=0.7,
                         temperature=1):
        tokens = self.tokenizer([prompt], return_tensors="np")
        input_ids = tokens['input_ids']
        position_ids = tokens['position_ids']
        past_key_values = None
        output_tokens = []
        new_position_id = np.copy(position_ids[..., -1:])
        while True:
            inputs = {"input_ids": input_ids}
            if past_key_values is not None:
                new_position_id += 1
                inputs["position_ids"] = new_position_id
                inputs.update(past_key_values)
            else:
                inputs["position_ids"] = position_ids
                shape_input_ids = input_ids.shape
                for input_name in past_names:
                    model_inputs = self.model.input(input_name)
                    shape = model_inputs.get_partial_shape()
                    if shape[0].is_dynamic:
                        shape[0] = 0
                    if shape[1].is_dynamic:
                        shape[1] = shape_input_ids[0]
                    inputs[input_name] = Tensor(
                        model_inputs.get_element_type(), shape.get_shape())

            self.request.start_async(inputs, shared_memory=True)
            self.request.wait()
            logits = self.request.get_tensor("logits").data
            past_key_values = tuple(
                self.request.get_tensor(key).data for key in present_names)
            past_key_values = {
                k: v
                for k, v in zip(past_names, past_key_values)
            }
            next_token = self.sample_next_token(logits[0, -1],
                                                top_k=top_k,
                                                top_p=top_p,
                                                temperature=temperature)

            output_tokens += [next_token]

            if next_token == self.eos_token_id or len(
                    output_tokens) > max_generated_tokens:
                break

            input_ids = np.array([[next_token]], dtype=np.longlong)

            yield process_response(self.tokenizer.decode(output_tokens))
        return process_response(self.tokenizer.decode(output_tokens))
