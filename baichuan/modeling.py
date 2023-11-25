import sys
import numpy as np
import time
from transformers import AutoTokenizer
from openvino.runtime import Core, Tensor
from pathlib import Path
from typing import List, Tuple

utils_file_path = Path('.')
sys.path.append(str(utils_file_path))
from utils import sample_next_token

class Baichuan2Model():

    def __init__(self,
                 model_path='./baichuan/baichuan2',
                 device='CPU') -> None:

        ir_model_path = Path(model_path)
        ir_model = ir_model_path / "openvino_model.xml"

        print(" --- loading tokenizer --- ")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True)
        core = Core()

        print(" --- reading model --- ")
        # read the model and corresponding weights from file
        self.model = core.read_model(ir_model)
        # input & output names
        input_names = {
            key.get_any_name(): idx
            for idx, key in enumerate(self.model.inputs)
        }
        output_names = {
            key.get_any_name(): idx
            for idx, key in enumerate(self.model.outputs)
        }
        self.key_value_input_names = [
            key for key in input_names if "key_values" in key
        ]
        self.key_value_output_names = [
            key for key in output_names if "present" in key
        ]
        print(" --- model compiling --- ")
        # compile the model for CPU devices
        self.request = core.compile_model(
            model=self.model, device_name=device).create_infer_request()
        self.eos_token_id = self.tokenizer.eos_token_id

    def build_inputs(self,
                     history: List[Tuple[str, str]],
                     query: str,
                     system: str = "",
                     max_input_tokens: int = 2048):
        input_tokens = self.tokenizer([system],
                                      return_tensors="np")['input_ids']
        for (old_query, response) in history:
            old_query_token = self.tokenizer([old_query],
                                             return_tensors="np")['input_ids']
            response_token = self.tokenizer([response],
                                            return_tensors="np")['input_ids']
            input_tokens = np.concatenate(
                (input_tokens, [[195]], old_query_token,
                 [[196]], response_token),
                axis=-1)
        query_token = self.tokenizer([query], return_tensors="np")['input_ids']
        input_tokens = np.concatenate(
            (input_tokens, [[195]], query_token, [[196]]), axis=-1)
        input_tokens = input_tokens[:][-max_input_tokens:]
        return input_tokens

    def process_response(self, output, history):
        return output, [history, output]

    def build_memory(self, memory, query):
        return memory[0] + [(query, memory[1])]

    def generate_sequence(self,
                          input_ids,
                          max_generated_tokens=100,
                          top_k=50,
                          top_p=0.85,
                          temperature=1):
        attention_mask = np.ones((input_ids.shape[0], input_ids.shape[1]),
                                 dtype=np.int64)
        past_key_values = None
        num_iteration = 0
        other_latency = 0
        output_tokens = []
        while True:
            inputs = {"input_ids": input_ids}
            if past_key_values is not None:
                inputs.update(past_key_values)
            else:
                shape_input_ids = input_ids.shape
                for input_name in self.key_value_input_names:
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
            before = time.perf_counter()
            self.request.start_async(inputs, share_inputs=True)
            self.request.wait()
            after = time.perf_counter()
            if num_iteration == 0:
                first_latency = after - before
            else:
                other_latency += after - before
            num_iteration += 1
            logits = self.request.get_tensor("logits").data
            past_key_values = tuple(
                self.request.get_tensor(key).data
                for key in self.key_value_output_names)
            past_key_values = {
                k: v
                for k, v in zip(self.key_value_input_names, past_key_values)
            }
            next_token = sample_next_token(logits[0, -1],
                                           top_k=top_k,
                                           top_p=top_p,
                                           temperature=temperature)
            output_tokens += [next_token]

            if next_token == self.eos_token_id or len(
                    output_tokens) > max_generated_tokens:
                break
            attention_mask = np.concatenate((attention_mask, [[1]]), axis=-1)
            input_ids = np.array([[next_token]], dtype=np.longlong)
        return output_tokens, num_iteration, (first_latency, other_latency)

    def generate_iterate(self,
                         input_ids,
                         history,
                         max_generated_tokens,
                         top_k=20,
                         top_p=0.7,
                         temperature=1):
        attention_mask = np.ones((input_ids.shape[0], input_ids.shape[1]),
                                 dtype=np.int64)
        past_key_values = None
        output_tokens = []
        while True:
            inputs = {"input_ids": input_ids}
            if past_key_values is not None:
                inputs.update(past_key_values)
            else:
                shape_input_ids = input_ids.shape
                for input_name in self.key_value_input_names:
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
            self.request.start_async(inputs, share_inputs=True)
            self.request.wait()
            logits = self.request.get_tensor("logits").data
            past_key_values = tuple(
                self.request.get_tensor(key).data
                for key in self.key_value_output_names)
            past_key_values = {
                k: v
                for k, v in zip(self.key_value_input_names, past_key_values)
            }
            next_token = sample_next_token(logits[0, -1],
                                           top_k=top_k,
                                           top_p=top_p,
                                           temperature=temperature)
            output_tokens += [next_token]
            if next_token == self.eos_token_id or len(
                    output_tokens) > max_generated_tokens:
                break
            attention_mask = np.concatenate((attention_mask, [[1]]), axis=-1)
            input_ids = np.array([[next_token]], dtype=np.longlong)

            yield self.process_response(self.tokenizer.decode(output_tokens), history)
        return self.process_response(self.tokenizer.decode(output_tokens), history)
