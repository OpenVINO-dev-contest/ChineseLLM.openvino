import sys
import numpy as np
from transformers import AutoTokenizer
import openvino as ov
from pathlib import Path
from typing import List, Tuple
from copy import deepcopy
import time

utils_file_path = Path('.')
sys.path.append(str(utils_file_path))
from utils import sample_next_token

class ChatGLM2Model():

    def __init__(self,
                 model_path='./chatglm/chatglm2',
                 device='CPU') -> None:

        ir_model_path = Path(model_path)
        ir_model = ir_model_path / "openvino_model.xml"

        print(" --- loading tokenizer --- ")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True)
        core = ov.Core()

        print(" --- reading model --- ")
        # read the model and corresponding weights from file
        self.model = core.read_model(ir_model)
        # input & output names
        self.input_names = {
            key.get_any_name(): idx
            for idx, key in enumerate(self.model.inputs)
        }
        self.output_names = {
            key.get_any_name(): idx
            for idx, key in enumerate(self.model.outputs)
        }
        self.key_value_input_names = [
            key for key in self.input_names if "key_values" in key
        ]
        self.key_value_output_names = [
            key for key in self.output_names if "present" in key
        ]
        print(" --- model compiling --- ")
        # compile the model for CPU devices
        self.request = core.compile_model(
            model=self.model, device_name=device).create_infer_request()
        self.eos_token_id = [self.tokenizer.eos_token_id]

    def build_inputs(self,
                     history: List[Tuple[str, str]],
                     query: str,
                     system: str = "",
                     max_input_tokens: int = 2048):
        prompt = self.tokenizer.build_prompt(query, history=history)
        inputs  = self.tokenizer([prompt], return_tensors="np")
        input_tokens = inputs['input_ids'][:][-max_input_tokens:]
        return input_tokens

    def process_response(self, output, history):
        output = output.strip()
        output = output.replace("[[训练时间]]", "2023年")
        return output, [history, output]

    def build_memory(self, memory, query):
        return memory[0] + [(query, memory[1])]

    def generate_sequence(self,
                          input_ids,
                          max_generated_tokens=100,
                          top_k=20,
                          top_p=0.7,
                          temperature=1):
        attention_mask = np.ones((input_ids.shape[0], input_ids.shape[1]),
                                 dtype=np.int64)
        position_ids = np.arange(0, input_ids.shape[1], dtype=np.int64)
        position_ids = np.expand_dims(position_ids, axis=0)
        past_key_values = None
        num_iteration = 0
        other_latency = 0
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
                for input_name in self.key_value_input_names:
                    model_inputs = self.model.input(input_name)
                    shape = model_inputs.get_partial_shape()
                    if shape[0].is_dynamic:
                        shape[0] = 0
                    if shape[1].is_dynamic:
                        shape[1] = shape_input_ids[0]
                    inputs[input_name] = ov.Tensor(
                        model_inputs.get_element_type(), shape.get_shape())
            if "attention_mask" in self.input_names and attention_mask is not None:
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
            if next_token in self.eos_token_id or len(
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
        position_ids = np.arange(0, input_ids.shape[1], dtype=np.int64)
        position_ids = np.expand_dims(position_ids, axis=0)
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
                for input_name in self.key_value_input_names:
                    model_inputs = self.model.input(input_name)
                    shape = model_inputs.get_partial_shape()
                    if shape[0].is_dynamic:
                        shape[0] = 0
                    if shape[1].is_dynamic:
                        shape[1] = shape_input_ids[0]
                    inputs[input_name] = ov.Tensor(
                        model_inputs.get_element_type(), shape.get_shape())
            if "attention_mask" in self.input_names and attention_mask is not None:
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
            if next_token in self.eos_token_id or len(
                    output_tokens) > max_generated_tokens:
                break
            attention_mask = np.concatenate((attention_mask, [[1]]), axis=-1)
            input_ids = np.array([[next_token]], dtype=np.longlong)
            yield self.process_response(self.tokenizer.decode(output_tokens, skip_special_tokens=True), history)
        return self.process_response(self.tokenizer.decode(output_tokens, skip_special_tokens=True), history)


class ChatGLM3Model(ChatGLM2Model):

    def __init__(self,
                 model_path='./chatglm/chatglm3_model',
                 device='CPU') -> None:
        ChatGLM2Model.__init__(self, model_path, device)
        self.eos_token_id = [self.tokenizer.eos_token_id, self.tokenizer.get_command("<|user|>"),
                             self.tokenizer.get_command("<|observation|>")]

    def build_inputs(self,
                     history: List[Tuple[str, str]],
                     query: str,
                     system: str = "",
                     max_input_tokens: int = 2048):
        inputs = self.tokenizer.build_chat_input(
            query, history=history, role="user")
        input_tokens = inputs['input_ids'].numpy()
        input_tokens = input_tokens[:][-max_input_tokens:]
        return input_tokens

    def process_response(self, output, history):
        content = ""
        history = deepcopy(history)
        for response in output.split("<|assistant|>"):
            content = response.strip()
            history.append(
                {"role": "assistant", "metadata": '', "content": content})
            content = content.replace("[[训练时间]]", "2023年")
        return content, history

    def build_memory(self, memory, query):
        memory.append({"role": "user", "content": query})
        return memory
