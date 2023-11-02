import sys
import numpy as np
from transformers import AutoTokenizer
from openvino.runtime import Core, Tensor
from pathlib import Path
from typing import List, Tuple

utils_file_path = Path('.')
sys.path.append(str(utils_file_path))
from utils import process_response, sample_next_token


class QwenModel():

    def __init__(self,
                 model_path='./qwen/ir_model',
                 device='CPU') -> None:
        
        ir_model_path = Path(model_path)
        ir_model = ir_model_path / "qwen.xml"
        
        print(" --- loading tokenizer --- ")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
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
        self.im_end_id = self.tokenizer.im_end_id

    def build_inputs(
        self,
        history: List[Tuple[str, str]],
        query: str,
        system: str = "",
        max_input_tokens: int = 6144,
        chat_format: str = "chatml",
    ):
        if history is None:
            history = []
        if chat_format == "chatml":
            im_start, im_end = "<|im_start|>", "<|im_end|>"
            
            def _to_str(role, content):
                return f"{role}\n{content}"
            system_text = _to_str("system", system)
            raw_text = ""
            for turn_query, turn_response in reversed(history):
                query_text = _to_str("user", turn_query)
                response_text = _to_str("assistant", turn_response)
                prev_chat = (
                    f"\n{im_start}{query_text}{im_end}\n{im_start}{response_text}{im_end}"
                )
                if len(raw_text) < max_input_tokens:
                    raw_text = prev_chat + raw_text
                else:
                    break
            raw_text = f"{im_start}{system_text}{im_end}" + raw_text
            raw_text += f"\n{im_start}user\n{query}{im_end}\n{im_start}assistant\n"
        elif chat_format == "raw":
            raw_text = query
        else:
            raise NotImplementedError(f"Unknown chat format {chat_format!r}")
        tokens = self.tokenizer([raw_text], return_tensors="np")
        input_tokens = tokens['input_ids']
        return input_tokens


    def generate_sequence(self,
                          input_ids,
                          max_generated_tokens=100,
                          top_k=20,
                          top_p=0.8,
                          temperature=1):
        attention_mask = np.ones((input_ids.shape[0], input_ids.shape[1]),
                            dtype=np.int64)
        past_key_values = None
        num_iteration = 0
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
                    if shape[1].is_dynamic:
                        shape[1] = 0
                    inputs[input_name] = Tensor(
                        model_inputs.get_element_type(), shape.get_shape())
            
            if attention_mask is not None:
                inputs["attention_mask"] = attention_mask
            self.request.start_async(inputs, share_inputs=True)
            self.request.wait()
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

            if next_token == self.im_end_id or len(
                    output_tokens) > max_generated_tokens:
                break
            output_tokens += [next_token]
            attention_mask = np.concatenate((attention_mask, [[1]]), axis=-1)
            input_ids = np.array([[next_token]], dtype=np.longlong)
        return output_tokens, num_iteration

    def generate_iterate(self,
                         input_ids,
                         max_generated_tokens,
                         top_k=20,
                         top_p=0.7,
                         temperature=1):
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
                    if shape[1].is_dynamic:
                        shape[1] = 0
                    inputs[input_name] = Tensor(
                        model_inputs.get_element_type(), shape.get_shape())
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

            if next_token == self.im_end_id or len(
                    output_tokens) > max_generated_tokens:
                break
            input_ids = np.array([[next_token]], dtype=np.longlong)

            yield process_response(self.tokenizer.decode(output_tokens))
        return process_response(self.tokenizer.decode(output_tokens))