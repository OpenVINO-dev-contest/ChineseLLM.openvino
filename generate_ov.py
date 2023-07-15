import re
import numpy as np
from transformers import AutoTokenizer
from openvino.runtime import Core
import numpy as np
import argparse
import time

# input & output names
past_names = [
    f"past_key_values.{i}.{name}" for i in range(28)
    for name in ["key", "value"]
]
present_names = [
    f"present_key_values.{i}.{name}" for i in range(28)
    for name in ["key", "value"]
]
output_names = ["logits"] + present_names
default_past_key_values = {
    k: np.zeros((1, 1, 2, 128), dtype=np.float32)
    for k in past_names
}

# default kv_cache for first inference
default_past_key_values = {
    k: np.zeros((1, 1, 2, 128), dtype=np.float32)
    for k in past_names
}


def chat_template(history: list[tuple[str, str]], current: str):
    prompt = ""
    chat_round = 0
    for question, answer in history:
        prompt += f"[Round {chat_round}]\n问：{question}\n答：{answer}\n"
        chat_round += 1
    prompt += f"[Round {chat_round}]\n问：{current}\n答："
    return prompt


def process_response(response: str):
    response = response.strip()
    response = response.replace("[[训练时间]]", "2023年")
    punkts = [
        [",", "，"],
        ["!", "！"],
        [":", "："],
        [";", "；"],
        ["\?", "？"],
    ]
    for item in punkts:
        response = re.sub(r"([\u4e00-\u9fff])%s" % item[0], r"\1%s" % item[1],
                          response)
        response = re.sub(r"%s([\u4e00-\u9fff])" % item[0], r"%s\1" % item[1],
                          response)
    return response


class ChatGLMModel():

    def __init__(self,
                 tokenizer_path,
                 onnx_model_path='./onnx_model/chatglm.onnx') -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path,
                                                       trust_remote_code=True)
        core = Core()

        print(" --- reading model --- ")
        # read the model and corresponding weights from file
        model = core.read_model(onnx_model_path)

        print(" --- model compiling --- ")
        # compile the model for CPU devices
        self.request = core.compile_model(
            model=model, device_name=args.device).create_infer_request()
        self.eos_token_id = self.tokenizer.eos_token_id

    def sample_next_token(self,
                          logits: np.ndarray,
                          top_k=50,
                          top_p=0.7,
                          temperature=1):
        # softmax with temperature
        exp_logits = np.exp(logits / temperature)
        probs = exp_logits / np.sum(exp_logits)

        # top k
        top_k_idx = np.argsort(-probs)[:top_k]
        top_k_probs = probs[top_k_idx]

        # top p
        cumsum_probs = np.cumsum(top_k_probs)
        top_k_probs[(cumsum_probs - top_k_probs) > top_p] = 0.0
        top_k_probs = top_k_probs / np.sum(top_k_probs)

        # sample
        next_token = np.random.choice(top_k_idx, size=1, p=top_k_probs)
        return next_token[0].item()

    def generate_iterate(self,
                         prompt: str,
                         max_generated_tokens,
                         top_k=20,
                         top_p=0.7,
                         temperature=1):
        inputs = self.tokenizer([prompt], return_tensors="np")
        input_ids = inputs['input_ids']
        # attention_mask = inputs['attention_mask']
        position_ids = inputs['position_ids']
        past_key_values = default_past_key_values
        output_tokens = []
        while True:
            inputs = {
                "input_ids": input_ids,
                "position_ids": position_ids,
            }
            new_position_id = position_ids[..., -1:]
            new_position_id += 1
            position_ids = np.concatenate((position_ids, new_position_id),
                                          axis=-1)

            inputs.update(past_key_values)

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
            # attention_mask = np.concatenate([attention_mask, np.array([[0]], dtype=np.longlong)], axis=1)

            yield process_response(self.tokenizer.decode(output_tokens))
        return process_response(self.tokenizer.decode(output_tokens))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-h',
                        '--help',
                        action='help',
                        help='Show this help message and exit.')
    parser.add_argument('-m',
                        '--model_id',
                        required=True,
                        type=str,
                        help='Required. huggingface model id')
    parser.add_argument('-p',
                        '--prompt',
                        required=True,
                        type=str,
                        help='Required. prompt sentence')
    parser.add_argument('-l',
                        '--max_sequence_length',
                        default=128,
                        required=False,
                        type=int,
                        help='Required. maximun lengh of output')
    parser.add_argument('-d',
                        '--device',
                        default='CPU',
                        required=False,
                        type=str,
                        help='Required. device for inference')
    args = parser.parse_args()

    ov_chatglm = ChatGLMModel(args.model_id)

    print(" --- start generation --- ")
    start = time.perf_counter()
    for text in ov_chatglm.generate_iterate(args.prompt,
                                            args.max_sequence_length):
        print(text)
    end = time.perf_counter()
    print(f"Generation took {end - start:.3f} s")