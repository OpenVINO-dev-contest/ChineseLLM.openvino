from qwen.modeling import QwenModel
from chatglm2.modeling import ChatGLMModel
from qwen.modeling import QwenModel
import argparse
import time
import re

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

    ov_model = QwenModel(args.model_id, args.device)

    print(" --- start generating --- ")
    start = time.perf_counter()
    response, num_tokens = ov_model.generate_sequence(
        prompt=args.prompt, max_generated_tokens=args.max_sequence_length)
    end = time.perf_counter()
    answer = process_response(ov_model.tokenizer.decode(response))
    print(answer)
    print(f"Generated {num_tokens} tokens in {end - start:.3f} s")