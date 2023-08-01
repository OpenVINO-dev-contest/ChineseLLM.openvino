from model import ChatGLMModel, process_response
import argparse
import time

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

    ov_chatglm = ChatGLMModel(args.model_id, args.device)

    print(" --- start generating --- ")
    start = time.perf_counter()
    response, num_tokens = ov_chatglm.generate_sequence(
        prompt=args.prompt, max_generated_tokens=args.max_sequence_length)
    end = time.perf_counter()
    answer = process_response(ov_chatglm.tokenizer.decode(response))
    print(answer)
    print(f"Generated {num_tokens} tokens in {end - start:.3f} s")