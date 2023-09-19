import numpy as np
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


def sample_next_token(logits: np.ndarray, top_k=20, top_p=0.7, temperature=1):
    # softmax with temperature
    logits = logits - np.max(logits, axis=-1, keepdims=True)
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