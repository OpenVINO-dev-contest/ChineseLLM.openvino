import re
import numpy as np
from transformers import AutoTokenizer
from openvino.runtime import Core
import numpy as np
import argparse
import streamlit as st
from streamlit_chat import message

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


def chat_template(history: list[tuple[str, str]], query: str):
    if history == []:
        prompt = ""
        for i, (old_query, response) in enumerate(history):
            prompt += "[Round {}]\n问：{}\n答：{}\n".format(
                i + 1, old_query, response)
        prompt += "[Round {}]\n问：{}\n答：".format(len(history) + 1, query)
    else:
        prompt = query
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
                 model_path='./ir_model/chatglm2.xml') -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path,
                                                       trust_remote_code=True)
        core = Core()

        print(" --- reading model --- ")
        # read the model and corresponding weights from file
        model = core.read_model(model_path)

        print(" --- model compiling --- ")
        # compile the model for CPU devices
        self.request = core.compile_model(
            model=model, device_name='CPU').create_infer_request()
        self.eos_token_id = self.tokenizer.eos_token_id

    def sample_next_token(self, logits: np.ndarray, top_k, top_p, temperature):
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

    def generate_iterate(self, prompt: str, max_generated_tokens, top_k, top_p,
                         temperature):
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


@st.cache_resource
def create_model():
    return ChatGLMModel("THUDM/chatglm2-6b")


with st.spinner("加载模型中..."):
    ov_chatglm = create_model()

if 'history' not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.markdown("## 选择参数")

    max_tokens = st.number_input("max_tokens",
                                 min_value=1,
                                 max_value=500,
                                 value=200)
    temperature = st.number_input("temperature",
                                  min_value=0.1,
                                  max_value=4.0,
                                  value=0.8)
    top_p = st.number_input("top_p", min_value=0.1, max_value=1.0, value=0.8)
    top_k = st.number_input("top_k", min_value=1, max_value=500, value=20)

    if st.button("清空上下文"):
        st.session_state.message = ""
        st.session_state.history = []

st.markdown("## OpenVINO Chat Robot based on ChatGLM2")

history: list[tuple[str, str]] = st.session_state.history

if len(history) == 0:
    st.caption("请在下方输入消息开始会话")

for idx, (question, answer) in enumerate(history):
    message(question, is_user=True, key=f"history_question_{idx}")
    st.write(answer)
    st.markdown("---")

next_answer = st.container()

question = st.text_area(label="消息", key="message")

if st.button("发送") and len(question.strip()):
    with next_answer:
        message(question, is_user=True, key="message_question")
        with st.spinner("正在回复中"):
            with st.empty():
                prompt = chat_template(history, question)
                for answer in ov_chatglm.generate_iterate(
                        prompt,
                        max_generated_tokens=max_tokens,
                        top_k=top_k,
                        top_p=top_p,
                        temperature=temperature,
                ):
                    st.write(answer)
        st.markdown("---")

    st.session_state.history = history + [(question, answer)]