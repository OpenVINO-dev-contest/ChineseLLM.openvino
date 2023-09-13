import streamlit as st
from streamlit_chat import message
from chatglm2.modeling import ChatGLMModel
from qwen.modeling import QwenModel
from baichuan2.modeling import BaichuanModel
import argparse


@st.cache_resource
def create_model():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-h',
                        '--help',
                        action='help',
                        help='Show this help message and exit.')
    parser.add_argument('-m',
                        '--model_id',
                        required=True,
                        type=str,)
    parser.add_argument('-d',
                    '--device',
                    default='CPU',
                    required=False,
                    type=str,
                    help='Required. device for inference')
    args = parser.parse_args()
    model_id = args.model_id
    if 'chatglm' in model_id:
        ov_model = ChatGLMModel(model_id, args.device)
    elif 'Qwen' in model_id:
        ov_model = QwenModel(model_id, args.device)
    elif 'baichuan' in model_id:
        ov_model = BaichuanModel(model_id, args.device)
    else:
        raise NotImplementedError(f"Unsupported model id {model_id!r}")
    return ov_model


with st.spinner("加载模型中..."):
     chat_model = create_model()

if 'history' not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    system = st.text_area("系统提示词", value="你是一个友好、诚实、善良的聊天助手，可以回答任何问题")
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
    top_k = st.number_input("top_k", min_value=1, max_value=500, value=50)

    if st.button("清空上下文"):
        st.session_state.message = ""
        st.session_state.history = []

st.markdown("## OpenVINO中文聊天助手")

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
                prompt = chat_model.build_inputs(history, question, system)
                for answer in chat_model.generate_iterate(
                        prompt,
                        max_generated_tokens=max_tokens,
                        top_k=top_k,
                        top_p=top_p,
                        temperature=temperature,
                ):
                    st.write(answer)
        st.markdown("---")

    st.session_state.history = history + [(question, answer)]
