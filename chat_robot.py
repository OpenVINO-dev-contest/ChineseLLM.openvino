
import streamlit as st
from streamlit_chat import message
from generate_ov import ChatGLMModel, build_inputs


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
                prompt = build_inputs(history, question)
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