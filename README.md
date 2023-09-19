# ChineseLLM.openvino

This sample shows how to implement trending Chinese LLM model with OpenVINO runtime.


<img width="1110" alt="image" src="https://github.com/OpenVINO-dev-contest/chatglm2.openvino/assets/91237924/6cdfbc45-f70c-42d4-b748-27113d8fe3a8">

## Requirements

- Linux, Windows, MacOS
- Python >= 3.7.0
- CPU or GPU compatible with OpenVINO.
- RAM >= 32GB
- vRAM >= 16GB

## How to run it?

**1. Set-up the environments:**

```python3 -m venv openvino_env```

```source openvino_env/bin/activate```

```python3 -m pip install --upgrade pip```

```pip install wheel setuptools```

```pip install -r requirements.txt```

**2. Run tasks:**

|                              |                               **ChatGLM2**                              |                                     **Baichuan2**                                    |                                 **Qwen**                                |
|------------------------------|:-----------------------------------------------------------------------:|:------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------:|
| **Export FP16 IR**           | ```python3 chatglm2/export_ir.py```                                   | ```python3 baichuan2/export_ir.py```                                                 | ```python3 qwen/export_onnx.py```                                       |
| **Export INT8 IR(Optional)** | ```python3 chatglm2/export_ir.py -cw=True```                          | ```python3 baichuan2/export_ir.py -cw=True```                                        | ```python3 qwen/export_onnx.py -cw=True```                              |
| **Run text generation**      | ```python3 generate_ov.py -m 'THUDM/chatglm2-6b' -p '请介绍一下上海'``` | ```python3 generate_ov.py -m 'baichuan-inc/Baichuan2-7B-Chat' -p '请介绍一下上海'``` | ```python3 generate_ov.py -m 'Qwen/Qwen-7B-Chat' -p '请介绍一下上海'``` |
| **Run chatbot**              | ```streamlit run chatbot.py -- -m 'THUDM/chatglm2-6b'```                | ```streamlit run chatbot.py -- -m 'baichuan-inc/Baichuan2-7B-Chat'```                | ```streamlit run chatbot.py -- -m 'Qwen/Qwen-7B-Chat'```                |
