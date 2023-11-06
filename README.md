# ChineseLLM.openvino

This sample shows how to implement trending Chinese LLM model with OpenVINO runtime.


<img width="1110" alt="image" src="https://github.com/OpenVINO-dev-contest/chatglm2.openvino/assets/91237924/6cdfbc45-f70c-42d4-b748-27113d8fe3a8">

## Supported models

- ChatGLM2/3
- Baichuan2
- Qwen
- InternLM

## Requirements

- Linux, Windows, MacOS
- Python >= 3.7.0
- CPU or GPU compatible with OpenVINO.
- RAM >= 32GB
- vRAM >= 16GB

## How to run it?

**1. Set-up the environments:**

```
python3 -m venv openvino_env

source openvino_env/bin/activate

python3 -m pip install --upgrade pip

pip install wheel setuptools

pip install -r requirements.txt
```

**2. Run tasks:**

|                              |                               **ChatGLM2/3**                              |                                     **Baichuan2**                                    |                                 **Qwen**                                |                                   **InternLM**                                  |
|------------------------------|:-----------------------------------------------------------------------:|:------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------:|:-------------------------------------------------------------------------------:|
| **Export FP16 IR**           | ```python3 chatglm/export_ir.py```                                     | ```python3 baichuan/export_ir.py```                                                 | ```python3 qwen/export_ir.py```                                         | ```python3 internlm/export_ir.py```                                             |
| **Export INT8 IR(Optional)** | ```python3 chatglm/export_ir.py -cw=True```                            | ```python3 baichuan/export_ir.py -cw=True```                                        | ```python3 qwen/export_ir.py -cw=True```                                | ```python3 Internlm/export_ir.py -cw=True```                                    |
| **Run text generation**      | ```python3 generate_ov.py -m 'chatglm/chatglm3' -p '请介绍一下上海'``` | ```python3 generate_ov.py -m 'baichuan/baichuan2' -p '请介绍一下上海'``` | ```python3 generate_ov.py -m 'qwen/qwen' -p '请介绍一下上海'``` | ```python3 generate_ov.py -m 'internlm/internlm' -p '请介绍一下上海'``` |
| **Run chatbot**              | ```streamlit run chatbot.py -- -m 'chatglm/chatglm3'```                | ```streamlit run chatbot.py -- -m 'baichuan/baichuan2'```                | ```streamlit run chatbot.py -- -m 'qwen/qwen'```                | ```streamlit run chatbot.py -- -m 'internlm/internlm'```                |