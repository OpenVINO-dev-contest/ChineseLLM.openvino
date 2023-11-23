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

<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-9wq8{border-color:inherit;text-align:center;vertical-align:middle}
.tg .tg-0lax{text-align:left;vertical-align:top}
.tg .tg-uzvj{border-color:inherit;font-weight:bold;text-align:center;vertical-align:middle}
.tg .tg-nrix{text-align:center;vertical-align:middle}
</style>
<table class="tg">
<thead>
  <tr>
    <th class="tg-9wq8"></th>
    <th class="tg-9wq8"><span style="font-weight:bold">ChatGLM2/3</span></th>
    <th class="tg-9wq8"><span style="font-weight:bold">Baichuan2</span></th>
    <th class="tg-9wq8"><span style="font-weight:bold">Qwen</span></th>
    <th class="tg-9wq8"><span style="font-weight:bold">Internlm</span></th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-uzvj">Export FP16 IR</td>
    <td class="tg-9wq8">python3 chatglm/export_ir.py</td>
    <td class="tg-9wq8">python3 baichuan/export_ir.py</td>
    <td class="tg-9wq8">python3 qwen/export_ir.py</td>
    <td class="tg-9wq8">python3 internlm/export_ir.py</td>
  </tr>
  <tr>
    <td class="tg-9wq8"><span style="font-weight:bold">Export INT8 IR(Optional)</span></td>
    <td class="tg-9wq8" colspan="4">python3 quantize.py -m "orignal model dir" -o "output model dir" </td>
  </tr>
  <tr>
    <td class="tg-9wq8"><span style="font-weight:bold">Run text generation</span></td>
    <td class="tg-9wq8">python3 generate_ov.py -m 'chatglm/chatglm3' -p '请介绍一下上海'</td>
    <td class="tg-9wq8">python3 generate_ov.py -m 'baichuan/baichuan2' -p '请介绍一下上海</td>
    <td class="tg-9wq8">python3 generate_ov.py -m 'qwen/qwen' -p '请介绍一下上海'</td>
    <td class="tg-9wq8">python3 generate_ov.py -m 'internlm/internlm' -p '请介绍一下上海'</td>
  </tr>
  <tr>
    <td class="tg-9wq8"><span style="font-weight:bold">Run chatbot</span></td>
    <td class="tg-9wq8">streamlit run chatbot.py -- -m 'chatglm/chatglm3'</td>
    <td class="tg-9wq8">streamlit run chatbot.py -- -m 'baichuan/baichuan2'</td>
    <td class="tg-9wq8">streamlit run chatbot.py -- -m 'qwen/qwen'</td>
    <td class="tg-9wq8">streamlit run chatbot.py -- -m 'internlm/internlm'</td>
  </tr>
</tbody>
</table>

For more information on quantization configuration, please refer to [weight compression](https://github.com/openvinotoolkit/nncf/blob/release_v270/docs/compression_algorithms/CompressWeights.md)