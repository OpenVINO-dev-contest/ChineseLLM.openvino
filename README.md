# chatglm2.openvino

This sample shows how to implement a chatglm2-based model with OpenVINO runtime.


<img width="1110" alt="image" src="https://github.com/OpenVINO-dev-contest/chatglm2.openvino/assets/91237924/6cdfbc45-f70c-42d4-b748-27113d8fe3a8">


**Please notice this repository is only for a functional test, and you can try to quantize the model to further optimize the performance of it**

## How to run it?
1. Install the requirements:

    ```$python3 -m venv openvino_env```

    ```$source openvino_env/bin/activate```

    ```$pip install -r requirements.txt```

2. Export the ONNX model from HuggingFace pipeline:

    ```$mkdir onnx_model```

    ```$python3 export_onnx.py -m "THUDM/chatglm2-6b" -o ./onnx_model/chatglm2.onnx```

    ***please follow the Licence on HuggingFace and get the approval before downloading ChatGLM2 checkpoints***

3. Convert ONNX model to OpenVINO IR:

    ```$mkdir ir_model```

    ```$mo -m ./onnx_model/chatglm2.onnx -o ./ir_model/ --compress_to_fp16```
    
    ```$rm ./onnx_model -rf```

4. Run restructured native OpenVINO pipeline:

    ```$python3 generate_ov.py -m  "THUDM/chatglm2-6b" -p "请介绍一下上海？" ```

5. Run chat mode with web UI:

    ```$streamlit run chat_robot.py ```
    
