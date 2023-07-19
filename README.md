# chatglm2.openvino

This sample shows how to implement a chatglm2-based model with OpenVINO runtime.

**Please notice this repository is only for a functional test, and you can try to quantize the model to further optimize the performance of it**

## How to run it?
1. Install the requirements:

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

    ```$python3 generate_ov.py -m  "THUDM/chatglm2-6b" -p "请讲一个有趣的故事" ```