FROM agrigorev/model-2025-hairstyle:v1

USER root

RUN pip install onnxruntime pillow torchvision

ARG MODEL_NAME=hair_classifier_empty.onnx
ENV MODEL_NAME=${MODEL_NAME}

COPY lambda_function.py ./

CMD ["lambda_function.lambda_handler"]