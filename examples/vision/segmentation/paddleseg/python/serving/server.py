import fastdeploy as fd
import os
import logging

logging.getLogger().setLevel(logging.INFO)

# Get arguments from envrionment variables
model_dir = 'mobileseg_mobilenetv2/'
device = 'gpu'
use_trt = True

model_file = os.path.join(model_dir, "model.pdmodel")
params_file = os.path.join(model_dir, "model.pdiparams")
config_file = os.path.join(model_dir, "deploy.yaml")

# Setup runtime option to select hardware, backend, etc.
option = fd.RuntimeOption()
if device.lower() == 'gpu':
    option.use_gpu()
if use_trt:
    option._option.backend = fd.C.Backend.TRT
    option.enable_paddle_to_trt()
    option.set_trt_cache_file('mobileseg_mobilenetv2')
    option.set_trt_input_shape("x", [1, 3, 256, 512])
    option.enable_trt_fp16()
    option.enable_paddle_trt_collect_shape()

# Create model instance
model_instance = fd.vision.segmentation.PaddleSegModel(
    model_file=model_file,
    params_file=params_file,
    config_file=config_file,
    runtime_option=option)

# Create server, setup REST API
app = fd.serving.SimpleServer()
app.register(
    task_name="fd/mobileseg",
    model_handler=fd.serving.handler.VisionModelHandler,
    predictor=model_instance)
