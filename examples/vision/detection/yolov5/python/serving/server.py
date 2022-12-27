import fastdeploy as fd
import os
import logging

logging.getLogger().setLevel(logging.INFO)

# Get arguments from envrionment variables
model_dir = os.environ.get('MODEL_DIR')
device = os.environ.get('DEVICE', 'cpu')
use_trt = os.environ.get('USE_TRT', False)

# Prepare model
model_file = os.path.join(model_dir, "model.pdmodel")
params_file = os.path.join(model_dir, "model.pdiparams")

# Setup runtime option to select hardware, backend, etc.
option = fd.RuntimeOption()
if device.lower() == 'gpu':
    option.use_gpu()
if use_trt:
    option.use_trt_backend()
    option.set_trt_input_shape("images", [1, 3, 640, 640])
    option.set_trt_cache_file('yolov5s.trt')

# Create model instance
model_instance = fd.vision.detection.YOLOv5(
    model_file,
    params_file,
    runtime_option=option,
    model_format=fd.ModelFormat.PADDLE)

# Create server, setup REST API
app = fd.serving.SimpleServer()
app.register(
    task_name="fd/yolov5s",
    model_handler=fd.serving.handler.VisionModelHandler,
    predictor=model_instance)
