// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "fastdeploy/vision.h"
#include "preprocess.h"


#ifndef CUDA_CHECK
#define CUDA_CHECK(callstr)\
    {\
        cudaError_t error_code = callstr;\
        if (error_code != cudaSuccess) {\
            std::cerr << "CUDA error " << error_code << " at " << __FILE__ << ":" << __LINE__;\
            assert(0);\
        }\
    }
#endif  // CUDA_CHECK


int MAX_IMAGE_INPUT_SIZE_THRESH = 1920 * 1080;
uint8_t* img_host = nullptr;
uint8_t* img_device = nullptr;
float* input_buffer_gpu = nullptr;

void CpuInfer(const std::string& model_file, const std::string& image_file) {
  auto model = fastdeploy::vision::detection::YOLOv5(model_file);
  if (!model.Initialized()) {
    std::cerr << "Failed to initialize." << std::endl;
    return;
  }

  auto im = cv::imread(image_file);
  auto im_bak = im.clone();

  fastdeploy::vision::DetectionResult res;
  if (!model.Predict(&im, &res)) {
    std::cerr << "Failed to predict." << std::endl;
    return;
  }
  std::cout << res.Str() << std::endl;

  auto vis_im = fastdeploy::vision::Visualize::VisDetection(im_bak, res);
  cv::imwrite("vis_result.jpg", vis_im);
  std::cout << "Visualized result saved in ./vis_result.jpg" << std::endl;
}

void GpuInfer(const std::string& model_file, const std::string& image_file) {
  auto option = fastdeploy::RuntimeOption();
  option.UseGpu();
  auto model = fastdeploy::vision::detection::YOLOv5(model_file, "", option);
  if (!model.Initialized()) {
    std::cerr << "Failed to initialize." << std::endl;
    return;
  }

  auto im = cv::imread(image_file);
  auto im_bak = im.clone();

  fastdeploy::vision::DetectionResult res;
  if (!model.Predict(&im, &res)) {
    std::cerr << "Failed to predict." << std::endl;
    return;
  }
  std::cout << res.Str() << std::endl;

  auto vis_im = fastdeploy::vision::Visualize::VisDetection(im_bak, res);
  cv::imwrite("vis_result.jpg", vis_im);
  std::cout << "Visualized result saved in ./vis_result.jpg" << std::endl;
}

bool PreprocessGPU(fastdeploy::vision::Mat* mat, fastdeploy::FDTensor* output,
                        std::map<std::string, std::array<float, 2>>* im_info,
                        const std::vector<int>& size,
                        const std::vector<float> padding_value,
                        bool is_mini_pad, bool is_no_pad, bool is_scale_up,
                        int stride, float max_wh, bool multi_label) {
  // Record the shape of image and the shape of preprocessed image
  (*im_info)["input_shape"] = {static_cast<float>(mat->Height()),
                               static_cast<float>(mat->Width())};
  (*im_info)["output_shape"] = {static_cast<float>(mat->Height()),
                                static_cast<float>(mat->Width())};

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  memcpy(img_host, mat->GetCpuMat()->data, mat->Height() * mat->Width() * 3);
  CUDA_CHECK(cudaMemcpyAsync(img_device, img_host, mat->Height() * mat->Width() * 3, cudaMemcpyHostToDevice, stream));
  preprocess_kernel_img(img_device, mat->Width(), mat->Height(), input_buffer_gpu, size[1], size[0], stream);
  cudaStreamSynchronize(stream);

  // Record output shape of preprocessed image
  (*im_info)["output_shape"] = {640, 640};

  output->SetExternalData({3, 640, 640}, fastdeploy::FDDataType::FP32,
                          input_buffer_gpu);
  output->device = fastdeploy::Device::GPU;
  output->shape.insert(output->shape.begin(), 1);  // reshape to n, h, w, c
  return true;
}

bool MyPredict(fastdeploy::vision::detection::YOLOv5& model, cv::Mat* im, fastdeploy::vision::DetectionResult* result, float conf_threshold,
                     float nms_iou_threshold) {

  fastdeploy::vision::Mat mat(*im);
  std::vector<fastdeploy::FDTensor> input_tensors(1);

  std::map<std::string, std::array<float, 2>> im_info;

  auto start = std::chrono::system_clock::now();

  if (true) {  // gpu
    if (!PreprocessGPU(&mat, &input_tensors[0], &im_info, model.size_, model.padding_value_,
                    model.is_mini_pad_, model.is_no_pad_, model.is_scale_up_, model.stride_, model.max_wh_,
                    model.multi_label_)) {
      fastdeploy::FDERROR << "Failed to preprocess input image." << std::endl;
      return false;
    }
  } else {
    if (!model.Preprocess(&mat, &input_tensors[0], &im_info, model.size_, model.padding_value_,
                    model.is_mini_pad_, model.is_no_pad_, model.is_scale_up_, model.stride_, model.max_wh_,
                    model.multi_label_)) {
      fastdeploy::FDERROR << "Failed to preprocess input image." << std::endl;
      return false;
    }
  }

  auto end = std::chrono::system_clock::now();
  std::cout << "my preprocessing time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

  input_tensors[0].name = model.InputInfoOfRuntime(0).name;
  std::vector<fastdeploy::FDTensor> output_tensors;
  if (!model.Infer(input_tensors, &output_tensors)) {
    fastdeploy::FDERROR << "Failed to inference." << std::endl;
    return false;
  }

  if (!model.Postprocess(output_tensors, result, im_info, conf_threshold,
                   nms_iou_threshold, model.multi_label_)) {
    fastdeploy::FDERROR << "Failed to post process." << std::endl;
    return false;
  }
  return true;
}

void TrtInfer(const std::string& model_file, const std::string& image_file) {
  auto option = fastdeploy::RuntimeOption();
  option.UseGpu(5);
  option.UseTrtBackend();
  option.SetTrtInputShape("images", {1, 3, 640, 640});
  option.SetTrtCacheFile("yolov5.engine");
  auto model = fastdeploy::vision::detection::YOLOv5(model_file, "", option);
  if (!model.Initialized()) {
    std::cerr << "Failed to initialize." << std::endl;
    return;
  }

  auto im = cv::imread(image_file);
  auto im_bak = im.clone();

  // prepare input data cache in pinned memory 
  CUDA_CHECK(cudaMallocHost((void**)&img_host, MAX_IMAGE_INPUT_SIZE_THRESH * 3));
  // prepare input data cache in device memory
  CUDA_CHECK(cudaMalloc((void**)&img_device, MAX_IMAGE_INPUT_SIZE_THRESH * 3));
  CUDA_CHECK(cudaMalloc((void**)&input_buffer_gpu, 3 * 640 * 640 * sizeof(float)));

  fastdeploy::vision::DetectionResult res;
  for (int i = 0; i < 100; i++) {
  auto start = std::chrono::system_clock::now();
  if (!MyPredict(model, &im, &res, 0.25, 0.5)) {
    std::cerr << "Failed to predict." << std::endl;
    return;
  }
  // if (!model.Predict(&im, &res)) {
  //   std::cerr << "Failed to predict." << std::endl;
  //   return;
  // }
  auto end = std::chrono::system_clock::now();
  std::cout << "inference time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
  }
  std::cout << res.Str() << std::endl;

  auto vis_im = fastdeploy::vision::Visualize::VisDetection(im_bak, res);
  cv::imwrite("vis_result.jpg", vis_im);
  std::cout << "Visualized result saved in ./vis_result.jpg" << std::endl;
}

int main(int argc, char* argv[]) {
  if (argc < 4) {
    std::cout << "Usage: infer_demo path/to/model path/to/image run_option, "
                 "e.g ./infer_model ./yolov5.onnx ./test.jpeg 0"
              << std::endl;
    std::cout << "The data type of run_option is int, 0: run with cpu; 1: run "
                 "with gpu; 2: run with gpu and use tensorrt backend."
              << std::endl;
    return -1;
  }

  if (std::atoi(argv[3]) == 0) {
    CpuInfer(argv[1], argv[2]);
  } else if (std::atoi(argv[3]) == 1) {
    GpuInfer(argv[1], argv[2]);
  } else if (std::atoi(argv[3]) == 2) {
    TrtInfer(argv[1], argv[2]);
  }
  return 0;
}
