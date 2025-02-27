# Integrate TAO model with DeepStream SDK

- [Integrate TAO model with DeepStream SDK](#integrate-tao-model-with-deepstream-sdk)
  - [Description](#description)
  - [Prerequisites](#prerequisites)
  - [Download](#download)
    - [1. Download Source Code with SSH or HTTPS](#1-download-source-code-with-ssh-or-https)
    - [2. Download Models](#2-download-models)
  - [Triton Inference Server](#triton-inference-server)
  - [Build](#build)
    - [Build Sample Application](#build-sample-application)
  - [Run](#run)
  - [Information for Customization](#information-for-customization)
    - [TAO Models](#tao-models)
    - [Label Files](#label-files)
    - [DeepStream configuration file](#deepstream-configuration-file)
    - [Model Outputs](#model-outputs)
      - [1~4. Yolov3 / YoloV4 / Yolov4-tiny / Yolov5](#14-yolov3--yolov4--yolov4-tiny--yolov5)
      - [5~8. RetinaNet / DSSD / SSD/ FasterRCNN](#58-retinanet--dssd--ssd-fasterrcnn)
      - [9. PeopleSegNet](#9-peoplesegnet)
      - [10~12. UNET/PeopleSemSegNet/CitySemSegFormer](#1012-unetpeoplesemsegnetcitysemsegformer)
      - [13. multi_task](#13-multitask)
      - [14~15. EfficientDet / Retail Object Detection](#1415-efficientdet--retail-object-detection)
      - [16~23. FaceDetect / Facial Landmarks Estimation / EmotionNet / Gaze Estimation / GestureNet / HeartRateNet / BodyPoseNet/ PoseClassification](#1623-facedetect--facial-landmarks-estimation--emotionnet--gaze-estimation--gesturenet--heartratenet--bodyposenet-poseclassification)
      - [24. PeopleNet Transformer](#24-peoplenet-transformer)
      - [25~26. Re-Identification / Retail Item Recognition](#2526-re-identification--retail-item-recognition)
      - [27~28. OCDNet / OCRNet](#2728-ocdnet--ocrnet)
  - [FAQ](#faq)
    - [Measure The Inference Perf](#measure-the-inference-perf)
    - [About misc folder](#about-misc-folder)
  - [Others Models](#others-models)
  - [Graph Composer Samples](#graph-composer-samples)
  - [Known issues](#known-issues)

## Description

This repository provides a DeepStream sample application based on [NVIDIA DeepStream SDK](https://developer.nvidia.com/deepstream-sdk) to run eleven TAO models (**Faster-RCNN** / **YoloV3** / **YoloV4** / **YoloV5** /**SSD** / **DSSD** / **RetinaNet**/ **PeopleSegNet**/ **UNET**/ **multi_task**/ **peopleSemSegNet**) with below files:

- **apps**: sample application for detection models and segmentation models
- **configs**: DeepStream nvinfer configure file and label files
- **post_processor**: include inference postprocessor for the models
- **graphs**: DeepStream sample graphs based on the Graph Composer tools.
- **models**: The models which will be used as samples.
- **TRT-OSS**: The OSS nvinfer plugin build and download instructions. The OSS plugins are needed for some models with DeepStream 6.4 GA.

The pipeline of the sample:

```
                                                                           |-->filesink(save the output in local dir)
                                                            |--> encode -->
                                                                           |-->fakesink(use -f option)
uridecoderbin -->streammux-->nvinfer(detection)-->nvosd-->
                                                            |--> display
```

## Prerequisites

* [DeepStream SDK 6.4 GA](https://developer.nvidia.com/deepstream-sdk)

   Make sure deepstream-test1 sample can run successful to verify your installation. According to the
   [document](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_docker_containers.html),
   please run below command to install additional audio video packages.

  ```
  /opt/nvidia/deepstream/deepstream/user_additional_install.sh
  ```
* Eigen development packages
  ```
    sudo apt install libeigen3-dev
    cd /usr/include
    sudo ln -sf eigen3/Eigen Eigen
  ```

## Download

### 1. Download Source Code with SSH or HTTPS

```
sudo apt update
sudo apt install git-lfs
git lfs install --skip-repo
// SSH
git clone git@github.com:NVIDIA-AI-IOT/deepstream_tao_apps.git
// or HTTPS
git clone https://github.com/NVIDIA-AI-IOT/deepstream_tao_apps.git
```
### 2. Download Models
Run below script to download models except multi_task and YoloV5 models.

```
sudo ./download_models.sh  # (sudo not required in case of docker containers)
```

For multi_task, refer to https://docs.nvidia.com/tao/tao-toolkit/text/multitask_image_classification.html to train and generate the model.

For yolov5, refer to [yolov5_gpu_optimization](https://github.com/NVIDIA-AI-IOT/yolov5_gpu_optimization) to generate the onnx model

**Note:** We deliver new trained SSD/DSSD/FasterRCNN models for the demo purpose with TAO 5.1 release. The output of the new models will not be excatly same as the previous models. For example, you will notice that more cars can be detected in the DeepStream sample video with the new SSD/DSSD.

### 3. Download Pre-built TensorRT OSS nvinfer plugin library

Please download the TensorRT OSS plugin according to your platforms

[x86 platform TRT OSS plugin download instruction](TRT-OSS/x86/README.md)

[Jetson platform TRT OSS plugin download instruction](TRT-OSS/Jetson/README.md)

## Triton Inference Server

The sample provides three inferencing methods. For the TensorRT based gst-nvinfer inferencing, please skip this part.

The DeepStream sample application can work as Triton client with the [Triton Inference Server](https://developer.nvidia.com/nvidia-triton-inference-server), one of the following two methods can be used to set up the Triton Inference Server before starting a gst-nvinferserver inferncing DeepStream application.

 - Native Triton Inference Server, please refer to [Triton Server](https://github.com/NVIDIA-AI-IOT/deepstream_tao_apps/blob/master/triton_server.md)
 - Stand-alone Triton Inference server, please refer to [Triton grpc server](https://github.com/NVIDIA-AI-IOT/deepstream_tao_apps/blob/master/triton_server_grpc.md)

For the TAO sample applications, please enable Triton or Triton gRPC inferencing with the app YAML configurations.

E.G. With apps/tao_detection/ds-tao-detection, the "primary-gie" part in configs/app/det_app_frcnn.yml can be modified as following:

```
primary-gie:
  #0:nvinfer, 1:nvinfeserver
  plugin-type: 1
  #dssd
  #config-file-path: ../nvinfer/dssd_tao/pgie_dssd_tao_config.yml
  config-file-path: ../triton/dssd_tao/pgie_dssd_tao_config.yml
  #config-file-path: ../triton-grpc/dssd_tao/pgie_dssd_tao_config.yml

```
And then run the app with the command:

```
./apps/tao_detection/ds-tao-detection configs/app/det_app_frcnn.yml
```

## Build

### Build Sample Application

```
export CUDA_MODULE_LOADING=LAZY
export CUDA_VER=xy.z                                      // xy.z is CUDA version, e.g. 10.2
make
```
## Run

```

1.Usage: ds-tao-detection -c pgie_config_file -i <H264 or JPEG file uri> [-b BATCH] [-d] [-f] [-l]
    -h: print help info
    -c: pgie config file, e.g. pgie_frcnn_tao_config.txt
    -i: uri of the input file, start with the file:///, e.g. file:///.../video.mp4
    -b: batch size, this will override the value of "batch-size" in pgie config file
    -d: enable display, otherwise it will dump to output MP4 or JPEG file without -f option
    -f: use fakesink mode
    -l: use loop mode

2.Usage: ds-tao-detection <yaml file uri>
  e.g.
  ./apps/tao_detection/ds-tao-detection configs/app/det_app_frcnn.yml


note: If you want use multi-source, you can input multi -i input(e.g., -i uri -i uri...) 
      Only YAML configurations support Triton and Triton gRPC inferencing.
```
For detailed model information, pleasing refer to the following table:

note:  
  The default $DS_SRC_PATH is /opt/nvidia/deepstream/deepstream

|Model Type|Tao Model|Demo|
|-----------|----------|----|
|detector|dssd, peoplenet_transformer, efficientdet, frcnn, retinanet, retail_detector_100, retail_detector_binary, ssd, yolov3, yolov4-tiny, yolov4, yolov5|./apps/tao_detection/ds-tao-detection -c configs/dssd_tao/pgie_dssd_tao_config.txt -i file:///$DS_SRC_PATH/samples/streams/sample_720p.mp4<br>or<br>./apps/tao_detection/ds-tao-detection configs/app/det_app_frcnn.yml|
|classifier|multi-task|./apps/tao_classifier/ds-tao-classifier -c configs/multi_task_tao/pgie_multi_task_tao_config.txt -i file:///$DS_SRC_PATH/samples/streams/sample_720p.mp4<br>or<br>./apps/tao_classifier/ds-tao-classifier configs/app/multi_task_app_config.yml|
|segmentation|peopleSemSegNet, unet, citySemSegFormer|./apps/tao_segmentation/ds-tao-segmentation -c configs/peopleSemSegNet_tao/pgie_peopleSemSegNet_tao_config.txt -i file:///$DS_SRC_PATH/samples/streams/sample_720p.mp4<br>or<br>./apps/tao_segmentation/ds-tao-segmentation configs/app/seg_app_unet.yml|
|instance segmentation|peopleSegNet|export SHOW_MASK=1; ./apps/tao_detection/ds-tao-detection -c configs/peopleSegNet_tao/pgie_peopleSegNet_tao_config.txt -i file:///$DS_SRC_PATH/samples/streams/sample_720p.mp4<br>or<br>export SHOW_MASK=1; ./apps/tao_detection/ds-tao-detection configs/app/ins_seg_app_peopleSegNet.yml|
|others|FaceDetect, Facial Landmarks Estimation, EmotionNet, Gaze Estimation, GestureNet, HeartRateNet, BodyPoseNet,Re-identification, Retail Object Recognition, PoseClassificationNet, OCDNet, OCRNet|refer detailed [README](https://github.com/NVIDIA-AI-IOT/deepstream_tao_apps/blob/master/apps/tao_others/README.md) for how to configure and run the model|

Building the TensorRT engine of citySemSegFormer consumes a lot of device memory. Please `export CUDA_MODULE_LOADING=LAZY` to reduce device memory consumption. Please read [CUDA Environment Variables](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#env-vars) for details.

## Information for Customization

If you want to do some customization, such as training your own TAO models, running the model in other DeepStream pipeline, you should read below sections.
### TAO Models

To download the sample models that we have trained with [NVIDIA TAO Toolkit SDK](https://developer.nvidia.com/tao-toolkit) , run `wget https://nvidia.box.com/shared/static/hzrhk33vijf31w9nxb9c93gctu1w0spd -O models.zip`

Refer [TAO Doc](https://docs.nvidia.com/tao/tao-toolkit/text/overview.html) for how to train the models, after training finishes, run `tao-export` to generate an `.etlt` model. This .etlt model can be deployed into DeepStream for fast inference as this sample shows.
This DeepStream sample app also supports the TensorRT engine(plan) file generated by running the `tao-converter` tool on the `.etlt` model.
The TensorRT engine file is hardware dependent, while the `.etlt` model is not. You may specify either a TensorRT engine file or a `.etlt` model in the DeepStream configuration file.

Note, for Unet/peopleSemSegNet/yolov3/yolov4/yolov5 model, you can also convert the etlt model to TensorRT engine file using `tao-converter` like following:

```
tao-converter -e models/unet/unet_resnet18.etlt_b1_gpu0_fp16.engine -p input_1,1x3x608x960,1x3x608x960,1x3x608x960 -t fp16 -k tlt_encode -m 1 tlt_encode models/unet/unet_resnet18.etlt
```
### Label Files

The label file includes the list of class names for a model, which content varies for different models.  
User can find the detailed label information for the MODEL in the README.md and the label file under *configs/$(MODEL)_tao/*, e.g. ssd label informantion under *configs/ssd_tao/*
  
Note, for some models like FasterRCNN, DON'T forget to include "background" lable and change num-detected-classes in pgie configure file accordingly 

### DeepStream configuration file

The DeepStream configuration file includes some runtime parameters for DeepStream **nvinfer** plugin or **nvinferserver** plugin, such as model path, label file path, TensorRT inference precision, input and output node names, input dimensions and so on.  
In this sample, each model has its own DeepStream configuration file, e.g. pgie_dssd_tao_config.txt for DSSD model.
Please refer to [DeepStream Development Guide](https://docs.nvidia.com/metropolis/deepstream/dev-guide/index.html#page/DeepStream_Development_Guide%2Fdeepstream_app_config.3.2.html) for detailed explanations of those parameters.

### Model Outputs

#### 1~4. Yolov3 / YoloV4 / Yolov4-tiny / Yolov5

The model has the following four outputs:

- **num_detections**: A [batch_size] tensor containing the INT32 scalar indicating the number of valid detections per batch item. It can be less than keepTopK. Only the top num_detections[i] entries in nmsed_boxes[i], nmsed_scores[i] and nmsed_classes[i] are valid
- **nmsed_boxes**: A [batch_size, keepTopK, 4] float32 tensor containing the coordinates of non-max suppressed boxes
- **nmsed_scores**: A [batch_size, keepTopK] float32 tensor containing the scores for the boxes
- **nmsed_classes**: A [batch_size, keepTopK] float32 tensor containing the classes for the boxes

#### 5~8. RetinaNet / DSSD / SSD/ FasterRCNN

These three models have the same output layer named NMS which implementation can refer to TRT OSS [nmsPlugin](https://github.com/NVIDIA/TensorRT/tree/master/plugin/nmsPlugin):

* an output of shape [batchSize, 1, keepTopK, 7] which contains nmsed box class IDs(1 value), nmsed box scores(1 value) and nmsed box locations(4 value)
* another output of shape [batchSize, 1, 1, 1] which contains the output nmsed box count.

#### 9. PeopleSegNet

The model has the following two outputs:

- **generate_detections**: A [batchSize, keepTopK, C*6] tensor containing the bounding box, class id, score
- **mask_head/mask_fcn_logits/BiasAdd**:  A [batchSize, keepTopK, C+1, 28*28] tensor containing the masks

#### 10~12. UNET/PeopleSemSegNet/CitySemSegFormer

- **argmax_1/output**: A [batchSize, H, W, 1] tensor containing the class id per pixel location

#### 13. multi_task
- refer detailed [README](./configs/nvinfer/multi_task_tao/README.md) for how to configure and run the model

#### 14~15. EfficientDet / Retail Object Detection

Please note there are two `Retail Object Detection` models. These models have the following four outputs:

- **num_detections**: This is a [batch_size, 1] tensor of data type int32. The last dimension is a scalar indicating the number of valid detections per batch image. It can be less than max_output_boxes. Only the top num_detections[i] entries in nms_boxes[i], nms_scores[i] and nms_classes[i] are valid.
- **detection_boxes**: This is a [batch_size, max_output_boxes, 4] tensor of data type float32 or float16, containing the coordinates of non-max suppressed boxes. The output coordinates will always be in BoxCorner format, regardless of the input code type.
- **detection_scores**: This is a [batch_size, max_output_boxes] tensor of data type float32 or float16, containing the scores for the boxes.
- **detection_classes**: This is a [batch_size, max_output_boxes] tensor of data type int32, containing the classes for the boxes.

#### 16~23. FaceDetect / Facial Landmarks Estimation / EmotionNet / Gaze Estimation / GestureNet / HeartRateNet / BodyPoseNet/ PoseClassification
- refer detailed [README](https://github.com/NVIDIA-AI-IOT/deepstream_tao_apps/blob/master/apps/tao_others/README.md) for how to configure and run the model

#### 24. PeopleNet Transformer

The model has the following two outputs:

- **pred_logits**: This is a [batch_size, num_queries, num_classes] tensor of data type float32. The
tensor contains probability values of each class.
- **pred_boxes**: This is a [batch_size, num_queries, 4] tensor of data type float32. The tensor
represents the 2D bounding box coordinates in the format of [center_x, center_y, width, height].

#### 25~26. Re-Identification / Retail Item Recognition

These models are trained to extract the embedding vector from an image. The image is the cropped area of a
bounding box from a primary-gie task, like people detection by `PeopleNet Transformer` or retail item detection
by `Retail Object Detection`. These embedding extraction models are typically arranged
as the secondary GIE module in a Deepstream pipeline.

##### Re-Identification uses ResNet50 backbone.
The output layer is:
- **fc_pred**: This is a [batch_size, embedding_size] tensor of data type float32. The tensor
contains the embedding vector of size `embedding_size = 256`.

##### Retail Item Recognition uses ResNet101 backbone.
The output layer is:
- **outputs**: This is a [batch_size, 2048] tensor of data type float32. The tensor contains the embedding
vector of size `2048`.

#### 27~28. OCDNet / OCRNet

##### OCDNet output layer
- **pred**:This is a [batchSize, H, W , 1] float tensor containing the probability of a pixel belongs to text
##### OCRNet output layer
- **output_prob**:This is a [batchSize, W/4] float tensor containing the confidence of each character in the text
- **output_id**:This is a [batchSize, W/4] integer tensor containing each character index of the text. This index can be mapped to character through the OCRNet charater list

## FAQ

### Measure The Inference Perf

```CQL
# 1.  Build TensorRT Engine through this smample, for example, build YoloV3 with batch_size=2
./ds-tao -c pgie_yolov3_tao_config.txt -i /opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.h264 -b 2
## after this is done, it will generate the TRT engine file under models/$(MODEL), e.g. models/yolov3/ for above command.
# 2. Measure the Inference Perf with trtexec, following above example
cd models/yolov3/
trtexec --batch=2 --useSpinWait --loadEngine=yolo_resnet18.etlt_b2_gpu0_fp16.engine
## then you can find the per *BATCH* inference time in the trtexec output log
```

### About misc folder

```CQL
# The files in the folder are used by TAO dev blogs:
## 1.  Training State-Of-The-Art Models for Classification and Object Detection with NVIDIA TAO Toolkit
## 2.  Real time vehicle license plate detection and recognition using NVIDIA TAO Toolkit
```
## Others Models

There are some special models which are not exactly detector, classifier or segmetation. The sample application of these special models are put in apps/tao_others. These samples should run on DeepStream 6.1 or above versions. Please refer to apps/tao_others/README.md document for details.

## Graph Composer Samples

Some special models needs special deepstream pipeline for running. The deepstream sample graphs for them are put in graphs/tao_others. Please refer to graphs/README.md file for more details.

## Known issues

1. For some yolo models, some layers of the models should use FP32 precision. This is a network characteristics that the accuracy drops rapidly when maximum layers are run in INT8 precision. Please refer the [layer-device-precision](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvinfer.html) for more details.
2. Currently the citySemSegFormer model only supports batch-size 1.
