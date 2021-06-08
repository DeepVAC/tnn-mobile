// Tencent is pleased to support the open source community by making TNN available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#ifndef TNN_FACE_DETECT_H_
#define TNN_FACE_DETECT_H_

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>
#include "tnn_sdk_sample.h"
#include "tnn/utils/mat_utils.h"
#include "tnn/utils/dims_vector_utils.h"


namespace TNN_NS {

class FaceDetectInput : public TNNSDKInput {
public:
    FaceDetectInput(std::shared_ptr<Mat> mat = nullptr) : TNNSDKInput(mat) {};
    virtual ~FaceDetectInput() {}
};

class FaceDetectOutput : public TNNSDKOutput {
public:
    FaceDetectOutput(std::shared_ptr<Mat> mat = nullptr) : TNNSDKOutput(mat) {};
    virtual ~FaceDetectOutput() {};
};

class FaceDetectOption : public TNNSDKOption {
public:
    FaceDetectOption() {}
    virtual ~FaceDetectOption() {}
    int input_width;
    int input_height;
    int num_thread = 1;
    // the processing mode of output mask
    int mode = 0;
};

#define hard_nms 1
#define blending_nms 2 /* mix nms was been proposaled in paper blaze face, aims to minimize the temporal jitter*/
#define clip(x, y) (x < 0 ? 0 : (x > y ? y : x))
typedef struct FaceInfo {
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int l;
    int t;
    int w;
    int h;
} FaceInfo;


class FaceDetect : public TNN_NS::TNNSDKSample {
public:

    std::vector<FaceInfo> faceList;

    ~FaceDetect();
    virtual Status Init(std::shared_ptr<TNNSDKOption> option);
    virtual std::shared_ptr<Mat> ProcessSDKInputMat(std::shared_ptr<Mat> mat, std::string name = kTNNSDKDefaultName);
    virtual MatConvertParam GetConvertParamForInput(std::string name = "");
    virtual std::shared_ptr<TNNSDKOutput> CreateSDKOutput();
    virtual Status ProcessSDKOutput(std::shared_ptr<TNNSDKOutput> output);

private:

    // the original input image shape
    DimsVector orig_dims;
    // the original input image
    //std::shared_ptr<Mat> input_image;

    int dx = 0;
    int dy = 0;
    float scale = 1;

    int inputWidth = 0;
    int inputHeight = 0;

    std::vector<std::vector<float>> priors = {};

    int calcPriorWidth = -1;
    int calcPriorHeight = -1;

    void calcPriors();
    void nms(std::vector<FaceInfo> &input, std::vector<FaceInfo> &output, int type);

    const float score_threshold = 0.7;
    const float iou_threshold = 0.3;
};

}
#endif //TNN_FACE_DETECT_H_
