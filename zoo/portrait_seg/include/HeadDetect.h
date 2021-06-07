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

#ifndef TNN_HEAD_DETECT_H_
#define TNN_HEAD_DETECT_H_

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>
#include "tnn_sdk_sample.h"
#include "tnn/utils/mat_utils.h"
#include "tnn/utils/dims_vector_utils.h"
#include "FaceDetect.h"

namespace TNN_NS {

#define HEAD_INPUT_BATCH 1

class HeadDetectInput : public TNNSDKInput {
public:
    HeadDetectInput(std::shared_ptr<Mat> mat = nullptr) : TNNSDKInput(mat) {};
    virtual ~HeadDetectInput() {}
};

class HeadDetectOutput : public TNNSDKOutput {
public:
    HeadDetectOutput(std::shared_ptr<Mat> mat = nullptr) : TNNSDKOutput(mat) {};
    virtual ~HeadDetectOutput() {};
};

class HeadDetectOption : public TNNSDKOption {
public:
    HeadDetectOption() {}
    virtual ~HeadDetectOption() {}
    int input_width;
    int input_height;
    int num_thread = 1;
    // the processing mode of output mask
    int mode = 0;
};


class HeadDetect : public TNN_NS::TNNSDKSample {
public:

    int srcInputWidth = 0; // 最初的输入大小
    int srcInputHeight = 0;
    int* maskData;
    bool isDetectedBody = false;
    std::vector<tnn::FaceInfo> faceList;

    int modelInputWidth = 0;
    int modelInputHeight = 0;

    ~HeadDetect();
    virtual Status Init(std::shared_ptr<TNNSDKOption> option);
    virtual std::shared_ptr<Mat> ProcessSDKInputMat(std::shared_ptr<Mat> mat, std::string name = kTNNSDKDefaultName);
    virtual MatConvertParam GetConvertParamForInput(std::string name = "");
    virtual std::shared_ptr<TNNSDKOutput> CreateSDKOutput();
    virtual Status ProcessSDKOutput(std::shared_ptr<TNNSDKOutput> output);

private:


    DimsVector orig_dims;

    u_char * rmaskData = NULL;

    float scaleX{1.0f};
    float scaleY{1.0f};

    int inputWidth = 0;
    int inputHeight = 0;

};

}
#endif //TNN_HEAD_DETECT_H_
