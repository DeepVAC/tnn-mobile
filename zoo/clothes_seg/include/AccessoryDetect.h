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

#ifndef TNN_BODY_DETECT_H_
#define TNN_BODY_DETECT_H_

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>
#include "tnn_sdk_sample.h"
#include "tnn/utils/mat_utils.h"
#include "tnn/utils/dims_vector_utils.h"

namespace TNN_NS {

class AccessoryDetectInput : public TNNSDKInput {
public:
    AccessoryDetectInput(std::shared_ptr<Mat> mat = nullptr) : TNNSDKInput(mat) {};
    virtual ~AccessoryDetectInput() {}
};

class AccessoryDetectOutput : public TNNSDKOutput {
public:
    AccessoryDetectOutput(std::shared_ptr<Mat> mat = nullptr) : TNNSDKOutput(mat) {};
    virtual ~AccessoryDetectOutput() {};
};

class AccessoryDetectOption : public TNNSDKOption {
public:
    AccessoryDetectOption() {}
    virtual ~AccessoryDetectOption() {}
    int input_width;
    int input_height;
    int num_thread = 1;
    // the processing mode of output mask
    int mode = 0;
};

class AccessoryDetect : public TNN_NS::TNNSDKSample {
public:
    ~AccessoryDetect();
    virtual Status Init(std::shared_ptr<TNNSDKOption> option);
    virtual std::shared_ptr<Mat> ProcessSDKInputMat(std::shared_ptr<Mat> mat, std::string name = kTNNSDKDefaultName);
    virtual MatConvertParam GetConvertParamForInput(std::string name = "");
    virtual std::shared_ptr<TNNSDKOutput> CreateSDKOutput();
    virtual Status ProcessSDKOutput(std::shared_ptr<TNNSDKOutput> output);

    int* maskData;

private:
    u_char* OFD(const int size);

private:

    DimsVector orig_dims;
    //std::shared_ptr<Mat> input_image;
    u_char * rmaskData = nullptr;
    u_char* p_pre_mask = nullptr;
    u_char* p_cur_mask = nullptr;
    u_char* p_next_mask = nullptr;

    int srcInputWidth = 0;
    int srcInputHeight = 0;
    float scaleX{1.0f};
    float scaleY{1.0f};
};

}
#endif //TNN_BODY_DETECT_H_
