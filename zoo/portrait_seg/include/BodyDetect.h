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
#include <queue>
#include <string>
#include <vector>
#include "tnn_sdk_sample.h"
#include "tnn/utils/mat_utils.h"
#include "tnn/utils/dims_vector_utils.h"

namespace TNN_NS {

#define USE_NEW_MODEL true

class BodyDetectInput : public TNNSDKInput {
public:
    BodyDetectInput(std::shared_ptr<Mat> mat = nullptr) : TNNSDKInput(mat) {};
    virtual ~BodyDetectInput() {}
};

class BodyDetectOutput : public TNNSDKOutput {
public:
    BodyDetectOutput(std::shared_ptr<Mat> mat = nullptr) : TNNSDKOutput(mat) {};
    virtual ~BodyDetectOutput() {};
};

class BodyDetectOption : public TNNSDKOption {
public:
    BodyDetectOption() {}
    virtual ~BodyDetectOption() {}
    int input_width;
    int input_height;
    int num_thread = 1;
    // the processing mode of output mask
    int mode = 0;
};

class BodyDetect : public TNN_NS::TNNSDKSample {
public:
    ~BodyDetect();
    virtual Status Init(std::shared_ptr<TNNSDKOption> option);
    virtual std::shared_ptr<Mat> ProcessSDKInputMat(std::shared_ptr<Mat> mat, std::string name = kTNNSDKDefaultName);
    virtual MatConvertParam GetConvertParamForInput(std::string name = "");
    virtual std::shared_ptr<TNNSDKOutput> CreateSDKOutput();
    virtual Status ProcessSDKOutput(std::shared_ptr<TNNSDKOutput> output);

    void setOFDStatus(bool b) {
        m_enable_ofd = b;
    }

    void setThreshold(float thres) {
        m_thres = thres;
    }

    int humRectLeft;
    int humRectTop;
    int humRectWidth;
    int humRectHeight;
    int* maskData;

private:
    u_char* OFD(const int size);

    // float* OFDV1(const int size);
private:

    DimsVector orig_dims;
    u_char * rmaskData = NULL;

    u_char* p_pre_mask = nullptr;
    u_char* p_cur_mask = nullptr;
    u_char* p_next_mask = nullptr;

    // float* p_pre_conf = nullptr;
    // float* p_cur_conf = nullptr;
    // float* p_next_conf = nullptr;

    bool m_enable_ofd = true;

    int srcInputWidth = 0;
    int srcInputHeight = 0;
    float scaleX{1.0f};
    float scaleY{1.0f};

    float m_thres = 0.9;
};

}
#endif //TNN_BODY_DETECT_H_
