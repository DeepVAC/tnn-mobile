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

#include "AccessoryDetect.h"
#include <sys/time.h>
#include <cmath>
#include <cstring>

namespace TNN_NS {

AccessoryDetect::~AccessoryDetect() {
    if(rmaskData!=nullptr){
        free(rmaskData);
        rmaskData = nullptr;
    }
    if(p_pre_mask!= nullptr){
        free(p_pre_mask);
        p_pre_mask = nullptr;
    }

    if(p_cur_mask!=nullptr){
        free(p_cur_mask);
        p_cur_mask = nullptr;
    }

    if(p_next_mask!=nullptr){
        free(p_next_mask);
        p_next_mask = nullptr;
    }
}

MatConvertParam AccessoryDetect::GetConvertParamForInput(std::string tag) {
    MatConvertParam input_cvt_param;
    // input_cvt_param.scale = {1.0 / (255 * 63.906185), 1.0 / (255 * 63.37611), 1.0 / (255 * 64.270676), 0.0};//{1.0 / (255 * 62.1744), 1.0 / (255 * 62.69237), 1.0 / (255 * 62.24606), 0.0};
    // input_cvt_param.bias  = {-113.00495 /  (255 * 63.906185), -122.11948 /  (255 * 63.37611), -130.61388 /(255 * 64.270676), 0.0};//{-142.01215 /  (255 * 62.1744), -146.44064 /  (255 * 62.69237), -154.8 /(255 * 62.24606), 0.0};
    input_cvt_param.scale = {1.0 / (255 * 64.1041), 1.0 / (255 * 63.424324), 1.0 / (255 *  64.38638), 0.0};//{1.0 / (255 * 62.1744), 1.0 / (255 * 62.69237), 1.0 / (255 * 62.24606), 0.0};
    input_cvt_param.bias  = {-111.38878 /  (255 * 64.1041), -120.52692 /  (255 * 63.424324), -128.46246 /(255 * 64.38638), 0.0};//{-142.01215 /  (255 * 62.1744), -146.44064 /  (255 * 62.69237), -154.8 /(255 * 62.24606), 0.0};
    input_cvt_param.reverse_channel = true;
    return input_cvt_param;
}


Status AccessoryDetect::Init(std::shared_ptr<TNNSDKOption> option_i) {
    Status status = TNN_OK;
    auto option = dynamic_cast<AccessoryDetectOption *>(option_i.get());
    RETURN_VALUE_ON_NEQ(!option, false, Status(TNNERR_PARAM_ERR, "TNNSDKOption is invalid"));

    status = TNNSDKSample::Init(option_i);
    RETURN_ON_NEQ(status, TNN_OK);

    auto input_dims = GetInputShape();
    option->input_height = input_dims[2];
    option->input_width  = input_dims[3];

    if(rmaskData!=NULL){
        free(rmaskData);
    }
    LOGE("input w:%d, h:%d",input_dims[2], input_dims[3]);
    auto size = sizeof(u_char)* input_dims[2]*input_dims[3];

    rmaskData = (u_char*)malloc(size);
    p_pre_mask = (u_char*)malloc(size);
    p_cur_mask = (u_char*)malloc(size);
    p_next_mask = (u_char*)malloc(size);

    return status;
}

std::shared_ptr<TNNSDKOutput> AccessoryDetect::CreateSDKOutput() {
    return std::make_shared<AccessoryDetectOutput>();
}

u_char* AccessoryDetect::OFD(const int size) {
    static int count = 0;
    if(!m_enable_ofd) {
        count = 0;
        return p_next_mask;
    }

    auto f = [=] {
        auto* temp = p_pre_mask;
        auto* temp1 = p_cur_mask;
        p_cur_mask = p_next_mask;
        p_pre_mask = temp1;
        p_next_mask = temp;
    };

    ++count;
    if(count < 3) {
        f();
        return p_cur_mask;
    }

    for(int i = 0; i < size; ++i) {
        if(p_pre_mask[i] == p_next_mask[i]) {
            p_cur_mask[i] = p_next_mask[i];
        }
    }

    f();
    return p_pre_mask;

}

std::shared_ptr<Mat> AccessoryDetect::ProcessSDKInputMat(std::shared_ptr<Mat> input_image, std::string name) {
    RETURN_VALUE_ON_NEQ(input_image->GetMatType(), N8UC3, nullptr);
    this->orig_dims = input_image->GetDims();
    // save input image mat for merging
    auto dims = input_image->GetDims();

    auto status = TNN_OK;

    auto target_dims = GetInputShape(name);
    auto input_height = input_image->GetHeight();
    auto input_width = input_image->GetWidth();

    srcInputWidth = input_width;
    srcInputHeight = input_height;

    // 强制resize
    if (target_dims.size() >= 4 && (input_height != target_dims[2] || input_width != target_dims[3])) {
        auto target_mat = std::make_shared<TNN_NS::Mat>(input_image->GetDeviceType(), input_image->GetMatType(), target_dims);
        scaleX = (float)target_dims[3] / (float) input_width;
        scaleY = (float)target_dims[2] / (float) input_height;
        auto status = Resize(input_image, target_mat,TNNInterpLinear);
        LOGE("Body Detect Resize to [%d,%d,%d,%d]\n", target_dims[0],target_dims[1],target_dims[2],target_dims[3]);
        if (status == TNN_OK) {
            return target_mat;
        } else {
            LOGE("Body Detect Resize error:%s\n", status.description().c_str());
            return nullptr;
        }
    }else{
        scaleX = 1;
        scaleY = 1;
    }
    return input_image;
}
#define E 2.718281828459045

inline float a_sigmoid(float x){
//    y * (1.0f + exp(-x)) = 1;
//    y  + y * exp(-x) = 1;
//    exp(-x) = (1-y)/y;
//     x = -log((1-y)/y) / log(e);
    return -log((1.0-x)/x) / log(E);
}

Status AccessoryDetect::ProcessSDKOutput(std::shared_ptr<TNNSDKOutput> output_) {
    Status status = TNN_OK;
    auto option = dynamic_cast<AccessoryDetectOption *>(option_.get());
    RETURN_VALUE_ON_NEQ(!option, false, Status(TNNERR_PARAM_ERR, "Body TNNOption is invalid"));

    auto output = dynamic_cast<AccessoryDetectOutput *>(output_.get());
    RETURN_VALUE_ON_NEQ(!output, false, Status(TNNERR_PARAM_ERR, "Body TNNSDKOutput is invalid"));

    auto bg = output->GetMat("background");
    auto hat = output->GetMat("hats");
    auto upper = output->GetMat("upper_clothes");
    auto lower = output->GetMat("lower_clothes");

    float* bgData = (float *)bg->GetData();
    int ow = bg->GetWidth();
    int oh = bg->GetHeight();
    float* hatData = (float *)hat->GetData();
    float* upperData = (float *)upper->GetData();
    float* lowerData = (float *)lower->GetData();

    auto p1=std::chrono::steady_clock::now();

    long total = ow * oh;
    memset(p_next_mask, 0, sizeof(u_char) * total);

    unsigned char color = 0;
    int bgCount = 0;
    int hatCount = 0;
    int upperCount = 0;
    for (long i = 0; i < total; ++i) {
        float bgScore = bgData[i];
        float hatScore = hatData[i];
        float upperScore = upperData[i];
        float lowerScore = lowerData[i];
        if(bgScore >= hatScore && bgScore >= upperScore && bgScore>=lowerScore){
            color = 0; // bg
            bgCount++;
        }else if(hatScore >= bgScore && hatScore >= upperScore && hatScore>=lowerScore){
            color = 0xff; // hat
            hatCount ++;
        }else if(upperScore >= bgScore && upperScore >= hatScore && upperScore>=lowerScore){
            color = 0xaf; // upperScore
            upperCount ++;
        }else{
            color = 0x5f; // lowerScore
        }
        p_next_mask[i] = color;
    }
    float bgRate = bgCount/(float)total;
    float hatRate = hatCount/(float)total;
    float upRate = upperCount/(float)total;
    LOGE("detect output bg占比:%f hat:%f up:%f down:%f", bgRate, hatRate, upRate, 1.0f -bgRate-hatRate-upRate);

    auto* mask_human = OFD(total);

    TNN_NS::DeviceType dt = TNN_NS::DEVICE_ARM;
    TNN_NS::DimsVector r_dims = {1, 1, oh, ow};
    auto rMaskSize = std::make_shared<TNN_NS::Mat>(dt, TNN_NS::NGRAY, r_dims, mask_human);


    TNN_NS::DimsVector target_dims = {1, 1, orig_dims[2], orig_dims[3]};
    auto target_mat = std::make_shared<TNN_NS::Mat>(dt, TNN_NS::NGRAY, target_dims);
    status = Resize(rMaskSize, target_mat,TNNInterpLinear);
    if(status == TNN_OK){
        total = orig_dims[2] * orig_dims[3];
        //LOGE("target_dims h:%d, w:%d total:%d",orig_dims[2],orig_dims[3],total);
        u_char * alpha = (u_char*)target_mat->GetData();
        for (long i = 0; i < total; ++i) {
            float av = alpha[i];
            //maskData[i] = (alpha[i]<<16) | 0x7f000000;
            if(av>0xaf){
                maskData[i] = 0x7f7f0000;
            }else if(av > 0x5f){
                maskData[i] = 0x7f007f00;
            }else if(av >0){
                maskData[i] = 0x7f00007f;
            }
            // maskData[i] = 0x3fff0000;
        }
    }else{
        LOGE("detect output resize error!");
    }
    auto p2=std::chrono::steady_clock::now();
    double p_s=std::chrono::duration<double>(p2-p1).count(); //秒
    LOGE("Postprocess coast:%f",p_s);

    return status;
}

}
