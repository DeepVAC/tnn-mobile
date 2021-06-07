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

#include "BodyDetect.h"
#include <sys/time.h>
#include <cmath>
#include <cstring>
#include <chrono>


namespace TNN_NS {

BodyDetect::~BodyDetect() {
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

MatConvertParam BodyDetect::GetConvertParamForInput(std::string tag) {
    MatConvertParam input_cvt_param;
    input_cvt_param.scale = {1.0 / (255 * 60.225704), 1.0 / (255 * 60.16983), 1.0 / (255 * 62.044025), 0.0};//{1 / 0.5 / 255.f, 1 / 0.5 / 255.f, 1 / 0.5 / 255.f, 0.0};//
    input_cvt_param.bias  = {-116.05706 /  (255 * 60.225704), -122.36507 /  (255 * 60.16983), -127.62416 /(255 * 62.044025), 0.0};//{-0.5 / 0.5, -0.5 / 0.5, -0.5 / 0.5, 0.0};//
    input_cvt_param.reverse_channel = true;

    return input_cvt_param;
}

Status BodyDetect::Init(std::shared_ptr<TNNSDKOption> option_i) {
    Status status = TNN_OK;
    auto option = dynamic_cast<BodyDetectOption *>(option_i.get());
    RETURN_VALUE_ON_NEQ(!option, false, Status(TNNERR_PARAM_ERR, "TNNSDKOption is invalid"));

    status = TNNSDKSample::Init(option_i);
    RETURN_ON_NEQ(status, TNN_OK);

    auto input_dims = GetInputShape();
    option->input_height = input_dims[2];
    option->input_width  = input_dims[3];

    LOGE("input w:%d, h:%d",input_dims[2], input_dims[3]);
    auto size = sizeof(u_char)* input_dims[2]*input_dims[3];
    rmaskData = (u_char*)malloc(size);
    p_pre_mask = (u_char*)malloc(size);
    p_cur_mask = (u_char*)malloc(size);
    p_next_mask = (u_char*)malloc(size);

    return status;
}

std::shared_ptr<TNNSDKOutput> BodyDetect::CreateSDKOutput() {
    return std::make_shared<BodyDetectOutput>();
}

std::shared_ptr<Mat> BodyDetect::ProcessSDKInputMat(std::shared_ptr<Mat> input_image, std::string name) {
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

    if(fabs(input_width - humRectWidth)>5 || fabs(input_height - humRectHeight)>5){
        LOGE("Crop input image to detect human rect!");
        TNN_NS::DimsVector crop_dims = {1, dims[1], humRectHeight, humRectWidth}; // 转成3通道的
        auto crop_mat = std::make_shared<TNN_NS::Mat>(input_image->GetDeviceType(), input_image->GetMatType(), crop_dims);
        Crop(input_image, crop_mat, humRectLeft, humRectTop);
        input_image = crop_mat;
        input_width = humRectWidth;
        input_height = humRectHeight;
        this->orig_dims[2] = humRectHeight;
        this->orig_dims[3] = humRectWidth;
    }else{ // 微小差距就不调节了
        humRectTop = 0;
        humRectLeft = 0;
        humRectWidth = srcInputWidth;
        humRectHeight = srcInputHeight;
    }

    // 强制Resize到256*256
    if (target_dims.size() >= 4 && (input_height != target_dims[2] || input_width != target_dims[3])) {
        auto target_mat = std::make_shared<TNN_NS::Mat>(input_image->GetDeviceType(), input_image->GetMatType(), target_dims);
        scaleX = (float)target_dims[3] / (float) input_width;
        scaleY = (float)target_dims[2] / (float) input_height;
        auto status = Resize(input_image, target_mat,TNNInterpLinear);
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

u_char* BodyDetect::OFD(const int size) {
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

Status BodyDetect::ProcessSDKOutput(std::shared_ptr<TNNSDKOutput> output_) {
    Status status = TNN_OK;
    auto option = dynamic_cast<BodyDetectOption *>(option_.get());
    RETURN_VALUE_ON_NEQ(!option, false, Status(TNNERR_PARAM_ERR, "Body TNNOption is invalid"));

    auto output = dynamic_cast<BodyDetectOutput *>(output_.get());
    RETURN_VALUE_ON_NEQ(!output, false, Status(TNNERR_PARAM_ERR, "Body TNNSDKOutput is invalid"));

    std::shared_ptr<TNN_NS::Mat> out = output->GetMat("human");

    float *outData = (float *) out->GetData();
    int ow = out->GetWidth();
    int oh = out->GetHeight();
    long total = ow * oh;
    LOGE("output w:%d, h:%d", ow, oh);

    bool hasFound = false;
    memset(p_next_mask, 0, sizeof(u_char) * total);
    for (int i = 0; i < total; ++i) {
        if (outData[i] < m_thres) { // 阈值
            hasFound = true;
            p_next_mask[i] = 0xff;// alpha
        }
    }

    auto* mask_human = OFD(total);

    LOGE("isFound human:%d",hasFound?1:0);
    if (hasFound) {
        // 强制Resize到输入的大小
        TNN_NS::DeviceType dt = TNN_NS::DEVICE_ARM;
        TNN_NS::DimsVector r_dims = {1, 1, oh, ow}; // 转成3通道的
        auto rMaskSize = std::make_shared<TNN_NS::Mat>(dt, TNN_NS::NGRAY, r_dims, mask_human);

        TNN_NS::DimsVector target_dims = {1, 1, orig_dims[2], orig_dims[3]}; // 转成3通道的
        auto target_mat = std::make_shared<TNN_NS::Mat>(dt, TNN_NS::NGRAY, target_dims);
        auto status = Resize(rMaskSize, target_mat, TNNInterpLinear);
        if (status == TNN_OK) {
            total = orig_dims[2] * orig_dims[3];
            int height = orig_dims[2];
            int width = orig_dims[3];
            u_char *alpha = (u_char *) target_mat->GetData();
            for (int j = 0; j < height; ++j) {
                for (int i = 0; i < width; ++i) {
                    int index = (j + humRectTop) * srcInputWidth + (i + humRectLeft);
                    maskData[index] = (alpha[j * width + i] << 24) | 0xffffff;
                }
            }
        } else {
            return Status(TNNERR_NO_RESULT, "Not Found Body! Resize Failure!");
        }
    } else {
        return Status(TNNERR_NO_RESULT, "Not Found Body!");
    }

    return status;
}
}
