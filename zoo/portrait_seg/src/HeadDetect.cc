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

#include "HeadDetect.h"
#include <sys/time.h>
#include <cmath>
#include <cstring>

namespace TNN_NS {

HeadDetect::~HeadDetect() {

}

MatConvertParam HeadDetect::GetConvertParamForInput(std::string tag) {
    MatConvertParam input_cvt_param;
    input_cvt_param.scale = {1.0 / 127.5, 1.0 / 127.5, 1.0 / 127.5, 0.0};
    input_cvt_param.bias = {-1.0, -1.0, -1.0, 0.0};
    return input_cvt_param;
}

Status HeadDetect::Init(std::shared_ptr<TNNSDKOption> option_i) {
    Status status = TNN_OK;
    auto option = dynamic_cast<HeadDetectOption *>(option_i.get());
    RETURN_VALUE_ON_NEQ(!option, false, Status(TNNERR_PARAM_ERR, "TNNSDKOption is invalid"));

    status = TNNSDKSample::Init(option_i);
    RETURN_ON_NEQ(status, TNN_OK);

    auto input_dims = GetInputShape();
    option->input_height = input_dims[2];
    option->input_width  = input_dims[3];

    modelInputWidth = option->input_width;
    modelInputHeight = option->input_height;

    if(rmaskData!=NULL){
        free(rmaskData);
    }
    rmaskData = (u_char*)malloc(modelInputWidth * modelInputHeight * HEAD_INPUT_BATCH); // 固定3个batch维度

    return status;
}



std::shared_ptr<Mat> HeadDetect::ProcessSDKInputMat(std::shared_ptr<Mat> input_image, std::string name) {
    RETURN_VALUE_ON_NEQ(input_image->GetMatType(), N8UC3, nullptr);
    this->orig_dims = input_image->GetDims();
//    // save input image mat for merging
//    auto dims = input_image->GetDims();
//
//    auto status = TNN_OK;
//
//    auto target_dims = GetInputShape(name);
//    auto input_height = input_image->GetHeight();
//    auto input_width = input_image->GetWidth();
//
//    // 强制Resize到256*256
//    if (target_dims.size() >= 4 && (input_height != target_dims[2] || input_width != target_dims[3])) {
//
//        auto target_mat = std::make_shared<TNN_NS::Mat>(input_image->GetDeviceType(), input_image->GetMatType(), target_dims);
//
//        scaleX = (float)target_dims[3] / (float) input_width;
//        scaleY = (float)target_dims[2] / (float) input_height;
//
//        auto status = Resize(input_image, target_mat,TNNInterpLinear);
//
//        if (status == TNN_OK) {
//            return target_mat;
//        } else {
//            LOGE("Body Detect Resize error:%s\n", status.description().c_str());
//            return nullptr;
//        }
//    }else{
//        scaleX = 1;
//        scaleY = 1;
//    }
    return input_image;
}

std::shared_ptr<TNNSDKOutput> HeadDetect::CreateSDKOutput() {
    return std::make_shared<HeadDetectOutput>();
}

#define E 2.718281828459045

inline float a_sigmoid(float x){
//    y * (1.0f + exp(-x)) = 1;
//    y  + y * exp(-x) = 1;
//    exp(-x) = (1-y)/y;
//     x = -log((1-y)/y) / log(e);
    return -log((1.0-x)/x) / log(E);
}


Status HeadDetect::ProcessSDKOutput(std::shared_ptr<TNNSDKOutput> output_) {
    Status status = TNN_OK;
    // LOGE("HeadDetect ProcessSDKOutput !!! ");
    auto option = dynamic_cast<HeadDetectOption *>(option_.get());
    RETURN_VALUE_ON_NEQ(!option, false, Status(TNNERR_PARAM_ERR, "TNNOption is invalid"));

    auto output = dynamic_cast<HeadDetectOutput *>(output_.get());
    RETURN_VALUE_ON_NEQ(!output, false, Status(TNNERR_PARAM_ERR, "TNNSDKOutput is invalid"));


    auto output0 = output->GetMat("output"); // [1,1,256,256]
    float* outData = (float *)output0->GetData();

    int ow = output0->GetWidth();
    int oh = output0->GetHeight();
    int oc = output0->GetChannel();
    int outBatch = output0->GetBatch();
    // LOGE("Detect batch:%d",outBatch);
    int batch = faceList.size();
    long total = ow * oh;
    //u_char * rmask = (u_char*)malloc(sizeof(u_char)* total);
    //u_char* rmask = new u_char[total];
    memset(rmaskData, 0, HEAD_INPUT_BATCH*modelInputHeight*modelInputWidth);
    if(HEAD_INPUT_BATCH==1){
        batch = 1;
    }
    long allTotal = batch * total;
    bool hasFound = false;
    float threshold = a_sigmoid(0.5);
    // LOGE("Detect threshold:%f allTotal:%d",threshold,allTotal);
    for (int i = 0; i < allTotal; ++i) {
        if(outData[i] > threshold){ // 阈值
            hasFound = true;
            rmaskData[i] = 0xff;
        }
    }
    if(hasFound) {
        void * command_queue = nullptr;
        status = GetCommandQueue(&command_queue);
        if (status != TNN_NS::TNN_OK) {
            LOGE("HeadDetect output process getCommandQueue failed with:%s\n", status.description().c_str());
            return status;
        }
        if(HEAD_INPUT_BATCH==1){
            batch = 1;
        }
        auto t1=std::chrono::steady_clock::now();
        int total2 = srcInputHeight * srcInputWidth;
        for (int i = 0; i < batch; i++) {
            FaceInfo faceInfo = faceList[i];
            // 先resize到原始的大小
            TNN_NS::DeviceType dt = TNN_NS::DEVICE_ARM;
            TNN_NS::DimsVector r_dims = {1, 1, oh, ow}; // 转成3通道的
            auto rMaskSize = std::make_shared<TNN_NS::Mat>(dt, TNN_NS::NGRAY, r_dims, rmaskData + i*total); // + i*total

//            LOGE("Detect total:%d",total);
//            TNN_NS::DimsVector resize_dims = {1, 1, srcInputHeight, srcInputWidth};
//            auto resize_mat = std::make_shared<TNN_NS::Mat>(dt, TNN_NS::NGRAY, resize_dims);
//            auto status = Resize(rMaskSize, resize_mat, TNNInterpLinear);
//            if (status == TNN_OK) {
//                u_char *alpha = (u_char *) resize_mat->GetData();
//                for (int i = 0; i < total2; ++i) {
//                    //maskData[i] = (alpha[i] << 24) | 0x00ff00;
//                    maskData[i] = maskData[i] | ((alpha[i]>>1) << 24) | alpha[i] << 8;
//                }
//            }

            TNN_NS::DimsVector resize_dims = {1, 1, faceInfo.h, faceInfo.w};
            auto resize_mat = std::make_shared<TNN_NS::Mat>(dt, TNN_NS::NGRAY, resize_dims);
            auto status = Resize(rMaskSize, resize_mat, TNNInterpLinear);
            if (status == TNN_OK) {
                // 填充到原始大小
                CopyMakeBorderParam param1;
                param1.border_type = BORDER_TYPE_CONSTANT;
                param1.border_val = 0;
                param1.left = faceInfo.l;
                param1.top = faceInfo.t;
                param1.right = srcInputWidth - faceInfo.l - faceInfo.w;
                param1.bottom = srcInputHeight - faceInfo.t - faceInfo.h;

                TNN_NS::DimsVector dst_dims = {1, 1, srcInputHeight, srcInputWidth}; // 转成3通道的
                auto dst_mat = std::make_shared<TNN_NS::Mat>(dt, TNN_NS::NGRAY, dst_dims);

                status = MatUtils::CopyMakeBorder(*(resize_mat.get()), *(dst_mat.get()), param1, command_queue);
                if (status != TNN_NS::TNN_OK){
                    LOGE("Resize to Src Input Size failed with:%s\n", status.description().c_str());
                }

                u_char *alpha = (u_char *) dst_mat->GetData();
                if(isDetectedBody){
                    for (int i = 0; i < total2; ++i) {
                        maskData[i] = maskData[i] | ((alpha[i]>>1) << 24) | alpha[i] << 8; // alpha-> alpha[i]>>1    alpha = alpha[i]/2 ~ 7f
                        //if(alpha[i]>0)maskData[i] = (alpha[i] << 24) | 0x00ff00;
                    }
                }else {
                    for (int i = 0; i < total2; ++i) {
                        maskData[i] = ((alpha[i]>>1) << 24) | 0x00ff00;
                    }
                }
                //memcpy(target_mat->GetData(), maskData, sizeof(u_int) * orig_dims[2] * orig_dims[3]); // copy到mask
            } else {
                //free(rmask);
                return Status(TNNERR_NO_RESULT, "Not Found Body! Resize Failure!");
            }
        }
        auto t2=std::chrono::steady_clock::now();
        double dr_s=std::chrono::duration<double>(t2-t1).count(); //秒
        LOGE("Head Output process coast:%fs", dr_s);
    }
    return status;
}

}
