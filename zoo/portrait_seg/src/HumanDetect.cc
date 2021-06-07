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

#include "HumanDetect.h"
#include <sys/time.h>
#include <cmath>

namespace TNN_NS {

HumanDetect::~HumanDetect() {

}

MatConvertParam HumanDetect::GetConvertParamForInput(std::string tag) {
    MatConvertParam input_cvt_param;
    input_cvt_param.scale = {1.0 / 127.5, 1.0 / 127.5, 1.0 / 127.5, 0.0};
    input_cvt_param.bias = {-1.0, -1.0, -1.0, 0.0};
    return input_cvt_param;
}

Status HumanDetect::Init(std::shared_ptr<TNNSDKOption> option_i) {
    Status status = TNN_OK;
    auto option = dynamic_cast<HumanDetectOption *>(option_i.get());
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

std::shared_ptr<Mat> HumanDetect::ProcessSDKInputMat(std::shared_ptr<Mat> input_image, std::string name) {
    RETURN_VALUE_ON_NEQ(input_image->GetMatType(), N8UC3, nullptr);
    this->orig_dims = input_image->GetDims();
    // save input image mat for merging
    auto dims = input_image->GetDims();

    auto status = TNN_OK;

    auto target_dims = GetInputShape(name);
    auto input_height = input_image->GetHeight();
    auto input_width = input_image->GetWidth();

    // 强制Resize到128*128
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

std::shared_ptr<TNNSDKOutput> HumanDetect::CreateSDKOutput() {
    return std::make_shared<HumanDetectOutput>();
}

Status HumanDetect::ProcessSDKOutput(std::shared_ptr<TNNSDKOutput> output_) {
    Status status = TNN_OK;
    // LOGE("HeadDetect ProcessSDKOutput !!! ");
    auto option = dynamic_cast<HumanDetectOption *>(option_.get());
    RETURN_VALUE_ON_NEQ(!option, false, Status(TNNERR_PARAM_ERR, "TNNOption is invalid"));

    auto output = dynamic_cast<HumanDetectOutput *>(output_.get());
    RETURN_VALUE_ON_NEQ(!output, false, Status(TNNERR_PARAM_ERR, "TNNSDKOutput is invalid"));


    auto centerPos = output->GetMat("739"); // [1,2,32,32]
    float* centerPosData = (float *)centerPos->GetData();

    int ow = centerPos->GetWidth();
    int oh = centerPos->GetHeight();
    int oc = centerPos->GetChannel();
    int outBatch = centerPos->GetBatch();

    int x = -1;
    int y = -1;
    float maxScore = 0;
    int total = ow * oh;
    for(int i=0;i<oh;i++){
        for (int j = 0; j < ow; ++j) {
            float score = centerPosData[i*ow+j];
            if(score > 0.5 && score > maxScore){ //
                maxScore = score;
                x = j;
                y = i;
            }
            score = centerPosData[i*ow+j + total];
            if(score > 0.5 && score > maxScore){ // score > 0.5 &&
                maxScore = score;
                x = j;
                y = i;
            }
        }
    }
    LOGE("Found human maxScore:%f", maxScore);

    if(maxScore>0.5){
        //找到
        auto offset = output->GetMat("743"); // [1,2,32,32]
        float* offsetData = (float *)offset->GetData();


        auto size = output->GetMat("747"); // [1,2,32,32]
        float* sizeData = (float *)size->GetData();

        cropWidth = sizeData[y*ow + x];
        cropHeight = sizeData[y*ow + x + total];
        cropX = x + offsetData[y*ow + x] - cropWidth/2.0f;
        cropY = y + offsetData[y*ow + x + total] - cropHeight/2.0f;

        cropWidth = MAX(cropWidth, cropHeight) * 1.1f;
        cropHeight = cropWidth;

        if(cropX < 0){
            cropWidth = cropWidth + cropX;
            cropX = 0;
        }
        if(cropY < 0){
            cropHeight = cropHeight + cropY;
            cropY = 0;
        }
        float resizeSize = 32.0f;
        if(cropX + cropWidth > resizeSize){
            cropWidth = resizeSize - cropX;
        }
        if(cropY + cropHeight > resizeSize){
            cropHeight = resizeSize - cropY;
        }
        cropX = cropX / resizeSize * srcInputWidth;
        cropY = cropY / resizeSize * srcInputHeight;
        cropWidth = cropWidth / resizeSize * srcInputWidth;
        cropHeight = cropHeight / resizeSize * srcInputHeight;
        LOGE("Find Human at (%f,%f,%f,%f)",cropX, cropY, cropWidth, cropHeight);
    }else{
        cropX = 0;
        cropY = 0;
        cropWidth = 0;
        cropHeight = 0;
    }
    LOGE("ow:%d oh:%d oc:%d batch:%d",ow,oh,oc,outBatch);
    return status;
}

}
