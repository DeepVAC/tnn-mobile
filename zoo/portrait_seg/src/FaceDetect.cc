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

#include "FaceDetect.h"
#include <sys/time.h>
#include <cmath>
#include <cstring>


namespace TNN_NS {

    FaceDetect::~FaceDetect() {

    }

    MatConvertParam FaceDetect::GetConvertParamForInput(std::string tag) {
        MatConvertParam input_cvt_param;
        input_cvt_param.scale = {1.0 / 127.5, 1.0 / 127.5, 1.0 / 127.5, 0.0};
        input_cvt_param.bias = {-1.0, -1.0, -1.0, 0.0};
        return input_cvt_param;
    }

    Status FaceDetect::Init(std::shared_ptr<TNNSDKOption> option_i) {
        Status status = TNN_OK;
        auto option = dynamic_cast<FaceDetectOption *>(option_i.get());
        RETURN_VALUE_ON_NEQ(!option, false, Status(TNNERR_PARAM_ERR, "TNNSDKOption is invalid"));

        status = TNNSDKSample::Init(option_i);
        RETURN_ON_NEQ(status, TNN_OK);

        auto input_dims = GetInputShape();
        option->input_height = input_dims[2];
        option->input_width = input_dims[3];

        //------------------------------------------------------------------
        if (option->input_width != calcPriorWidth || option->input_height != calcPriorHeight) {
            calcPriorWidth = option->input_width;
            calcPriorHeight = option->input_height;
            calcPriors();
        }

        return status;
    }


    std::shared_ptr<Mat>
    FaceDetect::ProcessSDKInputMat(std::shared_ptr<Mat> input_image, std::string name) {
        RETURN_VALUE_ON_NEQ(input_image->GetMatType(), N8UC3, nullptr);

        this->orig_dims = input_image->GetDims();
        auto dims = input_image->GetDims();
        auto target_dims = GetInputShape(name);
        auto input_height = input_image->GetHeight();
        auto input_width = input_image->GetWidth();

        inputWidth = input_width;
        inputHeight = input_height;

        if (target_dims.size() >= 4 &&
            (input_height != target_dims[2] || input_width != target_dims[3])) {
            auto target_mat = std::make_shared<TNN_NS::Mat>(input_image->GetDeviceType(),
                                                            input_image->GetMatType(), target_dims);

            int ow = target_dims[3];
            int oh = target_dims[2];
            int iw = dims[3];
            int ih = dims[2];
            int nw = ow;
            int nh = nw * ih / iw;
            scale = ow * 1.0 / static_cast<float>(iw);
            if (nh > oh) {
                nh = oh;
                nw = nh * iw / ih;
                scale = oh * 1.0 / static_cast<float>(ih);
            }
            dx = (ow - nw) / 2;
            dy = (oh - nh) / 2;


            auto status = ResizeAndMakeBorder(input_image, target_mat);
            if (status == TNN_OK) {
                return target_mat;
            } else {
                LOGE("ResizeAndMakeBorder error:%s\n", status.description().c_str());
                return nullptr;
            }
        } else {
            scale = 1;
            dx = 0;
            dy = 0;
        }
        return input_image;
    }

    std::shared_ptr<TNNSDKOutput> FaceDetect::CreateSDKOutput() {
        return std::make_shared<FaceDetectOutput>();
    }


    void FaceDetect::calcPriors() {
        priors.clear();

        std::vector<int> w_h_list = {calcPriorWidth, calcPriorHeight};
        const std::vector<float> strides = {8.0, 16.0, 32.0, 64.0};
        const int num_featuremap = 4;
        std::vector<std::vector<float>> featuremap_size;
        std::vector<std::vector<float>> shrinkage_size;
        const std::vector<std::vector<float>> min_boxes = {
                {10.0f,  16.0f,  24.0f},
                {32.0f,  48.0f},
                {64.0f,  96.0f},
                {128.0f, 192.0f, 256.0f}};
        for (auto size : w_h_list) {
            std::vector<float> fm_item;
            for (float stride : strides) {
                fm_item.push_back(ceil(size / stride));
            }
            featuremap_size.push_back(fm_item);
        }

        for (auto size : w_h_list) {
            shrinkage_size.push_back(strides);
        }
        /* generate prior anchors */
        for (int index = 0; index < num_featuremap; index++) {
            float scale_w = calcPriorWidth / shrinkage_size[0][index];
            float scale_h = calcPriorHeight / shrinkage_size[1][index];
            for (int j = 0; j < featuremap_size[1][index]; j++) {
                for (int i = 0; i < featuremap_size[0][index]; i++) {
                    float x_center = (i + 0.5) / scale_w;
                    float y_center = (j + 0.5) / scale_h;

                    for (float k : min_boxes[index]) {
                        float w = k / calcPriorWidth;
                        float h = k / calcPriorHeight;
                        priors.push_back(
                                {clip(x_center, 1), clip(y_center, 1), clip(w, 1), clip(h, 1)});
                    }
                }
            }
        }
    }

    void FaceDetect::nms(std::vector<FaceInfo> &input, std::vector<FaceInfo> &output, int type) {
        std::sort(input.begin(), input.end(),
                  [](const FaceInfo &a, const FaceInfo &b) { return a.score > b.score; });

        int box_num = input.size();

        std::vector<int> merged(box_num, 0);

        for (int i = 0; i < box_num; i++) {
            if (merged[i])
                continue;
            std::vector<FaceInfo> buf;

            buf.push_back(input[i]);
            merged[i] = 1;

            float h0 = input[i].y2 - input[i].y1 + 1;
            float w0 = input[i].x2 - input[i].x1 + 1;

            float area0 = h0 * w0;

            for (int j = i + 1; j < box_num; j++) {
                if (merged[j])
                    continue;

                float inner_x0 = input[i].x1 > input[j].x1 ? input[i].x1 : input[j].x1;
                float inner_y0 = input[i].y1 > input[j].y1 ? input[i].y1 : input[j].y1;

                float inner_x1 = input[i].x2 < input[j].x2 ? input[i].x2 : input[j].x2;
                float inner_y1 = input[i].y2 < input[j].y2 ? input[i].y2 : input[j].y2;

                float inner_h = inner_y1 - inner_y0 + 1;
                float inner_w = inner_x1 - inner_x0 + 1;

                if (inner_h <= 0 || inner_w <= 0)
                    continue;

                float inner_area = inner_h * inner_w;

                float h1 = input[j].y2 - input[j].y1 + 1;
                float w1 = input[j].x2 - input[j].x1 + 1;

                float area1 = h1 * w1;

                float score;

                score = inner_area / (area0 + area1 - inner_area);

                if (score > iou_threshold) {
                    merged[j] = 1;
                    buf.push_back(input[j]);
                }
            }
            switch (type) {
                case hard_nms: {
                    output.push_back(buf[0]);
                    break;
                }
                case blending_nms: {
                    float total = 0;
                    for (int i = 0; i < buf.size(); i++) {
                        total += exp(buf[i].score);
                    }
                    FaceInfo rects;
                    memset(&rects, 0, sizeof(rects));
                    for (int i = 0; i < buf.size(); i++) {
                        float rate = exp(buf[i].score) / total;
                        rects.x1 += buf[i].x1 * rate;
                        rects.y1 += buf[i].y1 * rate;
                        rects.x2 += buf[i].x2 * rate;
                        rects.y2 += buf[i].y2 * rate;
                        rects.score += buf[i].score * rate;
                    }
                    output.push_back(rects);
                    break;
                }
                default: {
                    printf("wrong type of nms.");
                    exit(-1);
                }
            }
        }
    }

    const std::string EMPTY_RESULT = "";

    Status FaceDetect::ProcessSDKOutput(std::shared_ptr<TNNSDKOutput> output_) {
        Status status = TNN_OK;
        //LOGE("FaceDetect ProcessSDKOutput !!! ");
        auto option = dynamic_cast<FaceDetectOption *>(option_.get());
        RETURN_VALUE_ON_NEQ(!option, false, Status(TNNERR_PARAM_ERR, "TNNOption is invalid"));

        auto output = dynamic_cast<FaceDetectOutput *>(output_.get());
        RETURN_VALUE_ON_NEQ(!output, false, Status(TNNERR_PARAM_ERR, "TNNSDKOutput is invalid"));

        auto output0 = output->GetMat("boxes"); // [1,4420,4,1]
        auto output1 = output->GetMat("scores"); // [1,4420,2,1]
        float *boxes = (float *) output0->GetData();
        float *scores = (float *) output1->GetData();

        int ow = output0->GetWidth();
        int oh = output0->GetHeight();
        int oc = output0->GetChannel();


        int num_anchors = oc; // 4420

        const float center_variance = 0.1;
        const float size_variance = 0.2;

        /* generate prior anchors finished */
        //int num_anchors = priors.size();
        //LOGE("num_anchors:%d",num_anchors); // num_anchors:4420
        //------------------------------------------------------------------

        bool isFound = false;
        std::vector<FaceInfo> bbox_collection;
        for (int i = 0; i < num_anchors; i++) {
            float score = scores[i * 2 + 1];
            if (score > score_threshold) {
                isFound = true;
                FaceInfo rects;
                float x_center = boxes[i * 4] * center_variance * priors[i][2] + priors[i][0];
                float y_center = boxes[i * 4 + 1] * center_variance * priors[i][3] + priors[i][1];
                float w = exp(boxes[i * 4 + 2] * size_variance) * priors[i][2];
                float h = exp(boxes[i * 4 + 3] * size_variance) * priors[i][3];

                rects.x1 = clip(x_center - w / 2.0, 1) * calcPriorWidth;
                rects.y1 = clip(y_center - h / 2.0, 1) * calcPriorHeight;
                rects.x2 = clip(x_center + w / 2.0, 1) * calcPriorWidth;
                rects.y2 = clip(y_center + h / 2.0, 1) * calcPriorHeight;
                rects.score = clip(score, 1);
                bbox_collection.push_back(rects);
            }
        }

        std::vector<FaceInfo> infoList;
        faceList.clear();
        if (isFound) {
            isFound = false;
//            FaceInfo maxScoreRect;
            if (bbox_collection.size() > 1) {
                //nms(bbox_collection, infoList, blending_nms);
                nms(bbox_collection, infoList, blending_nms);
                if(infoList.size()>3) {
                    // 找最大的3张人脸图
                    int box_num = infoList.size();
                    FaceInfo tmpFace;
                    FaceInfo face1 = infoList.at(0);
                    FaceInfo face2 = infoList.at(1);
                    FaceInfo face3 = infoList.at(2);
                    if(face1.score > face2.score){
                        tmpFace = face1;
                        face1 = face2;
                        face2 = tmpFace;
                    }
                    if(face3.score < face1.score){
                        tmpFace = face3;
                        face3 = face2;
                        face2 = face1;
                        face1 = tmpFace;
                    }else if(face3.score < face2.score){
                        tmpFace = face3;
                        face3 = face2;
                        face2 = tmpFace;
                    }
                    for (int i = 3; i < box_num; i++) {
                        FaceInfo rect = infoList[i];
                        if (rect.score >= face3.score) {
                            face1 = face2;
                            face2 = face3;
                            face3 = rect;
                        }else if(rect.score >= face2.score){
                            face1 = face2;
                            face2 = rect;
                        }else if(rect.score >= face1.score){
                            face1 = rect;
                        }
                    }
                    infoList.clear();
                    infoList.push_back(face3);
                    infoList.push_back(face2);
                    infoList.push_back(face1);
                }
            } else {
                //isFound = true;
                //maxScoreRect = bbox_collection.at(0);
                infoList.push_back(bbox_collection.at(0));
            }
            //LOGE("Found infoList size:%d", infoList.size());

            // 裁切出脸的图片
            if (infoList.size()>0) {
                int box_num = infoList.size();
                //LOGE("Found faceList size:%d", box_num);
                scale = 1.0f / scale;

                float top_shift = 0;
                float amplifier = 2.5;
                for (int k = 0; k < box_num; k++) {
                    FaceInfo rect = infoList[k];
                    // 还原比例
                    //LOGE("Face %d x1:%f, y1:%f, x2:%f, y2:%f dx:%d dy:%d scale:%f", k, rect.x1, rect.y1, rect.x2, rect.y2, dx, dy, scale);
                    int x1 = (int)((rect.x1 - dx) * scale);
                    int y1 = (int)((rect.y1 - dy) * scale);
                    int x2 = (int)((rect.x2 - dx) * scale);
                    int y2 = (int)((rect.y2 - dy) * scale);

                    // 裁剪出脸部的最大边框
                    int w = x2 - x1;
                    int h = y2 - y1;
                    if (w < h) {
                        w = h;
                    }
//                    w = (int) (1.5 * w); // 扩大
//                    h = w;

                    int cx = (x2 + x1) / 2;
                    int cy = (y2 + y1) / 2;
//                    cy = (int) (0.8 * cy); //向上偏移
                    cy = y1 + (h / 2 * (1 + top_shift));
                    w = MAX(w, h) * amplifier;
                    h = w;


                    x1 = MAX(cx - w / 2, 0);
                    y1 = MAX(cy - h / 2, 0);

                    if (x1 + w > inputWidth) {
                        w = inputWidth - x1;
                    }
                    if (y1 + h > inputHeight) {
                        h = inputHeight - y1;
                    }
                    rect.x1 = x1;
                    rect.y1 = y1;
                    rect.x2 = x1 + w;
                    rect.y2 = x1 + h;
                    rect.l = x1;
                    rect.t = y1;
                    rect.w = w;
                    rect.h = h;
                    faceList.push_back(rect);
                    //LOGE("2 Face %d x1:%f, y1:%f, x2:%f, y2:%f dx:%d dy:%d scale:%f", k, rect.x1, rect.y1, rect.x2, rect.y2, dx, dy, scale);
                    //LOGE("2 Face %d l:%d, t:%d, w:%d, h:%d dx:%d dy:%d scale:%f", k, rect.l, rect.t, rect.w, rect.h, dx, dy, scale);
                }
            }else{
                return Status(TNNERR_NO_RESULT, "Not Found Face!");
            }
        }else{
            return Status(TNNERR_NO_RESULT, "Not Found Face!");
        }
        return status;
    }
}