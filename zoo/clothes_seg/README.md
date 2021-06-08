## 配饰分割推理代码

### API

```

/**
    获取均值/方差等参数信息

    @return MatConvertParam: 参数信息
*/
MatConvertParam GetConvertParamForInput(std::string name = "")

/**
    模型推理后处理
    @param output: 模型推理结果
    
    @return Status: 状态码 TNN_OK 代表执行成功
*/
Status ProcessSDKOutput(std::shared_ptr<TNNSDKOutput> output)


```