# Agrotech视觉与软件组 旧版C++目标识别定位ROS功能包
###
### 该功能包为视觉与软件组初期开发目标识别定位&目标测距功能时使用C++17编写的ROS-noetic功能包，目前已经全部改用Python3进行重构和改写，此仓库与功能包已存档。

### 功能包目录简介：
> CalibrationData：存放用于相机标定的数据
###
> CalibrationImages：存放相机拍摄的照片，用于标定
###
> inlcude/detection：存放代码头文件
>> 注：头文件内容与存放细则详见源文件命名与存放规则
###
> models：存放经pytorch训练导出后的onnx格式的神经网络模型
###
> msg：存放ROS的话题消息文件
###
> src：存放代码的源文件
>> AppDomain：主函数所在的源文件 <br/>
>> Calibration：标定功能的实现所在的源文件 <br/>
>> Detection：实现读取神经网络模型、推理与图像匹配功能的源文件 <br/>
>> Distance_Estimate：实现目标的距离测算，以及（简单的）立体视觉&深度匹配功能的源文件 <br/>
>> Read_Calibration_Data.cpp：读取标定数据函数所在的源文件 <br/>
>> Save_Calibration_Data.cpp：保存标定数据函数所在的源文件 <br/>
>> Split_Image.cpp：实现对图像进行局部/整体分割提取处理功能的源文件 <br/>
### <br/>
### <br/>
### Copyright (C) 2024 Agrotech, Dept. Vision and Software