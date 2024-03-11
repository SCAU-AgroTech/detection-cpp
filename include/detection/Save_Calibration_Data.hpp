#ifndef SAVE_CALIBRATION_DATA
#define SAVE_CALIBRATION_DATA

#include <iostream>
#include <opencv2/opencv.hpp>
#include <ros/ros.h>

namespace Calibration_Related
{

    // 该枚举用于指示保存的格式
    enum SaveFormat
    {
        YAML = 0,
        XML = 1,
    };

    // 该类主要用于保存OpenCV标定的数据文件（以yaml或xml格式进行保存）
    class SaveCalibrationData
    {
    public:
        // 主构造函数，需要传入保存数据的目录与格式
        SaveCalibrationData(const std::string &saveDirectory, Calibration_Related::SaveFormat &saveFormat);

        // 该函数用于执行保存标定数据的过程
        void SaveData(const std::string &fileName, const cv::Mat &cameraMatrix, const cv::Mat &distCoeffs);

        // 终结器
        ~SaveCalibrationData();

    private:
        // 保存标定数据的目录的字段
        std::string directory = "";

        // 保存标定数据格式的字段
        Calibration_Related::SaveFormat format = Calibration_Related::SaveFormat::YAML;
    };
}
#endif
