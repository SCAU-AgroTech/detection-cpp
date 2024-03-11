#ifndef READ_CALIBRATION_DATA
#define READ_CALIBRATION_DATA

#include <iostream>
#include <opencv2/opencv.hpp>

namespace Calibration_Related
{
    class ReadCalibrationData
    {
    public:
        // 类的主构构造函数，需要传入标定数据文件所在的路径
        ReadCalibrationData(const std::string &dataPath);

        // 该函数用于从标定数据中获取CamMat内参矩阵
        cv::Mat GetCameraMatrix();

        // 该函数用于从标定数据中获取DistCoeffs内参矩阵
        cv::Mat GetDistCoeffs();

        // 终结器
        ~ReadCalibrationData();

    private:
        // 内参矩阵 cameraMatrix
        cv::Mat cameraMatrix;

        // 内参矩阵 distCoeffs
        cv::Mat distCoeffs;
    };
}

#endif
