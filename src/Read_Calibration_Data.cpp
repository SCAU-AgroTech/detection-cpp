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

#pragma region ReadCalibrationData的函数定义

    // 类的主构构造函数，需要传入标定数据文件所在的路径
    ReadCalibrationData::ReadCalibrationData(const std::string &dataPath)
    {
        cv::FileStorage calibFile(dataPath, cv::FileStorage::READ);
        // 判空
        if (!calibFile.isOpened())
        {
            std::cerr << "读取标定数据失败！文件路径：" << dataPath << std::endl;
            return;
        }
        // 若打开成功即开始读取‘
        calibFile["camera_matrix"] >> cameraMatrix;
        calibFile["distortion_coefficients"] >> distCoeffs;
        // 释放非托管资源
        calibFile.release();
    }

    // 该函数用于从标定数据中获取CamMat内参矩阵
    cv::Mat ReadCalibrationData::GetCameraMatrix()
    {
        return cameraMatrix;
    }

    // 该函数用于从标定数据中获取DistCoeffs内参矩阵
    cv::Mat ReadCalibrationData::GetDistCoeffs()
    {
        return distCoeffs;
    }

    // 终结器
    ReadCalibrationData::~ReadCalibrationData() {}
#pragma endregion ReadCalibrationData的函数定义
}