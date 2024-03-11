#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <ros/ros.h>
#include <math.h>

namespace Distance_Estimate
{
    // 该类主要用于进行双目的三维重建，获取点云/深度坐标
    class Depth3D
    {
    public:
        // 类的主构造函数，需要传入双目相机的各个内参矩阵
        Depth3D(cv::Mat &cameraMatrix1, cv::Mat &distCoeffs1, cv::Mat &cameraMatrix2, cv::Mat &distCoeffs2, cv::Mat &R, cv::Mat &T, cv::Mat &E, cv::Mat &F, cv::Size &imgSize);

        // 该函数用于生成双目视差图，需要传入双目相机的两个原始画面
        cv::Mat GetStereoDisparity(const cv::Mat &leftFrame, const cv::Mat &rightFrame);

        // 该函数用于从双目视差图中进行推断生成三维坐标点云图
        cv::Mat Estimate3DCoordinates(const cv::Mat &disparity);

        // 该函数用于比较两个不同方向而映射到三维空间的三维坐标点之间的欧几里德距离与阈值之间的关系
        bool IsPointCloseEnough(const cv::Vec3i &point1, const cv::Vec3i &point2, double threshold);

        // 终结器
        ~Depth3D();

    private:
        // 双目画面的双目外参矩阵们
        cv::Mat R1, R2;
        cv::Mat P1, P2;
        cv::Mat Q;
    };

#pragma region Depth3D类的函数定义

    // 类的主构造函数，需要传入双目相机的各个内参矩阵
    Depth3D::Depth3D(cv::Mat &cameraMatrix1, cv::Mat &distCoeffs1, cv::Mat &cameraMatrix2, cv::Mat &distCoeffs2, cv::Mat &R, cv::Mat &T, cv::Mat &E, cv::Mat &F, cv::Size &imgSize)
    {
        cv::stereoRectify(
            cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, imgSize, R, T,
            R1, R2, P1, P2, Q, cv::CALIB_ZERO_DISPARITY, 1.0, imgSize);
    }

    // 该函数用于生成双目视差图，需要传入双目相机的两个原始画面
    cv::Mat Depth3D::GetStereoDisparity(const cv::Mat &leftFrame, const cv::Mat &rightFrame)
    {
        // 图像预处理成灰度图
        cv::Mat leftGrayFrame, rightGrayFrame;
        cv::cvtColor(leftFrame, leftGrayFrame, cv::COLOR_BGR2GRAY);
        cv::cvtColor(rightFrame, rightGrayFrame, cv::COLOR_BGR2GRAY);
        // 建立立体匹配对象，create函数第1个参数范围[7, 9]，第2个参数范围[16, 32]
        cv::Ptr<cv::StereoBM> stereo = cv::StereoBM::create(9, 21);
        // 计算视差图像
        cv::Mat disparity;
        stereo->compute(leftGrayFrame, rightGrayFrame, disparity);
        return disparity;
    }

    // 该函数用于从双目视差图中进行推断生成三维坐标点云图
    cv::Mat Depth3D::Estimate3DCoordinates(const cv::Mat &disparity)
    {
        // 利用cv::reprojectImageTo3D()函数从视差图计算三维点云图
        cv::Mat pointCloudMatrix;
        cv::reprojectImageTo3D(disparity, pointCloudMatrix, Q, false, -1);
        return pointCloudMatrix;
    }

    // 该函数用于比较两个不同方向而映射到三维空间的三维坐标点之间的欧几里德距离与阈值之间的关系
    bool IsPointCloseEnough(const cv::Vec3i &point1, const cv::Vec3i &point2, double threshold)
    {
        double distance3D = cv::norm(point1, point2);
        return distance3D <= threshold;
    }

    // 终结器
    Depth3D::~Depth3D() {}

#pragma endregion Depth3D类的函数定义

    // 该类用于计算物体距离双目相机的距离，通过相似三角形的原理进行目标距离的计算
    class StereoDistCalc
    {
    public:
        // 类的主构造函数，需要传入物体在左、右相机中的中心点坐标，双目相机镜头的基线距离，相机镜头的焦距
        StereoDistCalc(const cv::Point &leftObjectPoint, const cv::Point &rightObjectPoint, double distBetweenCameras, double FocalLength);

        // 该函数用于计算距离，并返回距离值
        double DistCalc();

        // 终结器
        ~StereoDistCalc();

    private:
        // 中心点在不同画面中x方向上的像素差值
        double delta_X_Pixel;

        // 双目相机的基线距离
        double two_cameras_dist;

        // 相机的焦距
        double focal_length;
    };

#pragma region StereoDistCalc类的函数定义

    // 类的主构造函数，需要传入物体在左、右相机中的中心点坐标，双目相机镜头的基线距离，相机镜头的焦距
    StereoDistCalc::StereoDistCalc(const cv::Point &leftObjectPoint, const cv::Point &rightObjectPoint, double distBetweenCameras, double FocalLength)
    {
        // 开始填充private字段
        delta_X_Pixel = std::abs(leftObjectPoint.x - rightObjectPoint.x);
        two_cameras_dist = distBetweenCameras;
        focal_length = FocalLength;
    }

    // 该函数用于计算距离，并返回距离值
    double StereoDistCalc::DistCalc()
    {
        // 三角形相似原理计算距离Z
        // 需要注意各个量之间的单位统一！
        double Z = (focal_length * two_cameras_dist) / delta_X_Pixel;
        return Z;
    }

    // 终结器
    StereoDistCalc::~StereoDistCalc() {}

#pragma endregion StereoDistCalc类的函数定义
}