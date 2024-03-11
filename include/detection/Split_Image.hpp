#ifndef SPLIT_IMAGE
#define SPLIT_IMAGE

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <ros/ros.h>

namespace Detection_Related
{
    //  该类提供分割双目相机合并画面的一些API
    class SplitImage
    {
    public:
        // 类的主构造函数，需要传入一个图像
        SplitImage(const cv::Mat &originalFrame);

        // 该函数返回双目合并画面的左相机画面
        cv::Mat GetLeftImage();

        // 该函数返回双目合并画面的右相机画面
        cv::Mat GetRightImage();

        // 终结器
        ~SplitImage();

    private:
        // 双目相机原始画面
        cv::Mat frame;

        // 双目相机左相机画面
        cv::Mat leftImageMatrix;

        // 双目相机右相机画面
        cv::Mat rightImageMatrix;

        // 指示是否已经进行分割
        bool isSplited;

        // 内部的分割函数
        void Split();
    };

    // 该函数用于提取特定区域的画面，需要传入一个区域的左上角x,y坐标与宽高度
    cv::Mat ExtractImage(cv::Mat originalImage, int x1, int y1, int width, int height);

    // 该函数用于将图像调整至一个特定的尺寸，不足的像素点使用特定的颜色进行填充
    // 源见：https://github.com/Hexmagic/ONNX-yolov5/blob/master/src/test.cpp
    cv::Mat ResizeImage2New(cv::Mat &image, const cv::Size &newShapeSize, const cv::Scalar &color, bool isAuto, bool isScaleFill, bool isScaleUp, int stride);
}
#endif
