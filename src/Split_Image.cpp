#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <math.h>
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

#pragma endregion SplitImage类的函数定义

    /* 先定义private域的函数，再定义public域的函数 */
    void SplitImage::Split()
    {
        // 取中点
        const int SPLIT_POINT = frame.cols / 2;
        // 判空
        if (frame.empty())
        {
            std::cerr << "目标识别过程：无可用于分割的图像！" << std::endl;
            return;
        }
        cv::Rect leftFrame(0, 0, SPLIT_POINT, frame.rows);
        cv::Rect rightFrame(SPLIT_POINT, 0, frame.cols - SPLIT_POINT, frame.rows);

        cv::Mat leftImage = frame(leftFrame);
        cv::Mat rightImage = frame(rightFrame);

        leftImageMatrix = leftImage;
        rightImageMatrix = rightImage;

        isSplited = true;
    }

    SplitImage::SplitImage(const cv::Mat &originalFrame)
    {
        // 设定初始状态
        isSplited = false;
        frame = originalFrame;
        Split();
    }

    cv::Mat SplitImage::GetLeftImage()
    {
        // 若已经分割则返回分割结果，若无则返回一个空矩阵
        return isSplited ? leftImageMatrix : cv::Mat();
    }

    cv::Mat SplitImage::GetRightImage()
    {
        return isSplited ? rightImageMatrix : cv::Mat();
    }

    SplitImage::~SplitImage() {}

#pragma endregion SplitImage类的函数定义

#pragma region 游离的函数

    // 该函数用于提取特定区域的画面，需要传入一个区域的左上角x,y坐标与宽高度
    cv::Mat ExtractImage(cv::Mat originalImage, int x1, int y1, int width, int height)
    {
        cv::Rect roi = cv::Rect(x1, y1, width, height);
        return (originalImage(roi));
    }

    // 该函数用于将图像调整至一个特定的尺寸，不足的像素点使用特定的颜色进行填充
    // 源见：https://github.com/Hexmagic/ONNX-yolov5/blob/master/src/test.cpp
    cv::Mat ResizeImage2New(cv::Mat &image, const cv::Size &newShapeSize, const cv::Scalar &color, bool isAuto, bool isScaleFill, bool isScaleUp, int stride)
    {
        float width = image.cols;
        float height = image.rows;
        // 找到新旧图像缩放比例的最小值
        float scl = cv::min(newShapeSize.width / width, newShapeSize.height / height);
        // 是否允许放大（否）
        if (!isScaleUp)
        {
            scl = cv::min(scl, 1.0f);
        }
        // 找到要调整到的目标大小
        int new_unpad_W = int(std::round(width * scl));
        int new_unpad_H = int(std::round(height * scl));
        // 计算变化量
        int dW = newShapeSize.width - new_unpad_W;
        int dH = newShapeSize.height - new_unpad_H;
        // 是否自动判断缩放到stride的整数比（是）
        if (isAuto)
        {
            dW %= stride;
            dH %= stride;
        }
        // （上、下）各取一半
        dW /= 2, dH /= 2;
        // 开始操作
        cv::Mat dst;
        cv::resize(image, dst, cv::Size(new_unpad_W, new_unpad_H), 0.0, 0.0, cv::INTER_LINEAR);
        // 修正误差
        int top = int(std::round(dH - 0.1));
        int buttom = int(std::round(dH + 0.1));
        int left = int(round(dW - 0.1));
        int right = int(round(dW + 0.1));
        // 拷贝Border
        cv::copyMakeBorder(dst, dst, top, buttom, left, right, cv::BORDER_CONSTANT, color);
        return dst;
    }

#pragma endregion 游离的函数
}
