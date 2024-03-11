#ifndef DETECTION
#define DETECTION
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <math.h>
#include <string.h>
#include <sys/stat.h>

namespace Detection_Related
{
    // 默认的输入图像宽度
    const int INPUT_WIDTH = 640;

    // 默认的输入图像高度
    const int INPUT_HEIGHT = 640;

    // 默认的分数阈值
    const float SCORE_THRESHOLD = 0.55F;

    // 默认的NMS过滤阈值
    const float NMS_THRESHOLD = 0.45F;

    // 默认的置信度的阈值
    const float CONFIDENCE_THRESHOLD = 0.45F;

    // YOLO模型的维度
    const int MODEL_DIMENSIONS = 85;

    // YOLO模型矩阵的条目量
    const int YOLO_ROWS = 25200;

    // 字体的Scale
    const float FONT_SCALE = 0.70F;

    // 默认的scale factor
    const double SCALE_FACTOR = 1.0 / 255.0;

    // 字体的Interface
    const float FONT_FACE = cv::FONT_HERSHEY_SIMPLEX;

    // 字体的Thickness
    const int THICKNESS = 1;

    // 使用cv::Scalar()定义的一些颜色Scalar常量
    // 黑色
    const cv::Scalar BLACK = cv::Scalar(0, 0, 0);

    // 蓝色
    const cv::Scalar BLUE = cv::Scalar(255, 178, 50);

    // 红色
    const cv::Scalar RED = cv::Scalar(0, 0, 255);

    // 黄色
    const cv::Scalar YELLOW = cv::Scalar(0, 255, 255);

    // 用于表示目标识别框的结构体
    struct BoundingBox
    {
        // 中心点x坐标
        int x;

        // 中心点y坐标
        int y;

        // 目标框的宽度
        int width;

        // 目标框的高度
        int height;

        // 目标框的序号
        int index;
    };

    // 用于检测目标的类
    class DetectObject
    {
    public:
        // 经过label后的图像
        cv::Mat labeledImg;

        // 单幅画面中所有bounding box的抽象的向量
        std::vector<BoundingBox> bBoxVector;

        // 经过NMS非最大抑制过滤等算法后单幅画面的bouding box的OpenCV数据结构抽象的向量
        std::vector<cv::Rect> bBoxRectVector;

        /* public的实例函数 */

        // 构造函数，需要传入onnx模型的绝对路径、图像的cv::Mat类型矩阵以及训练模型时候设定的所有类型名
        DetectObject(std::string &modelPath, cv::Mat imageMatrix, const std::vector<std::string> &classesNames);

        // 构造函数，需要传入onnx模型的绝对路径、图像的绝对路径以及训练模型时候设定的所有类型名
        // DetectObject(std::string& modelPath, std::string& imagePath, const std::vector<std::string>& classesNames);

        // 核心函数，用于实现目标识别检测与数据处理的主函数
        bool Detect();

        // 获取最终目标框中大小最大的bounding box在vector中的序号
        unsigned int GetMaxBoxIndex();

        // 终结器
        ~DetectObject();

    private:
        // 模型的绝对路径
        std::string model_path;

        // 图像的绝对路径
        // std::string image_path;

        // 图像矩阵
        cv::Mat img_matrix;

        // 所有类别ID的向量
        std::vector<std::string> class_names;

        // 指示是否从绝对路径获取的图像
        // bool is_image_from_path;

        /* private的函数 */
        /* static函数 */

        // 在特定图像上的特定绘制矩形边框，且绘制特定的ASCII字符文字
        static bool DrawLabel(cv::Mat &inputImage, std::string &label, int left, int top);

        // 加载神经网络及其初始化
        static cv::dnn::Net LoadNet(const std::string &path);

        // 判断类内部的图像是否已经加载成功
        static bool IsImageMatrixLoaded(const cv::Mat &img);

        // 判断类内部的模型路径是否为有效路径
        static bool IsModelPathValid(const std::string &path);

        // 在进行前向推理前对图像进行预处理
        static bool PreProcess(cv::Mat &inputImage, cv::dnn::Net &model);

        // 进行推理
        static std::vector<cv::Mat> DetectForward(cv::dnn::Net &model);

        // 对推理数据进行读取暂存，并对多重相近且重叠的bounding box进行NMS非最大抑制的过滤处理
        // 读取onnx格式的神经网络模型的核心函数
        static cv::Mat PostProcess(cv::Mat inputImage, std::vector<cv::Mat> &outputs, const std::vector<std::string> &classesNames, std::vector<BoundingBox> &bBoxVect, std::vector<cv::Rect> &bBoxRectVect);
    };

    // 该类主要用于匹配两张图像上的相同特征点并维护计数
    class MatchFeature
    {
    public:
        // 类构造函数，需要传入需要比对的两个图像。请勿传入双目相机的两个原始图像！
        MatchFeature(cv::Mat image1, cv::Mat image2);

        // 该函数通过ORB特征匹配法进行特征匹配
        void DoMatch(double distanceThreshold = 30.0, long unsigned int goodMatchesThreshold = 10, int H_MatrixThreshold = 20);
        
        // 指示是否匹配
        bool IsMatched();

        // 指示H单应性矩阵是否有效
        bool IsH_MatrixValid();

        // 该函数用于将两张经过ORB特征匹配的图匹配描述点之间连线显示
        cv::Mat GetMatchesDrawing();

        // 终结器
        ~MatchFeature();

    private:
        cv::Mat leftImage, rightImage, matchesDrawing;
        bool isMatched;
        bool isH_MatrixValid;
    };
}

#endif