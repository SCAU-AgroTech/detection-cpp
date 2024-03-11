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

#pragma region DetectObject类的函数定义

    /* public可见级的函数定义 */

    // 构造函数，需要传入onnx模型的绝对路径、图像的cv::Mat类型矩阵以及训练模型时候设定的所有类型名
    DetectObject::DetectObject(std::string &modelPath, cv::Mat imageMatrix, const std::vector<std::string> &classesNames)
    {
        // 对字段进行初始化
        bBoxVector = std::vector<BoundingBox>();
        bBoxRectVector = std::vector<cv::Rect>();
        labeledImg = cv::Mat();
        model_path = modelPath;
        img_matrix = imageMatrix;
        class_names = classesNames;
    }

    // 核心函数，用于实现目标识别检测与数据处理的主函数
    bool DetectObject::Detect()
    {
        try
        {
            // 先判空
            if (!(DetectObject::IsImageMatrixLoaded(img_matrix) && DetectObject::IsModelPathValid(model_path)))
            {
                throw "目标检测的类构造时图像未正确加载或模型路径不合法";
            }
            // 加载模型
            cv::dnn::Net model = DetectObject::LoadNet(model_path);
            // 图像预处理
            DetectObject::PreProcess(img_matrix, model);
            // 获取推理矩阵
            auto resultMatVect = DetectObject::DetectForward(model);
            // 获取推理数据并且拿到处理后的图像
            cv::Mat labeledImage = DetectObject::PostProcess(img_matrix, resultMatVect, class_names, bBoxVector, bBoxRectVector);
            labeledImg = labeledImage;
            // 返回
            return true;
        }
        catch (const std::exception &e)
        {
            std::cerr << "目标检测时出现错误：" << e.what() << '\n';
            return false;
        }
    }

    // 获取最终目标框中大小最大的bounding box在vector中的序号
    unsigned int DetectObject::GetMaxBoxIndex()
    {
        /*
         *   按理来说，为了提高查找效率，应该使用平衡二叉树查找这样时间复杂度为O(log2(N))的查找算法
         *   但由于开发效率与前期调试效率等问题，暂时使用顺序查找
         *   后续重构
         */
        int tempIdx = 0, tempSum = 0;
        for (long unsigned int i = 0; i < bBoxVector.size(); i++)
        {
            // 若总和大小（宽度+高度）小于先前的计数，并且点的范围在图像大小范围内
            if ((tempSum < bBoxVector[i].width + bBoxVector[i].height) && (bBoxVector[i].x <= INPUT_WIDTH) && (bBoxVector[i].y <= INPUT_HEIGHT))
            {
                // 更新最大总和大小和index号
                tempSum = bBoxVector[i].width + bBoxVector[i].height;
                tempIdx = i;
            }
        }
    }

    // 终结器
    DetectObject::~DetectObject() {}

    /* private可见级的函数定义 */

    // 在特定图像上的特定绘制矩形边框，且在bouding box的顶部显示文字（ASCII字符集）
    bool DetectObject::DrawLabel(cv::Mat &inputImage, std::string &label, int left, int top)
    {
        try
        {
            // 在bouding box的顶部显示文字
            int baseLine;
            cv::Size labelSize = cv::getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS, &baseLine);
            top = cv::max(top, labelSize.height);
            // 左上角顶点坐标
            cv::Point topLeftPoint = cv::Point(left, top);
            // 右下角顶点坐标
            cv::Point bottomRightPoint = cv::Point(left + labelSize.width, top + labelSize.height + baseLine);
            // 绘制黄色矩形框
            cv::rectangle(inputImage, topLeftPoint, bottomRightPoint, YELLOW, cv::FILLED);
            // 在矩形框上方绘制文字
            cv::putText(inputImage, label, cv::Point(left, top + labelSize.height), FONT_FACE, FONT_SCALE, YELLOW, THICKNESS);
            return true;
        }
        catch (const std::exception &e)
        {
            // 打印报错
            std::cerr << "在绘制bounding box时候出现错误：" << e.what() << '\n';
            return false;
        }
    }

    // 加载神经网络及其初始化
    cv::dnn::Net DetectObject::LoadNet(const std::string &path)
    {
        try
        {
            cv::dnn::Net net = cv::dnn::readNetFromONNX(path);
            return net;
        }
        catch (const std::exception &e)
        {
            std::cerr << "加载ONNX格式的神经网络模型时出现错误：" << e.what() << '\n';
            // 直接退出，返回错误码
            // exit();
        }
    }

    // 判断类内部的图像是否已经加载成功
    bool DetectObject::IsImageMatrixLoaded(const cv::Mat &img)
    {
        try
        {
            return !img.empty();
        }
        catch (const std::exception &e)
        {
            std::cerr << "内部图像加载验证时出错：" << e.what() << '\n';
            return false;
        }
    }

    // 判断类内部的模型路径是否为有效路径
    bool DetectObject::IsModelPathValid(const std::string &path)
    {
        try
        {
            // 利用stat.h中的stat结构体API判断模型文件是否存在
            struct stat buffer;
            return (stat(path.c_str(), &buffer) == 0);
        }
        catch (const std::exception &e)
        {
            std::cerr << "内部模型路径有效性验证时出错：" << e.what() << '\n';
            return false;
        }
    }

    // 在进行前向推理前对图像进行预处理
    bool DetectObject::PreProcess(cv::Mat &inputImage, cv::dnn::Net &model)
    {
        try
        {
            // 转换为blob
            cv::Mat blob;
            cv::dnn::blobFromImage(inputImage, blob, SCALE_FACTOR, cv::Size(INPUT_WIDTH, INPUT_HEIGHT), cv::Scalar(), true, false);
            model.setInput(blob);
            // 默认采用CPU进行推理
            return true;
        }
        catch (const std::exception &e)
        {
            std::cerr << "进行前向推理前的预处理过程出现错误：" << e.what() << '\n';
            return false;
        }
    }

    // 进行推理
    std::vector<cv::Mat> DetectObject::DetectForward(cv::dnn::Net &model)
    {
        try
        {
            // 推理结果的矩阵
            std::vector<cv::Mat> outputs;
            // 进行推理
            model.forward(outputs, model.getUnconnectedOutLayersNames());
        }
        catch (const std::exception &e)
        {
            std::cerr << "进行前向推理时候发生错误：" << e.what() << '\n';
            // 直接退出，返回错误码
            // exit();
        }
    }

    // 对推理数据进行读取暂存，并对多重相近且重叠的bounding box进行NMS非最大抑制的过滤处理
    // 读取onnx格式的神经网络模型的核心函数
    // 对于inputImage参数，请不要使用左值引用以避免修改被传参量的值
    cv::Mat DetectObject::PostProcess(cv::Mat inputImage, std::vector<cv::Mat> &outputs, const std::vector<std::string> &classesNames, std::vector<BoundingBox> &bBoxVect, std::vector<cv::Rect> &bBoxRectVect)
    {
        /*  说明：

         *    这个函数返回的对象是一个二维数组，输出取决于输入的大小。例如，当默认输入大小为 640 时，我们得到一个大小为 25200 × 85（行和列）的 2D 矩阵
         *    行表示检测次数
         *    因此，每次网络运行时，它都会预测 25200 个边界框。每个边界框都有一个包含 85 个条目的一维数组，用于说明检测的质量，此信息足以筛选出所需的检测

         *    数据的排列大概长这样：
         *        X   |   Y   |   W   |   H   |   目标置信度（Confidence）  |   （最多）80个类别物体的得分权重
         *    ^~~~~~~~^~~~~~~~^~~~~~~~^~~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
         *    数据第0位 数据第1位 数据第2位 数据第3位         数据第4位                                 数据第5-85位

         *    网络根据 blob 的输入大小（默认 640 × 640）生成输出坐标。因此，应将坐标乘以调整大小系数以获得实际输出

         */

        // 初始化一些向量用于后续的数据处理

        // 类别的ID号
        std::vector<int> class_ids;
        // 置信度
        std::vector<float> confidences;
        // 以cv::Rect形式抽象出的bounding box;
        std::vector<cv::Rect> bBoxes;

        // 计算缩放因子
        // 由于模型是基于blob输入的640 × 640尺寸图像进行推理的，因此数据也是640 × 640尺寸坐标系下的
        // 因此我们需要计算缩放比，在最后的时候逆推回原图形的x, y ,w, h参量
        float x_factor = (1.0F * inputImage.cols) / (1.0F * INPUT_WIDTH);
        float y_factor = (1.0F * inputImage.rows) / (1.0F * INPUT_HEIGHT);

        // 获取数据指针
        // “为什么要使用float*而不使用usigned char* => usigned int*?”
        float *data = (float *)outputs[0].data;

        try
        {
            // 开始从0行遍历到25200行
            for (int row = 0; row < YOLO_ROWS; row++)
            {
                // 获取置信度
                float confidence = data[4];
                // 筛掉低于置信度要求的数据
                if (confidence >= CONFIDENCE_THRESHOLD)
                {
                    float *classes_scores = data + 5;
                    // 创建一个1 × {类别数量}规格的矩阵用来存储80个类别的权重分数
                    cv::Mat scores_matrix(1, classesNames.size(), CV_32FC1, classes_scores);
                    // 执行cv::minMaxLoc函数并获取最佳类别分数的索引
                    cv::Point class_id;
                    double max_class_score;
                    cv::minMaxLoc(scores_matrix, 0, &max_class_score, 0, &class_id);
                    // 再度筛选超过分数阈值的类别
                    if (max_class_score >= SCORE_THRESHOLD)
                    {
                        // 将类别 ID 和置信度存储在先前定义的各个向量中
                        confidences.push_back(confidence);
                        class_ids.push_back(class_id.x);
                        // 分别获取（640大小）下的bounding box中心x, y坐标以及宽高尺寸
                        // original在此解释为从模型中读取的数据是"original"的
                        float center_x = data[0];
                        float center_y = data[1];
                        float original_w = data[2];
                        float original_h = data[3];
                        // 逆推原画面的边界框的左上角顶点坐标以及bounding box宽高
                        int left = int((center_x - original_w / 2) * x_factor);
                        int top = int((center_y - original_h / 2) * y_factor);
                        int width = int(original_w * x_factor);
                        int height = int(original_h * y_factor);
                        // 存储较为符合预期的bounding box到向量中
                        bBoxes.push_back(cv::Rect(left, top, width, height));
                    }
                }
                // 指针前移85个单元以指向下一行
                data += 85;
            }

            // 由于一张图像上有25200个预推理生成的bounding box，难免会出现多个检测框重叠的情况
            // 在此处使用非最大抑制 NMS算法进行过滤，计算出较为简明的IOU，然后存储到向量并且绘制bounding box和文字到图像
            std::vector<int> indices;
            // 运行NMS算法
            cv::dnn::NMSBoxes(bBoxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, indices);
            // 循环以筛选出最终结果
            for (long unsigned int i = 0; i < indices.size(); i++)
            {
                // 拿到筛选后的序列ID
                int index = indices[i];
                // 拿到筛选后ID对应的bounding box
                cv::Rect bBox = bBoxes[index];
                // 解析输出
                int final_left = bBox.x;
                int final_top = bBox.y;
                int final_width = bBox.width;
                int final_height = bBox.height;
                // 加入到类的静态变量中
                bBoxRectVect.push_back(cv::Rect(final_left, final_top, final_width, final_height));

                BoundingBox mybBox;
                mybBox.x = final_left + final_width / 2;
                mybBox.y = final_top + final_height / 2;
                mybBox.width = final_width;
                mybBox.height = final_height;
                bBoxVect.push_back(mybBox);

                // 绘制bounding box
                cv::rectangle(inputImage, cv::Point(final_left, final_top), cv::Point(final_left + final_width, final_top + final_height), YELLOW, 3 * THICKNESS);
                // 获取类名称及其置信度的标签
                std::string label = cv::format("%.2f", confidences[index]);
                label = classesNames[class_ids[index]] + ": " + label;
                // 绘制类别的标签
                DrawLabel(inputImage, label, final_left, final_top);
            }
            return inputImage;
        }
        catch (const std::exception &e)
        {
            std::cerr << "后处理网络模型数据和图像数据时出现错误：" << e.what() << '\n';
            // 直接退出
            // exit();
        }
    }
#pragma endregion DetectObject类的函数定义

    // 该类主要用于匹配两张图像上的相同特征点并维护计数
    class MatchFeature
    {
    public:
        // 类构造函数，需要传入需要比对的两个图像。请勿传入双目相机的两个原始图像！
        MatchFeature(cv::Mat image1, cv::Mat image2);

        // 该函数通过ORB特征匹配法进行特征匹配
        void DoMatch(double distanceThreshold, long unsigned int goodMatchesThreshold, int H_MatrixThreshold);

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

#pragma region MatchFeature类的函数定义

    // 类构造函数，需要传入需要比对的两个图像。请勿传入双目相机的两个原始图像
    // 请勿使用左值引用传参
    MatchFeature::MatchFeature(cv::Mat image1, cv::Mat image2)
    {
        // 先将图像转换为灰度图
        cv::cvtColor(image1, leftImage, cv::COLOR_BGR2GRAY);
        cv::cvtColor(image2, rightImage, cv::COLOR_BGR2GRAY);
        // 初始化字段
        isMatched = false;
        isH_MatrixValid = false;
    }

    // 该函数通过ORB特征匹配法进行特征匹配
    void MatchFeature::DoMatch(double distanceThreshold, long unsigned int goodMatchesThreshold, int H_MatrixThreshold)
    {
        cv::Mat left = leftImage, right = rightImage;
        // 初始化ORB检测器
        cv::Ptr<cv::ORB> orbDetector = cv::ORB::create();
        // 检测关键点并计算关键点的描述符
        std::vector<cv::KeyPoint> leftImageKeyPoints, rightImageKeyPoints;
        cv::Mat leftImageDescriptors, rightImageDescriptors;
        orbDetector->detectAndCompute(leftImage, cv::Mat(), leftImageKeyPoints, leftImageDescriptors);
        orbDetector->detectAndCompute(rightImage, cv::Mat(), rightImageKeyPoints, rightImageDescriptors);
        // 匹配描述符
        cv::BFMatcher matcher(cv::NORM_HAMMING);
        std::vector<cv::DMatch> matches;
        matcher.match(leftImageDescriptors, rightImageDescriptors, matches);
        // 筛选良好的匹配结果，遍历获取汉明距离极大与极小
        double maximunDistance = 0, minimumDistance = 100.0;
        for (long unsigned int i = 0; i < matches.capacity(); i++)
        {
            double dist = matches[i].distance;
            if (dist > maximunDistance)
            {
                maximunDistance = dist;
            }
            if (dist < minimumDistance)
            {
                minimumDistance = dist;
            }
        }
        std::vector<cv::DMatch> goodMatches;
        for (long unsigned int i = 0; i < matches.capacity(); i++)
        {
            if (matches[i].distance <= cv::max<float>(2 * minimumDistance, distanceThreshold))
            {
                goodMatches.push_back(matches[i]);
            }
        }
        // 指示是否匹配
        this->isMatched = goodMatches.size() >= goodMatchesThreshold;

        // 计算两张图片中匹配的描述符并连线
        cv::drawMatches(left, leftImageKeyPoints, right, rightImageKeyPoints, goodMatches, matchesDrawing);

        // 使用RANSAC算法计算单应性矩阵
        std::vector<cv::Point2f> leftScene, rightScene;
        for (long unsigned int i = 0; i < goodMatches.size(); i++)
        {
            leftScene.push_back(leftImageKeyPoints[goodMatches[i].queryIdx].pt);
            rightScene.push_back(rightImageKeyPoints[goodMatches[i].trainIdx].pt);
        }
        cv::Mat outMask;
        cv::Mat H = cv::findHomography(leftScene, rightScene, cv::RANSAC, 3.0, outMask);
        int inliers = cv::countNonZero(outMask);
        this->isH_MatrixValid = inliers >= H_MatrixThreshold;
    }

    // 指示是否匹配
    bool MatchFeature::IsMatched()
    {
        return isMatched;
    }

    // 指示H单应性矩阵是否有效
    bool MatchFeature::IsH_MatrixValid()
    {
        return isH_MatrixValid;
    }

    // 该函数用于将两张经过ORB特征匹配的图匹配描述点之间连线显示
    cv::Mat MatchFeature::GetMatchesDrawing()
    {
        return matchesDrawing;
    }

    // 终结器
    MatchFeature::~MatchFeature() {}

#pragma endregion MatchFeature类的函数定义
}
