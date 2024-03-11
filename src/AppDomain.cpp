#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <ros/ros.h>
#include <ros/package.h>
#include <detection/Detection.hpp>             // 与目标检测相关的函数
#include <detection/Distance_Estimate.hpp>     // 与目标距离测算相关的函数
#include <detection/Split_Image.hpp>           // 与图像提取、分割相关的函数
#include <detection/Read_Calibration_Data.hpp> // 与读取相机标定文件（默认是相机内参）相关的函数
#include <detection/VisionInfo.h>              // ROS消息相关

// 入口点主函数
int main(int argc, char *argv[])
{
    // 初始化ROS节点创建句柄
    setlocale(LC_ALL, "");
    ros::init(argc, argv, "Stereo_Vision");
    ros::NodeHandle nHnd;
    ros::Publisher publisher = nHnd.advertise<detection::VisionInfo>("visionInfo", 10);
    detection::VisionInfo info;

    // 通过ROS的运行时来找到自己的目录，进而拿到模型所在的目录与相机标定数据所在的目录
    std::string packageName = "detection";
    std::string packageDirectory = ros::package::getPath(packageName);

    // 拿到模型路径
    /* packageDirectory.empty() ? "NaN" : packageDirectory + "/models/" */
    /*  packageDirectory.empty() ? "NaN" : packageDirectory + "/CalibrationImages/" */
    // std::string calibrationImageDirectory = "/home/agrotech/detection_ws/src/detection/CalibrationImages/";
    std::string modelsDirectory = "/home/agrotech/detection_ws/src/detection/models/";

    /*
        应加入对相机序列号的判断
    */

    // 打开相机，使用双面分辨率进行新读取
    cv::VideoCapture camera(4);
    if (!camera.isOpened())
    {
        std::cerr << "打开摄像头时出错！"
                  << "\n";
        return -1;
    }
    // 设定自动曝光、读取分辨率等
    camera.set(cv::CAP_PROP_AUTO_EXPOSURE, 1.0);
    camera.set(cv::CAP_PROP_FPS, 30);
    camera.set(cv::CAP_PROP_FOURCC, cv::CAP_OPENCV_MJPEG);
    camera.set(cv::CAP_PROP_FRAME_WIDTH, 2160);
    camera.set(cv::CAP_PROP_FRAME_HEIGHT, 720);

    // 利用Detetction_Related命名空间下的一些函数对图像进行处理，分别获得左右相机的图像
    cv::Mat stereoFrame, stereoLeftFrame, stereoRightFrame;

    // 设置用于显示图像的窗口
    cv::namedWindow("Left", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Right", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Matches", cv::WINDOW_AUTOSIZE);

    // 设置窗口大小
    cv::resizeWindow("Left", 1280, 720);
    cv::resizeWindow("Right", 1280, 720);
    cv::resizeWindow("Matches", 640, 480);

    // 循环以不断进行检测
    ros::Rate rate(1);
    while (ros::ok())
    {
        // 获取图像并进行分割处理
        camera >> stereoFrame;
        Detection_Related::SplitImage split(stereoFrame);

        // 获得左右图像
        stereoLeftFrame = split.GetLeftImage();
        stereoRightFrame = split.GetRightImage();

        // 从模型所在目录获取ONNX模型，然后输入类别名称
        std::string modelPath = modelsDirectory + "A.onnx";
        std::vector<std::string> cls_names;
        cls_names.push_back("Peppers-rBaC");

        // 构建检测对象并分别进行检测
        Detection_Related::DetectObject detectLeft(modelPath, stereoLeftFrame, cls_names);
        Detection_Related::DetectObject detectRight(modelPath, stereoRightFrame, cls_names);

        // 进行检测
        detectLeft.Detect();
        detectRight.Detect();

        // 分别获取当前图像中检测目标最大的物体对应的bouding box;
        // 在理想环境下，若左、右相机都检测到了目标的存在，那么根据近大远小的原理，在两个画面中“类似且都最大的物体”很可能就是同一个物体，且该物体距离相机最近
        unsigned int leftMax = detectLeft.GetMaxBoxIndex();
        unsigned int rightMax = detectRight.GetMaxBoxIndex();

        // 获取两个实例的bounding box向量的容量
        size_t leftVectSize = detectLeft.bBoxVector.size();
        size_t rightVectSize = detectRight.bBoxVector.size();
        size_t leftRectVectSize = detectLeft.bBoxRectVector.size();
        size_t rightRectVectSize = detectRight.bBoxRectVector.size();

        // 若两个Vector的大小都大于0，则说明有货
        if (leftVectSize > 0 && rightVectSize > 0)
        {
            // 获得左相机中标号最大的物体的中心点坐标、宽度与高度
            int left_cx = detectLeft.bBoxVector[leftMax].x;
            int left_cy = detectLeft.bBoxVector[leftMax].y;
            int left_width = detectLeft.bBoxVector[leftMax].width;
            int left_height = detectLeft.bBoxVector[leftMax].height;

            // 获得右相机中标号最大的物体的中心点坐标、宽度与高度
            int right_cx = detectRight.bBoxVector[rightMax].x;
            int right_cy = detectRight.bBoxVector[rightMax].y;
            int right_width = detectRight.bBoxVector[rightMax].width;
            int right_height = detectRight.bBoxVector[rightMax].height;

            // 分别获得左、右相机画面中目标物体的检测框的左上角顶点坐标
            int left_l = detectLeft.bBoxRectVector[leftMax].x;
            int left_t = detectLeft.bBoxRectVector[leftMax].y;

            int right_l = detectRight.bBoxRectVector[rightMax].x;
            int right_t = detectRight.bBoxRectVector[rightMax].y;

            // 将检测框画面进行裁切，获取“该物体”的局部图像
            cv::Mat leftbBoxImg = Detection_Related::ExtractImage(stereoLeftFrame, left_l, left_t, left_width, left_height);
            cv::Mat rightbBoxImg = Detection_Related::ExtractImage(stereoRightFrame, right_l, right_t, right_width, right_height);

            // 对这两个局部画面使用OBR特征匹配算法进行保暴力匹配
            Detection_Related::MatchFeature match(leftbBoxImg, rightbBoxImg);
            match.DoMatch();

            // 距离单位：米
            double distance = 0.00;

            // 若使用ORB特征匹配后判定为匹配，则说明他们是同一个物体，进行后续的继续操作
            if (match.IsMatched())
            {
                // 传入该物体在两个相机画面对应各自画面的坐标
                // 已知双目相机基线为 60 mm，即 0.06 m；每个相机镜头的焦距为 3.0 mm，即 0.003 m，填入参数
                Distance_Estimate::StereoDistCalc sdc(cv::Point(left_cx, left_cy), cv::Point(right_cx, right_cy), 0.060, 0.003);

                // 计算物体距离相机的距离
                distance = sdc.DistCalc();

                // 在各个显示图像的窗口上绘制与更新图像
                cv::imshow("Left", detectLeft.labeledImg);
                cv::imshow("Right", detectRight.labeledImg);
                cv::imshow("Matches", match.GetMatchesDrawing());

                // 发布距离信息
                info.distance = distance * 1000;
                // info.x = 
                // info.y = 
                // info.z = 
                publisher.publish(info);
                rate.sleep();
                ros::spinOnce();

                // 打印
                std::cout << "目标距离为：" << distance << "\r\n";
            }
            // 若不匹配，则输出原图画
            else
            {
                cv::imshow("Left", stereoLeftFrame);
                cv::imshow("Right", stereoRightFrame);
                std::cout << "目标距离为：" << 0.000 << "\n";
            }
            
            // 若按下Q键，则退出
            auto key = cv::waitKey(1);
            if (key == 'q' || key == 'Q')
            {
                goto PROGRAM_END;
            }
        }
    }
PROGRAM_END:
    // 释放资源
    camera.release();
    cv::destroyAllWindows();
    return 0;
}