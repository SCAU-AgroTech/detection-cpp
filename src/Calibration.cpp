// #include <iostream>
// #include <opencv2/opencv.hpp>
// #include "DoSaveCalibrationData.hpp"

// namespace Calibration_Related
// {
//     int main(int argc, char* agrv[])
//     {
//         int chessBoardRows = 0, chessBoardColumns = 0, cameraIndex = 0;
//         std::cout << "请输入棋盘标定板的纵列格子数与横列格子数：\n" << std::endl;
//         std::cin >> chessBoardRows >> chessBoardColumns;
//         if (!(chessBoardRows || chessBoardColumns))
//         {
//             std::cout << "已接收：" << chessBoardRows << ", " << chessBoardColumns << std::endl;
//         }
//         else
//         {
//             std::cerr << "无效输入：" << chessBoardRows << ", " << chessBoardColumns << std::endl;
//         }

//         std::cout << "请输入摄像头在该操作系统中的编号：\n" << std::endl;
//         std::cin >> cameraIndex;
//         if (cameraIndex < 0)
//         {
//             std::cerr << "无效输入：" << cameraIndex << std::endl;
//         }
//         else
//         {
//             std::cout << "已接收：" << cameraIndex << std::endl;
//         }
//         cv::VideoCapture camera(cameraIndex);
//         if (!camera.isOpened())
//         {
//             std::cerr << "无法打开相机，请检查后再重新运行该功能包！" << std::endl;
//         }

//         cv::Size chessBoardSize(chessBoardRows, chessBoardColumns);
//         std::vector<std::vector<cv::Point2f>> imagePoints2D;
//         std::vector<cv::Point3f> object3D;

//         for (int rowIndex = 0; rowIndex < chessBoardSize.height; rowIndex++)
//         {
//             for (int columnIndex = 0; columnIndex < chessBoardSize.width; columnIndex++)
//             {
//                 object3D.push_back(cv::Point3f(columnIndex, rowIndex, 0.0F));
//             }
//         }

//         cv::Mat originalFrame;
//         cv::namedWindow("Calibration Image", cv::WINDOW_NORMAL);
//         while (cv::waitKey(1000 / 30) >= 0)
//         {
//             camera >> originalFrame;
//             if (originalFrame.empty())
//             {
//                 return -1;
//             }

//             std::vector<cv::Point2f> corners;
//             bool isFound = cv::findChessboardCorners(originalFrame, chessBoardSize, corners);

//             if (isFound)
//             {
//                 cv::cornerSubPix
//                 (
//                     originalFrame,
//                     corners,
//                     cv::Size(11, 11),
//                     cv::Size(-1, -1),
//                     cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.01)
//                 );
//                 imagePoints2D.push_back(corners);
//                 cv::drawChessboardCorners(originalFrame, chessBoardSize, corners, isFound);
//             }

//             cv::imshow("Calibration Image", originalFrame);
//         }

//         return 0;
//     }
// }
