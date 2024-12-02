#include "cmd_line_parser.h"
#include "logger.h"
#include "ocr.h"
#include <chrono>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/opencv.hpp>

int main(int argc, char *argv[]) {
    CommandLineArguments arguments;

    std::string logLevelStr = getLogLevelFromEnvironment();
    spdlog::level::level_enum logLevel = toSpdlogLevel("info");
    spdlog::set_level(logLevel);

    // 如果命令行参数为空，设置默认值
    if (argc == 1) {
        // 使用 const char* 数组来代替原始 argv 赋值
        const char* new_argv[] = {
            "",  // 通常是程序名
            "--det_onnx_model", "..\\..\\..\\models\\det_model.onnx",
            "--input", "..\\..\\..\\inputs\\12.jpg",
            "--rec_onnx_model", "..\\..\\..\\models\\rec_model.onnx"
        };
        argc = 7;
        argv = const_cast<char**>(new_argv);  // 将新数组传递给 argv
    }

    // 解析命令行参数
    if (!parseArguments(argc, argv, arguments)) {
        return -1;
    }

    // 使用命令行参数提供的输入路径，若未提供则使用默认路径
    std::string inputImage = arguments.inputPath.empty() ? "..\\..\\..\\inputs\\12.jpg" : arguments.inputPath;

    if (!arguments.inputPath.empty()) {
        inputImage = arguments.inputPath;
    }

    // 读取输入图像
    cv::Mat cpuImg = cv::imread(inputImage);
    if (cpuImg.empty()) {
        const std::string msg = "Unable to read image at path: " + inputImage;
        spdlog::error(msg);
        throw std::runtime_error(msg);
    }

    // 创建 OCR 实例并初始化
    ocr instance;    
    vector<double> ocr_times;

    instance.Model_Init(arguments.det_trt_model, arguments.det_onnx_model, arguments.rec_trt_model, arguments.rec_onnx_model);

    #ifdef _MSC_VER
    // 在 Windows 下启用内存泄漏调试
    _CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
    #endif

    // 执行模型推理
    instance.Model_Infer(cpuImg, ocr_times);

    #ifdef _MSC_VER
    // 在 Windows 下打印内存泄漏信息
    _CrtDumpMemoryLeaks();
    #endif

    // 在 Windows 下暂停，Linux 可替代为 std::cin.get()
    #ifdef _MSC_VER
    system("pause");
    #else
    std::cout << "Press Enter to continue..." << std::endl;
    std::cin.get();
    #endif

    return 0;
}
