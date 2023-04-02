
#include "logger.h"
#include <NvInfer.h>
#include "yolo.h"
#include <opencv2/opencv.hpp>
#include <chrono>

const std::vector<std::string> CLASS_NAMES = {
	"person", "bicycle", "car", "motorcycle", "airplane", "bus",
	"train", "truck", "boat", "traffic light", "fire hydrant",
	"stop sign", "parking meter", "bench", "bird", "cat",
	"dog", "horse", "sheep", "cow", "elephant",
	"bear", "zebra", "giraffe", "backpack", "umbrella",
	"handbag", "tie", "suitcase", "frisbee", "skis",
	"snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
	"skateboard", "surfboard", "tennis racket", "bottle", "wine glass",
	"cup", "fork", "knife", "spoon", "bowl",
	"banana", "apple", "sandwich", "orange", "broccoli",
	"carrot", "hot dog", "pizza", "donut", "cake",
	"chair", "couch", "potted plant", "bed", "dining table",
	"toilet", "tv", "laptop", "mouse", "remote",
	"keyboard", "cell phone", "microwave", "oven",
	"toaster", "sink", "refrigerator", "book", "clock", "vase",
	"scissors", "teddy bear", "hair drier", "toothbrush"};

const std::vector<std::vector<unsigned int>> COLORS = {
	{0, 114, 189}, {217, 83, 25}, {237, 177, 32}, {126, 47, 142}, {119, 172, 48}, {77, 190, 238}, {162, 20, 47}, {76, 76, 76}, {153, 153, 153}, {255, 0, 0}, {255, 128, 0}, {191, 191, 0}, {0, 255, 0}, {0, 0, 255}, {170, 0, 255}, {85, 85, 0}, {85, 170, 0}, {85, 255, 0}, {170, 85, 0}, {170, 170, 0}, {170, 255, 0}, {255, 85, 0}, {255, 170, 0}, {255, 255, 0}, {0, 85, 128}, {0, 170, 128}, {0, 255, 128}, {85, 0, 128}, {85, 85, 128}, {85, 170, 128}, {85, 255, 128}, {170, 0, 128}, {170, 85, 128}, {170, 170, 128}, {170, 255, 128}, {255, 0, 128}, {255, 85, 128}, {255, 170, 128}, {255, 255, 128}, {0, 85, 255}, {0, 170, 255}, {0, 255, 255}, {85, 0, 255}, {85, 85, 255}, {85, 170, 255}, {85, 255, 255}, {170, 0, 255}, {170, 85, 255}, {170, 170, 255}, {170, 255, 255}, {255, 0, 255}, {255, 85, 255}, {255, 170, 255}, {85, 0, 0}, {128, 0, 0}, {170, 0, 0}, {212, 0, 0}, {255, 0, 0}, {0, 43, 0}, {0, 85, 0}, {0, 128, 0}, {0, 170, 0}, {0, 212, 0}, {0, 255, 0}, {0, 0, 43}, {0, 0, 85}, {0, 0, 128}, {0, 0, 170}, {0, 0, 212}, {0, 0, 255}, {0, 0, 0}, {36, 36, 36}, {73, 73, 73}, {109, 109, 109}, {146, 146, 146}, {182, 182, 182}, {219, 219, 219}, {0, 114, 189}, {80, 183, 189}, {128, 128, 0}};

sample::Logger logger;

static  const int num =100;

int main(int argc, char **argv)
{
	if (argc != 3)
	{
		std::cerr << "usage: ./infer [trt_model_path] [img_path]" << std::endl;
		return -1;
	}
	char *trt_file_path = argv[1];
	char *img_path = argv[2];
	yolo model(trt_file_path, true);

	cv::Mat res;
	cv::namedWindow("result", cv::WINDOW_AUTOSIZE);

	cv::Mat img = cv::imread(img_path, cv::IMREAD_COLOR);
	auto start = std::chrono::system_clock::now();
	auto results = model.predict_GPU(img);
	auto end = std::chrono::system_clock::now();
	auto tc = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.;
	for (auto result : results)
	{
		if (result.class_id >= CLASS_NAMES.size() || result.class_id < 0)
		{
			cout << "wrong class_id " << result.class_id << endl;
			continue;
		}
		cout << CLASS_NAMES[result.class_id] << endl;
		for (int i = 0; i < 4; i++)
			cout << result.bbox[i] << endl;
		cout << result.conf << endl;
	}
	model.draw_objects(img, results, COLORS, CLASS_NAMES);
	cv::imshow("result", img);
	cv::waitKey(0);

	return 0;
}