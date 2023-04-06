
#include "logger.h"
#include <NvInfer.h>
#include "yolo.h"
#include <opencv2/opencv.hpp>
#include <chrono>
#include "options.h"
#include "jsoncpp/json/json.h"

vector<string> names;
vector<vector<unsigned int>> colors;
sample::Logger logger;

static const int num = 100;

void parseFromJson(const string &file_path)
{
	ifstream ifile;
	ifile.open(file_path.c_str());
	if(!ifile.is_open()){
		cerr << "please check the config file's name" << endl;
		exit(-1);
	};

	Json::Value root;
	Json::Reader reader;

	string strerr;
	reader.parse(ifile, root);
	int nc = root["nc"].asInt();
	// auto namesValue = root["names"];
	for (int i = 0; i < nc; i++)
	{
		names.push_back(root["names"][i].asString());
	}
	for (int i = 0; i < nc; i++)
	{
		colors.push_back({root["color"][i][0].asUInt(),
						  root["color"][i][1].asUInt(),
						  root["color"][i][2].asUInt()});
	}
	return;
}


static Options options = Options({"Usage: infer [options...]",
                                  "detect objects using serialized engine file",
                                  "",
                                  "Options:",
                                  "  -(i|-input)         <input>      the input onnx file path",
                                  "  -(e|-engine)        <engine>     the path for serialized engine file",
                                  "  -(c|-config)       <config>      the config file of tje input engine file,include name and color",
                                  "  -(h|-help)                       print the message and exit",
                                  "",
                                  "Examples:",
                                  " ./infer -i=bus.jpg -e=yolov5l_plugin.trt -c=coco.json"});

int main(int argc, const char *argv[])
{
	options.parse(argc,argv);
	if(options.has("h")){
		options.show_usage();
		return 0;
	}
	parseFromJson(options.get("c","./coco.json"));
	string trt_file_path = options.get("e","./yolov8l_plugin.trt");
	string img_path = options.get("i","./bus.jpg");
	yolo model(trt_file_path.c_str(), true);

	cv::Mat res;
	cv::namedWindow("result", cv::WINDOW_AUTOSIZE);

	cv::Mat img = cv::imread(img_path.c_str(), cv::IMREAD_COLOR);
	
	auto start = std::chrono::system_clock::now();
	auto results = model.predict_GPU(img);
	auto end = std::chrono::system_clock::now();
	auto tc = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.;
	for (auto result : results)
	{
		if (result.class_id >= names.size() || result.class_id < 0)
		{
			cout << "wrong class_id " << result.class_id << endl;
			continue;
		}
		cout << names[result.class_id] << endl;
		for (int i = 0; i < 4; i++)
			cout << result.bbox[i] << endl;
		cout << result.conf << endl;
	}
	model.draw_objects(img, results, colors, names);
	cv::imshow("result", img);
	cv::waitKey(0);

	return 0;
}