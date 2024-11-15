#include <json/json.h>
#include <fstream>
#include <filesystem>
#include <json/value.h>
#include <vector>
#include <opencv4/opencv2/opencv.hpp>
#include <windows.h>
#include <thread>

namespace fs = std::filesystem;

struct cimg {
    int x, y, n;
    std::string name;
};

void createcimg(std::string val, Json::Value &root, std::vector<cimg> &img);
void fixdcimg(std::vector<cimg> &img, int w, int h, int a, int basex);
Json::Value readjson(std::string file);
cv::Mat cutimg(cv::Mat img, int x, int y, int w, int h);
cv::Mat coverimg(cv::Mat baseimg, cv::Mat faceimg, int x, int y);
void processImage(const std::string &pngfile, const std::string &inputfile, const std::string &basename, const std::vector<cimg> &eyemap, const std::vector<cimg> &lipmap, int w, int h, int eyew, int eyeh, int eyex, int eyey, int lipx, int lipy, int lipw, int liph);

Json::Value readjson(std::string file) {
    std::ifstream jsonfile(file, std::ifstream::binary);
    Json::Value root;
    Json::CharReaderBuilder readerBuilder;
    std::string errs;
    Json::parseFromStream(readerBuilder, jsonfile, &root, &errs);
    return root;
}

void createcimg(std::string val, Json::Value &root, std::vector<cimg> &img) {
    Json::Value::Members members = root[val.c_str()].getMemberNames();
    int num = 0;
    for (auto &strKey : members) {
        cimg temp;
        if (root[val.c_str()][strKey.c_str()].isNull()) {
            temp.name = strKey;
            temp.n = -1;
        } else {
            temp.name = strKey;
            temp.n = num;
        }
        img.push_back(temp);
        num++;
    }
}

void fixdcimg(std::vector<cimg> &img, int w, int h, int a, int basex) {
    for (auto &entry : img) {
        if (entry.n == -1) {
            entry.x = -1;
            entry.y = -1;
        } else {
            entry.x = basex + (entry.n / a) * w;
            entry.y = (entry.n % a) * h;
        }
    }
}

cv::Mat cutimg(cv::Mat img, int x, int y, int w, int h) {
    cv::Rect cropRegion(x, y, w, h);
    cropRegion = cropRegion & cv::Rect(0, 0, img.cols, img.rows);
    return img(cropRegion).clone();
}

cv::Mat coverimg(cv::Mat baseimg, cv::Mat faceimg, int x, int y) {
    cv::Mat output = baseimg.clone();
    for (int i = 0; i < faceimg.rows; ++i) {
        for (int j = 0; j < faceimg.cols; ++j) {
            int targetX = x + j;
            int targetY = y + i;
            if (targetX >= 0 && targetX < baseimg.cols && targetY >= 0 && targetY < baseimg.rows) {
                cv::Vec4b facePixel = faceimg.at<cv::Vec4b>(i, j);
                if (facePixel[3] > 0) {  
                    output.at<cv::Vec4b>(targetY, targetX) = facePixel;
                }
            }
        }
    }
    return output;
}

void processImage(const std::string &pngfile, const std::string &inputfile, const std::string &basename, const std::vector<cimg> &eyemap, const std::vector<cimg> &lipmap, int w, int h, int eyew, int eyeh, int eyex, int eyey, int lipx, int lipy, int lipw, int liph) {

    cv::Mat img = cv::imread(pngfile, cv::IMREAD_UNCHANGED);
    cv::Mat baseimg = cutimg(img, 0, 0, w, h);  

    int num = eyemap.size();
    for (int i = 0; i < num; ++i) {
        cv::Mat out = baseimg.clone();

        if (eyemap[i].n != -1) {
            cv::Mat eyecut = cutimg(img, eyemap[i].x, eyemap[i].y, eyew, eyeh);
            out = coverimg(out, eyecut, eyex, eyey);
        }

        if (lipmap[i].n != -1) {
            cv::Mat lipcut = cutimg(img, lipmap[i].x, lipmap[i].y, lipw, liph);
            out = coverimg(out, lipcut, lipx, lipy);
        }

        fs::path filename = "./output/" + basename + "_" + eyemap[i].name + "_" + lipmap[i].name + ".png";
        cv::imwrite(filename.string(), out);
    }
}

void work(const std::string &pngfile, const std::string &inputfile, const std::string &basename) {
    Json::Value root = readjson(inputfile);

    int eyex = abs(root["crop"]["x"].asInt() - root["eyediff"]["x"].asInt()) - 1;
    int eyey = abs(root["crop"]["y"].asInt() - root["eyediff"]["y"].asInt()) - 1;

    int h = root["crop"]["h"].asInt();
    int w = root["crop"]["w"].asInt();
    int eyew = root["eyediff"]["w"].asInt() + 2;
    int eyeh = root["eyediff"]["h"].asInt() + 2;
    int eyediffbase = root["eyediffbase"].asInt();

    std::vector<cimg> eyemap;
    createcimg("eyemap", root, eyemap);
    fixdcimg(eyemap, eyew, eyeh, h / eyeh, eyediffbase);

    int lipx = -1, lipy = -1, liph = -1, lipw = -1, lipdiffbase = -1;
    std::vector<cimg> lipmap;
    createcimg("lipmap", root, lipmap);
    if (!root["lipdiffbase"].isNull()) {
        lipx = abs(root["crop"]["x"].asInt() - root["lipdiff"]["x"].asInt());
        lipy = abs(root["crop"]["y"].asInt() - root["lipdiff"]["y"].asInt());
        liph = root["lipdiff"]["h"].asInt() + 2;
        lipw = root["lipdiff"]["w"].asInt() + 2;
        lipdiffbase = root["lipdiffbase"].asInt();
        fixdcimg(lipmap, lipw, liph, h / liph, lipdiffbase);
    }

    
    std::vector<std::thread> threads;
    threads.push_back(std::thread(processImage, pngfile, inputfile, basename, std::cref(eyemap), std::cref(lipmap), w, h, eyew, eyeh, eyex, eyey, lipx, lipy, lipw, liph));

    
    for (auto &t : threads) {
        t.join();
    }
}

int main(int argc, char *argv[]) {
    fs::create_directory("output");
    for (const auto &entry : fs::directory_iterator(".")) {
        if (fs::is_regular_file(entry.status())) {
            fs::path file_path = entry.path();
            std::string filename = file_path.filename().string();
            if (filename.find(".psb.m.json") != std::string::npos) {
                std::string inputfile = filename;
                std::string basename = filename.substr(0, filename.size() - 11);
                std::string pngfile = "./" + basename + ".psb.m/" + basename + ".png";
                work(pngfile, inputfile, basename);
            }
        }
    }
    return 0;
}
