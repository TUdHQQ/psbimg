#include <json/json.h>
#include <fstream>
#include <filesystem>
#include <json/value.h>
#include <vector>
#include <opencv4/opencv2/opencv.hpp>
#include <windows.h>

namespace fs = std::filesystem;

struct cimg{
    int x, y, n;
    std::string name;
};

void createcimg(std::string val, Json::Value &root, std::vector<cimg> &img);

void fixdcimg(std::vector<cimg> &img, int w, int h, int a, int basex);

Json::Value readjson(std::string file);

cv::Mat cutimg(cv::Mat img, int x, int y, int w, int h);

cv::Mat coverimg(cv::Mat baseimg, cv::Mat faceimg, int x, int y);

void work(std::string pngfile, std::string inputfile, std::string basename);

int main(int argc,char* argv[]){
    fs::create_directory("output");
    for (const auto& entry : fs::directory_iterator(".")) {
        if (fs::is_regular_file(entry.status())) {
            fs::path file_path = entry.path();
            std::string filename = file_path.filename().string();
            if (filename.find(".psb.m.json") != std::string::npos) {
                std::string inputfile = filename;
                std::string basename = filename.substr(0,filename.size() - 11);
                std::string pngfile = "./" + basename + ".psb.m/" + basename + ".png";
                work(pngfile, inputfile, basename);
            }
        }
    }
    return 0;
}

Json::Value readjson(std::string file){
    std::ifstream jsonfile(file, std::ifstream::binary);
    Json::Value root;
    Json::CharReaderBuilder readerBuilder;
    std::string errs;
    Json::parseFromStream(readerBuilder, jsonfile, &root, &errs);
    return root;
}

void createcimg(std::string val,Json::Value &root,std::vector<cimg> &img){
    Json::Value::Members members;  
    members = root[val.c_str()].getMemberNames();
    int num = 0;
    for (Json::Value::Members::iterator iterMember = members.begin(); iterMember != members.end(); iterMember++){
        cimg temp;
        std::string strKey = *iterMember;
        if (root[val.c_str()][strKey.c_str()].isNull())  {  
            temp.name = strKey.c_str();
            temp.n = -1;
            num--;
        }
        else{
            temp.name = strKey.c_str();
            temp.n = num;
        }
        img.push_back(temp);
        num ++;
    }
    return ;
}

void fixdcimg(std::vector<cimg> &img, int w, int h, int a, int basex){
    for (int i = 0; i < img.size(); i++){
        if(img[i].n == -1){
            img[i].x = -1;
            img[i].y = -1;
        }
        else{
            img[i].x = basex + (img[i].n / a) * w;
            img[i].y = (img[i].n % a) * h;
        }
    }
}

cv::Mat cutimg(cv::Mat img, int x, int y, int w, int h){
    cv::Rect cropRegion(x, y, w, h);
    cropRegion = cropRegion & cv::Rect(0, 0, img.cols, img.rows);
    return img(cropRegion).clone();
}

cv::Mat coverimg(cv::Mat baseimg, cv::Mat faceimg, int x, int y){
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

void work(std::string pngfile, std::string inputfile, std::string basename){

    //读取json文件
    Json::Value root = readjson(inputfile);
    
    //eyex,eyey为眼睛与基低的差值
    int eyex = abs(root["crop"]["x"].asInt() - root["eyediff"]["x"].asInt()) -1;
    int eyey = abs(root["crop"]["y"].asInt() - root["eyediff"]["y"].asInt()) -1;
    
    //h,w代表裁剪后的图片高度和宽度,eyediffbase,eyediffend为x坐标的起始值和终止值
    int h = root["crop"]["h"].asInt(),
        w = root["crop"]["w"].asInt(),
        eyeh = root["eyediff"]["h"].asInt() + 2,
        eyew = root["eyediff"]["w"].asInt() + 2,
        eyediffbase = root["eyediffbase"].asInt(),
        eyediffend = root["h"].asInt();
    
    //创建一个cimg类型，用来存储eye的各种数据
    std::vector<cimg> eyemap;
    createcimg("eyemap",root,eyemap);
    fixdcimg(eyemap,eyew,eyeh,h/eyeh,eyediffbase);
    //for(int i = 0; i < eyemap.size(); i++){
    //    std::cout << eyemap[i].name << " " << eyemap[i].x << " " << eyemap[i].y << std::endl;
    //}
    
    //创建一个cimg类型，用来存储lip的各种数据
    int lipx, lipy, liph, lipw, lipdiffbase;
    std::vector<cimg> lipmap;
    createcimg("lipmap",root,lipmap);

    //判断lip是否存在
    if(root["lipdiffbase"].isNull()) {
        lipx = -1,
        lipy = -1,
        liph = -1,
        lipw = -1,
        lipdiffbase = -1;
    }
    else{
        lipx = abs(root["crop"]["x"].asInt() - root["lipdiff"]["x"].asInt()),
        lipy = abs(root["crop"]["y"].asInt() - root["lipdiff"]["y"].asInt()),
        liph = root["lipdiff"]["h"].asInt() + 2,
        lipw = root["lipdiff"]["w"].asInt() + 2,
        lipdiffbase = root["lipdiffbase"].asInt();
        fixdcimg(lipmap, lipw, liph, h/liph, lipdiffbase);
    }
    
    //for(int i = 0; i < lipmap.size(); i++){
    //    std::cout << lipmap[i].name << " " << lipmap[i].x << " " << lipmap[i].y << std::endl;
    //}

    //读入baseimg
    cv::Mat img = cv::imread(pngfile, cv::IMREAD_UNCHANGED);
    int num = eyemap.size();
    cv::Mat baseimg = cutimg(img, 0, 0, w, h);
    for(int i = 0; i < num; i++){
        cv::Mat out;
        if(eyemap[i].n != -1 && lipmap[i].n != -1){
            cv::Mat eyecut = cutimg(img, eyemap[i].x, eyemap[i].y, eyew, eyeh);
            cv::Mat lipcut = cutimg(img, lipmap[i].x, lipmap[i].y, lipw, liph);
            out = coverimg(baseimg, eyecut, eyex, eyey);
            out = coverimg(out, lipcut, lipx, lipy);
        }
        else if(eyemap[i].n != -1 && lipmap[i].n == -1){
            cv::Mat eyecut = cutimg(img, eyemap[i].x, eyemap[i].y, eyew, eyeh);
            out = coverimg(baseimg, eyecut, eyex, eyey);
        }
        else{
            out = baseimg;
        }

        fs::path filename = "./output/"+ basename + "_" + eyemap[i].name + "_" + lipmap[i].name + ".png";
        cv::imwrite(filename.string(), out);
    }
    
}
