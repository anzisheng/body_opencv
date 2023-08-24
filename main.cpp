
//#include <torch/torch.h>
#define _USE_MATH_DEFINES
#include <k4a/k4a.h>
#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <k4a/k4a.h>
#include <k4abt.h>
#include <math.h>
#include <fstream>
#include <chrono>
#include <thread>
#include <time.h>
#include <cmath>
#include <string>

#include <torch/torch.h>
#include <cam/smplcam.h>

#include "definition/def.h"
#include "toolbox/Singleton.hpp"
using namespace std;
//#include <Device.h>
#define SINGLE_SMPL smpl::Singleton<smpl::SMPL>

//#include "main.h"

// OpenCV
#include <opencv2/opencv.hpp>
// Kinect DK
#include <k4a.hpp>
#define VERIFY(result, error)                                                                            \
    if(result != K4A_RESULT_SUCCEEDED)                                                                   \
    {                                                                                                    \
        printf("%s \n - (File: %s, Function: %s, Line: %d)\n", error, __FILE__, __FUNCTION__, __LINE__); \
        exit(1);                                                                                         \
    }                                                                                                    \

double get_angle(double x1, double y1, double x2, double y2, double x3, double y3)
{
    double theta = atan2(x1 - x3, y1 - y3) - atan2(x2 - x3, y2 - y3);
    if (theta > M_PI)
        theta -= 2 * M_PI;
    if (theta < -M_PI)
        theta += 2 * M_PI;

    theta = abs(theta * 180.0 / M_PI);
    return theta;
}

double get_line_equation(double x1, double y1, double x2, double y2)
{
    double k, b;
    k = (y2 - y1) / (x2 - x1);
    b = y1 - k * x1;
    return k;
}

// 计算两条直线的夹角（弧度制）
double get_lines_angle(double k1, double k2)
{
    double angle = atan(abs((k2 - k1) / (1 + k1 * k2))) * 180.0 / M_PI;
    return angle;
}


std::string Time2Str()

{

    time_t tm;

    time(&tm); //获取time_t类型的当前时间

    char tmp[64];

    strftime(tmp, sizeof(tmp), "%Y-%m-%d %H:%M:%S", localtime(&tm));

    return tmp;

}

const char* filename = "E:/data.csv";



torch::Tensor k4a2torch_float( float x,float y, float z)
{
    std::vector<float> v;
    v.push_back(x);
    v.push_back(y);
    v.push_back(z);
    torch::Tensor convert = torch::from_blob(v.data(), { 1, 3 });// torch::tensor({ x, y, z });
   //torch::Tensor convert = torch::tensor({ k4a.v[0], k4a.v[1], k4a.v[2]}); //torch::from_blob(k4a.v, { 1,3 });
    /*try
    {
        convert = torch::from_blob(k4a.v, {1,3});
        std::cout << "hello " << convert << std::endl;
    }
    catch (const exception& e)
    {
        std::cout << e.what() << std::endl;
        throw;
    }*/
    
    if (SHOWOUT)
    {
        std::cout << "convert " << convert << std::endl;

    }
    //std::cout << "convert " << convert << std::endl;
    return convert.clone();

}

std::vector<torch::Tensor>  convert25_29(std::vector<k4a_float3_t> source25)
{
    std::vector<torch::Tensor> target29;// (29);//np.zeros((1, 29, 3))
    //float x = 
    //torch::Tensor c1 = k4a2torch_float(1.0f, 2.0f, 3.0f);;//source25[8].xyz.x, source25[8].xyz.y, source25[8].xyz.z);

    //std::vector<float> v;
    //v.push_back(source25[8].xyz.x);
    //v.push_back(source25[8].xyz.y);
    //v.push_back(source25[8].xyz.z);
    //torch::Tensor convert = torch::from_blob(v.data(), { 1, 3 });
    //if (SHOWOUT)
    //{
    //    std::cout << "convert " << convert << std::endl;

    //}

    torch::Tensor convert = k4a2torch_float(source25[8].xyz.x, source25[8].xyz.y, source25[8].xyz.z);  // pelvis = MidHip
    if (SHOWOUT)
    {
        std::cout << "convert1 " << convert << std::endl;

    }
    target29.push_back(convert);        
    if (SHOWOUT)
    {
        std::cout << "convert2 " << convert << std::endl;

    }
    //target29[1] = source25[12];  //left_hip = LHip
    convert = k4a2torch_float(source25[12].xyz.x, source25[12].xyz.y, source25[12].xyz.z);  // pelvis = MidHip
    target29.push_back(convert);
    
    //target29[2] = source25[9];  // right_hip = RHip
    convert = k4a2torch_float(source25[9].xyz.x, source25[9].xyz.y, source25[9].xyz.z);  // pelvis = MidHip
    target29.push_back(convert);


    //spine = source25[:, 1] - source25[:, 8]  # spine = (Neck - MidHip)

    k4a_float3_t spine; //source25[1] - source25[8]  // spine = (Neck - MidHip)
    spine.v[0] = (source25[1].v[0] - source25[8].v[0]);
    spine.v[1] = (source25[1].v[1] - source25[8].v[1]); 
    spine.v[2] = (source25[1].v[2] - source25[8].v[2]);

    //target29[:,3] = source25[:,8] + spine / 4  # spine1 = MidHip + (Neck - MidHip)/4
    /*target29[3].v[0] = source25[8].v[0] + spine.v[0] / 4;
    target29[3].v[1] = source25[8].v[1] + spine.v[1] / 4;
    target29[3].v[2] = source25[8].v[2] + spine.v[2] / 4;*/
    convert = k4a2torch_float(source25[8].v[0] + spine.v[0]/ 4, source25[8].v[1] + spine.v[1] / 4, source25[8].v[2] + spine.v[2] / 4);
    target29.push_back(convert);

    //target29[4] = source25[13];  // left_knee = 13, “LKnee”
    convert = k4a2torch_float(source25[13].xyz.x, source25[13].xyz.y, source25[13].xyz.z);  // pelvis = MidHip
    target29.push_back(convert);
   
    //target29[5] = source25[10];  // right_knee = 10, “RKnee”
    convert = k4a2torch_float(source25[10].xyz.x, source25[10].xyz.y, source25[10].xyz.z);  // pelvis = MidHip
    target29.push_back(convert);

    //target29[:, 6] = source25[:, 8] + spine / 3  # spine2 = MidHip + (Neck - MidHip) / 3
    /*target29[6].v[0] = source25[8].v[0] + spine.v[0] / 3;
    target29[6].v[1] = source25[8].v[1] + spine.v[1] / 3;
    target29[6].v[2] = source25[8].v[2] + spine.v[2] / 3;*/
    convert = k4a2torch_float(source25[8].v[0] + spine.v[0] / 3, source25[8].v[1] + spine.v[1] / 3, source25[8].v[2] + spine.v[2] / 3);  // pelvis = MidHip
    target29.push_back(convert);


    //target29[7] = source25[14];  // left_ankle = 14, “LAnkle”
    convert = k4a2torch_float(source25[14].xyz.x, source25[14].xyz.y, source25[14].xyz.z);  // pelvis = MidHip
    target29.push_back(convert);

    //target29[8] = source25[11];  // right_ankle = 11, “RAnkle”
    convert = k4a2torch_float(source25[11].xyz.x, source25[11].xyz.y, source25[11].xyz.z);  // pelvis = MidHip
    target29.push_back(convert);


    //target29[:, 9] = source25[:, 8] + spine / 2  # spine3 = MidHip + (Neck - MidHip) / 2
    /*target29[9].v[0] = source25[8].v[0] + spine.v[0] / 2;
    target29[9].v[1] = source25[8].v[1] + spine.v[1] / 2;
    target29[9].v[2] = source25[8].v[2] + spine.v[2] / 2;*/
    convert = k4a2torch_float(source25[8].v[0] + spine.v[0] / 2, source25[8].v[1] + spine.v[1] / 2, source25[8].v[2] + spine.v[2] / 2);  // pelvis = MidHip
    target29.push_back(convert);
    
    //target29[7] = source25[14];
    // target29[4] = source25[13]；
    k4a_float3_t left_leg; // = target29[:, 7] - target29[:, 4]
    //left_leg.v[0] = target29[7].v[0] - target29[4].v[0];
    //left_leg.v[1] = target29[7].v[1] - target29[4].v[1];
    //left_leg.v[2] = target29[7].v[2] - target29[4].v[2];
    //改成：
    left_leg.v[0] = source25[14].v[0] - source25[13].v[0];
    left_leg.v[1] = source25[14].v[1] - source25[13].v[1];
    left_leg.v[2] = source25[14].v[2] - source25[13].v[2];

    //convert = k4a2torch_float(left_leg.xyz.x, left_leg.xyz.y, left_leg.xyz.z);  // pelvis = MidHip
    //target29.push_back(convert);



   //target29[8] = source25[11];
   //target29[5] = source25[10];

    k4a_float3_t  right_leg;// = target29[:, 8] - target29[:, 5]
    //right_leg.v[0] = target29[8].v[0] - target29[5].v[0];
    //right_leg.v[1] = target29[8].v[1] - target29[5].v[1];
    //right_leg.v[2] = target29[8].v[2] - target29[5].v[2];
    //改成：
    right_leg.v[0] = source25[11].v[0] - source25[10].v[0];
    right_leg.v[1] = source25[11].v[1] - source25[10].v[1];
    right_leg.v[2] = source25[11].v[2] - source25[10].v[2];


    ////target29[:, 10] = target29[:, 7] + left_leg * 0.01
    //target29[10].v[0] = target29[7].v[0] + left_leg.v[0] * 0.01;
    //target29[10].v[1] = target29[7].v[1] + left_leg.v[1] * 0.01;
    //target29[10].v[2] = target29[7].v[2] + left_leg.v[2] * 0.01;

    convert = k4a2torch_float(source25[14].v[0] + left_leg.v[0] * 0.01, source25[14].v[1] + left_leg.v[1] * 0.01, source25[14].v[2] + left_leg.v[2] * 0.01);  // pelvis = MidHip
    target29.push_back(convert);


    ////target29[:, 11] = target29[:, 8] + right_leg * 0.01;
    //target29[11].v[0] = target29[8].v[0] + right_leg.v[0] * 0.01;
    //target29[11].v[1] = target29[8].v[1] + right_leg.v[1] * 0.01;
    //target29[11].v[2] = target29[8].v[2] + right_leg.v[2] * 0.01;
    convert = k4a2torch_float(source25[11].v[0] + right_leg.v[0] * 0.01, source25[11].v[1] + right_leg.v[1] * 0.01, source25[11].v[1] + right_leg.v[2] * 0.01);  // pelvis = MidHip
    target29.push_back(convert);



    ////#temp_left = target29[:, 7] + left_leg * 0.0001
    ////#target29[:, 10] = target29[:, 7] * 0.9#  + target29[:, 4] * 0.2
    ////target29[:, 10] = target29[:, 7] + left_leg * 0.01 #(source25[:, 21] * 1.0 + source25[:, 14] * 0.0) #target29[:, 7] + left_leg * 0.000001 #source25[:, 21]#(source25[:, 19] + source25[:, 21]) / 2  # left_foot = 21, “LHeel” 14, “LAnkle”  19, “LBigToe
    ////target29[:, 11] = target29[:, 8] + right_leg * 0.01#(source25[:, 24] * 1.0 + source25[:, 11] * 0.0) #target29[:, 8] + right_leg * 0.000001 #source25[:, 24]#(source25[:, 22] + source25[:, 24]) / 2  # right_foot = 24, “RHeel" 11, “RAnkle”  22, “RBigToe”

    //target29[12] = source25[1];  //# neck = 1, “Neck”
    convert = k4a2torch_float(source25[1].v[0], source25[1].v[1], source25[1].v[2]);  // pelvis = MidHip
    target29.push_back(convert);


    ////left_collar_low = (target29[:, 9] + source25[:, 5]) / 2  // 5, “LShoulder”
    ////target29[9].v[0] = source25[8].v[0] + spine.v[0] / 2; 
    //k4a_float3_t left_collar_low; 
    //left_collar_low.v[0] = (target29[9].v[0] + source25[5].v[0]) / 2;
    //left_collar_low.v[1] = (target29[9].v[1] + source25[5].v[1]) / 2;
    //left_collar_low.v[2] = (target29[9].v[2] + source25[5].v[2]) / 2;
    //

    ////# left_collar_high = (neck + left_shoulder) / 2
    //k4a_float3_t left_collar_high;
    //left_collar_high.v[0] = (source25[1].v[0] + source25[5].v[0]) / 2;
    //left_collar_high.v[1] = (source25[1].v[1] + source25[5].v[1]) / 2;
    //left_collar_high.v[2] = (source25[1].v[2] + source25[5].v[2]) / 2;


    //k4a_float3_t left_collar;
    //left_collar.v[0] = (left_collar_low.v[0] + left_collar_high.v[0]) / 2;
    //left_collar.v[1] = (left_collar_low.v[1] + left_collar_high.v[1]) / 2;
    //left_collar.v[2] = (left_collar_low.v[2] + left_collar_high.v[2]) / 2;
    //改成：
    k4a_float3_t left_collar;
    left_collar.v[0] = ((source25[8].v[0] + spine.v[0] / 2 + source25[5].v[0]) / 2 + (source25[1].v[0] + source25[5].v[0]) / 2) / 2;
    left_collar.v[1] = ((source25[8].v[1] + spine.v[1] / 2 + source25[5].v[1]) / 2 + (source25[1].v[1] + source25[5].v[1]) / 2) / 2;
    left_collar.v[2] = ((source25[8].v[2] + spine.v[2] / 2 + source25[5].v[2]) / 2 + (source25[1].v[2] + source25[5].v[2]) / 2) / 2;



    //target29[13] = left_collar; //  # 'left_collar'
    convert = k4a2torch_float(left_collar.v[0], left_collar.v[1], left_collar.v[2]);  // pelvis = MidHip ...
    target29.push_back(convert);


    ////# right_collar_low = (spin3 + right_shoulder) / 2
    //k4a_float3_t right_collar_low;
    //right_collar_low.v[0] = (target29[9].v[0] + source25[2].v[0]) / 2;  //# 2, “RShoulder”
    //right_collar_low.v[1] = (target29[9].v[1] + source25[2].v[1]) / 2;
    //right_collar_low.v[2] = (target29[9].v[2] + source25[2].v[2]) / 2;
    //改成：
    k4a_float3_t right_collar_low;
    right_collar_low.v[0] = (source25[8].v[0] + spine.v[0] / 2 + source25[2].v[0]) / 2;  //# 2, “rshoulder”
    right_collar_low.v[1] = (source25[8].v[1] + spine.v[1] / 2 + source25[2].v[1]) / 2;
    right_collar_low.v[2] = (source25[8].v[2] + spine.v[2] / 2 + source25[2].v[2]) / 2;


    //
    ////# right_collar_high = (neck + right_shoulder) / 2
    //k4a_float3_t right_collar_high;
    //right_collar_high.v[0] = (source25[1].v[0] + source25[2].v[0]) / 2;
    //right_collar_high.v[1] = (source25[1].v[1] + source25[2].v[1]) / 2;
    //right_collar_high.v[2] = (source25[1].v[2] + source25[2].v[2]) / 2;
    //改成：
    k4a_float3_t right_collar_high;
    right_collar_high.v[0] = (source25[1].v[0] + source25[2].v[0]) / 2;
    right_collar_high.v[1] = (source25[1].v[1] + source25[2].v[1]) / 2;
    right_collar_high.v[2] = (source25[1].v[2] + source25[2].v[2]) / 2;


    //k4a_float3_t right_collar;
    //right_collar.v[0] = (right_collar_low.v[0] + right_collar_high.v[0]) / 2;
    //right_collar.v[1] = (right_collar_low.v[1] + right_collar_high.v[1]) / 2;
    //right_collar.v[2] = (right_collar_low.v[2] + right_collar_high.v[2]) / 2;
    //改成：
    k4a_float3_t right_collar;
    right_collar.v[0] = (right_collar_low.v[0] + right_collar_high.v[0]) / 2;
    right_collar.v[1] = (right_collar_low.v[1] + right_collar_high.v[1]) / 2;
    right_collar.v[2] = (right_collar_low.v[2] + right_collar_high.v[2]) / 2;

    //target29[14] = right_collar;  //# 'right_collar'
    convert = k4a2torch_float(right_collar.v[0], right_collar.v[1], right_collar.v[2]);  // pelvis = MidHip
    target29.push_back(convert);

    //target29[15].v[0] = (source25[0].v[0] + source25[1].v[0]) / 2; //# 'jaw' = { 0, “Nose” }, { 1, “Neck” },
    //target29[15].v[1] = (source25[0].v[1] + source25[1].v[1]) / 2;
    //target29[15].v[2] = (source25[0].v[2] + source25[1].v[2]) / 2;

    convert = k4a2torch_float((source25[0].v[0] + source25[1].v[0]) / 2, (source25[0].v[1] + source25[1].v[1]) / 2, (source25[0].v[2] + source25[1].v[2]) / 2);  // pelvis = MidHip
    target29.push_back(convert);

    //
    //target29[16] = source25[5];    // 'left_shoulder', 5, “LShoulder”
    convert = k4a2torch_float(source25[5].v[0] , source25[5].v[1], source25[5].v[2]);  // pelvis = MidHip
    target29.push_back(convert);


    //target29[17] = source25[2];   // 'right_shoulder', 2, “RShoulder”
    convert = k4a2torch_float(source25[2].v[0], source25[2].v[1], source25[2].v[2]);  // pelvis = MidHip
    target29.push_back(convert);


    //target29[18] = source25[6];  // 'left_elbow', 6, “LElbow”
    convert = k4a2torch_float(source25[6].v[0], source25[6].v[1], source25[6].v[2]);  // pelvis = MidHip
    target29.push_back(convert);

    //target29[19] = source25[3];  //# 'right_elbow', 3, “RElbow”
    convert = k4a2torch_float(source25[3].v[0], source25[3].v[1], source25[3].v[2]);  // pelvis = MidHip
    target29.push_back(convert);

    //target29[20] = source25[7];  //'left_wrist', 7, “LWrist”
    convert = k4a2torch_float(source25[7].v[0], source25[7].v[1], source25[7].v[2]);  // pelvis = MidHip
    target29.push_back(convert);

    //target29[21] = source25[4];  // 'right_wrist', 4, “RWrist”
    convert = k4a2torch_float(source25[4].v[0], source25[4].v[1], source25[4].v[2]);  // pelvis = MidHip
    target29.push_back(convert);

    //后来加的
    //# left_thumb = extend elbow and wrist
    // extend = (wrist - elbow)
    //target29[20] = source25[7];
    //target29[18] = source25[6]
    k4a_float3_t extend_left;// = source25[7]- source25[6]
    extend_left.v[0] = source25[7].v[0] - source25[6].v[0];
    extend_left.v[1] = source25[7].v[1] - source25[6].v[1];
    extend_left.v[2] = source25[7].v[2] - source25[6].v[2];


    //#extend 0.3 * extend
    //target29[20] = source25[7];

    //target29[:, 22] = target29[:, 20] + 0.2 * extend_left#source25[:, 7]  # 'left_thumb', 7, “LWrist”
    convert = k4a2torch_float(source25[7].v[0] + 0.2 * extend_left.v[0], source25[7].v[1] + 0.2 * extend_left.v[1], source25[7].v[2] + 0.2 * extend_left.v[2]);  // pelvis = MidHip
    target29.push_back(convert);

    //target29[21] = source25[4];  
    //target29[19] = source25[3];
    k4a_float3_t extend_right;// = target29[:, 21] - target29[:, 19]
    extend_right.v[0] = source25[4].v[0] - source25[3].v[0];
    extend_right.v[1] = source25[4].v[1] - source25[3].v[1];
    extend_right.v[2] = source25[4].v[2] - source25[3].v[2];
       
    //target29[:, 23] = target29[:, 21] + 0.2 * extend_right#source25[:, 4]  # 'right_thumb', 4, “RWrist”
    convert = k4a2torch_float(source25[4].v[0] + 0.2 * extend_right.v[0], source25[4].v[1] + 0.2 * extend_right.v[1], source25[4].v[2] + 0.2 * extend_right.v[2]);  // pelvis = MidHip
    target29.push_back(convert);


    //target29[:, 24] = (source25[:, 15] + source25[:, 16]) / 2  // 'head', { 15, “REye” }, { 16, “LEye” },
    convert = k4a2torch_float((source25[15].v[0] + source25[16].v[0]) / 2, (source25[15].v[1] + source25[16].v[1]) / 2, (source25[15].v[2] + source25[16].v[2]) / 2);  // pelvis = MidHip
    target29.push_back(convert);


    //target29[:, 25] = target29[:, 22] + 0.2 * extend_left; //left_middle = # left_middle = left_thumb
    convert = k4a2torch_float((source25[15].v[0] + source25[16].v[0]) / 2, (source25[15].v[1] + source25[16].v[1]) / 2, (source25[15].v[2] + source25[16].v[2]) / 2);  // pelvis = MidHip
    target29.push_back(convert);

    //target29[21] = source25[4]
    //target29[:, 23] = target29[:, 21] + 0.2 * extend_right

    //target29[:, 26] = target29[:, 23] + 0.2 * extend_right;  // right_middle = right_thumb
    convert = k4a2torch_float(source25[4].v[0] + 0.2 * extend_right.v[0], source25[4].v[1] + 0.2 * extend_right.v[1], source25[4].v[2] + 0.2 * extend_right.v[2]);  // pelvis = MidHip
    target29.push_back(convert);


    //target29[:, 27] = (source25[:, 19] + source25[:, 20]) / 2; //source25[:, 21] + source25[:, 19] #left_bigtoe = 19, “LBigToe” 20, “LSmallToe”
    convert = k4a2torch_float((source25[19].v[0] + source25[20].v[0]) / 2, (source25[19].v[1] + source25[20].v[1]) / 2, (source25[19].v[2] + source25[20].v[2]) / 2);  // pelvis = MidHip
    target29.push_back(convert);


    //target29[:, 28] = (source25[:, 22] + source25[:, 23]) / 2; //source25[:, 24] + source25[:, 22] # right_bigtoe = 22, “RBigToe” 23, “RSmallToe”
    convert = k4a2torch_float((source25[22].v[0] + source25[23].v[0]) / 2, (source25[22].v[1] + source25[23].v[1]) / 2, (source25[22].v[2] + source25[23].v[2]) / 2);
    target29.push_back(convert);
    return target29;
}


    

using namespace std;
//using namespace torch;

int main(int argc, char const* argv[])
{
    //cuda device    
    
// 	torch::Device cuda(torch::kCUDA);    
// 	cuda.set_index(0);



    k4a_device_t device = NULL;
    VERIFY(k4a_device_open(0, &device), "Open K4A Device failed!");

    const uint32_t device_count = k4a_device_get_installed_count();
    if (1 == device_count)
    {
        std::cout << "Found " << device_count << " connected devices. " << std::endl;
    }
    else
    {
        std::cout << "Error: more than one K4A devices found. " << std::endl;
    }

    //打开设备
    k4a_device_open(0, &device);
    std::cout << "Done: open device. " << std::endl;

    // Start camera. Make sure depth camera is enabled.
    k4a_device_configuration_t deviceConfig = K4A_DEVICE_CONFIG_INIT_DISABLE_ALL;
    deviceConfig.depth_mode = K4A_DEPTH_MODE_NFOV_2X2BINNED;
    deviceConfig.color_resolution = K4A_COLOR_RESOLUTION_720P;
    deviceConfig.camera_fps = K4A_FRAMES_PER_SECOND_30;
    deviceConfig.color_format = K4A_IMAGE_FORMAT_COLOR_BGRA32;
    deviceConfig.synchronized_images_only = true;// ensures that depth and color images are both available in the capture

    //开始相机
    //k4a_device_start_cameras(device, &deviceConfig);
    VERIFY(k4a_device_start_cameras(device, &deviceConfig), "Start K4A cameras failed!");
    std::cout << "Done: start camera." << std::endl;

    //查询传感器校准
    k4a_calibration_t sensor_calibration;
    k4a_device_get_calibration(device, deviceConfig.depth_mode, deviceConfig.color_resolution, &sensor_calibration);
    VERIFY(k4a_device_get_calibration(device, deviceConfig.depth_mode, deviceConfig.color_resolution, &sensor_calibration),
        "Get depth camera calibration failed!");
    //创建人体跟踪器
    k4abt_tracker_t tracker = NULL;
    k4abt_tracker_configuration_t tracker_config = K4ABT_TRACKER_CONFIG_DEFAULT;
    k4abt_tracker_create(&sensor_calibration, tracker_config, &tracker);
    VERIFY(k4abt_tracker_create(&sensor_calibration, tracker_config, &tracker), "Body tracker initialization failed!");


    cv::Mat cv_rgbImage_with_alpha;
    cv::Mat cv_rgbImage_no_alpha;
    cv::Mat cv_depth;
    cv::Mat cv_depth_8U;

    std::ofstream outfile(filename, std::ios::app);
    std::cout << outfile.is_open();
    cv::VideoWriter writer;




    std::string output_filename = "E:/output.avi";
    int codec = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
    int fps = 15;
    cv::Size frame_size(1280, 720);

    int frame_count = 0;
    writer.open(output_filename, codec, fps, frame_size);
    if (!writer.isOpened())
    {
        std::cerr << "Failed to open output file: " << output_filename << std::endl;
        return -1;
    }

    int frameId = 0;

    while(frameId < 100)
    {
        k4a_capture_t sensor_capture;
        k4a_wait_result_t get_capture_result = k4a_device_get_capture(device, &sensor_capture, K4A_WAIT_INFINITE);
        k4a_wait_result_t queue_capture_result = k4abt_tracker_enqueue_capture(tracker, sensor_capture, K4A_WAIT_INFINITE);

        //获取RGB和depth图像
        k4a_image_t rgbImage = k4a_capture_get_color_image(sensor_capture);
        k4a_image_t depthImage = k4a_capture_get_depth_image(sensor_capture);

        
        //RGB
        cv_rgbImage_with_alpha = cv::Mat(k4a_image_get_height_pixels(rgbImage), k4a_image_get_width_pixels(rgbImage), CV_8UC4, k4a_image_get_buffer(rgbImage));
        cout << k4a_image_get_height_pixels(rgbImage) << k4a_image_get_width_pixels(rgbImage) << endl;
        cvtColor(cv_rgbImage_with_alpha, cv_rgbImage_no_alpha, cv::COLOR_BGRA2BGR);

        //depth
        cv_depth = cv::Mat(k4a_image_get_height_pixels(depthImage), k4a_image_get_width_pixels(depthImage), CV_16U, k4a_image_get_buffer(depthImage), k4a_image_get_stride_bytes(depthImage));
        cv_depth.convertTo(cv_depth_8U, CV_8U, 1);
        k4a_capture_release(sensor_capture); // Remember to release the sensor capture once you finish using it

        if (queue_capture_result == K4A_WAIT_RESULT_TIMEOUT)
        {
            // It should never hit timeout when K4A_WAIT_INFINITE is set.
            printf("Error! Add capture to tracker process queue timeout!\n");

        }
        else if (queue_capture_result == K4A_WAIT_RESULT_FAILED)
        {
            printf("Error! Add capture to tracker process queue failed!\n");
        }

        writer.write(cv_rgbImage_no_alpha);
        //弹出结果
        k4abt_frame_t body_frame = NULL;
        k4a_wait_result_t pop_frame_result = k4abt_tracker_pop_result(tracker, &body_frame, K4A_WAIT_INFINITE);
        if (pop_frame_result == K4A_WAIT_RESULT_SUCCEEDED)
        {
            // Successfully popped the body tracking result. Start your processing
            //检测人体数
            size_t num_bodies = k4abt_frame_get_num_bodies(body_frame); //取帧
            std::vector<SMPL::person> g_persons;// (num_bodies);
            //依次计算每个人的smpl pose，global_trans 和global_trans, 保存到person中
            for (size_t i = 0; i < num_bodies; i++)
            { 
                //获取人体框架
                k4abt_skeleton_t skeleton;
                k4abt_frame_get_body_skeleton(body_frame, i, &skeleton);
                k4a_float2_t P_HEAD_2D;
                k4a_float2_t P_NECK_2D;
                k4a_float2_t P_CHEST_2D;
                k4a_float2_t P_HIP_2D;
                k4a_float2_t P_SHOULDER_RIGHT_2D;
                k4a_float2_t P_SHOULDER_LEFT_2D;
                k4a_float2_t P_HIP_RIGHT_2D;
                k4a_float2_t P_HIP_LEFT_2D;
                k4a_float2_t P_KNEE_LEFT_2D;
                k4a_float2_t P_KNEE_RIGHT_2D;
                k4a_float2_t P_ANKLE_RIGHT_2D;
                k4a_float2_t P_ANKLE_LEFT_2D;
                k4a_float2_t P_ELBOW_RIGHT_2D;
                k4a_float2_t P_ELBOW_LEFT_2D;
                k4a_float2_t P_WRIST_RIGHT_2D;
                k4a_float2_t P_WRIST_LEFT_2D;

                

                int result;
                //头部
                k4abt_joint_t  P_NOSE = skeleton.joints[K4ABT_JOINT_NOSE];
                //3D转2D，并在color中画出
                k4a_calibration_3d_to_2d(&sensor_calibration, &P_NOSE.position, K4A_CALIBRATION_TYPE_DEPTH, K4A_CALIBRATION_TYPE_COLOR, &P_HEAD_2D, &result);
                cv::circle(cv_rgbImage_no_alpha, cv::Point(P_HEAD_2D.xy.x, P_HEAD_2D.xy.y), 3, cv::Scalar(0, 255, 255));

                //颈部
                k4abt_joint_t  P_NECK = skeleton.joints[K4ABT_JOINT_NECK];
                //3D转2D，并在color中画出
                k4a_calibration_3d_to_2d(&sensor_calibration, &P_NECK.position, K4A_CALIBRATION_TYPE_DEPTH, K4A_CALIBRATION_TYPE_COLOR, &P_NECK_2D, &result);
                cv::circle(cv_rgbImage_no_alpha, cv::Point(P_NECK_2D.xy.x, P_NECK_2D.xy.y), 3, cv::Scalar(0, 255, 255));


                //胸部
                k4abt_joint_t  P_CHEST = skeleton.joints[K4ABT_JOINT_SPINE_CHEST];
                //3D转2D，并在color中画出
                k4a_calibration_3d_to_2d(&sensor_calibration, &P_CHEST.position, K4A_CALIBRATION_TYPE_DEPTH, K4A_CALIBRATION_TYPE_COLOR, &P_CHEST_2D, &result);
                cv::circle(cv_rgbImage_no_alpha, cv::Point(P_CHEST_2D.xy.x, P_CHEST_2D.xy.y), 3, cv::Scalar(0, 255, 255));

                //髋部
                k4abt_joint_t  P_HIP = skeleton.joints[K4ABT_JOINT_PELVIS];
                //3D转2D，并在color中画出
                k4a_calibration_3d_to_2d(&sensor_calibration, &P_HIP.position, K4A_CALIBRATION_TYPE_DEPTH, K4A_CALIBRATION_TYPE_COLOR, &P_HIP_2D, &result);
                cv::circle(cv_rgbImage_no_alpha, cv::Point(P_HIP_2D.xy.x, P_HIP_2D.xy.y), 3, cv::Scalar(0, 255, 255));

                //右肩
                k4abt_joint_t  P_SHOULDER_RIGHT = skeleton.joints[K4ABT_JOINT_SHOULDER_RIGHT];
                //3D转2D，并在color中画出
                k4a_calibration_3d_to_2d(&sensor_calibration, &P_SHOULDER_RIGHT.position, K4A_CALIBRATION_TYPE_DEPTH, K4A_CALIBRATION_TYPE_COLOR, &P_SHOULDER_RIGHT_2D, &result);
                cv::circle(cv_rgbImage_no_alpha, cv::Point(P_SHOULDER_RIGHT_2D.xy.x, P_SHOULDER_RIGHT_2D.xy.y), 3, cv::Scalar(0, 255, 255));

                //右髋

                k4abt_joint_t  P_HIP_RIGHT = skeleton.joints[K4ABT_JOINT_HIP_RIGHT];
                //3D转2D，并在color中画出,并画出右肩到右髋的连线
                k4a_calibration_3d_to_2d(&sensor_calibration, &P_HIP_RIGHT.position, K4A_CALIBRATION_TYPE_DEPTH, K4A_CALIBRATION_TYPE_COLOR, &P_HIP_RIGHT_2D, &result);
                cv::circle(cv_rgbImage_no_alpha, cv::Point(P_HIP_RIGHT_2D.xy.x, P_HIP_RIGHT_2D.xy.y), 3, cv::Scalar(0, 255, 255));

                //右膝
                k4abt_joint_t  P_KNEE_RIGHT = skeleton.joints[K4ABT_JOINT_KNEE_RIGHT];
                //3D转2D，并在color中画出,并画出右肩到右膝、右髋到右膝的连线
                k4a_calibration_3d_to_2d(&sensor_calibration, &P_KNEE_RIGHT.position, K4A_CALIBRATION_TYPE_DEPTH, K4A_CALIBRATION_TYPE_COLOR, &P_KNEE_RIGHT_2D, &result);
                cv::circle(cv_rgbImage_no_alpha, cv::Point(P_KNEE_RIGHT_2D.xy.x, P_KNEE_RIGHT_2D.xy.y), 3, cv::Scalar(0, 255, 255));

                //左肩
                k4abt_joint_t  P_SHOULDER_LEFT = skeleton.joints[K4ABT_JOINT_SHOULDER_LEFT];
                //3D转2D，并在color中画出
                k4a_calibration_3d_to_2d(&sensor_calibration, &P_SHOULDER_LEFT.position, K4A_CALIBRATION_TYPE_DEPTH, K4A_CALIBRATION_TYPE_COLOR, &P_SHOULDER_LEFT_2D, &result);
                cv::circle(cv_rgbImage_no_alpha, cv::Point(P_SHOULDER_LEFT_2D.xy.x, P_SHOULDER_LEFT_2D.xy.y), 3, cv::Scalar(0, 255, 255));



                //左髋
                k4abt_joint_t  P_HIP_LEFT = skeleton.joints[K4ABT_JOINT_HIP_LEFT];
                //3D转2D，并在color中画出,并画出左肩到左髋的连线
                k4a_calibration_3d_to_2d(&sensor_calibration, &P_HIP_LEFT.position, K4A_CALIBRATION_TYPE_DEPTH, K4A_CALIBRATION_TYPE_COLOR, &P_HIP_LEFT_2D, &result);
                cv::circle(cv_rgbImage_no_alpha, cv::Point(P_HIP_LEFT_2D.xy.x, P_HIP_LEFT_2D.xy.y), 3, cv::Scalar(0, 255, 255));

                //左膝
                k4abt_joint_t  P_KNEE_LEFT = skeleton.joints[K4ABT_JOINT_KNEE_LEFT];
                //3D转2D，并在color中画出,并画出左肩到左膝、左髋到左膝的连线
                k4a_calibration_3d_to_2d(&sensor_calibration, &P_KNEE_LEFT.position, K4A_CALIBRATION_TYPE_DEPTH, K4A_CALIBRATION_TYPE_COLOR, &P_KNEE_LEFT_2D, &result);
                cv::circle(cv_rgbImage_no_alpha, cv::Point(P_KNEE_LEFT_2D.xy.x, P_KNEE_LEFT_2D.xy.y), 3, cv::Scalar(0, 255, 255));

                //左脚腕
                k4abt_joint_t  P_ANKLE_LEFT = skeleton.joints[K4ABT_JOINT_ANKLE_LEFT];
                k4a_calibration_3d_to_2d(&sensor_calibration, &P_ANKLE_LEFT.position, K4A_CALIBRATION_TYPE_DEPTH, K4A_CALIBRATION_TYPE_COLOR, &P_ANKLE_LEFT_2D, &result);
                cv::circle(cv_rgbImage_no_alpha, cv::Point(P_ANKLE_LEFT_2D.xy.x, P_ANKLE_LEFT_2D.xy.y), 3, cv::Scalar(0, 255, 255));



                //右脚腕
                k4abt_joint_t  P_ANKLE_RIGHT = skeleton.joints[K4ABT_JOINT_ANKLE_RIGHT];
                k4a_calibration_3d_to_2d(&sensor_calibration, &P_ANKLE_RIGHT.position, K4A_CALIBRATION_TYPE_DEPTH, K4A_CALIBRATION_TYPE_COLOR, &P_ANKLE_RIGHT_2D, &result);
                cv::circle(cv_rgbImage_no_alpha, cv::Point(P_ANKLE_RIGHT_2D.xy.x, P_ANKLE_RIGHT_2D.xy.y), 3, cv::Scalar(0, 255, 255));

                //左手腕
                k4abt_joint_t  P_WRIST_LEFT = skeleton.joints[K4ABT_JOINT_WRIST_LEFT];
                k4a_calibration_3d_to_2d(&sensor_calibration, &P_WRIST_LEFT.position, K4A_CALIBRATION_TYPE_DEPTH, K4A_CALIBRATION_TYPE_COLOR, &P_WRIST_LEFT_2D, &result);
                cv::circle(cv_rgbImage_no_alpha, cv::Point(P_WRIST_LEFT_2D.xy.x, P_WRIST_LEFT_2D.xy.y), 3, cv::Scalar(0, 255, 255));

                //右手腕
                k4abt_joint_t  P_WRIST_RIGHT = skeleton.joints[K4ABT_JOINT_WRIST_RIGHT];
                k4a_calibration_3d_to_2d(&sensor_calibration, &P_WRIST_RIGHT.position, K4A_CALIBRATION_TYPE_DEPTH, K4A_CALIBRATION_TYPE_COLOR, &P_WRIST_RIGHT_2D, &result);
                cv::circle(cv_rgbImage_no_alpha, cv::Point(P_WRIST_RIGHT_2D.xy.x, P_WRIST_RIGHT_2D.xy.y), 3, cv::Scalar(0, 255, 255));

                //左手肘
                k4abt_joint_t  P_ELBOW_LEFT = skeleton.joints[K4ABT_JOINT_ELBOW_LEFT];
                k4a_calibration_3d_to_2d(&sensor_calibration, &P_ELBOW_LEFT.position, K4A_CALIBRATION_TYPE_DEPTH, K4A_CALIBRATION_TYPE_COLOR, &P_ELBOW_LEFT_2D, &result);
                cv::circle(cv_rgbImage_no_alpha, cv::Point(P_ELBOW_LEFT_2D.xy.x, P_ELBOW_LEFT_2D.xy.y), 3, cv::Scalar(0, 255, 255));

                //右手肘
                k4abt_joint_t  P_ELBOW_RIGHT = skeleton.joints[K4ABT_JOINT_ELBOW_RIGHT];
                k4a_calibration_3d_to_2d(&sensor_calibration, &P_ELBOW_RIGHT.position, K4A_CALIBRATION_TYPE_DEPTH, K4A_CALIBRATION_TYPE_COLOR, &P_ELBOW_RIGHT_2D, &result);
                cv::circle(cv_rgbImage_no_alpha, cv::Point(P_ELBOW_RIGHT_2D.xy.x, P_ELBOW_RIGHT_2D.xy.y), 3, cv::Scalar(0, 255, 255));

                //Eyes
                k4abt_joint_t  P_EYE_RIGHT = skeleton.joints[K4ABT_JOINT_EYE_RIGHT];
                k4abt_joint_t  P_EYE_LEFT= skeleton.joints[K4ABT_JOINT_EYE_LEFT];
                //EARS
                k4abt_joint_t  P_EAR_RIGHT = skeleton.joints[K4ABT_JOINT_EAR_RIGHT];
                k4abt_joint_t  P_EAR_LEFT = skeleton.joints[K4ABT_JOINT_EAR_LEFT];

                //Foot
                k4abt_joint_t  P_FOOT_RIGHT = skeleton.joints[K4ABT_JOINT_FOOT_RIGHT];
                k4abt_joint_t  P_FOOT_LEFT = skeleton.joints[K4ABT_JOINT_FOOT_LEFT];

                //Heel
                //k4abt_joint_t  P_HEEL_RIGHT = skeleton.joints[K4ABT_JOINT_FOOT_RIGHT];

                


                //关节点连线
                cv::line(cv_rgbImage_no_alpha, cv::Point(P_HEAD_2D.xy.x, P_HEAD_2D.xy.y), cv::Point(P_NECK_2D.xy.x, P_NECK_2D.xy.y), cv::Scalar(0, 0, 255), 2);
                cv::line(cv_rgbImage_no_alpha, cv::Point(P_NECK_2D.xy.x, P_NECK_2D.xy.y), cv::Point(P_SHOULDER_LEFT_2D.xy.x, P_SHOULDER_LEFT_2D.xy.y), cv::Scalar(0, 0, 255), 2);
                cv::line(cv_rgbImage_no_alpha, cv::Point(P_NECK_2D.xy.x, P_NECK_2D.xy.y), cv::Point(P_SHOULDER_RIGHT_2D.xy.x, P_SHOULDER_RIGHT_2D.xy.y), cv::Scalar(0, 0, 255), 2);
                cv::line(cv_rgbImage_no_alpha, cv::Point(P_ELBOW_RIGHT_2D.xy.x, P_ELBOW_RIGHT_2D.xy.y), cv::Point(P_SHOULDER_RIGHT_2D.xy.x, P_SHOULDER_RIGHT_2D.xy.y), cv::Scalar(0, 0, 255), 2);
                cv::line(cv_rgbImage_no_alpha, cv::Point(P_ELBOW_RIGHT_2D.xy.x, P_ELBOW_RIGHT_2D.xy.y), cv::Point(P_WRIST_RIGHT_2D.xy.x, P_WRIST_RIGHT_2D.xy.y), cv::Scalar(0, 0, 255), 2);
                cv::line(cv_rgbImage_no_alpha, cv::Point(P_WRIST_LEFT_2D.xy.x, P_WRIST_LEFT_2D.xy.y), cv::Point(P_ELBOW_LEFT_2D.xy.x, P_ELBOW_LEFT_2D.xy.y), cv::Scalar(0, 0, 255), 2);
                cv::line(cv_rgbImage_no_alpha, cv::Point(P_ELBOW_LEFT_2D.xy.x, P_ELBOW_LEFT_2D.xy.y), cv::Point(P_SHOULDER_LEFT_2D.xy.x, P_SHOULDER_LEFT_2D.xy.y), cv::Scalar(0, 0, 255), 2);
                cv::line(cv_rgbImage_no_alpha, cv::Point(P_CHEST_2D.xy.x, P_CHEST_2D.xy.y), cv::Point(P_NECK_2D.xy.x, P_NECK_2D.xy.y), cv::Scalar(0, 0, 255), 2);
                cv::line(cv_rgbImage_no_alpha, cv::Point(P_CHEST_2D.xy.x, P_CHEST_2D.xy.y), cv::Point(P_HIP_2D.xy.x, P_HIP_2D.xy.y), cv::Scalar(0, 0, 255), 2);
                cv::line(cv_rgbImage_no_alpha, cv::Point(P_HIP_LEFT_2D.xy.x, P_HIP_LEFT_2D.xy.y), cv::Point(P_HIP_2D.xy.x, P_HIP_2D.xy.y), cv::Scalar(0, 0, 255), 2);
                cv::line(cv_rgbImage_no_alpha, cv::Point(P_HIP_RIGHT_2D.xy.x, P_HIP_RIGHT_2D.xy.y), cv::Point(P_HIP_2D.xy.x, P_HIP_2D.xy.y), cv::Scalar(0, 0, 255), 2);
                cv::line(cv_rgbImage_no_alpha, cv::Point(P_HIP_RIGHT_2D.xy.x, P_HIP_RIGHT_2D.xy.y), cv::Point(P_KNEE_RIGHT_2D.xy.x, P_KNEE_RIGHT_2D.xy.y), cv::Scalar(0, 0, 255), 2);
                cv::line(cv_rgbImage_no_alpha, cv::Point(P_HIP_LEFT_2D.xy.x, P_HIP_LEFT_2D.xy.y), cv::Point(P_KNEE_LEFT_2D.xy.x, P_KNEE_LEFT_2D.xy.y), cv::Scalar(0, 0, 255), 2);
                cv::line(cv_rgbImage_no_alpha, cv::Point(P_ANKLE_RIGHT_2D.xy.x, P_ANKLE_RIGHT_2D.xy.y), cv::Point(P_KNEE_RIGHT_2D.xy.x, P_KNEE_RIGHT_2D.xy.y), cv::Scalar(0, 0, 255), 2);
                cv::line(cv_rgbImage_no_alpha, cv::Point(P_ANKLE_LEFT_2D.xy.x, P_ANKLE_LEFT_2D.xy.y), cv::Point(P_KNEE_LEFT_2D.xy.x, P_KNEE_LEFT_2D.xy.y), cv::Scalar(0, 0, 255), 2);

                // 保存25個點（按照openpose 順序）
               /*
               * 
                {0, “Nose”},
                {1, “Neck”},
                {2, “RShoulder”},
                {3, “RElbow”},
                {4, “RWrist”},
                {5, “LShoulder”},
                {6, “LElbow”},
                {7, “LWrist”},
                {8, “MidHip”},
                {9, “RHip”},
                {10, “RKnee”},
                {11, “RAnkle”},
                {12, “LHip”},
                {13, “LKnee”},
                {14, “LAnkle”},
                {15, “REye”},
                {16, “LEye”},
                {17, “REar”},
                {18, “LEar”},
                {19, “LBigToe”},
                {20, “LSmallToe”},
                {21, “LHeel”},
                {22, “RBigToe”},
                {23, “RSmallToe”},
                {24, “RHeel”}

               */
                
                //anzs
                //step1: get 25 kps of openpose
                std::vector< k4a_float3_t> source25;
                source25.push_back(P_NOSE.position); //0, “Nose”
                source25.push_back(P_NECK.position); //1, “Neck”
                source25.push_back(P_SHOULDER_RIGHT.position); //2, “RShoulder”
                source25.push_back(P_ELBOW_RIGHT.position);//3, “RElbow”                
                source25.push_back(P_WRIST_RIGHT.position); //4, “RWrist”
                source25.push_back(P_SHOULDER_LEFT.position);//5, “LShoulder”
                source25.push_back(P_ELBOW_LEFT.position); //6, “LElbow”
                source25.push_back(P_WRIST_LEFT.position);//7, “LWrist”
                source25.push_back(P_HIP.position);//8, “MidHip”
                source25.push_back(P_HIP_RIGHT.position);//9, “RHip”
                source25.push_back(P_KNEE_RIGHT.position);//10, “RKnee”
                source25.push_back(P_ANKLE_RIGHT.position);//11, “RAnkle”
                source25.push_back(P_HIP_LEFT.position);//12, “LHip”
                source25.push_back(P_KNEE_LEFT.position);//13, “LKnee”
                source25.push_back(P_ANKLE_LEFT.position);//14, “LAnkle”
                source25.push_back(P_EYE_RIGHT.position);//15, “REye”
                source25.push_back(P_EYE_LEFT.position);//16, “LEye”
                source25.push_back(P_EAR_RIGHT.position);//17, “REar”
                source25.push_back(P_EAR_LEFT.position);//18, “LEar”
                
                source25.push_back(P_FOOT_LEFT.position);//19, “LBigToe”
                source25.push_back(P_FOOT_LEFT.position);//20, “LSmallToe”

                source25.push_back(P_ANKLE_LEFT.position);//21, “LHeel”= 14, “LAnkle”

                source25.push_back(P_FOOT_RIGHT.position);//22, “RBigToe”
                source25.push_back(P_FOOT_RIGHT.position);//23, “RSmallToe””
                source25.push_back(P_ANKLE_RIGHT.position);//24, “RHeel” = 11, “RAnkle”


                //step2: convert25-->29.
                std::vector<torch::Tensor> target29 = convert25_29(source25);
                torch::Tensor target29_tensor = torch::stack(target29, 1);
                target29_tensor = target29_tensor / 800;

                //target29_tensor = target29_tensor.index({ Slice(),Slice(1),Slice(2) });
                if (SHOWOUT)
                {
                    std::cout << "target29 " << target29_tensor << std::endl;
                }
                

                //////////////////////////////////////////////////////////////////////////

				torch::DeviceType device_type;

				if (torch::cuda::is_available())
				{
                    device_type = torch::kCPU;// torch::kCUDA;
				}
				else
				{
					device_type = torch::kCPU;
				}
				torch::Device device_cuda(device_type, 0);
				device_cuda.set_index(0);

                std::string modelPath = "x64\\debug\\data\\basicModel_neutral_lbs_10_207_0_v1.0.0.npz";
                smplcam* p_smplcam = new smplcam(device_cuda);
                p_smplcam->m_smpl = SINGLE_SMPL::get();
                SINGLE_SMPL::get()->setDevice(device_cuda);
                SINGLE_SMPL::get()->setModelPath(modelPath);
                SINGLE_SMPL::get()->init();

                torch::Tensor vertices;
                torch::Tensor beta0;
                torch::Tensor theta0;

                beta0 = 0.3 * torch::rand({ BATCH_SIZE, SHAPE_BASIS_DIM }).to(device_cuda);

                float pose_rand_amplitude0 = 0.0;
                theta0 = pose_rand_amplitude0 * torch::rand({ BATCH_SIZE, JOINT_NUM, 3 }) - pose_rand_amplitude0 / 2 * torch::ones({ BATCH_SIZE, JOINT_NUM, 3 });

                SINGLE_SMPL::get()->launch(beta0, theta0);
                torch::Tensor joints = SINGLE_SMPL::get()->getRestJoint();
                //std::cout << "joints " << joints << std::endl;
                if (SHOWOUT)
                {
                    std::cout << "joint2d " << joints << std::endl;                  

                }


                p_smplcam->call_forward(target29_tensor, joints,frameId); //.hybrik(); // .skinning();

                
                //////////////////////////////////////////////////////////////////////////
                string time = Time2Str();
                outfile << time.c_str() << ",";

                //输出显示，关键点坐标（skeleton.joints_HEAD->position.v为头部坐标点，数据结构float[3]）
                std::cout << "鼻子坐标：";
                std::cout << "Joint " << i << ": (" << P_NOSE.position.v[0] << ", "
                    << P_NOSE.position.v[1] << ", "
                    << P_NOSE.position.v[2] << ")" << std::endl;
                outfile << P_NOSE.position.v[0] << ", "
                    << P_NOSE.position.v[1] << ", "
                    << P_NOSE.position.v[2] << ",";

                std::cout << "颈部坐标：";
                std::cout << "Joint " << i << ": (" << P_NECK.position.v[0] << ", "
                    << P_NECK.position.v[1] << ", "
                    << P_NECK.position.v[2] << ")" << std::endl;
                outfile << P_NECK.position.v[0] << ", "
                    << P_NECK.position.v[1] << ", "
                    << P_NECK.position.v[2] << ",";

                std::cout << "胸部坐标：";
                std::cout << "Joint " << i << ": (" << P_CHEST.position.v[0] << ", "
                    << P_CHEST.position.v[1] << ", "
                    << P_CHEST.position.v[2] << ")" << std::endl;
                outfile << P_CHEST.position.v[0] << ", "
                    << P_CHEST.position.v[1] << ", "
                    << P_CHEST.position.v[2] << ",";

                std::cout << "髋部坐标：";
                std::cout << "Joint " << i << ": (" << P_HIP.position.v[0] << ", "
                    << P_HIP.position.v[1] << ", "
                    << P_HIP.position.v[2] << ")" << std::endl;
                outfile << P_HIP.position.v[0] << ", "
                    << P_HIP.position.v[1] << ", "
                    << P_HIP.position.v[2] << ",";

                std::cout << "左髋坐标：";
                std::cout << "Joint " << i << ": (" << P_HIP_LEFT.position.v[0] << ", "
                    << P_HIP_LEFT.position.v[1] << ", "
                    << P_HIP_LEFT.position.v[2] << ")" << std::endl;
                outfile << P_HIP_LEFT.position.v[0] << ", "
                    << P_HIP_LEFT.position.v[1] << ", "
                    << P_HIP_LEFT.position.v[2] << ",";
                

                std::cout << "右髋坐标：";
                std::cout << "Joint " << i << ": (" << P_HIP_RIGHT.position.v[0] << ", "
                    << P_HIP_RIGHT.position.v[1] << ", "
                    << P_HIP_RIGHT.position.v[2] << ")" << std::endl;
                outfile << P_HIP_LEFT.position.v[0] << ", "
                    << P_HIP_LEFT.position.v[1] << ", "
                    << P_HIP_LEFT.position.v[2] << ",";

                std::cout << "左膝坐标：";
                std::cout << "Joint " << i << ": (" << P_KNEE_LEFT.position.v[0] << ", "
                    << P_KNEE_LEFT.position.v[1] << ", "
                    << P_KNEE_LEFT.position.v[2] << ")" << std::endl;
                outfile << P_KNEE_LEFT.position.v[0] << ", "
                    << P_KNEE_LEFT.position.v[1] << ", "
                    << P_KNEE_LEFT.position.v[2] << ",";

                std::cout << "右膝坐标：";
                std::cout << "Joint " << i << ": (" << P_KNEE_RIGHT.position.v[0] << ", "
                    << P_KNEE_RIGHT.position.v[1] << ", "
                    << P_KNEE_RIGHT.position.v[2] << ")" << std::endl;
                outfile << P_KNEE_RIGHT.position.v[0] << ", "
                    << P_KNEE_RIGHT.position.v[1] << ", "
                    << P_KNEE_RIGHT.position.v[2] << ",";

                std::cout << "左腕坐标：";
                std::cout << "Joint " << i << ": (" << P_WRIST_LEFT.position.v[0] << ", "
                    << P_WRIST_LEFT.position.v[1] << ", "
                    << P_WRIST_LEFT.position.v[2] << ")" << std::endl;
                outfile << P_WRIST_LEFT.position.v[0] << ", "
                    << P_WRIST_LEFT.position.v[1] << ", "
                    << P_WRIST_LEFT.position.v[2] << ",";

                std::cout << "右腕坐标：";
                std::cout << "Joint " << i << ": (" << P_WRIST_RIGHT.position.v[0] << ", "
                    << P_WRIST_RIGHT.position.v[1] << ", "
                    << P_WRIST_RIGHT.position.v[2] << ")" << std::endl;
                outfile << P_WRIST_RIGHT.position.v[0] << ", "
                    << P_WRIST_RIGHT.position.v[1] << ", "
                    << P_WRIST_RIGHT.position.v[2] << ",";

                std::cout << "左肘坐标：";
                std::cout << "Joint " << i << ": (" << P_ELBOW_LEFT.position.v[0] << ", "
                    << P_ELBOW_LEFT.position.v[1] << ", "
                    << P_ELBOW_LEFT.position.v[2] << ")" << std::endl;
                outfile << P_ELBOW_LEFT.position.v[0] << ", "
                    << P_ELBOW_LEFT.position.v[1] << ", "
                    << P_ELBOW_LEFT.position.v[2] << ",";

                std::cout << "右肘坐标：";
                std::cout << "Joint " << i << ": (" << P_ELBOW_RIGHT.position.v[0] << ", "
                    << P_ELBOW_RIGHT.position.v[1] << ", "
                    << P_ELBOW_RIGHT.position.v[2] << ")" << std::endl;
                outfile << P_ELBOW_RIGHT.position.v[0] << ", "
                    << P_ELBOW_RIGHT.position.v[1] << ", "
                    << P_ELBOW_RIGHT.position.v[2] << ",";

                std::cout << "左脚腕坐标：";
                std::cout << "Joint " << i << ": (" << P_ANKLE_LEFT.position.v[0] << ", "
                    << P_ANKLE_LEFT.position.v[1] << ", "
                    << P_ANKLE_LEFT.position.v[2] << ")" << std::endl;
                outfile << "(" << P_ANKLE_LEFT.position.v[0] << ", "
                    << P_ANKLE_LEFT.position.v[1] << ", "
                    << P_ANKLE_LEFT.position.v[2] << ",";

                std::cout << "右脚腕坐标：";
                std::cout << "Joint " << i << ": (" << P_ANKLE_RIGHT.position.v[0] << ", "
                    << P_ANKLE_RIGHT.position.v[1] << ", "
                    << P_ANKLE_RIGHT.position.v[2] << ")" << std::endl;
                outfile << P_ANKLE_LEFT.position.v[0] << ", "
                    << P_ANKLE_LEFT.position.v[1] << ", "
                    << P_ANKLE_LEFT.position.v[2] << ",";

                std::cout << "右肩坐标：";
                std::cout << "Joint " << i << ": (" << P_SHOULDER_RIGHT.position.v[0] << ", "
                    << P_SHOULDER_RIGHT.position.v[1] << ", "
                    << P_SHOULDER_RIGHT.position.v[2] << ")" << std::endl;
                outfile << P_SHOULDER_RIGHT.position.v[0] << ", "
                    << P_SHOULDER_RIGHT.position.v[1] << ", "
                    << P_SHOULDER_RIGHT.position.v[2] << ",";

                std::cout << "左肩坐标：";
                std::cout << "Joint " << i << ": (" << P_SHOULDER_LEFT.position.v[0] << ", "
                    << P_SHOULDER_LEFT.position.v[1] << ", "
                    << P_SHOULDER_LEFT.position.v[2] << ")" << std::endl;
                outfile << P_SHOULDER_RIGHT.position.v[0] << ", "
                    << P_SHOULDER_RIGHT.position.v[1] << ", "
                    << P_SHOULDER_RIGHT.position.v[2] << ",";


                // 左肘， 右肘角度
                double left_elbow_angle = get_angle(P_WRIST_LEFT_2D.xy.x, P_WRIST_LEFT_2D.xy.y, P_ELBOW_LEFT_2D.xy.x, P_ELBOW_LEFT_2D.xy.y, P_SHOULDER_LEFT_2D.xy.x, P_SHOULDER_LEFT_2D.xy.y);
                double right_elbow_angle = get_angle(P_WRIST_RIGHT_2D.xy.x, P_WRIST_RIGHT_2D.xy.y, P_ELBOW_RIGHT_2D.xy.x, P_ELBOW_RIGHT_2D.xy.y, P_SHOULDER_RIGHT_2D.xy.x, P_SHOULDER_RIGHT_2D.xy.y);

                //左膝，右膝角度
                double left_knee_angle = get_angle(P_HIP_LEFT_2D.xy.x, P_HIP_LEFT_2D.xy.y, P_KNEE_LEFT_2D.xy.x, P_KNEE_LEFT_2D.xy.y, P_ANKLE_LEFT_2D.xy.x, P_ANKLE_LEFT_2D.xy.y);
                double right_knee_angle = get_angle(P_HIP_RIGHT_2D.xy.x, P_HIP_RIGHT_2D.xy.y, P_ELBOW_RIGHT_2D.xy.x, P_ELBOW_RIGHT_2D.xy.y, P_SHOULDER_RIGHT_2D.xy.x, P_SHOULDER_RIGHT_2D.xy.y);

                //左肩，右肩角度
                double k1 = get_line_equation(P_NECK_2D.xy.x, P_NECK_2D.xy.y, P_CHEST_2D.xy.x, P_CHEST_2D.xy.y);
                double k2 = get_line_equation(P_SHOULDER_LEFT_2D.xy.x, P_SHOULDER_LEFT_2D.xy.y, P_ELBOW_LEFT_2D.xy.x, P_ELBOW_LEFT_2D.xy.y);
                double k3 = get_line_equation(P_SHOULDER_RIGHT_2D.xy.x, P_SHOULDER_RIGHT_2D.xy.y, P_ELBOW_RIGHT_2D.xy.x, P_ELBOW_RIGHT_2D.xy.y);
                double left_shoulder_angle = get_lines_angle(k1, k2);
                double right_shoulder_angle = get_lines_angle(k1, k3);

                outfile << left_elbow_angle << "," << right_elbow_angle << "," << left_knee_angle << "," << right_knee_angle << ","
                    << left_shoulder_angle << "," << right_shoulder_angle << endl;

                //按smpl 格式 保存關節點，並計算pose


            }

            

        }
        //std::this_thread::sleep_for(std::chrono::seconds(5));
        // show image
        imshow("color", cv_rgbImage_no_alpha);
        //imshow("depth", cv_depth_8U);
        cv::waitKey(1);
        k4a_image_release(rgbImage);
        
        frameId++;
    }

    return 0;
}