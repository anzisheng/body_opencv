
//#include <torch/torch.h>
//#include <torch/extension.h>
//#include <torch/extension.h>
#define _USE_MATH_DEFINES
#include <k4a/k4a.h>
#define _USE_MATH_DEFINES
#include <stdio.h>
//__global__ void kernel();
//#include "hiredis.h"
#include <stdlib.h>
#include <iostream>
//#include <tbb/parallel_for.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <torch/extension.h>

#include <assert.h>
#include <winsock2.h>
#include <string.h>
#include <tuple>
#include <iostream>
#include <sstream>
#include <string.h>
#include"hiredis/hiredis.h"


#include <stdio.h>
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
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include "torch/script.h"
#include "nlohmann/json.hpp"
//#include <json/json.h>

//#include "C:\Program Files (x86)\Intel\oneAPI\tbb\2021.10.0\include\tbb\parallel_for.h"

//#include <tbb/parallel_for.h>
//#include <tbb/parallel_for.h>
//#include "main.h"
using namespace torch::indexing;
using namespace std;
using namespace cv;
//#include <Device.h>
//#define SINGLE_SMPL smpl::Singleton<smpl::SMPL>
//smpl::SMPL* g_smpl = new smpl::SMPL;

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






void write_json(ofstream& myfile, const int id, const torch::Tensor &Rh, const torch::Tensor &Th, const torch::Tensor &poses, const torch::Tensor &shapes)
{
    /*
    * quat = quat.to(torch::kCPU);
    ofstream  myfile("data/000000.json");
    //out_text.append('[\n')
    myfile << "[\n";

    double* ptr = (double*)quat.data_ptr();
    for (size_t i = 0; i < 72; i++) {
        try
        {
            //std::cout << *((ptr + i)) << std::endl;
            myfile << *((ptr + i));
            myfile << ", ";

        }
        catch (const std::exception& e)
        {
            std::cout << e.what() << std::endl;
            throw;
        }

    }
    myfile << "]\n";
    myfile.close();

    */
    myfile << "{\n";
    myfile << "\"id\":" << id << ",\n";
    myfile << "\"Rh\": " << "[\n";
    myfile << "[";
    if (SHOWOUT)
    {
        std::cout << "Rh" << Rh << std::endl;
        std::cout << "Th" << Th << std::endl;
        std::cout << "poses" << poses << std::endl;
        std::cout << "shapes" << shapes << std::endl;

    }
    float* ptr_Rh = (float*)Rh.data_ptr();
    for (size_t i = 0; i < 2; i++)
    {
        myfile << (float)*((ptr_Rh + i));
        myfile << ", ";
    }
    myfile << *((ptr_Rh + 2));
    myfile << "]\n],\n";

    myfile << "\"Th\": " << "[\n";
    myfile << "[";
    float* ptr_Th = (float*)Th.data_ptr();
    for (size_t i = 0; i < 2; i++)
    {
        myfile << (float)*((ptr_Th + i));
        myfile << ", ";
    }
    myfile << (float)*((ptr_Th + 2));
    myfile << "]\n],\n";

    myfile << "\"poses\": " << "[\n";
    myfile << "[";
    double* ptr_poses = (double*)poses.data_ptr();
    for (size_t i = 0; i < 71; i++)
    {
        myfile << *((ptr_poses + i));
        myfile << ", ";
    }
    myfile << *((ptr_poses + 71));
    myfile << "]\n],\n";

    myfile << "\"shapes\": " << "[\n";
    myfile << "[";
    float* shapes_ptr = (float*)shapes.data_ptr();
    for (size_t i = 0; i < 9; i++)
    {
        myfile << *((shapes_ptr + i));
        myfile << ", ";
    }
    myfile << *((shapes_ptr + 9));
    myfile << "]\n]\n";
    myfile << "}\n";


}

/////////////////////////////////////////////////////////

class RedisConnect {
public:
    RedisConnect() :redisCon(nullptr), reply(nullptr) {}
    bool Init(const std::string& ip, int port) {
        if (nullptr != redisCon) {
            return false;
        }
        redisCon = redisConnect(ip.c_str(), port);
        if (redisCon->err) {
            std::cerr << "error code : " << redisCon->err << ". " << redisCon->errstr << std::endl;
            return false;
        }

        return true;
    }
    void freeReply()
    {
        if (nullptr != reply)
        {
            ::freeReplyObject(reply);
            reply = nullptr;
        }
    }

    template<class T, class... Args>
    bool HashSet(const std::string command, T head, Args... rest) {
        std::stringstream ss;
        ss << command << " " << head << " ";
        return HashSetInner(ss, rest...);
    }

    template<typename T>
    bool Set(const std::string& key, const T& value)
    {
        bool bret = false;
        std::stringstream ss;
        ss << "SET " << key << " " << value;
        //ss << "PUBLISH " << key << " " << value;
        std::string s;
        getline(ss, s);
        return Set(s);
    }

    template<typename T>
    bool Publish(const std::string& key, const T& value)
    {
        bool bret = false;
        std::stringstream ss;
        ss << "PUBLISH " << key << " " << value;
        //ss << "PUBLISH " << key << " " << value;
        std::string s;
        getline(ss, s);
        return Publish(s);
    }



    bool InitWithTimeout(const std::string& ip, int port, int seconds) {
        if (nullptr != redisCon) {
            return false;
        }
        struct timeval tv;
        tv.tv_sec = seconds;
        tv.tv_usec = 0;
        redisCon = redisConnectWithTimeout(ip.c_str(), port, tv);
        if (redisCon->err) {
            std::cerr << "error code : " << redisCon->err << ". " << redisCon->errstr << std::endl;
            return false;
        }
        return true;
    }

    ~RedisConnect() {
        freeReply();
        if (nullptr == redisCon) {
            redisFree(redisCon);
            redisCon = nullptr;
        }
    }
private:
    bool HashSetInner(std::stringstream& ss)
    {
        std::string data;
        getline(ss, data);
        //std::cout << __FUNCTION__ << " " << data << std::endl;
        bool bret = false;
        freeReply();
        reply = (redisReply*)::redisCommand(redisCon, data.c_str());

        if (reply->type == REDIS_REPLY_ERROR ||
            (reply->type == REDIS_REPLY_STATUS && _stricmp(reply->str, "OK") != 0))
        {
            if (reply->str != nullptr) {
                std::cout << reply->str << std::endl;
            }
            std::cout << "Failed to execute " << __FUNCTION__ << std::endl << std::endl;
            return bret;
        }

        bret = true;
        return bret;
    }

    template<class T, class... Args>
    bool HashSetInner(std::stringstream& ss, T head, Args... rest)
    {
        ss << head << " ";
        return HashSetInner(ss, rest...);
    }

    bool Set(std::string data)
    {
        bool bret = false;
        freeReply();
        reply = (redisReply*)::redisCommand(redisCon, data.c_str());

        if (!(reply->type == REDIS_REPLY_STATUS && _stricmp(reply->str, "OK") == 0))
        {
            std::cout << reply->str << std::endl;
            std::cout << "Failed to execute " << __FUNCTION__ << std::endl;
            return bret;
        }
        bret = true;
        return bret;
    }

    bool Publish(std::string data)
    {
        //typedef unsigned char byte;
        bool bret = false;
        freeReply();
        //byte* px = (byte*)data.c_str();
        reply = (redisReply*)::redisCommand(redisCon, data.c_str());

        /*        if (!(reply->type == REDIS_REPLY_STATUS && _stricmp(reply->str, "OK") == 0))
                {
                    std::cout << reply->str << std::endl;
                    std::cout << "Failed to execute " << __FUNCTION__ << std::endl;
                    return bret;
                }
        */

        bret = true;
        return bret;
    }



    redisContext* redisCon;
    redisReply* reply;
};






void write_redis(std::vector<SMPL::person*> persons, RedisConnect r, int timestamp, int frame_id)
{
    //string name = "dddd"+ string(obj_id);
    //int timestamp = 1111113;
    //int obj_id = 4;


    //torch::Tensor Thtemp = torch::ones({ 72 });
    using json = nlohmann::json;
    for (std::vector<SMPL::person*>::iterator iter = persons.begin(); iter != persons.end(); iter++)
    {
        SMPL::person* p = *iter;
        std::vector<float>  thetas(72);
        float* ptr_Th = (float*)p->m_poses.data_ptr();
        for (size_t i = 0; i < 72; i++)
        {
            thetas[i] = (float)*((ptr_Th + i));
            //myfile << (float)*((ptr_Th + i));
        }

        std::vector<float>  trans(3);
        ptr_Th = (float*)p->m_Th.data_ptr();
        for (size_t i = 0; i < 3; i++)
        {
            trans[i] = (float)*((ptr_Th + i));
        }


        ostringstream oss;
        oss << "audiChinaHeadquater_" << timestamp << "_" << p->m_id;
        nlohmann::json json_data;
        json_data["type"] = "setFrame";
        nlohmann::json data_all;
        data_all["modelName"] = oss.str();
        data_all["boneNames"] = { "Pelvis","L_Hip","R_Hip","Spine1","L_Knee", "R_Knee", "Spine2", "L_Ankle","R_Ankle","Spine3", "L_Foot", "R_Foot", "Neck", "L_Collar", "R_Collar", "Head", "L_Shoulder",
        "R_Shoulder","L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist","L_Hand", "R_Hand" };
        data_all["scene"] = "audiChinaHeadquater";
        nlohmann::json frameData;
        frameData["poses"] = thetas;
        frameData["trans"] = trans;
        data_all["frameData"] = frameData;
        data_all["frame_id"] = frame_id;
        data_all["modelUrl"] = "/models/unionavatar/biden_smpl.glb";
        data_all["create"] = true;
        data_all["clamp"] = true;

        json_data["data"] = data_all;

        std::string s2 = json_data.dump();// .encode('utf-8');
        cout << json_data << endl;
        //r.Publish(" message", s);
        r.Publish(" message", s2);

     
    }
   

 
}

void write_persons(std::vector<SMPL::person*> persons, ofstream& file)
{
    file << "[\n";
    int num = persons.size();
    int index = 0;
    for (std::vector<SMPL::person*>::iterator iter = persons.begin(); iter != persons.end(); iter++)
    {
        SMPL::person* p = *iter;
        write_json(file, p->m_id, p->m_Rh, p->m_Th, p->m_poses, p->m_shapes);
        if (index != num - 1)
        {
            file << ",";
            index++;
        }
    }
    file << "]";

}


std::tuple<torch::Tensor, torch::Tensor> umeyama(torch::Tensor src, torch::Tensor dst)
{
    // 		"""Estimate N-D similarity transformation with or without scaling.
    // 			Parameters
    // 			----------
    // 			src : (M, N) array
    // 			Source coordinates.
    // 			dst : (M, N) array
    // 			Destination coordinates.
    // 			estimate_scale : bool
    // 			Whether to estimate scaling factor.
    // 			Returns
    // 			------ -
    // 			T : (N + 1, N + 1)
    // 			The homogeneous similarity transformation matrix.The matrix contains
    // 			NaN values only if the problem is not well - conditioned.

    int num = src.size(0);// [0] ;
    int dim = src.size(1);
    // Compute mean of src and dst.
    torch::Tensor src_mean = src.mean(0);
    if (SHOWOUT)
    {
        std::cout << "src_mean" << src_mean << std::endl;
    }
    torch::Tensor dst_mean = dst.mean(0);
    if (SHOWOUT)
    {
        std::cout << "src_mean" << dst_mean << std::endl;
    }

    //# Subtract mean from src and dst.
    torch::Tensor 	src_demean = src - src_mean;
    if (SHOWOUT)
    {
        std::cout << "src_demean " << src_demean << std::endl;
    }
    torch::Tensor 	dst_demean = dst - dst_mean;
    if (SHOWOUT)
    {
        std::cout << "dst_demean  " << dst_demean << std::endl;
    }


    torch::Tensor 	A = torch::mm(torch::t(dst_demean), src_demean) / num;
    if (SHOWOUT)
    {
        std::cout << "A" << A << std::endl;
    }
    torch::Tensor d = torch::ones({ dim });// , dtype = np.double)

    //torch::linalg.det(A);
    if (torch::linalg::det(A).item().toFloat() < 0)
    {
        d[dim - 1] = -1;
    }
    if (SHOWOUT)
    {
        std::cout << "d" << torch::linalg::det(A).item().toFloat() << "d " << d << std::endl;
    }

    torch::Tensor 	T = torch::eye(dim + 1);// , dtype = np.double)

    if (SHOWOUT)
    {
        std::cout << "T" << T << std::endl;
    }
    std::tuple<at::Tensor, at::Tensor, at::Tensor> t;
    t = torch::svd(A);

    torch::Tensor U = std::get<0>(t);
    torch::Tensor S = std::get<1>(t);
    torch::Tensor V = std::get<2>(t);
    V = V.t();
    if (SHOWOUT)
    {
        std::cout << "U" << U << std::endl;
        std::cout << "S" << S << std::endl;
        std::cout << "V" << V << std::endl;
    }


    int rank = torch::linalg::matrix_rank(A, 0, true).item().toInt();//anzs

    if (rank == dim - 1)
    {
        if ((torch::linalg::det(U) * torch::linalg::det(V)).item().toInt() > 0)
        {
            T.index({ Slice(None,dim), Slice(None,dim) }) = torch::dot(U, V);
        }
        else
        {
            torch::Tensor s = d[dim - 1];
            d[dim - 1] = -1;
            T.index({ Slice(None, dim), Slice(None, dim) }) = torch::dot(U, torch::dot(torch::diag(d), V));
            d[dim - 1] = s;
        }
    }
    else
    {
        T.index({ Slice(None,dim), Slice(None, dim) }) = torch::mm(U, torch::mm(torch::diag(d), V.transpose(0, 1)));
        if (SHOWOUT)
        {
            std::cout << "T" << T << std::endl;
        }

    }

    float scale = 1.0f;

    /* Is there a way to insert a tensor into an existing tensor?
    * https://discuss.pytorch.org/t/is-there-a-way-to-insert-a-tensor-into-an-existing-tensor/14642
    * a = torch.rand(3, 4)
    b = torch.zeros(1, 4)

    idx = 1
    c = torch.cat([a[:idx], b, a[idx:]], 0)
    */

    torch::Tensor homo_src;
    //torch::Tensor homo_src = torch::cat(src, 3, 1, 1).T; //在第三列上插入1
    torch::Tensor src_t = src.t();// index({ Slice(None,3) });
    torch::Tensor last_one_row = torch::ones({ 1, src_t.size(1) });
    homo_src = torch::cat({ src_t,last_one_row }, 0);
    if (SHOWOUT)
    {
        std::cout << "src" << src << std::endl;
        std::cout << "src_t" << src_t << std::endl;
        std::cout << "last_one_row" << last_one_row << std::endl;
        std::cout << "homo_src" << homo_src << std::endl;
    }

    torch::Tensor  rot = T.index({ Slice(None,dim),Slice(None,dim) });// [:dim, : dim] ;
    if (SHOWOUT)
    {
        std::cout << "rot" << rot << std::endl;
    }

    std::vector< torch::Tensor> rots;// = [];
    std::vector< float>	losses;// = []
    for (int i = 0; i < 2; i++)
    {
        if (i == 1)
        {
            //rot.index({Slice(), Slice(None,2)})/*[:, : 2] */ *= -1;
            rot.index({ Slice(), Slice(None,2) }) *= -1;
            if (SHOWOUT)
            {
                std::cout << "rot" << rot << std::endl;
            }
        }
        //transform = np.eye(dim + 1, dtype = np.double)
        torch::Tensor transform = torch::eye(dim + 1);
        transform.index({ Slice(None,dim),Slice(None,dim) }) = rot * scale;
        if (SHOWOUT)
        {
            std::cout << "transform" << transform << std::endl;

            std::cout << "src_mean.t()" << src_mean.t().reshape({ -1, 1 }) << std::endl;

            torch::Tensor ttt = T.index({ Slice(None,dim), Slice(None,dim) });
            std::cout << "ttt" << ttt << std::endl;

        }


        if (SHOWOUT)
        {
            torch::Tensor tttt = torch::mm(T.index({ Slice(None,dim), Slice(None,dim) }), src_mean.t().reshape({ -1, 1 }));
            //torch::Tensor ttt = T.index({ Slice(None,dim), Slice(None,dim) });
            std::cout << "ttt" << tttt << std::endl;
        }

        //transform[:dim,dim] = dst_mean - scale * np.dot(T[:dim, :dim], src_mean.T)
        try
        {
            transform.index({ Slice(None, dim),dim }) = dst_mean - scale * torch::mm(T.index({ Slice(None,dim), Slice(None,dim) }), src_mean.t().reshape({ -1, 1 })).squeeze();
        }
        catch (const std::exception& e)
        {
            std::cout << e.what() << std::endl;
            throw;
        }
        if (SHOWOUT)
        {
            std::cout << "transform" << transform << std::endl;
        }


        torch::Tensor transed = torch::mm(transform, homo_src).t().index({ Slice(), Slice(None, 3) });// [:, : 3]
        float loss = torch::norm(transed - dst, 2).item().toFloat();
        //losses.append(loss);
        losses.push_back(loss);
        //rots.append(rot.copy())
        rots.push_back(rot/*.clone()*/);

    }

    //# since only the smpl parameters is needed, we return rotation, translation and scale
//# since only the smpl parameters is needed, we return rotation, translation and scale
    torch::Tensor  trans;
    try {
        trans = dst_mean - scale * torch::mm(T.index({ Slice(None, dim),Slice(None,dim) }), src_mean.t().reshape({ -1, 1 })).squeeze();// [:dim, : dim] , src_mean.T)
    }
    catch (const std::exception& e)
    {
        std::cout << e.what() << std::endl;
        throw;
    }
    if (losses[0] > losses[1])
    {
        rot = rots[1];
    }
    else
    {
        rot = rots[0];
    }
    if (SHOWOUT)
    {
        std::cout << "trans" << trans << std::endl;
        std::cout << "rot" << rot << std::endl;
    }


    std::tuple<torch::Tensor, torch::Tensor> result = std::make_tuple(rot, trans);

    //, trans, scale
    return result;


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


cudaError_t addWithCuda();
cudaError_t umeyama_cuda(int thread_num, torch::Tensor src, torch::Tensor dst);

__device__ int ss(torch::Tensor src)
{
    //int num = src.sizes()[0];
    return 0;// num;
}
__global__ void umeyamaKernel(torch::Tensor src, torch::Tensor dst)
{
    printf("Hello, world from GPU!\n ");
    //src = src.to(torch::kCUDA).clone();
    //int s = src.size(0);
    // 
    // 
    //int tt = ss(dst);
    //int num = dst.size(0);// [0] ;
    //int dim = src.size(1);
/*
    if(SHOWOUT)
    {
        printf("Hello, world from GPU!\n %d, %d", num,dim);
    }
    // Compute mean of src and dst.
    torch::Tensor src_mean = src.mean(0);
*/

}

__global__ void addKernel()
{
    printf("Hello, world from GPU!\n");
}

//#include <torch/extension.h>


/*
template <typename scalar_t>
__global__ void trilinear_fw_kernel(
    const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> feats,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> points,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> feat_interp
) 
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int f = blockIdx.y * blockDim.y + threadIdx.y;
    if (n >= feats.size(0) || f >= feats.size(2)) return;
    // point -1~1
    const scalar_t u = (points[n][0] + 1) / 2;
    const scalar_t v = (points[n][1] + 1) / 2;
    const scalar_t w = (points[n][2] + 1) / 2;

    const scalar_t a = (1 - v) * (1 - w);
    const scalar_t b = (1 - v) * w;
    const scalar_t c = v * (1 - w);
    const scalar_t d = 1 - a - b - c;
    feat_interp[n][f] = (1 - u) * (a * feats[n][0][f] +
        b * feats[n][1][f] +
        c * feats[n][2][f] +
        d * feats[n][3][f]) +
        u * (a * feats[n][4][f] +
            b * feats[n][5][f] +
            c * feats[n][6][f] +
            d * feats[n][7][f]);
}
*/

template <typename scalar_t>
torch::Tensor trilinear_fw_cu(
    const torch::Tensor feats,
    const torch::Tensor points
) 
{
    
    const int N = feats.size(0), F = feats.size(2);

    torch::Tensor feat_interp = torch::empty({ N, F }, feats.options());

    const dim3 threads(16, 16);
    const dim3 blocks((N + threads.x - 1) / threads.x, (F + threads.y - 1) / threads.y);

    /*
    trilinear_fw_kernel<scalar_t> <<<blocks, threads >>> (
        feats.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
        points.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
        feat_interp.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>()
        );
        */
    return feats;

}

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor trilinear_interpolation_fw(
    const torch::Tensor feats,
    const torch::Tensor points
) {
    CHECK_INPUT(feats);
    CHECK_INPUT(points);

    return  trilinear_fw_cu<float>(feats, points);
}



using json = nlohmann::json;

int main(int argc, char const* argv[])
{
    //cuda device    
    
// 	torch::Device cuda(torch::kCUDA);    
// 	cuda.set_index(0);

    unsigned int j;
    //redisContext* c;
    //redisReply* reply;


    RedisConnect r;
    redisReply* reply;

    bool b = r.InitWithTimeout("127.0.0.1", 6379, 1);

    if (!b)
        return -1;

    int frame_id = 10;


    
    /*
    json ex3 = {
    {"type", "setFrame"},
    {"data", {"modelName",oss.str()},
    {"boneNames",{"Pelvis","L_Hip","R_Hip","Spine1","L_Knee", "R_Knee", "Spine2", "L_Ankle","R_Ankle","Spine3", "L_Foot", "R_Foot", "Neck", "L_Collar", "R_Collar", "Head", "L_Shoulder", 
    "R_Shoulder","L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist","L_Hand", "R_Hand"}},
    {"scene","audiChinaHeadquater"},
    {"frameData",{"poses",thetas},{"trans",{1,2,3}}},
     {"frame_id",frame_id},
     {"modelUrl", "/models/unionavatar/biden_smpl.glb"},
     {"create",true},
     {"clamp", true}
    },
    };*/
   
    /*
    json ex4 = {
        {"type", "setFrame"} ,
        {"boneNames", {"Pelvis","L_Hip","R_Hip","Spine1","L_Knee", "R_Knee", "Spine2", "L_Ankle","R_Ankle"}}};
    */
    

        //\"setFrame\"\n\"data\": {\"modelName\": f\"{" << "audiChinaHeadquater" << "}_{" << timestamp << "}_{" << obj_id << "},\n"
        //<< "\t \"boneNames\":" << bone_names.str() << "\n\t\"scene\": audiChinaHeadquater," << "\n \"frameData\": {\"poses\":";

    //ss << ""{"type": "setFrame"" << ""data" : {"modelName": f"{self.scene_name}_{self.timestamp}_{obj_id}"" < <std::endl;
    //ss << "SET " << key << " " << value;




  
    /*
    r.HashSet("hset", "myhash", "field1", 123.2342343);
    r.HashSet("hmset", "myhash", "field1", 1111, "field2", "f2");
    r.HashSet("hset", "myhash", "field1", 123.2342343);
    r.HashSet("hmset", "myhash", "field1", 1111, "field2", "f2");

    //wrong command
    r.HashSet("hset", "myhash", "field1", 1, 123.2342343);
    r.HashSet("hmset", "myhash", "field1", 1, 1111, "field2", "f2");

    */



    torch::DeviceType device_type_global;

    if (torch::cuda::is_available())
    {
        device_type_global = torch::kCUDA;// torch::kCUDA;
    }
    else
    {
        device_type_global = torch::kCPU;
    }
    torch::Device device_global(device_type_global, 0);
    device_global.set_index(0);

    /*
    const int N = 65536;
    const int F = 256;
    torch::Tensor rand = torch::rand({ N, 8, F }).to(device_global);// .to(torch::CUDA);// , device = 'cuda')
    torch::Tensor feats = rand.clone().requires_grad_();
    torch::Tensor feats2 = rand.clone().requires_grad_();
    //cout << feats << endl;
    torch::Tensor points = torch::rand({ N, 3 }).to(device_global);// , device = 'cuda') * 2 - 1
    //cout << points << endl << feats << endl;
    // anzs
    //torch::Tensor feat_interp =  trilinear_interpolation_fw(feats, points);
    //cout << feat_interp.sizes() << endl;
    */
	using ms = std::chrono::milliseconds;
	using clk = std::chrono::system_clock;

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
    

    smpl::SMPL* g_smpl = new smpl::SMPL;

    torch::DeviceType device_type;

    if (torch::cuda::is_available())
    {
        device_type = torch::kCPU;// CUDA;// torch::kCUDA;
    }
    else
    {
        device_type = torch::kCPU;
    }
    torch::Device device_cuda(device_type, 0);
    device_cuda.set_index(0);

    std::string modelPath = "x64\\debug\\data\\basicModel_neutral_lbs_10_207_0_v1.0.0.npz";
    //SINGLE_SMPL::get()->setDevice(device_cuda);
    g_smpl->setDevice(device_cuda);

    //SINGLE_SMPL::get()->setModelPath(modelPath);
    g_smpl->setModelPath(modelPath);

    //SINGLE_SMPL::get()->init();
    g_smpl->init();

    int frameId = 0;
    k4a_capture_t sensor_capture;

    while(true)
    {
        
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





        std::vector<SMPL::person*> g_persons;// (num_bodies);
        //弹出结果
        k4abt_frame_t body_frame = NULL;
        k4a_wait_result_t pop_frame_result = k4abt_tracker_pop_result(tracker, &body_frame, K4A_WAIT_INFINITE);
        if (pop_frame_result == K4A_WAIT_RESULT_SUCCEEDED)
        {
            // Successfully popped the body tracking result. Start your processing
            //检测人体数
            size_t num_bodies = k4abt_frame_get_num_bodies(body_frame); //取帧
            
            
            //依次计算每个人的smpl pose，global_trans 和global_trans, 保存到person中
            
            


            torch::Tensor vertices;
            torch::Tensor beta0;
            torch::Tensor theta0;

            beta0 = 0.3 * torch::rand({ BATCH_SIZE, SHAPE_BASIS_DIM }).to(device_cuda);

            float pose_rand_amplitude0 = 0.0;
            theta0 = pose_rand_amplitude0 * torch::rand({ BATCH_SIZE, JOINT_NUM, 3 }) - pose_rand_amplitude0 / 2 * torch::ones({ BATCH_SIZE, JOINT_NUM, 3 });

            //SINGLE_SMPL::get()->launch(beta0, theta0);
            g_smpl->launch(beta0, theta0);
            //torch::Tensor g_joints = SINGLE_SMPL::get()->getRestJoint();
            torch::Tensor g_joints = g_smpl->getRestJoint();
            //std::cout << "joints " << joints << std::endl;
            if (SHOWOUT)
            {
                std::cout << "g_joints" << g_joints << std::endl;

            }





           auto time_begin = clk::now();
//           const std::time_t t_c = std::chrono::system_clock::to_time_t(time_begin);
//           std::cout << "The system clock is currently at " << int(t_c);

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
                                
                smplcam* p_smplcam = new smplcam(device_cuda);
                torch::Tensor pose;
                p_smplcam->m_smpl = g_smpl;// SINGLE_SMPL::get();
               
                //SINGLE_SMPL::destroy();


                
				auto begin0 = clk::now();

                //auto duration = std::chrono::duration_cast<ms>(end0 - begin0);
                //std::cout << "Time duration to compute pose: " << (double)duration.count()  << " ms" << std::endl;

                //kernel<<<1, 3 >>>();
                //simpleD3DKernel();
                //addWithCuda();
                //umeyama_cuda(1,target29_tensor);

                pose = p_smplcam->call_forward(target29_tensor,/* g_joints ,*/frameId); //.hybrik(); // .skinning();

				//auto end0 = clk::now();

				auto duration = std::chrono::duration_cast<ms>(clk::now() - begin0);
				std::cout << frameId << "th frame  call_forward : " << (double)duration.count() << " ms" << std::endl;

                
                if (SHOWOUT)
                {
                    std::cout << pose << std::endl;
                }
                auto begin_umeyama = clk::now();
                //umeyama
                torch::Tensor b = torch::tensor({ 16, 17, 1, 2, 12, 0 });
                torch::Tensor joints = g_joints.index({0,{b}}).cpu();// [16] [17] .cpu();
                if (SHOWOUT)
                {
                    std::cout << "joints;" << joints << std::endl;
                    std::cout << "b:" << b << std::endl;
                }

                torch::Tensor joints3d = target29_tensor.index({ 0,{b}}).cpu()/*.clone()*/;//[:, [16, 17, 1, 2, 12, 0]])  #   [[5, 2, 12, 9]]
                if (SHOWOUT)
                {
                    std::cout << "joints3d;" << joints3d << std::endl;
                }
                std::tuple<torch::Tensor, torch::Tensor> rot_trans = umeyama(joints, joints3d);
                //umeyama_cuda(1, joints, joints3d);
                //torch::Tensor rot_global;
                //torch::Tensor trans_global;
                

                torch::Tensor rot_global = std::get<0>(rot_trans);
                //rot_global = rot_global.clone();// std::get<0>(rot_trans);
                torch::Tensor trans_global = std::get<1>(rot_trans);
                //trans_global = trans_global.clone();
                if (SHOWOUT)
                {
                    std::cout << "trans_global" << trans_global << std::endl;
                    std::cout << "rot_global" << rot_global << std::endl;
                }

                cv::Mat dst = (Mat_<float>(3, 1) << 0.0f, 0.0f, 0.0f );//anzs //cv::Mat::zeros(height, width, CV_32F);
                try
                {
                    rot_global = rot_global.reshape({ 9,1 });
                    if (SHOWOUT)
                    {
                        std::cout << "rot_global" << rot_global << std::endl;
                    }
                    auto tttt0 = rot_global.index({ Slice(0,9) }).to(torch::kFloat);// .item();
                    auto x0 = tttt0.index({ 0 }).item().toFloat();
                    auto x1 = tttt0.index({ 1 }).item().toFloat();
                    auto x2 = tttt0.index({ 2 }).item().toFloat();

                    auto x3 = tttt0.index({ 3 }).item().toFloat();
                    auto x4 = tttt0.index({ 4 }).item().toFloat();
                    auto x5 = tttt0.index({ 5 }).item().toFloat();

                    auto x6 = tttt0.index({ 6 }).item().toFloat();
                    auto x7 = tttt0.index({ 7 }).item().toFloat();
                    auto x8 = tttt0.index({ 8 }).item().toFloat();


                    cv::Mat src = (Mat_<float>(3, 3) << x0, x1, x2, x3, x4, x5, x6, x7, x8);
                    //src(0, 0) = 0;

                    cv::Rodrigues(src, dst);
                    if (SHOWOUT)
                    {
                        //std::cout << "tttt" << tttt0 << std::endl;
                        //if (SHOWOUT)
                        //{
                        //    std::cout << "dst_rot" << dst << std::endl;
                        //    //std::cout << "dst_rot" << dst(0).dims << std::endl;
                        //    //std::cout << "dst_rot" << dst(1).dims << std::endl;
                        //}
                    }


                }
                catch (const exception& e)
                {
                    std::cout << e.what() << std::endl;
                    throw;
                }                
                dst = dst.reshape(1, 3);// .clone();
                torch::Tensor rot = torch::from_blob(dst.data, { 1, 3 }, torch::kFloat);
                if (SHOWOUT)
                {
                    std::cout << "rot" << rot << std::endl;
                }

                rot = rot.clone();

                //int id = i;
                torch::Tensor Rh = rot;// torch::tensor({ 1.0f, 1.0f, 1.0f });
                torch::Tensor Th = trans_global;// torch::tensor({ 0.3, 0.3, 0.3 });
                torch::Tensor shapes = torch::zeros({ 10 });
                pose = pose.to(torch::kCPU);
                SMPL::person* p = new SMPL::person(i, Rh, Th, pose, shapes);
                g_persons.push_back(p);

                //id = 1;
                //Rh = torch::tensor({ 0.5f, 0.5f, 0.7f });
                //Th = torch::tensor({ 0.8, 0.9, 0.2 });
                //shapes = torch::zeros({ 10 });
                //pose = pose.to(torch::kCPU);
                //SMPL::person* p2 = new SMPL::person(id, Rh, Th, quat, shapes);
                //g_persons.push_back(p2);


                //char ss[7];
                //sprintf(ss, "%06d", frameId);
                ////return ss;

                //string file = string("x64\\debug\\data\\") + string(ss) + ".json";


                //ofstream myfile2(file);
                //write_persons(g_persons, myfile2);
                //myfile2.close();





                //generate a person
                //SMPL::person* p = new SMPL::person();

                
                //////////////////////////////////////////////////////////////////////////
                /*
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
                    */
                //按smpl 格式 保存關節點，並計算pose
                auto duration_ume = std::chrono::duration_cast<ms>(clk::now() - begin_umeyama);
                std::cout << "第" << frameId << "帧，共第" << i << "个人umeyama :话费 " << (double)duration_ume.count() << " ms" << std::endl;


            }  
			auto duration = std::chrono::duration_cast<ms>(clk::now() - time_begin);
			std::cout <<"第" << frameId << "帧，共"<<num_bodies << "个人 :话费 " << (double)duration.count() << " ms" << std::endl;

        }
        //std::this_thread::sleep_for(std::chrono::seconds(5));
        // show image
        imshow("color", cv_rgbImage_no_alpha);
        //imshow("depth", cv_depth_8U);
        cv::waitKey(1);
        k4a_image_release(rgbImage);
        

        char ss[7];
        sprintf(ss, "%06d", frameId);
        //return ss;

        string file = string("x64\\debug\\data\\") + string(ss) + ".json";


        ofstream myfile2(file);
        write_persons(g_persons, myfile2);
        

        //RedisConnect r;
        //redisReply* reply;


        auto timestamp = clk::now();
        const std::time_t t_c = std::chrono::system_clock::to_time_t(timestamp);
        int stamp = int(t_c);
        std::cout << "The system clock is currently at " << int(t_c);

        write_redis(g_persons,r, stamp, frameId);
        myfile2.close();

        //auto time_ = clk::now();
        //auto duration = 		


        frameId++;
        

    }
    delete g_smpl;
    //delete p_smplcam;
    writer.release();
    return 0;
}
// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda()
{
    //addKernel <<<1, 4 >>> ();
    return  (cudaError_t)0;// cudaStatus;
}
cudaError_t umeyama_cuda(int thread_num, torch::Tensor src, torch::Tensor dst)
{
    //src = src.to(torch::kCUDA);

    //umeyamaKernel<<<1, thread_num >>> ( src, dst);
    return  (cudaError_t)0;// cudaStatus;
}