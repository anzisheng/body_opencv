#pragma once
#include <vector>
#define _USE_MATH_DEFINES
#include <k4a/k4a.h>
#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <k4a/k4a.h>
#include <torch/torch.h>
#include <smpl/person.h>
using namespace std;
using namespace SMPL;
std::vector<k4a_float3_t>  convert25_29(std::vector<k4a_float3_t> source25);
void write_persons(std::vector<SMPL::person*> persons, std::ofstream& file);
void write_json(std::ofstream& file, const int id, const torch::Tensor& Rh, const torch::Tensor& Th, const torch::Tensor& poses, const torch::Tensor& shapes);
