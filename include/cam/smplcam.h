#pragma once
#ifndef __SMPLCAM__
#define __SMPLCAM__

#include "smpl/SMPL.h"
#include <torch/torch.h>
using namespace torch;
class smplcam
{
public:
	smplcam(torch::Device device);
	torch::Tensor call_forward(const torch::Tensor& m_pred_xyz_jts_29, /*const torch::Tensor& restJoints_24,*/int frameId);
public:
	smpl::SMPL* m_smpl;
	torch::Tensor m_pred_xyz_jts_29;
	torch::Tensor m_pred_shape;
// 	int  m_smpl_dtype;
// 	torch::Tensor m_h36m_jregressor;
};


#endif