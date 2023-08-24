#pragma once;
#include "torch/torch.h"

namespace SMPL
{
	struct person
	{
	public:
		int m_id;
		torch::Tensor m_Rh;
		torch::Tensor m_Th;
		torch::Tensor m_poses;
		torch::Tensor m_shapes;
		person(int id, torch::Tensor Rh, torch::Tensor Th, torch::Tensor poses, torch::Tensor shapes);
		virtual ~person() {};
	};
}