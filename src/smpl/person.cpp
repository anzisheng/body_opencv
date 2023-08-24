
#include "smpl/person.h"

namespace SMPL
{
    person::person(int id, torch::Tensor Rh, torch::Tensor Th, torch::Tensor poses, torch::Tensor shapes) :
        m_id(id), m_Rh(Rh), m_Th(Th), m_poses(poses), m_shapes(shapes)
    {

    }

}