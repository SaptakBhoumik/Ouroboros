#pragma once
#include "tensor.hpp"
namespace Ouroboros{
Tensor<bool> operator!(const Tensor<bool>& a);

Tensor<bool> operator==(const Tensor<bool>& a,const Tensor<bool>& b);
Tensor<bool> operator!=(const Tensor<bool>& a,const Tensor<bool>& b);
Tensor<bool> operator&&(const Tensor<bool>& a,const Tensor<bool>& b);
Tensor<bool> operator||(const Tensor<bool>& a,const Tensor<bool>& b);

Tensor<bool> operator==(const Tensor<bool>& a,bool b);
Tensor<bool> operator!=(const Tensor<bool>& a,bool b);
Tensor<bool> operator&&(const Tensor<bool>& a,bool b);
Tensor<bool> operator||(const Tensor<bool>& a,bool b);

Tensor<bool> operator==(bool a,const Tensor<bool>& b);
Tensor<bool> operator!=(bool a,const Tensor<bool>& b);
Tensor<bool> operator&&(bool a,const Tensor<bool>& b);
Tensor<bool> operator||(bool a,const Tensor<bool>& b);
}