#pragma once
#include "../macros.hpp"
#include "../tensor.hpp"
namespace Ouroboros{
namespace Scalar{
double erf(double x);
double erfc(double x);
double lerfc(double x);
double erf_Z(double x);
double erf_Q(double x);
double hazard(double x);
}
Tensor erf(const Tensor& t,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor erfc(const Tensor& t,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor lerfc(const Tensor& t,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor erf_Z(const Tensor& t,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor erf_Q(const Tensor& t,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor hazard(const Tensor& t,size_t min_count=__MIN__COUNT__FOR__THREAD__);
}