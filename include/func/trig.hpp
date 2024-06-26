#pragma once
#include "../macros.hpp"
#include "../tensor.hpp"
namespace Ouroboros{
namespace Scalar{
double sin(double x);
double cos(double x);
double tan(double x);
double cosec(double x);
double sec(double x);
double cot(double x);


double asin(double x);
double acos(double x);
double atan(double x);
double acosec(double x);
double asec(double x);
double acot(double x);
}
Tensor sin(const Tensor& t,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor cos(const Tensor& t,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor tan(const Tensor& t,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor cosec(const Tensor& t,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor sec(const Tensor& t,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor cot(const Tensor& t,size_t min_count=__MIN__COUNT__FOR__THREAD__);


Tensor asin(const Tensor& t,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor acos(const Tensor& t,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor atan(const Tensor& t,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor acosec(const Tensor& t,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor asec(const Tensor& t,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor acot(const Tensor& t,size_t min_count=__MIN__COUNT__FOR__THREAD__);
}