#pragma once
#include "../macros.hpp"
#include "../tensor.hpp"
namespace Ouroboros{
namespace Scalar{
double sinh(double x);
double cosh(double x);
double tanh(double x);
double cosech(double x);
double sech(double x);
double coth(double x);


double asinh(double x);
double acosh(double x);
double atanh(double x);
double acosech(double x);
double asech(double x);
double acoth(double x);
}

Tensor sinh(const Tensor& t,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor cosh(const Tensor& t,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor tanh(const Tensor& t,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor cosech(const Tensor& t,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor sech(const Tensor& t,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor coth(const Tensor& t,size_t min_count=__MIN__COUNT__FOR__THREAD__);


Tensor asinh(const Tensor& t,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor acosh(const Tensor& t,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor atanh(const Tensor& t,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor acosech(const Tensor& t,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor asech(const Tensor& t,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor acoth(const Tensor& t,size_t min_count=__MIN__COUNT__FOR__THREAD__);
}