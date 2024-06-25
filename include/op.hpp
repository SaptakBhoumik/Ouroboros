#pragma once
#include "tensor.hpp"

namespace Ouroboros{
Tensor operator-(const Tensor& a);

Tensor operator+(const Tensor& a,const Tensor& b);
Tensor operator-(const Tensor& a,const Tensor& b);
Tensor operator*(const Tensor& a,const Tensor& b);
Tensor operator/(const Tensor& a,const Tensor& b);

Tensor operator+(const Tensor& a,double b);
Tensor operator-(const Tensor& a,double b);
Tensor operator*(const Tensor& a,double b);
Tensor operator/(const Tensor& a,double b);

Tensor operator+(double a,const Tensor& b);
Tensor operator-(double a,const Tensor& b);
Tensor operator*(double a,const Tensor& b);
Tensor operator/(double a,const Tensor& b);

void operator+=(Tensor& a,const Tensor& b);
void operator-=(Tensor& a,const Tensor& b);
void operator*=(Tensor& a,const Tensor& b);
void operator/=(Tensor& a,const Tensor& b);

void operator+=(Tensor& a,double b);
void operator-=(Tensor& a,double b);
void operator*=(Tensor& a,double b);
void operator/=(Tensor& a,double b);

Tensor matmul(const Tensor& a,const Tensor& b);
Tensor matvecmul(const Tensor& a,const Tensor& b);
}