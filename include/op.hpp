#pragma once
#include "tensor.hpp"
#include "macros.hpp"

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

double cofactor(const Tensor& a,size_t row,size_t col);
double minor(const Tensor& a,size_t row,size_t col);
double determinant(const Tensor& a);

Tensor adjoint(const Tensor& a);


BoolTensor operator==(const Tensor& a,const Tensor& b);
BoolTensor operator!=(const Tensor& a,const Tensor& b);
BoolTensor operator<(const Tensor& a,const Tensor& b);
BoolTensor operator>(const Tensor& a,const Tensor& b);
BoolTensor operator<=(const Tensor& a,const Tensor& b);
BoolTensor operator>=(const Tensor& a,const Tensor& b);

BoolTensor operator==(const Tensor& a,double b);
BoolTensor operator!=(const Tensor& a,double b);
BoolTensor operator<(const Tensor& a,double b);
BoolTensor operator>(const Tensor& a,double b);
BoolTensor operator<=(const Tensor& a,double b);
BoolTensor operator>=(const Tensor& a,double b);

BoolTensor operator==(double a,const Tensor& b);
BoolTensor operator!=(double a,const Tensor& b);
BoolTensor operator<(double a,const Tensor& b);
BoolTensor operator>(double a,const Tensor& b);
BoolTensor operator<=(double a,const Tensor& b);
BoolTensor operator>=(double a,const Tensor& b);


BoolTensor operator!(const BoolTensor& a);

BoolTensor operator==(const BoolTensor& a,const BoolTensor& b);
BoolTensor operator!=(const BoolTensor& a,const BoolTensor& b);
BoolTensor operator&&(const BoolTensor& a,const BoolTensor& b);
BoolTensor operator||(const BoolTensor& a,const BoolTensor& b);

BoolTensor operator==(const BoolTensor& a,bool b);
BoolTensor operator!=(const BoolTensor& a,bool b);
BoolTensor operator&&(const BoolTensor& a,bool b);
BoolTensor operator||(const BoolTensor& a,bool b);

BoolTensor operator==(bool a,const BoolTensor& b);
BoolTensor operator!=(bool a,const BoolTensor& b);
BoolTensor operator&&(bool a,const BoolTensor& b);
BoolTensor operator||(bool a,const BoolTensor& b);
}