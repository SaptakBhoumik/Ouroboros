#include "tensor.hpp"
#include "../cpu/op.hpp"
#include <cblas.h>
namespace Ouroboros{
Tensor operator-(const Tensor& a){
    double* data=CPU::neg_ptr(a.data(),a.count());
    return Tensor(a.shape(),data);
}

Tensor operator+(const Tensor& a,const Tensor& b){
    #ifdef __OUROBOROS_CHECK__
    if(a.shape()!=b.shape()){
        throw std::invalid_argument("Shape mismatch");
    }
    #endif
    double* data=CPU::add_ptr(a.data(),b.data(),a.count());
    return Tensor(a.shape(),data);
}
Tensor operator-(const Tensor& a,const Tensor& b){
    #ifdef __OUROBOROS_CHECK__
    if(a.shape()!=b.shape()){
        throw std::invalid_argument("Shape mismatch");
    }
    #endif
    double* data=CPU::sub_ptr(a.data(),b.data(),a.count());
    return Tensor(a.shape(),data);
}
Tensor operator*(const Tensor& a,const Tensor& b){
    #ifdef __OUROBOROS_CHECK__
    if(a.shape()!=b.shape()){
        throw std::invalid_argument("Shape mismatch");
    }
    #endif
    double* data=CPU::mul_ptr(a.data(),b.data(),a.count());
    return Tensor(a.shape(),data);

}
Tensor operator/(const Tensor& a,const Tensor& b){
    #ifdef __OUROBOROS_CHECK__
    if(a.shape()!=b.shape()){
        throw std::invalid_argument("Shape mismatch");
    }
    #endif
    double* data=CPU::div_ptr(a.data(),b.data(),a.count());
    return Tensor(a.shape(),data);
}



Tensor operator+(const Tensor& a,double b){
    double* data=CPU::add_ptr(a.data(),b,a.count());
    return Tensor(a.shape(),data);
}
Tensor operator-(const Tensor& a,double b){
    double* data=CPU::sub_ptr(a.data(),b,a.count());
    return Tensor(a.shape(),data);
}
Tensor operator*(const Tensor& a,double b){
    double* data=CPU::mul_ptr(a.data(),b,a.count());
    return Tensor(a.shape(),data);
}
Tensor operator/(const Tensor& a,double b){
    double* data=CPU::div_ptr(a.data(),b,a.count());
    return Tensor(a.shape(),data);
}



Tensor operator+(double a,const Tensor& b){
    return b+a;
}
Tensor operator-(double a,const Tensor& b){
    double* data=CPU::sub_ptr(a,b.data(),b.count());
    return Tensor(b.shape(),data);
}
Tensor operator*(double a,const Tensor& b){
    return b*a;
}
Tensor operator/(double a,const Tensor& b){
    double* data=CPU::div_ptr(a,b.data(),b.count());
    return Tensor(b.shape(),data);
}



void operator+=(Tensor& a,const Tensor& b){
    #ifdef __OUROBOROS_CHECK__
    if(a.shape()!=b.shape()){
        throw std::invalid_argument("Shape mismatch");
    }
    #endif
    CPU::add_ptr_self(a.data(),b.data(),a.count());
}
void operator-=(Tensor& a,const Tensor& b){
    #ifdef __OUROBOROS_CHECK__
    if(a.shape()!=b.shape()){
        throw std::invalid_argument("Shape mismatch");
    }
    #endif
    CPU::sub_ptr_self(a.data(),b.data(),a.count());
}
void operator*=(Tensor& a,const Tensor& b){
    #ifdef __OUROBOROS_CHECK__
    if(a.shape()!=b.shape()){
        throw std::invalid_argument("Shape mismatch");
    }
    #endif
    CPU::mul_ptr_self(a.data(),b.data(),a.count());
}
void operator/=(Tensor& a,const Tensor& b){
    #ifdef __OUROBOROS_CHECK__
    if(a.shape()!=b.shape()){
        throw std::invalid_argument("Shape mismatch");
    }
    #endif
    CPU::div_ptr_self(a.data(),b.data(),a.count());
}



void operator+=(Tensor& a,double b){
    CPU::add_ptr_self(a.data(),b,a.count());
}
void operator-=(Tensor& a,double b){
    CPU::sub_ptr_self(a.data(),b,a.count());
}
void operator*=(Tensor& a,double b){
    CPU::mul_ptr_self(a.data(),b,a.count());
}
void operator/=(Tensor& a,double b){
    CPU::div_ptr_self(a.data(),b,a.count());
}

Tensor matmul(const Tensor& a,const Tensor& b){
    auto a_shape=a.shape();
    auto b_shape=b.shape();
    #ifdef __OUROBOROS_CHECK__
    if(a.shape().dim()!=2||b.shape().dim()!=2){
        throw std::invalid_argument("Invalid shape");
    }
    if(a_shape[1]!=b_shape[0]){
        throw std::invalid_argument("Shape mismatch");
    }
    #endif
    double* data=new double[a_shape[0]*b_shape[1]];
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                a_shape[0], b_shape[1], a_shape[1], 1.0, a.data(), 
                a_shape[1], b.data(), b_shape[1], 0.0, data, b_shape[1]);
    return Tensor({a_shape[0],b_shape[1]},data);
}

Tensor matvecmul(const Tensor& a,const Tensor& b){
    auto a_shape=a.shape();
    auto b_shape=b.shape();
    #ifdef __OUROBOROS_CHECK__
    if(a.shape().dim()!=2||b.shape().dim()!=1){
        throw std::invalid_argument("Invalid shape");
    }
    if(a_shape[1]!=b_shape[0]){
        throw std::invalid_argument("Shape mismatch");
    }
    #endif
    double* data=new double[a_shape[0]];
    cblas_dgemv(CblasRowMajor, CblasNoTrans, a_shape[0], a_shape[1], 1.0, 
                a.data(), a_shape[1], b.data(), 1, 0.0, data, 1);
    return Tensor({a_shape[0]},data);
}

BoolTensor operator==(const Tensor& a,const Tensor& b){
    #ifdef __OUROBOROS_CHECK__
    if(a.shape()!=b.shape()){
        throw std::invalid_argument("Shape mismatch");
    }
    #endif
    bool* data=CPU::eq_ptr(a.data(),b.data(),a.count());
    return BoolTensor(a.shape(),data);
}
BoolTensor operator!=(const Tensor& a,const Tensor& b){
    #ifdef __OUROBOROS_CHECK__
    if(a.shape()!=b.shape()){
        throw std::invalid_argument("Shape mismatch");
    }
    #endif
    bool* data=CPU::neq_ptr(a.data(),b.data(),a.count());
    return BoolTensor(a.shape(),data);
}
BoolTensor operator<(const Tensor& a,const Tensor& b){
    #ifdef __OUROBOROS_CHECK__
    if(a.shape()!=b.shape()){
        throw std::invalid_argument("Shape mismatch");
    }
    #endif
    bool* data=CPU::lt_ptr(a.data(),b.data(),a.count());
    return BoolTensor(a.shape(),data);
}
BoolTensor operator>(const Tensor& a,const Tensor& b){
    #ifdef __OUROBOROS_CHECK__
    if(a.shape()!=b.shape()){
        throw std::invalid_argument("Shape mismatch");
    }
    #endif
    bool* data=CPU::gt_ptr(a.data(),b.data(),a.count());
    return BoolTensor(a.shape(),data);
}
BoolTensor operator<=(const Tensor& a,const Tensor& b){
    #ifdef __OUROBOROS_CHECK__
    if(a.shape()!=b.shape()){
        throw std::invalid_argument("Shape mismatch");
    }
    #endif
    bool* data=CPU::lteq_ptr(a.data(),b.data(),a.count());
    return BoolTensor(a.shape(),data);
}
BoolTensor operator>=(const Tensor& a,const Tensor& b){
    #ifdef __OUROBOROS_CHECK__
    if(a.shape()!=b.shape()){
        throw std::invalid_argument("Shape mismatch");
    }
    #endif
    bool* data=CPU::gteq_ptr(a.data(),b.data(),a.count());
    return BoolTensor(a.shape(),data);
}


BoolTensor operator==(const Tensor& a,double b){
    bool* data=CPU::eq_ptr(a.data(),b,a.count());
    return BoolTensor(a.shape(),data);
}
BoolTensor operator!=(const Tensor& a,double b){
    bool* data=CPU::neq_ptr(a.data(),b,a.count());
    return BoolTensor(a.shape(),data);
}
BoolTensor operator<(const Tensor& a,double b){
    bool* data=CPU::lt_ptr(a.data(),b,a.count());
    return BoolTensor(a.shape(),data);
}
BoolTensor operator>(const Tensor& a,double b){
    bool* data=CPU::gt_ptr(a.data(),b,a.count());
    return BoolTensor(a.shape(),data);
}
BoolTensor operator<=(const Tensor& a,double b){
    bool* data=CPU::lteq_ptr(a.data(),b,a.count());
    return BoolTensor(a.shape(),data);
}
BoolTensor operator>=(const Tensor& a,double b){
    bool* data=CPU::gteq_ptr(a.data(),b,a.count());
    return BoolTensor(a.shape(),data);
}


BoolTensor operator==(double a,const Tensor& b){
    return b==a;
}
BoolTensor operator!=(double a,const Tensor& b){
    return b!=a;
}
BoolTensor operator<(double a,const Tensor& b){
    return b>a;
}
BoolTensor operator>(double a,const Tensor& b){
    return b<a;
}
BoolTensor operator<=(double a,const Tensor& b){
    return b>=a;
}
BoolTensor operator>=(double a,const Tensor& b){
    return b<=a;
}
}