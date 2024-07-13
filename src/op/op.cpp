#include "tensor.hpp"
#include "op.hpp"
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


double cofactor(const Tensor& a,std::size_t row,std::size_t col){
    return minor(a,row,col)*((row+col)%2==0?1:-1);
}
double minor(const Tensor& a,std::size_t row,std::size_t col){
    auto shape=a.shape();
    #ifdef __OUROBOROS_CHECK__
    if(shape.dim()!=2){
        throw std::invalid_argument("Invalid shape. Expected a square matrix");
    }
    else if(shape[0]<=1||shape[1]<=1){
        throw std::invalid_argument("Invalid shape.Should be atleast 2x2");
    }
    else if(shape[0]!=shape[1]){
        throw std::invalid_argument("Invalid shape. Expected a square matrix");
    }
    else if(row>=shape[0]||col>=shape[1]){
        throw std::invalid_argument("Invalid row and column index");
    }
    #endif
    std::size_t row_count=shape[0];
    Tensor m({row_count-1,row_count-1});
    double* data=m.data();
    const double* a_data=a.data();
    for(std::size_t i=0;i<row;i++){
        for(std::size_t j=0;j<col;j++){
            data[i*(row_count-1)+j]=a_data[i*row_count+j];
        }
        for(std::size_t j=col+1;j<row_count;j++){
            data[i*(row_count-1)+j-1]=a_data[i*row_count+j];
        }
    }
    for(std::size_t i=row+1;i<row_count;i++){
        for(std::size_t j=0;j<col;j++){
            data[(i-1)*(row_count-1)+j]=a_data[i*row_count+j];
        }
        for(std::size_t j=col+1;j<row_count;j++){
            data[(i-1)*(row_count-1)+j-1]=a_data[i*row_count+j];
        }
    }
    return determinant(m);
}
double determinant(const Tensor& a){
    auto shape=a.shape();
    #ifdef __OUROBOROS_CHECK__
    if(shape.dim()!=2){
        throw std::invalid_argument("Invalid shape. Expected a square matrix");
    }
    else if(shape[0]!=shape[1]){
        throw std::invalid_argument("Invalid shape. Expected a square matrix");
    }
    #endif
    std::size_t row_count=shape[0];
    if(row_count==1){
        return a[0];
    }
    else if(row_count==2){
        return a[0]*a[3]-a[1]*a[2];
    }
    else if(row_count==3){
        return a[0]*(a[4]*a[8]-a[5]*a[7])-a[1]*(a[3]*a[8]-a[5]*a[6])+a[2]*(a[3]*a[7]-a[4]*a[6]);
    }
    else{
        double det=0;
        for(std::size_t i=0;i<row_count;i++){
            det+=a[i]*cofactor(a,0,i);
        }
        return det;
    }
}
Tensor adjoint(const Tensor& a){
    auto shape=a.shape();
    #ifdef __OUROBOROS_CHECK__
    if(a.shape().dim()!=2){
        throw std::invalid_argument("Invalid shape. Expected a square matrix");
    }
    else if(shape[0]!=shape[1]){
        throw std::invalid_argument("Invalid shape. Expected a square matrix");
    }
    #endif
    std::size_t row_count=shape[0];
    Tensor adj({row_count,row_count});
    double* data=adj.data();
    for(std::size_t i=0;i<row_count;i++){
        for(std::size_t j=0;j<row_count;j++){
            data[j*row_count+i]=cofactor(a,i,j);
        }
    }
    return adj;
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