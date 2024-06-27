#include "tensor.hpp"
#include "func/func.hpp"
#include <functional>
#include <stdlib.h>
#include <random>
namespace Ouroboros{
Shape getStride(const Shape& shape){
    size_t* strides_data=new size_t[shape.dim()];
    size_t stride = 1;
    for (int i = shape.dim() - 1; i >= 0; --i) {
        strides_data[i] = stride;
        stride *= shape[i];
    }
    auto strides=Shape(shape.dim(), strides_data);
    delete[] strides_data;
    return strides;
}
void Tensor::fill(double value){
    for(size_t i=0;i<m_shape.count();i++){
        m_data[i]=value;
    }
}
void Tensor::fill(std::function<double()> func){
    for(size_t i=0;i<m_shape.count();i++){
        m_data[i]=func();
    }
}
void Tensor::zeros(){
    fill(0.0);
}
void Tensor::ones(){
    fill(1.0);
}
void Tensor::rand(double start,double end){
    std::random_device __dev;
    std::mt19937 __rng(__dev());
    std::uniform_real_distribution<double> dist(start,end);
    for(size_t i=0;i<m_shape.count();i++){
        m_data[i]=dist(__rng);
    }
}
void Tensor::clean(double a,double new_val){
    for(size_t i=0;i<m_shape.count();i++){
        if(m_data[i]<=a){
            m_data[i]=new_val;
        }
    }
}
void Tensor::clamp(double a,double b){
    for(size_t i=0;i<m_shape.count();i++){
        if(m_data[i]<a){
            m_data[i]=a;
        }
        else if(m_data[i]>b){
            m_data[i]=b;
        }
    }
}
void Tensor::clamp(double a,double b,double c){
    for(size_t i=0;i<m_shape.count();i++){
        if(m_data[i]<a){
            m_data[i]=a;
        }
        else if(m_data[i]>b){
            m_data[i]=b;
        }
        else{
            m_data[i]=c;
        }
    }
}
void Tensor::threshold(double a,double new_val){
    size_t count=m_shape.count();
    for(size_t i=0;i<count;i++){
        if(m_data[i]<=a){
            m_data[i]=new_val;
        }
    }
}
void Tensor::replace(double a,double b){
    auto epsilon=std::numeric_limits<double>::epsilon();
    for(size_t i=0;i<m_shape.count();i++){
        if(Scalar::abs(m_data[i]-a)<epsilon){
            m_data[i]=b;
        }
    }
}

void Tensor::fill_nan(double value){
    for(size_t i=0;i<m_shape.count();i++){
        if(std::isnan(m_data[i])){
            m_data[i]=value;
        }
    }
}
void Tensor::fill_inf(double value){
    for(size_t i=0;i<m_shape.count();i++){
        if(std::isinf(m_data[i])&&m_data[i]>0){
            m_data[i]=value;
        }
    }
}
void Tensor::fill_neg_inf(double value){
    for(size_t i=0;i<m_shape.count();i++){
        if(std::isinf(m_data[i])&&m_data[i]<0){
            m_data[i]=value;
        }
    }
}

void Tensor::fill_nan_inf(double value){
    for(size_t i=0;i<m_shape.count();i++){
        if(std::isnan(m_data[i])||(std::isinf(m_data[i])&&m_data[i]>0)){
            m_data[i]=value;
        }
    }
}
void Tensor::fill_nan_neg_inf(double value){
    for(size_t i=0;i<m_shape.count();i++){
        if(std::isnan(m_data[i])||(std::isinf(m_data[i])&&m_data[i]<0)){
            m_data[i]=value;
        }
    }
}
void Tensor::fill_inf_neg_inf(double value){
    for(size_t i=0;i<m_shape.count();i++){
        if(std::isinf(m_data[i])){
            m_data[i]=value;
        }
    }
}
void Tensor::fill_nan_inf_neg_inf(double value){
    for(size_t i=0;i<m_shape.count();i++){
        if(std::isnan(m_data[i])||std::isinf(m_data[i])){
            m_data[i]=value;
        }
    }
} 

bool Tensor::is_zero(){
    auto epsilon=std::numeric_limits<double>::epsilon();
    for(size_t i=0;i<m_shape.count();i++){
        if(Scalar::abs(m_data[i])>epsilon){
            return false;
        }
    }
    return true;

}
bool Tensor::is_finite(){
    for(size_t i=0;i<m_shape.count();i++){
        if(!std::isfinite(m_data[i])){
            return false;
        }
    }
    return true;
}
bool Tensor::has_nan(){
    for(size_t i=0;i<m_shape.count();i++){
        if(std::isnan(m_data[i])){
            return true;
        }
    }
    return false;
}
Tensor where(const BoolTensor& condition,const Tensor& x,const Tensor& y){
    auto shape=condition.shape();
    if(shape!=x.shape()||shape!=y.shape()){
        throw std::invalid_argument("Invalid shape");
    }
    auto result=Tensor(shape);
    size_t count=shape.count();
    double* result_data=result.data();
    const double* x_data=x.data();
    const double* y_data=y.data();
    const bool* condition_data=condition.data();
    for(size_t i=0;i<count;i++){
        result_data[i]=condition_data[i]?x_data[i]:y_data[i];
    }
    return result;
}
Tensor where(const BoolTensor& condition,const Tensor& x,double y){
    auto shape=condition.shape();
    if(shape!=x.shape()){
        throw std::invalid_argument("Invalid shape");
    }
    auto result=Tensor(shape);
    size_t count=shape.count();
    double* result_data=result.data();
    const double* x_data=x.data();
    const bool* condition_data=condition.data();
    for(size_t i=0;i<count;i++){
        result_data[i]=condition_data[i]?x_data[i]:y;
    }
    return result;
}
Tensor where(const BoolTensor& condition,double x,const Tensor& y){
    auto shape=condition.shape();
    if(shape!=y.shape()){
        throw std::invalid_argument("Invalid shape");
    }
    auto result=Tensor(shape);
    size_t count=shape.count();
    double* result_data=result.data();
    const double* y_data=y.data();
    const bool* condition_data=condition.data();
    for(size_t i=0;i<count;i++){
        result_data[i]=condition_data[i]?x:y_data[i];
    }
    return result;
}
Tensor where(const BoolTensor& condition,double x,double y){
    auto shape=condition.shape();
    auto result=Tensor(shape);
    size_t count=shape.count();
    double* result_data=result.data();
    const bool* condition_data=condition.data();
    for(size_t i=0;i<count;i++){
        result_data[i]=condition_data[i]?x:y;
    }
    return result;
}
}