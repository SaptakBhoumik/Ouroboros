#include "tensor.hpp"
#include <functional>
#include <stdlib.h>
#include <random>
namespace Ouroboros{
Shape getStride(const Shape& shape){
    if(shape.dim()==1){
        return {1};
    }
    size_t strides_data[shape.dim()];
    size_t stride = 1;
    for (int i = shape.dim() - 1; i >= 0; --i) {
        strides_data[i] = stride;
        stride *= shape[i];
    }
    auto strides=Shape(shape.dim(), strides_data);
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
        if(std::abs(m_data[i]-a)<epsilon){
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

bool Tensor::is_zero()const{
    auto epsilon=std::numeric_limits<double>::epsilon();
    for(size_t i=0;i<m_shape.count();i++){
        if(std::abs(m_data[i])>epsilon){
            return false;
        }
    }
    return true;

}
bool Tensor::is_finite()const{
    for(size_t i=0;i<m_shape.count();i++){
        if(!std::isfinite(m_data[i])){
            return false;
        }
    }
    return true;
}
bool Tensor::has_nan()const{
    for(size_t i=0;i<m_shape.count();i++){
        if(std::isnan(m_data[i])){
            return true;
        }
    }
    return false;
}
double Tensor::norm()const{
    return std::sqrt(norm2());
}
double Tensor::norm2()const{
    double res=0.0;
    for(size_t i=0;i<m_shape.count();i++){
        res+=m_data[i]*m_data[i];
    }
    return res;
}
double Tensor::sum()const{
    double res=0.0;
    for(size_t i=0;i<m_shape.count();i++){
        res+=m_data[i];
    }
    return res;
}
double Tensor::mean()const{
    return sum()/m_shape.count();
}
double Tensor::max()const{
    double res=m_data[0];
    for(size_t i=1;i<m_shape.count();i++){
        if(m_data[i]>res){
            res=m_data[i];
        }
    }
    return res;
}
double Tensor::min()const{
    double res=m_data[0];
    for(size_t i=1;i<m_shape.count();i++){
        if(m_data[i]<res){
            res=m_data[i];
        }
    }
    return res;
}
std::pair<double,size_t> Tensor::min_index()const{
    double res=m_data[0];
    size_t idx=0;
    for(size_t i=1;i<m_shape.count();i++){
        if(m_data[i]<res){
            res=m_data[i];
            idx=i;
        }
    }
    return std::make_pair(res,idx);

}
std::pair<double,size_t> Tensor::max_index()const{
    double res=m_data[0];
    size_t idx=0;
    for(size_t i=1;i<m_shape.count();i++){
        if(m_data[i]>res){
            res=m_data[i];
            idx=i;
        }
    }
    return std::make_pair(res,idx);
}
double Tensor::prod()const{
    double res=1.0;
    for(size_t i=0;i<m_shape.count();i++){
        res*=m_data[i];
    }
    return res;
}
}