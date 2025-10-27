#pragma once
#include "tensor.hpp"
namespace Ouroboros{
template<typename T>
__always_inline Tensor<bool> operator==(const Tensor<T>& a,const Tensor<T>& b){
    Tensor<bool> result(a.shape());
    for(std::size_t i=0;i<a.size();i++){
        result[i]=a[i]==b[i];
    }
    return result;
}
template<typename T>
__always_inline Tensor<bool> operator!=(const Tensor<T>& a,const Tensor<T>& b){
    Tensor<bool> result(a.shape());
    for(std::size_t i=0;i<a.size();i++){
        result[i]=a[i]!=b[i];
    }
    return result;
}
template<typename T>
__always_inline Tensor<bool> operator<(const Tensor<T>& a,const Tensor<T>& b){
    Tensor<bool> result(a.shape());
    for(std::size_t i=0;i<a.size();i++){
        result[i]=a[i]<b[i];
    }
    return result;
}
template<typename T>
__always_inline Tensor<bool> operator>(const Tensor<T>& a,const Tensor<T>& b){
    Tensor<bool> result(a.shape());
    for(std::size_t i=0;i<a.size();i++){
        result[i]=a[i]>b[i];
    }
    return result;
}
template<typename T>
__always_inline Tensor<bool> operator<=(const Tensor<T>& a,const Tensor<T>& b){
    Tensor<bool> result(a.shape());
    for(std::size_t i=0;i<a.size();i++){
        result[i]=a[i]<=b[i];
    }
    return result;
}
template<typename T>
__always_inline Tensor<bool> operator>=(const Tensor<T>& a,const Tensor<T>& b){
    Tensor<bool> result(a.shape());
    for(std::size_t i=0;i<a.size();i++){
        result[i]=a[i]>=b[i];
    }
    return result;
}

template<typename T>
__always_inline Tensor<bool> operator==(const Tensor<T>& a,double b){
    Tensor<bool> result(a.shape());
    for(std::size_t i=0;i<a.size();i++){
        result[i]=a[i]==b;
    }
    return result;
}
template<typename T>
__always_inline Tensor<bool> operator!=(const Tensor<T>& a,double b){
    Tensor<bool> result(a.shape());
    for(std::size_t i=0;i<a.size();i++){
        result[i]=a[i]!=b;
    }
    return result;
}
template<typename T>
__always_inline Tensor<bool> operator<(const Tensor<T>& a,double b){
    Tensor<bool> result(a.shape());
    for(std::size_t i=0;i<a.size();i++){
        result[i]=a[i]<b;
    }
    return result;
}
template<typename T>
__always_inline Tensor<bool> operator>(const Tensor<T>& a,double b){
    Tensor<bool> result(a.shape());
    for(std::size_t i=0;i<a.size();i++){
        result[i]=a[i]>b;
    }
    return result;
}
template<typename T>
__always_inline Tensor<bool> operator<=(const Tensor<T>& a,double b){
    Tensor<bool> result(a.shape());
    for(std::size_t i=0;i<a.size();i++){
        result[i]=a[i]<=b;
    }
    return result;
}
template<typename T>
__always_inline Tensor<bool> operator>=(const Tensor<T>& a,double b){
    Tensor<bool> result(a.shape());
    for(std::size_t i=0;i<a.size();i++){
        result[i]=a[i]>=b;
    }
    return result;
}

template<typename T>
__always_inline Tensor<bool> operator==(double a,const Tensor<T>& b){
    return b==a;
}
template<typename T>
__always_inline Tensor<bool> operator!=(double a,const Tensor<T>& b){
    return b!=a;
}
template<typename T>
__always_inline Tensor<bool> operator<(double a,const Tensor<T>& b){
    return b>a;
}
template<typename T>
__always_inline Tensor<bool> operator>(double a,const Tensor<T>& b){
    return b<a;
}
template<typename T>
__always_inline Tensor<bool> operator<=(double a,const Tensor<T>& b){
    return b>=a;
}
template<typename T>
__always_inline Tensor<bool> operator>=(double a,const Tensor<T>& b){
    return b<=a;
}
}