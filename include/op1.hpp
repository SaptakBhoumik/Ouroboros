#pragma once
#include "tensor.hpp"
#include <cstddef>
#include <sys/cdefs.h>

namespace Ouroboros{
template<typename T>
__always_inline Tensor<T> operator-(const Tensor<T>& a){
    Tensor<T> result(a.shape());
    for(std::size_t i=0;i<a.size();i++){
        result[i]=-a[i];
    }
    return result;
}

template<typename T>
__always_inline Tensor<T> operator+(const Tensor<T>& a,const Tensor<T>& b){
    Tensor<T> result(a.shape());
    for(std::size_t i=0;i<a.size();i++){
        result[i]=a[i]+b[i];
    }
    return result;
}
template<typename T>
__always_inline Tensor<T> operator-(const Tensor<T>& a,const Tensor<T>& b){
    Tensor<T> result(a.shape());
    for(std::size_t i=0;i<a.size();i++){
        result[i]=a[i]-b[i];
    }
    return result;
}
template<typename T>
__always_inline Tensor<T> operator*(const Tensor<T>& a,const Tensor<T>& b){
    Tensor<T> result(a.shape());
    for(std::size_t i=0;i<a.size();i++){
        result[i]=a[i]*b[i];
    }
    return result;
}
template<typename T>
__always_inline Tensor<T> operator/(const Tensor<T>& a,const Tensor<T>& b){
    Tensor<T> result(a.shape());
    for(std::size_t i=0;i<a.size();i++){
        result[i]=a[i]/b[i];
    }
    return result;
}

template<typename T>
__always_inline Tensor<T> operator+(const Tensor<T>& a,T b){
    Tensor<T> result(a.shape());
    for(std::size_t i=0;i<a.size();i++){
        result[i]=a[i]+b;
    }
    return result;
}
template<typename T>
__always_inline Tensor<T> operator-(const Tensor<T>& a,T b){
    return a+(-b);
}
template<typename T>
__always_inline Tensor<T> operator*(const Tensor<T>& a,T b){
    Tensor<T> result(a.shape());
    for(std::size_t i=0;i<a.size();i++){
        result[i]=a[i]*b;
    }
    return result;
}
template<typename T>
__always_inline Tensor<T> operator/(const Tensor<T>& a,T b){
    return a*((T)1/b);
}

template<typename T>
__always_inline Tensor<T> operator+(T a,const Tensor<T>& b){
    return b+a;
}
template<typename T>
__always_inline Tensor<T> operator-(T a,const Tensor<T>& b){
    Tensor<T> result(b.shape());
    for(std::size_t i=0;i<b.size();i++){
        result[i]=a-b[i];
    }
    return result;
}
template<typename T>
__always_inline Tensor<T> operator*(T a,const Tensor<T>& b){
    return b*a;
}
template<typename T>
__always_inline Tensor<T> operator/(T a,const Tensor<T>& b){
    Tensor<T> result(b.shape());
    for(std::size_t i=0;i<b.size();i++){
        result[i]=a/b[i];
    }
    return result;
}

template<typename T>
__always_inline void operator+=(Tensor<T>& a,const Tensor<T>& b){
    for(std::size_t i=0;i<a.size();i++){
        a[i]+=b[i];
    }
}
template<typename T>
__always_inline void operator-=(Tensor<T>& a,const Tensor<T>& b){
    for(std::size_t i=0;i<a.size();i++){
        a[i]-=b[i];
    }
}
template<typename T>
__always_inline void operator*=(Tensor<T>& a,const Tensor<T>& b){
    for(std::size_t i=0;i<a.size();i++){
        a[i]*=b[i];
    }
}
template<typename T>
__always_inline void operator/=(Tensor<T>& a,const Tensor<T>& b){
    for(std::size_t i=0;i<a.size();i++){
        a[i]/=b[i];
    }
}

template<typename T>
__always_inline void operator+=(Tensor<T>& a,T b){
    for(std::size_t i=0;i<a.size();i++){
        a[i]+=b;
    }
}
template<typename T>
__always_inline void operator-=(Tensor<T>& a,T b){
    a+=(-b);
}
template<typename T>
__always_inline void operator*=(Tensor<T>& a,T b){
    for(std::size_t i=0;i<a.size();i++){
        a[i]*=b;
    }
}
template<typename T>
__always_inline void operator/=(Tensor<T>& a,T b){
    a*=((T)1/b);
}
}