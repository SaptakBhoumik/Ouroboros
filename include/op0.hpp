#pragma once
#include "tensor.hpp"
namespace Ouroboros{
template<typename T>
Tensor<T> operator~(const Tensor<T>& a){
    Tensor<T> result(a.shape());
    for(std::uint64_t i=0;i<a.size();i++){
        result[i]=~a[i];
    }
    return result;
}

template<typename T>
Tensor<T> operator|(const Tensor<T>& a,const Tensor<T>& b){
    Tensor<T> result(a.shape());
    for(std::uint64_t i=0;i<a.size();i++){
        result[i]=a[i]|b[i];
    }
    return result;
}
template<typename T>
Tensor<T> operator&(const Tensor<T>& a,const Tensor<T>& b){
    Tensor<T> result(a.shape());
    for(std::uint64_t i=0;i<a.size();i++){
        result[i]=a[i]&b[i];
    }
    return result;
}

template<typename T>
Tensor<T> operator^(const Tensor<T>& a,const Tensor<T>& b){
    Tensor<T> result(a.shape());
    for(std::uint64_t i=0;i<a.size();i++){
        result[i]=a[i]^b[i];
    }
    return result;
}
template<typename T>
Tensor<T> operator<<(const Tensor<T>& a,const Tensor<T>& b){
    Tensor<T> result(a.shape());
    for(std::uint64_t i=0;i<a.size();i++){
        result[i]=a[i]<<b[i];
    }
    return result;
}
template<typename T>
Tensor<T> operator>>(const Tensor<T>& a,const Tensor<T>& b){
    Tensor<T> result(a.shape());
    for(std::uint64_t i=0;i<a.size();i++){
        result[i]=a[i]>>b[i];
    }
    return result;
}


template<typename T>
Tensor<T> operator|(const Tensor<T>& a,T b){
    Tensor<T> result(a.shape());
    for(std::uint64_t i=0;i<a.size();i++){
        result[i]=a[i]|b;
    }
    return result;
}
template<typename T>
Tensor<T> operator&(const Tensor<T>& a,T b){
    Tensor<T> result(a.shape());
    for(std::uint64_t i=0;i<a.size();i++){
        result[i]=a[i]&b;
    }
    return result;
}
template<typename T>
Tensor<T> operator^(const Tensor<T>& a,T b){
    Tensor<T> result(a.shape());
    for(std::uint64_t i=0;i<a.size();i++){
        result[i]=a[i]^b;
    }
    return result;
}
template<typename T>
Tensor<T> operator<<(const Tensor<T>& a,T b){
    Tensor<T> result(a.shape());
    for(std::uint64_t i=0;i<a.size();i++){
        result[i]=a[i]<<b;
    }
    return result;
}
template<typename T>
Tensor<T> operator>>(const Tensor<T>& a,T b){
    Tensor<T> result(a.shape());
    for(std::uint64_t i=0;i<a.size();i++){
        result[i]=a[i]>>b;
    }
    return result;
}


template<typename T>
Tensor<T> operator|(T a,const Tensor<T>& b){
    return b|a;
}
template<typename T>
Tensor<T> operator&(T a,const Tensor<T>& b){
    return b&a;
}
template<typename T>
Tensor<T> operator^(T a,const Tensor<T>& b){
    return b^a;
}
template<typename T>
Tensor<T> operator<<(T a,const Tensor<T>& b){
    Tensor<T> result(b.shape());
    for(std::uint64_t i=0;i<b.size();i++){
        result[i]=a<<b[i];
    }
    return result;
}
template<typename T>
Tensor<T> operator>>(T a,const Tensor<T>& b){
    Tensor<T> result(b.shape());
    for(std::uint64_t i=0;i<b.size();i++){
        result[i]=a>>b[i];
    }
    return result;
}
}
