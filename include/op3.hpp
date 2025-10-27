#pragma once
#include "tensor.hpp"
namespace Ouroboros{
__always_inline Tensor<bool> operator!(const Tensor<bool>& a){
    Tensor<bool> result(a.shape());
    for(std::size_t i=0;i<a.size();i++){
        result[i]=!a[i];
    }
    return result;
}

__always_inline Tensor<bool> operator==(const Tensor<bool>& a,const Tensor<bool>& b){
    Tensor<bool> result(a.shape());
    for(std::size_t i=0;i<a.size();i++){
        result[i]=a[i]==b[i];
    }
    return result;
}
__always_inline Tensor<bool> operator!=(const Tensor<bool>& a,const Tensor<bool>& b){
    Tensor<bool> result(a.shape());
    for(std::size_t i=0;i<a.size();i++){
        result[i]=a[i]!=b[i];
    }
    return result;
}
__always_inline Tensor<bool> operator&&(const Tensor<bool>& a,const Tensor<bool>& b){
    Tensor<bool> result(a.shape());
    for(std::size_t i=0;i<a.size();i++){
        result[i]=a[i]&&b[i];
    }
    return result;
}
__always_inline Tensor<bool> operator||(const Tensor<bool>& a,const Tensor<bool>& b){
    Tensor<bool> result(a.shape());
    for(std::size_t i=0;i<a.size();i++){
        result[i]=a[i]||b[i];
    }
    return result;
}

__always_inline Tensor<bool> operator==(const Tensor<bool>& a,bool b){
    Tensor<bool> result(a.shape());
    for(std::size_t i=0;i<a.size();i++){
        result[i]=a[i]==b;
    }
    return result;
}
__always_inline Tensor<bool> operator!=(const Tensor<bool>& a,bool b){
    Tensor<bool> result(a.shape());
    for(std::size_t i=0;i<a.size();i++){
        result[i]=a[i]!=b;
    }
    return result;
}
__always_inline Tensor<bool> operator&&(const Tensor<bool>& a,bool b){
    Tensor<bool> result(a.shape());
    for(std::size_t i=0;i<a.size();i++){
        result[i]=a[i]&&b;
    }
    return result;
}
__always_inline Tensor<bool> operator||(const Tensor<bool>& a,bool b){
    Tensor<bool> result(a.shape());
    for(std::size_t i=0;i<a.size();i++){
        result[i]=a[i]||b;
    }
    return result;
}

__always_inline Tensor<bool> operator==(bool a,const Tensor<bool>& b){
    return b==a;
}
__always_inline Tensor<bool> operator!=(bool a,const Tensor<bool>& b){
    return b!=a;
}
__always_inline Tensor<bool> operator&&(bool a,const Tensor<bool>& b){
    return b&&a;
}
__always_inline Tensor<bool> operator||(bool a,const Tensor<bool>& b){
    return b||a;
}
}