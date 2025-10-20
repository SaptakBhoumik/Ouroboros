#include "tensor.hpp"
namespace Ouroboros{
Tensor<bool> operator!(const Tensor<bool>& a){
    Tensor<bool> result(a.shape());
    for(std::uint64_t i=0;i<a.size();i++){
        result[i]=!a[i];
    }
    return result;
}

Tensor<bool> operator==(const Tensor<bool>& a,const Tensor<bool>& b){
    Tensor<bool> result(a.shape());
    for(std::uint64_t i=0;i<a.size();i++){
        result[i]=a[i]==b[i];
    }
    return result;
}
Tensor<bool> operator!=(const Tensor<bool>& a,const Tensor<bool>& b){
    Tensor<bool> result(a.shape());
    for(std::uint64_t i=0;i<a.size();i++){
        result[i]=a[i]!=b[i];
    }
    return result;
}
Tensor<bool> operator&&(const Tensor<bool>& a,const Tensor<bool>& b){
    Tensor<bool> result(a.shape());
    for(std::uint64_t i=0;i<a.size();i++){
        result[i]=a[i]&&b[i];
    }
    return result;
}
Tensor<bool> operator||(const Tensor<bool>& a,const Tensor<bool>& b){
    Tensor<bool> result(a.shape());
    for(std::uint64_t i=0;i<a.size();i++){
        result[i]=a[i]||b[i];
    }
    return result;
}

Tensor<bool> operator==(const Tensor<bool>& a,bool b){
    Tensor<bool> result(a.shape());
    for(std::uint64_t i=0;i<a.size();i++){
        result[i]=a[i]==b;
    }
    return result;
}
Tensor<bool> operator!=(const Tensor<bool>& a,bool b){
    Tensor<bool> result(a.shape());
    for(std::uint64_t i=0;i<a.size();i++){
        result[i]=a[i]!=b;
    }
    return result;
}
Tensor<bool> operator&&(const Tensor<bool>& a,bool b){
    Tensor<bool> result(a.shape());
    for(std::uint64_t i=0;i<a.size();i++){
        result[i]=a[i]&&b;
    }
    return result;
}
Tensor<bool> operator||(const Tensor<bool>& a,bool b){
    Tensor<bool> result(a.shape());
    for(std::uint64_t i=0;i<a.size();i++){
        result[i]=a[i]||b;
    }
    return result;
}

Tensor<bool> operator==(bool a,const Tensor<bool>& b){
    return b==a;
}
Tensor<bool> operator!=(bool a,const Tensor<bool>& b){
    return b!=a;
}
Tensor<bool> operator&&(bool a,const Tensor<bool>& b){
    return b&&a;
}
Tensor<bool> operator||(bool a,const Tensor<bool>& b){
    return b||a;
}
}