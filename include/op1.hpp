#pragma once
#include "tensor.hpp"
#include <cblas.h>

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

template<typename T>
__always_inline Tensor<T> matmul(const Tensor<T>& a,const Tensor<T>& b){
    Shape result_shape={a.shape()[0],b.shape()[1]};
    Tensor<T> result(result_shape);
    //Choose the right cblas function based on type T
    if constexpr (std::is_same<T,float>::value){
        cblas_sgemm(CBLAS_LAYOUT::CblasRowMajor,CBLAS_TRANSPOSE::CblasNoTrans,CBLAS_TRANSPOSE::CblasNoTrans,
                    a.shape()[0],b.shape()[1],a.shape()[1],
                    1.0f,a.data(),a.shape()[1],
                    b.data(),b.shape()[1],
                    0.0f,result.data(),result.shape()[1]);
    }
    else if constexpr (std::is_same<T,double>::value){
        cblas_dgemm(CBLAS_LAYOUT::CblasRowMajor,CBLAS_TRANSPOSE::CblasNoTrans,CBLAS_TRANSPOSE::CblasNoTrans,
                    a.shape()[0],b.shape()[1],a.shape()[1],
                    1.0,a.data(),a.shape()[1],
                    b.data(),b.shape()[1],
                    0.0,result.data(),result.shape()[1]);
    
    }
    else{
        static_assert("Type not supported for matmul");
    }
    return result;
}
template<typename T>
__always_inline Tensor<T> matvecmul(const Tensor<T>& a,const Tensor<T>& b){
    Shape result_shape={a.shape()[0]};
    Tensor<T> result(result_shape);
    //Choose the right cblas function based on type T
    if constexpr (std::is_same<T,float>::value){
        cblas_sgemv(CBLAS_LAYOUT::CblasRowMajor,CBLAS_TRANSPOSE::CblasNoTrans,
                    a.shape()[0],a.shape()[1],
                    1.0f,a.data(),a.shape()[1],
                    b.data(),1,
                    0.0f,result.data(),1);
    }
    else if constexpr (std::is_same<T,double>::value){
        cblas_dgemv(CBLAS_LAYOUT::CblasRowMajor,CBLAS_TRANSPOSE::CblasNoTrans,
                    a.shape()[0],a.shape()[1],
                    1.0,a.data(),a.shape()[1],
                    b.data(),1,
                    0.0,result.data(),1);
    
    }
    else{
        static_assert("Type not supported for matvecmul");
    }
    return result;
}

}