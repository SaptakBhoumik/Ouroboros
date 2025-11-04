#pragma once
#include "tensor.hpp"
#include <cblas.h>
#include <cstddef>
#include <sys/cdefs.h>
#include <lapacke.h>

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
        std::size_t row=a.shape()[0];
        std::size_t col=a.shape()[1];
        std::size_t bcol=b.shape()[1];
        for(std::size_t i=0;i<row;i++){
            for(std::size_t j=0;j<bcol;j++){
                result[i*bcol+j]=0;
                for(std::size_t k=0;k<col;k++){
                    result[i*bcol+j]+=a[i*col+k]*b[k*bcol+j];
                }
            }
        }
    }
    return result;
}
template<typename T>
__always_inline Tensor<T> matvecmul(const Tensor<T>& a,const Tensor<T>& b){
    /*
    a: (m,n)
    b: (n)
    result: (m)
    */
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
        std::size_t row=a.shape()[0];
        std::size_t col=a.shape()[1];
        for(std::size_t i=0;i<row;i++){
            result[i]=0;
            for(std::size_t j=0;j<col;j++){
                result[i]+=a[i*col+j]*b[j];
            }
        }
    }
    return result;
}

template<typename T>
T det(const Tensor<T>& a){
    /*
    a: (n,n)
    */
    T det=(T)1;
    Tensor<T> mat(a);
    std::size_t n=a.shape()[0];
    std::vector<int> ipiv(n);
    if constexpr (std::is_same<T,float>::value) {
        int info = LAPACKE_sgetrf(LAPACK_ROW_MAJOR, n, n, mat.data(), n, ipiv.data());
        if (info != 0) return 0.0;

        for (std::size_t i = 0; i < n; i++) {
            if (ipiv[i] != i + 1) det = -det;  // adjust for row swaps
            det *= a[i * n + i];
        }
        return det;
    }   
    else if constexpr (std::is_same<T,double>::value) {
        int info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, n, n, mat.data(), n, ipiv.data());
        if (info != 0) return 0.0;

        for (std::size_t i = 0; i < n; i++) {
            if (ipiv[i] != i + 1) det = -det;  // adjust for row swaps
            det *= a[i * n + i];
        }
        return det;
    }
    else {
        for(std::size_t i=0;i<n;i++){
            //Find pivot
            std::size_t pivot=i;
            for(std::size_t j=i+1;j<n;j++){
                if(std::abs(mat[j*n+i])>std::abs(mat[pivot*n+i])){
                    pivot=j;
                }
            }
            if(std::abs(mat[pivot*n+i])<1e-12){
                return 0;
            }
            if(pivot!=i){
                //Swap rows
                for(std::size_t j=0;j<n;j++){
                    std::swap(mat[i*n+j],mat[pivot*n+j]);
                }
                det=-det;
            }
            det*=mat[i*n+i];
            //Eliminate
            for(std::size_t j=i+1;j<n;j++){
                T factor=mat[j*n+i]/mat[i*n+i];
                for(std::size_t k=i;k<n;k++){
                    mat[j*n+k]-=factor*mat[i*n+k];
                }
            }
        }
        return det;
    }
}
template<typename T>
__always_inline Tensor<T> inverse(const Tensor<T>& a){
    std::size_t n=a.shape()[0];
    Tensor<T> inv(a);
    std::vector<int> ipiv(n);
    if constexpr (std::is_same<T,float>::value) {
        LAPACKE_sgetrf(LAPACK_ROW_MAJOR, n, n, inv.data(), n, ipiv.data());
        LAPACKE_sgetri(LAPACK_ROW_MAJOR, n, inv.data(), n, ipiv.data());
        return inv;
    }   
    else if constexpr (std::is_same<T,double>::value) {
        LAPACKE_dgetrf(LAPACK_ROW_MAJOR, n, n, inv.data(), n, ipiv.data());
        LAPACKE_dgetri(LAPACK_ROW_MAJOR, n, inv.data(), n, ipiv.data());
        return inv;}
    else {
        throw std::runtime_error("Inverse is only implemented for float and double types.");
    }
}
template<typename T>
__always_inline Tensor<T> adj(const Tensor<T>& a){
    return inverse(a)*det(a);
}
template<typename T>
__always_inline Tensor<T> solve(const Tensor<T>& A,const Tensor<T>& b){
    /*
    A: (n,n)
    b: (n) 
    return x: (n) where Ax=b
    */
    std::size_t n=A.shape()[0];
    Tensor<T> x(b);
    std::vector<int> ipiv(n);
    if constexpr (std::is_same<T,float>::value) {
        Tensor<T> A_copy(A);
        LAPACKE_sgetrf(LAPACK_ROW_MAJOR, n, n, A_copy.data(), n, ipiv.data());
        LAPACKE_sgetrs(LAPACK_ROW_MAJOR, 'N', n, 1, A_copy.data(), n, ipiv.data(), x.data(), 1);
        return x;
    }   
    else if constexpr (std::is_same<T,double>::value) {
        Tensor<T> A_copy(A);
        LAPACKE_dgetrf(LAPACK_ROW_MAJOR, n, n, A_copy.data(), n, ipiv.data());
        LAPACKE_dgetrs(LAPACK_ROW_MAJOR, 'N', n, 1, A_copy.data(), n, ipiv.data(), x.data(), 1);
        return x;
    }
    else {
        throw std::runtime_error("Solve is only implemented for float and double types.");
    }
}
}