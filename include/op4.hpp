#pragma once
#include <lapacke.h>
#include <cblas.h>
#include "tensor.hpp"
#include <cstddef>
#include <sys/cdefs.h>

namespace Ouroboros{
template<typename T>
__always_inline Tensor<T> matmul(const Tensor<T>& a,const Tensor<T>& b){
    Shape result_shape={a.shape()[0],b.shape()[1]};
    Tensor<T> result(result_shape);
    //Choose the right cblas function based on type T
    if constexpr (std::is_same<T,float>::value){
        cblas_sgemm(CblasRowMajor,CBLAS_TRANSPOSE::CblasNoTrans,CBLAS_TRANSPOSE::CblasNoTrans,
                    a.shape()[0],b.shape()[1],a.shape()[1],
                    1.0f,a.data(),a.shape()[1],
                    b.data(),b.shape()[1],
                    0.0f,result.data(),result.shape()[1]);
    }
    else if constexpr (std::is_same<T,double>::value){
        cblas_dgemm(CblasRowMajor,CBLAS_TRANSPOSE::CblasNoTrans,CBLAS_TRANSPOSE::CblasNoTrans,
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
        cblas_sgemv(CblasRowMajor,CBLAS_TRANSPOSE::CblasNoTrans,
                    a.shape()[0],a.shape()[1],
                    1.0f,a.data(),a.shape()[1],
                    b.data(),1,
                    0.0f,result.data(),1);
    }
    else if constexpr (std::is_same<T,double>::value){
        cblas_dgemv(CblasRowMajor,CBLAS_TRANSPOSE::CblasNoTrans,
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
__always_inline Tensor<T> solve(const Tensor<T>& A, const Tensor<T>& b) {
    /*
    Solves Ax = b for square matrix A (n×n)
    A: (n, n)
    b: (n)
    Returns: x (n)
    */
    std::size_t n = A.shape()[0];
    Tensor<T> A_copy(A); // LAPACK overwrites A
    Tensor<T> x(b);      // LAPACK overwrites b with the solution
    std::vector<int> ipiv(n);

    if constexpr (std::is_same<T, float>::value) {
        LAPACKE_sgesv(LAPACK_ROW_MAJOR, n, 1, A_copy.data(), n,
                      ipiv.data(), x.data(), 1);
        return x;
    }
    else if constexpr (std::is_same<T, double>::value) {
        LAPACKE_dgesv(LAPACK_ROW_MAJOR, n, 1, A_copy.data(), n,
                      ipiv.data(), x.data(), 1);
        return x;
    }
    else {
        throw std::runtime_error("Solve is only implemented for float and double types.");
    }
}
template<typename T>
Tensor<T> ls_solve(const Tensor<T>& A, const Tensor<T>& b) {
    int m = A.shape()[0];
    int n = A.shape()[1];

    Tensor<T> Acopy(A);
    Tensor<T> bcopy(b);   // overwritten with solution
    int nrhs = 1;

    if constexpr (std::is_same_v<T, float>) {
        LAPACKE_sgels(LAPACK_ROW_MAJOR, 'N',
                      m, n, nrhs,
                      Acopy.data(), n,
                      bcopy.data(), nrhs);
    } else if constexpr (std::is_same_v<T, double>) {
        LAPACKE_dgels(LAPACK_ROW_MAJOR, 'N',
                      m, n, nrhs,
                      Acopy.data(), n,
                      bcopy.data(), nrhs);
    } else {
        throw std::runtime_error("ls_solve only supports float/double");
    }

    Tensor<T> x(n);
    for (int i = 0; i < n; i++)
        x[i] = bcopy[i];

    return x;
}
template<typename T>
struct SVDResult {
    Tensor<T> U;
    Tensor<T> S;
    Tensor<T> VT;
};

template<typename T>
SVDResult<T> tensor_svd(const Tensor<T>& A) {
    int m = A.shape()[0];
    int n = A.shape()[1];

    Tensor<T> Acopy(A);
    Tensor<T> U(m, m);
    Tensor<T> S(std::min(m, n));
    Tensor<T> VT(n, n);

    if constexpr (std::is_same_v<T, float>) {
        LAPACKE_sgesdd(LAPACK_ROW_MAJOR, 'A',
                       m, n,
                       Acopy.data(), n,
                       S.data(),
                       U.data(), m,
                       VT.data(), n);
    } else if constexpr (std::is_same_v<T, double>) {
        LAPACKE_dgesdd(LAPACK_ROW_MAJOR, 'A',
                       m, n,
                       Acopy.data(), n,
                       S.data(),
                       U.data(), m,
                       VT.data(), n);
    } else {
        throw std::runtime_error("tensor_svd only supports float/double");
    }

    return {U, S, VT};
}

// //TODO:
// template<typename T>
// __always_inline std::vector<T> eigenvalues(const Tensor<T>& a){}
// template<typename T>
// __always_inline std::vector<std::pair<T,Tensor<T>>> eigenvectors(const Tensor<T>& a){//Return eigen value,eigen vector pairs
// }
// template<typename T>
// __always_inline std::vector<std::pair<T,Tensor<T>>> eigenvectors(const Tensor<T>& a,std::vector<T> eigen_values){//Return eigen value,eigen vector pairs for the values mentioned
// }
}
