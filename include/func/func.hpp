#pragma once
#include "op.hpp"
#include <cmath>
#include <functional>
#include <omp.h>
#include "basic.hpp"
#include "err_func.hpp"
#include "sf_func.hpp"
#include "trig.hpp"
#include "h_trig.hpp"
#ifndef __MIN__COUNT__FOR__THREAD__
#define __MIN__COUNT__FOR__THREAD__ 1000000
#endif
namespace Ouroboros{
template<double(*func)(double)>
Tensor transform(const Tensor& a,size_t min_size=__MIN__COUNT__FOR__THREAD__){
    Tensor res(a.shape());
    double* res_data=res.data();
    size_t count=a.count();
    if(count<=min_size){
        for(size_t i=0;i<count;i++){
            res_data[i]=func(a[i]);
        }
    }
    else{
        #pragma omp parallel for
        for(size_t i=0;i<count;i++){
            res_data[i]=func(a[i]);
        }
    }
    return res;
}
/*
//TODO:
cumprod(...): Compute the cumulative product of the tensor x along axis.

cumsum(...): Compute the cumulative sum of the tensor x along axis.

cumulative_logsumexp(...): Compute the cumulative log-sum-exp of the tensor x along axis.

*/
}
#undef __MIN__COUNT__FOR__THREAD__