#include "func/func.hpp"
namespace Ouroboros{
Tensor erf(const Tensor& t,size_t min_count){
    return transform(Scalar::erf,min_count,t);
}
Tensor erfc(const Tensor& t,size_t min_count){
    return transform(Scalar::erfc,min_count,t);
}
Tensor lerfc(const Tensor& t,size_t min_count){
    return transform(Scalar::lerfc,min_count,t);
}
Tensor erf_Z(const Tensor& t,size_t min_count){
    return transform(Scalar::erf_Z,min_count,t);
}
Tensor erf_Q(const Tensor& t,size_t min_count){
    return transform(Scalar::erf_Q,min_count,t);
}
Tensor hazard(const Tensor& t,size_t min_count){
    return transform(Scalar::hazard,min_count,t);
}
}