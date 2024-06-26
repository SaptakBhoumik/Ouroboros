#include "func/func.hpp"
namespace Ouroboros{
Tensor sin(const Tensor& t,size_t min_count){
    return transform(Scalar::sin,min_count,t);
}
Tensor cos(const Tensor& t,size_t min_count){
    return transform(Scalar::cos,min_count,t);
}
Tensor tan(const Tensor& t,size_t min_count){
    return transform(Scalar::tan,min_count,t);
}
Tensor cosec(const Tensor& t,size_t min_count){
    return transform(Scalar::cosec,min_count,t);
}
Tensor sec(const Tensor& t,size_t min_count){
    return transform(Scalar::sec,min_count,t);
}
Tensor cot(const Tensor& t,size_t min_count){
    return transform(Scalar::cot,min_count,t);
}


Tensor asin(const Tensor& t,size_t min_count){
    return transform(Scalar::asin,min_count,t);
}
Tensor acos(const Tensor& t,size_t min_count){
    return transform(Scalar::acos,min_count,t);
}
Tensor atan(const Tensor& t,size_t min_count){
    return transform(Scalar::atan,min_count,t);
}
Tensor acosec(const Tensor& t,size_t min_count){
    return transform(Scalar::acosec,min_count,t);
}
Tensor asec(const Tensor& t,size_t min_count){
    return transform(Scalar::asec,min_count,t);
}
Tensor acot(const Tensor& t,size_t min_count){
    return transform(Scalar::acot,min_count,t);
}
}