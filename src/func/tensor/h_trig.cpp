#include "func/func.hpp"
namespace Ouroboros{
Tensor sinh(const Tensor& t,size_t min_count){
    return transform(Scalar::sinh,min_count,t);
}
Tensor cosh(const Tensor& t,size_t min_count){
    return transform(Scalar::cosh,min_count,t);
}
Tensor tanh(const Tensor& t,size_t min_count){
    return transform(Scalar::tanh,min_count,t);
}
Tensor cosech(const Tensor& t,size_t min_count){
    return transform(Scalar::cosech,min_count,t);
}
Tensor sech(const Tensor& t,size_t min_count){
    return transform(Scalar::sech,min_count,t);
}
Tensor coth(const Tensor& t,size_t min_count){
    return transform(Scalar::coth,min_count,t);
}


Tensor asinh(const Tensor& t,size_t min_count){
    return transform(Scalar::asinh,min_count,t);
}
Tensor acosh(const Tensor& t,size_t min_count){
    return transform(Scalar::acosh,min_count,t);
}
Tensor atanh(const Tensor& t,size_t min_count){
    return transform(Scalar::atanh,min_count,t);
}
Tensor acosech(const Tensor& t,size_t min_count){
    return transform(Scalar::acosech,min_count,t);
}
Tensor asech(const Tensor& t,size_t min_count){
    return transform(Scalar::asech,min_count,t);
}
Tensor acoth(const Tensor& t,size_t min_count){
    return transform(Scalar::acoth,min_count,t);
}
}