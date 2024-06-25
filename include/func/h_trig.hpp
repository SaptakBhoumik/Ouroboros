#pragma once
#include <gsl/gsl_math.h>
#include <cmath>
namespace Ouroboros{
__always_inline double sinh(double x){
    return std::sinh(x);
}
__always_inline double cosh(double x){
    return std::cosh(x);
}
__always_inline double tanh(double x){
    return std::tanh(x);
}
__always_inline double cosech(double x){
    return 1.0/sinh(x);
}
__always_inline double sech(double x){
    return 1.0/cosh(x);
}
__always_inline double coth(double x){
    return 1.0/tanh(x);
}


__always_inline double asinh(double x){
    return gsl_asinh(x);
}
__always_inline double acosh(double x){
    return gsl_acosh(x);
}
__always_inline double atanh(double x){
    return gsl_atanh(x);
}
__always_inline double acosech(double x){
    return asinh(1.0/x);
}
__always_inline double asech(double x){
    return acosh(1.0/x);
}
__always_inline double acoth(double x){
    return atanh(1.0/x);
}
}