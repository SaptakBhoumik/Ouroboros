#include "func/h_trig.hpp"
#include <gsl/gsl_math.h>
#include <cmath>
namespace Ouroboros{
namespace Scalar{
double sinh(double x){
    return std::sinh(x);
}
double cosh(double x){
    return std::cosh(x);
}
double tanh(double x){
    return std::tanh(x);
}
double cosech(double x){
    return 1.0/sinh(x);
}
double sech(double x){
    return 1.0/cosh(x);
}
double coth(double x){
    return 1.0/tanh(x);
}


double asinh(double x){
    return gsl_asinh(x);
}
double acosh(double x){
    return gsl_acosh(x);
}
double atanh(double x){
    return gsl_atanh(x);
}
double acosech(double x){
    return asinh(1.0/x);
}
double asech(double x){
    return acosh(1.0/x);
}
double acoth(double x){
    return atanh(1.0/x);
}
}
}