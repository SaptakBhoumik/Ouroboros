#pragma once
#include <gsl/gsl_sf_trig.h>
#include <cmath>
namespace Ouroboros{
__always_inline double sin(double x){
    return gsl_sf_sin(x);
}
__always_inline double cos(double x){
    return gsl_sf_cos(x);
}
__always_inline double tan(double x){
    return std::tan(x);
}
__always_inline double cosec(double x){
    return 1.0/sin(x);
}
__always_inline double sec(double x){
    return 1.0/cos(x);
}
__always_inline double cot(double x){
    return 1.0/tan(x);
}


__always_inline double asin(double x){
    return std::asin(x);
}
__always_inline double acos(double x){
    return std::acos(x);
}
__always_inline double atan(double x){
    return std::atan(x);
}
__always_inline double acosec(double x){
    return asin(1.0/x);
}
__always_inline double asec(double x){
    return acos(1.0/x);
}
__always_inline double acot(double x){
    return atan(1.0/x);
}

}