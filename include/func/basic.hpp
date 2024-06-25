#pragma once
#include <gsl/gsl_math.h>
#include <gsl/gsl_sf_log.h>
#include <gsl/gsl_sf_exp.h>
#include <cmath>
namespace Ouroboros{
__always_inline double abs(double x){
    return std::abs(x);
}

__always_inline double exp(double x){
    return gsl_sf_exp(x);
}

__always_inline double ln(double x){
    return gsl_sf_log(x);
}

__always_inline double log10(double x){
    double ln_10=2.302585092994046;
    return ln(x)/ln_10;
}
__always_inline double log2(double x){
    double ln_2=0.6931471805599453;
    return ln(2)/ln_2;
}
__always_inline double log(double x,double base){
    return ln(x)/ln(base);
}


__always_inline double cbrt(double x){
    return std::cbrt(x);
}
__always_inline double sqrt(double x){
    return std::sqrt(x);
}
__always_inline double pow(double x,double y){
    return std::pow(x,y);
}
__always_inline double hypot2(double x,double y){
    return gsl_hypot(x,y);
}
__always_inline double hypot3(double x,double y,double z){
    return gsl_hypot3(x,y,z);
}
__always_inline double ceil(double x){
    return std::ceil(x);
}
__always_inline double floor(double x){
    return std::floor(x);
}
__always_inline double trunc(double x){
    return std::trunc(x);
}
__always_inline double nearbyint(double x){
    return std::nearbyint(x);
}
__always_inline double rint(double x){
    return std::rint(x);
}
__always_inline double round(double x){
    return std::round(x);
}
__always_inline double fmod(double x,double y){
    return std::fmod(x,y);
}


__always_inline double min(double x,double y){
    return std::min(x,y);
}
__always_inline double max(double x,double y){
    return std::max(x,y);
}
__always_inline double clamp(double x,double min,double max){
    if(x<min){
        return min;
    }
    if(x>max){
        return max;
    }
    return x;
}
__always_inline double clamp(double x,double min,double max,double c){
    if(x<min){
        return min;
    }
    if(x>max){
        return max;
    }
    return c;
}
__always_inline double sign(double x){
    return x>0?1:(x<0?-1:0);
}
__always_inline double fdim(double x,double y){
    return std::fdim(x,y);
}
}