#include "func/basic.hpp"
#include <gsl/gsl_math.h>
#include <gsl/gsl_sf_log.h>
#include <gsl/gsl_sf_exp.h>
#include <cmath>
namespace Ouroboros{
namespace Scalar{
double abs(double x){
    return std::abs(x);
}

double exp(double x){
    return gsl_sf_exp(x);
}

double ln(double x){
    return gsl_sf_log(x);
}

double log10(double x){
    double ln_10=2.302585092994046;
    return ln(x)/ln_10;
}
double log2(double x){
    double ln_2=0.6931471805599453;
    return ln(2)/ln_2;
}
double log(double x,double base){
    return ln(x)/ln(base);
}


double cbrt(double x){
    return std::cbrt(x);
}
double sqrt(double x){
    return std::sqrt(x);
}
double pow(double x,double y){
    return std::pow(x,y);
}
double hypot2(double x,double y){
    return gsl_hypot(x,y);
}
double hypot3(double x,double y,double z){
    return gsl_hypot3(x,y,z);
}
double ceil(double x){
    return std::ceil(x);
}
double floor(double x){
    return std::floor(x);
}
double trunc(double x){
    return std::trunc(x);
}
double nearbyint(double x){
    return std::nearbyint(x);
}
double rint(double x){
    return std::rint(x);
}
double round(double x){
    return std::round(x);
}
double fmod(double x,double y){
    return std::fmod(x,y);
}


double min(double x,double y){
    return std::min(x,y);
}
double max(double x,double y){
    return std::max(x,y);
}
double clamp(double x,double min,double max){
    if(x<min){
        return min;
    }
    if(x>max){
        return max;
    }
    return x;
}
double clamp(double x,double min,double max,double c){
    if(x<min){
        return min;
    }
    if(x>max){
        return max;
    }
    return c;
}
double sign(double x){
    return x>0?1:(x<0?-1:0);
}
double fdim(double x,double y){
    return std::fdim(x,y);
}
}
}