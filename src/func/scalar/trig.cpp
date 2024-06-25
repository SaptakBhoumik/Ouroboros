#include "func/h_trig.hpp"
#include <gsl/gsl_sf_trig.h>
#include <cmath>
namespace Ouroboros{
double sin(double x){
    return gsl_sf_sin(x);
}
double cos(double x){
    return gsl_sf_cos(x);
}
double tan(double x){
    return std::tan(x);
}
double cosec(double x){
    return 1.0/sin(x);
}
double sec(double x){
    return 1.0/cos(x);
}
double cot(double x){
    return 1.0/tan(x);
}


double asin(double x){
    return std::asin(x);
}
double acos(double x){
    return std::acos(x);
}
double atan(double x){
    return std::atan(x);
}
double acosec(double x){
    return asin(1.0/x);
}
double asec(double x){
    return acos(1.0/x);
}
double acot(double x){
    return atan(1.0/x);
}

}