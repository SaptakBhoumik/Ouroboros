#pragma once
#include <gsl/gsl_sf_erf.h>
#include <cmath>
namespace Ouroboros{
__always_inline double erf(double x){
    return gsl_sf_erf(x);
}
__always_inline double erfc(double x){
    return gsl_sf_erfc(x);
}
}