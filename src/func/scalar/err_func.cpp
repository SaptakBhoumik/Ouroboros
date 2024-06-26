#include "func/err_func.hpp"
#include <gsl/gsl_sf_erf.h>
#include <cmath>
namespace Ouroboros{
namespace Scalar{
double erf(double x){
    return gsl_sf_erf(x);
}
double erfc(double x){
    return gsl_sf_erfc(x);
}
double lerfc(double x){
    return gsl_sf_log_erfc(x);
}
double erf_Z(double x){
    return gsl_sf_erf_Z(x);
}
double erf_Q(double x){
    return gsl_sf_erf_Q(x);
}
double hazard(double x){
    return gsl_sf_hazard(x);
}
}
}