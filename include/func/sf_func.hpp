#pragma once
#include <cmath>
#include <functional>
#include <gsl/gsl_sf_psi.h>
#include <gsl/gsl_sf_trig.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_sf_airy.h>
#include <gsl/gsl_sf_bessel.h>
namespace Ouroboros{
//Airy Functions
__always_inline double Ai(double x){
    return gsl_sf_airy_Ai(x,GSL_PREC_DOUBLE);
}
__always_inline double Bi(double x){
    return gsl_sf_airy_Bi(x,GSL_PREC_DOUBLE);
}
__always_inline double Ai_scaled(double x){
    return gsl_sf_airy_Ai_scaled(x,GSL_PREC_DOUBLE);
}
__always_inline double Bi_scaled(double x){
    return gsl_sf_airy_Bi_scaled(x,GSL_PREC_DOUBLE);
}
__always_inline double Ai_grad(double x){
    return gsl_sf_airy_Ai_deriv(x,GSL_PREC_DOUBLE);
}
__always_inline double Bi_grad(double x){
    return gsl_sf_airy_Bi_deriv(x,GSL_PREC_DOUBLE);
}
__always_inline double Ai_grad_scaled(double x){
    return gsl_sf_airy_Ai_deriv_scaled(x,GSL_PREC_DOUBLE);
}
__always_inline double Bi_grad_scaled(double x){
    return gsl_sf_airy_Bi_deriv_scaled(x,GSL_PREC_DOUBLE);
}
__always_inline double Ai_zero(unsigned int n){
    return gsl_sf_airy_zero_Ai(n);
}
__always_inline double Bi_zero(unsigned int n){
    return gsl_sf_airy_zero_Bi(n);
}
__always_inline double Ai_grad_zero(unsigned int n){
    return gsl_sf_airy_zero_Ai_deriv(n);
}
__always_inline double Bi_grad_zero(unsigned int n){
    return gsl_sf_airy_zero_Bi_deriv(n);
}
//Bessel Functions
__always_inline double J0(double x){
    return gsl_sf_bessel_J0(x);
}
__always_inline double J1(double x){
    return gsl_sf_bessel_J1(x);
}
__always_inline double Jn(double x,int n){
    return gsl_sf_bessel_Jn(n,x);
}
__always_inline double J0_zero(unsigned int n){
    return gsl_sf_bessel_zero_J0(n);
}
__always_inline double J1_zero(unsigned int n){
    return gsl_sf_bessel_zero_J1(n);
}

__always_inline double Y0(double x){
    return gsl_sf_bessel_Y0(x);
}
__always_inline double Y1(double x){
    return gsl_sf_bessel_Y1(x);
}
__always_inline double Yn(double x,int n){
    return gsl_sf_bessel_Yn(n,x);
}

__always_inline double I0(double x){
    return gsl_sf_bessel_I0(x);
}
__always_inline double I1(double x){
    return gsl_sf_bessel_I1(x);
}
__always_inline double In(double x,int n){
    return gsl_sf_bessel_In(n,x);
}
__always_inline double I0_scaled(double x){
    return gsl_sf_bessel_I0_scaled(x);
}
__always_inline double I1_scaled(double x){
    return gsl_sf_bessel_I1_scaled(x);
}
__always_inline double In_scaled(double x,int n){
    return gsl_sf_bessel_In_scaled(n,x);
}

__always_inline double K0(double x){
    return gsl_sf_bessel_K0(x);
}
__always_inline double K1(double x){
    return gsl_sf_bessel_K1(x);
}
__always_inline double Kn(double x,int n){
    return gsl_sf_bessel_Kn(n,x);
}
__always_inline double K0_scaled(double x){
    return gsl_sf_bessel_K0_scaled(x);
}
__always_inline double K1_scaled(double x){
    return gsl_sf_bessel_K1_scaled(x);
}
__always_inline double Kn_scaled(double x,int n){
    return gsl_sf_bessel_Kn_scaled(n,x);
}

__always_inline double Ynu(double nu,double x){
    return gsl_sf_bessel_Ynu(nu,x);
}
__always_inline double Jnu(double nu,double x){
    return gsl_sf_bessel_Jnu(nu,x);
}
__always_inline double Jnu_zero(double nu,unsigned int n){
    return gsl_sf_bessel_zero_Jnu(nu,n);
}
__always_inline double Inu(double nu,double x){
    return gsl_sf_bessel_Inu(nu,x);
}
__always_inline double Knu(double nu,double x){
    return gsl_sf_bessel_Knu(nu,x);
}
__always_inline double Knu_scaled(double nu,double x){
    return gsl_sf_bessel_Knu_scaled(nu,x);
}
__always_inline double Knu_ln(double nu,double x){
    return gsl_sf_bessel_lnKnu(nu,x);
}
//From https://www.gnu.org/software/gsl/doc/html/specfunc.html#clausen-functions
/*
__always_inline double gamma(double x){
    return gsl_sf_gamma(x);
}
__always_inline double lgamma(double x){
    return gsl_sf_lngamma(x);
}
__always_inline double psi(double x){
    return gsl_sf_psi(x);
}
__always_inline double psi_3(double x){
    return gsl_sf_psi_1(x);
}
__always_inline double psi_n(double x,int n){
    return gsl_sf_psi_n(n,x);
}
*/
}