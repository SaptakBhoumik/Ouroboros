#include "func/sf_func.hpp"
#include <cmath>
#include <functional>
#include <gsl/gsl_sf_psi.h>
#include <gsl/gsl_sf_trig.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_sf_airy.h>
#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_sf_clausen.h>
#include <gsl/gsl_sf_dawson.h>
#include <gsl/gsl_sf_debye.h>
#include <gsl/gsl_sf_dilog.h>
#include <gsl/gsl_sf_ellint.h>
#include <gsl/gsl_sf_expint.h>
#include <gsl/gsl_sf_fermi_dirac.h>
#include <gsl/gsl_sf_gegenbauer.h>
#include <gsl/gsl_sf_hermite.h>
#include <gsl/gsl_sf_hyperg.h>
#include <gsl/gsl_sf_laguerre.h>
#include <gsl/gsl_sf_lambert.h>
#include <gsl/gsl_sf_legendre.h>
#include <gsl/gsl_sf_synchrotron.h>
#include <gsl/gsl_sf_transport.h>
#include <gsl/gsl_sf_zeta.h>
namespace Ouroboros{
//Airy Functions
double Ai(double x){
    return gsl_sf_airy_Ai(x,GSL_PREC_DOUBLE);
}
double Bi(double x){
    return gsl_sf_airy_Bi(x,GSL_PREC_DOUBLE);
}
double Ai_scaled(double x){
    return gsl_sf_airy_Ai_scaled(x,GSL_PREC_DOUBLE);
}
double Bi_scaled(double x){
    return gsl_sf_airy_Bi_scaled(x,GSL_PREC_DOUBLE);
}
double Ai_grad(double x){
    return gsl_sf_airy_Ai_deriv(x,GSL_PREC_DOUBLE);
}
double Bi_grad(double x){
    return gsl_sf_airy_Bi_deriv(x,GSL_PREC_DOUBLE);
}
double Ai_grad_scaled(double x){
    return gsl_sf_airy_Ai_deriv_scaled(x,GSL_PREC_DOUBLE);
}
double Bi_grad_scaled(double x){
    return gsl_sf_airy_Bi_deriv_scaled(x,GSL_PREC_DOUBLE);
}
double Ai_zero(unsigned int n){
    return gsl_sf_airy_zero_Ai(n);
}
double Bi_zero(unsigned int n){
    return gsl_sf_airy_zero_Bi(n);
}
double Ai_grad_zero(unsigned int n){
    return gsl_sf_airy_zero_Ai_deriv(n);
}
double Bi_grad_zero(unsigned int n){
    return gsl_sf_airy_zero_Bi_deriv(n);
}
//Bessel Functions
double J0(double x){
    return gsl_sf_bessel_J0(x);
}
double J1(double x){
    return gsl_sf_bessel_J1(x);
}
double Jn(double x,int n){
    return gsl_sf_bessel_Jn(n,x);
}
double J0_zero(unsigned int n){
    return gsl_sf_bessel_zero_J0(n);
}
double J1_zero(unsigned int n){
    return gsl_sf_bessel_zero_J1(n);
}

double Y0(double x){
    return gsl_sf_bessel_Y0(x);
}
double Y1(double x){
    return gsl_sf_bessel_Y1(x);
}
double Yn(double x,int n){
    return gsl_sf_bessel_Yn(n,x);
}

double I0(double x){
    return gsl_sf_bessel_I0(x);
}
double I1(double x){
    return gsl_sf_bessel_I1(x);
}
double In(double x,int n){
    return gsl_sf_bessel_In(n,x);
}
double I0_scaled(double x){
    return gsl_sf_bessel_I0_scaled(x);
}
double I1_scaled(double x){
    return gsl_sf_bessel_I1_scaled(x);
}
double In_scaled(double x,int n){
    return gsl_sf_bessel_In_scaled(n,x);
}

double K0(double x){
    return gsl_sf_bessel_K0(x);
}
double K1(double x){
    return gsl_sf_bessel_K1(x);
}
double Kn(double x,int n){
    return gsl_sf_bessel_Kn(n,x);
}
double K0_scaled(double x){
    return gsl_sf_bessel_K0_scaled(x);
}
double K1_scaled(double x){
    return gsl_sf_bessel_K1_scaled(x);
}
double Kn_scaled(double x,int n){
    return gsl_sf_bessel_Kn_scaled(n,x);
}

double Ynu(double nu,double x){
    return gsl_sf_bessel_Ynu(nu,x);
}
double Jnu(double nu,double x){
    return gsl_sf_bessel_Jnu(nu,x);
}
double Jnu_zero(double nu,unsigned int n){
    return gsl_sf_bessel_zero_Jnu(nu,n);
}
double Inu(double nu,double x){
    return gsl_sf_bessel_Inu(nu,x);
}
double Knu(double nu,double x){
    return gsl_sf_bessel_Knu(nu,x);
}
double Knu_scaled(double nu,double x){
    return gsl_sf_bessel_Knu_scaled(nu,x);
}
double lKnu(double nu,double x){
    return gsl_sf_bessel_lnKnu(nu,x);
}
//Clausen Functions
double clausen(double x){
    return gsl_sf_clausen(x);
}
//Dawson Function
double dawson(double x){
    return gsl_sf_dawson(x);
}
//Debye Functions
double debye1(double x){
    return gsl_sf_debye_1(x);
}
double debye2(double x){
    return gsl_sf_debye_2(x);
}
double debye3(double x){
    return gsl_sf_debye_3(x);
}
double debye4(double x){
    return gsl_sf_debye_4(x);
}
double debye5(double x){
    return gsl_sf_debye_5(x);
}
double debye6(double x){
    return gsl_sf_debye_6(x);
}
//Dilogarithm Function
double dilog(double x){
    return gsl_sf_dilog(x);
}
//Elliptic Integrals
double Kcomp(double k){
    return gsl_sf_ellint_Kcomp(k,GSL_PREC_DOUBLE);
}
double Ecomp(double k){
    return gsl_sf_ellint_Ecomp(k,GSL_PREC_DOUBLE);
}
double Pcomp(double k,double n){
    return gsl_sf_ellint_Pcomp(k,n,GSL_PREC_DOUBLE);
}
double F(double phi,double k){
    return gsl_sf_ellint_F(phi,k,GSL_PREC_DOUBLE);
}
double E(double phi,double k){
    return gsl_sf_ellint_E(phi,k,GSL_PREC_DOUBLE);
}
double P(double phi,double k,double n){
    return gsl_sf_ellint_P(phi,k,n,GSL_PREC_DOUBLE);
}
double D(double phi,double k){
    return gsl_sf_ellint_D(phi,k,GSL_PREC_DOUBLE);
}
double RC(double x,double y){
    return gsl_sf_ellint_RC(x,y,GSL_PREC_DOUBLE);
}
double RD(double x,double y,double z){
    return gsl_sf_ellint_RD(x,y,z,GSL_PREC_DOUBLE);
}
double RF(double x,double y,double z){
    return gsl_sf_ellint_RF(x,y,z,GSL_PREC_DOUBLE);
}
double RJ(double x,double y,double z,double p){
    return gsl_sf_ellint_RJ(x,y,z,p,GSL_PREC_DOUBLE);
}
//Exponential Integrals
double E1(double x){
    return gsl_sf_expint_E1(x);
}
double E2(double x){
    return gsl_sf_expint_E2(x);
}
double En(double x,int n){
    return gsl_sf_expint_En(n,x);
}
double Ei(double x){
    return gsl_sf_expint_Ei(x);
}
double Shi(double x){
    return gsl_sf_Shi(x);
}
double Chi(double x){
    return gsl_sf_Chi(x);
}
double Ei_3(double x){
    return gsl_sf_expint_3(x);
}
double Si(double x){
    return gsl_sf_Si(x);
}
double Ci(double x){
    return gsl_sf_Ci(x);
}
double atan_int(double x){
    return gsl_sf_atanint(x);
}
//Fermi-Dirac Functions
double fermi_dirac_m1(double x){
    return gsl_sf_fermi_dirac_m1(x);
}
double fermi_dirac_0(double x){
    return gsl_sf_fermi_dirac_0(x);
}
double fermi_dirac_1(double x){
    return gsl_sf_fermi_dirac_1(x);
}
double fermi_dirac(double x,int j){
    return gsl_sf_fermi_dirac_int(j,x);
}
double fermi_dirac_half(double x){
    return gsl_sf_fermi_dirac_half(x);
}
double fermi_dirac_mhalf(double x){
    return gsl_sf_fermi_dirac_mhalf(x);
}
double fermi_dirac_3half(double x){
    return gsl_sf_fermi_dirac_3half(x);
}
double fermi_dirac_inc_0(double x,double b){
    return gsl_sf_fermi_dirac_inc_0(x,b);
}
//Gamma and Related Functions
double gamma(double x){
    return gsl_sf_gamma(x);
}
double lgamma(double x){
    return gsl_sf_lngamma(x);
}
double gamma_star(double x){
    return gsl_sf_gammastar(x);
}
double gamma_inv(double x){
    return gsl_sf_gammainv(x);
}
double taylorcoeff(double x,int n){
    return gsl_sf_taylorcoeff(n,x);
}
double poch(double a, double x){
    return gsl_sf_poch(a,x);
}
double lpoch(double a, double x){
    return gsl_sf_lnpoch(a,x);
}
double pochrel(double a, double x){
    return gsl_sf_pochrel(a,x);
}
double gamma_inc(double a,double x){
    return gsl_sf_gamma_inc(a,x);
}
double gamma_inc_Q(double a,double x){
    return gsl_sf_gamma_inc_Q(a,x);
}
double gamma_inc_P(double a,double x){
    return gsl_sf_gamma_inc_P(a,x);
}
double beta(double a,double b){
    return gsl_sf_beta(a,b);
}
double lbeta(double a,double b){
    return gsl_sf_lnbeta(a,b);
}
double beta_inc(double a,double b,double x){
    return gsl_sf_beta_inc(a,b,x);
}
//Gegenbauer Functions
double gegenpoly1(double lambda,double x){
    return gsl_sf_gegenpoly_1(lambda,x);
}
double gegenpoly2(double lambda,double x){
    return gsl_sf_gegenpoly_2(lambda,x);
}
double gegenpoly3(double lambda,double x){
    return gsl_sf_gegenpoly_3(lambda,x);
}
double gegenpoly(double lambda,double x,int n){
    return gsl_sf_gegenpoly_n(n,lambda,x);
}
//Hermite Functions
double hermite(double x,int n){
    return gsl_sf_hermite(n,x);
}
double hermite_prob(double x,int n){
    return gsl_sf_hermite_prob(n,x);
}
double hermite_grad(double x,int n,int order){
    return gsl_sf_hermite_deriv(order,n,x);
}
double hermite_prob_grad(double x,int n,int order){
    return gsl_sf_hermite_prob_deriv(order,n,x);
}
double hermite_func(double x,int n){
    return gsl_sf_hermite_func(n,x);
}
double hermite_func_fast(double x,int n){
    return gsl_sf_hermite_func_fast(n,x);
}
double hermite_func_grad(double x,int n,int order){
    return gsl_sf_hermite_func_der(order,n,x);
}
double hermite_zero(int n,int s){
    return gsl_sf_hermite_zero(n,s);
}
double hermite_prob_zero(int n,int s){
    return gsl_sf_hermite_prob_zero(n,s);
}
double hermite_func_zero(int n,int s){
    return gsl_sf_hermite_func_zero(n,s);
}
//Hypergeometric Functions
double F01(double a,double b){
    return gsl_sf_hyperg_0F1(a,b);
}
double F11(double a,double b,double x){
    return gsl_sf_hyperg_1F1(a,b,x);
}
double U(double a,double b,double x){
    return gsl_sf_hyperg_U(a,b,x);
}
double F21(double a,double b,double c,double x){
    return gsl_sf_hyperg_2F1(a,b,c,x);
}
double F21_renorm(double a,double b,double c,double x){
    return gsl_sf_hyperg_2F1_renorm(a,b,c,x);
}
double F20(double a,double b,double x){
    return gsl_sf_hyperg_2F0(a,b,x);
}
//Laguerre Functions
double L1(double a,double x){
    return gsl_sf_laguerre_1(a,x);
}
double L2(double a,double x){
    return gsl_sf_laguerre_2(a,x);
}
double L3(double a,double x){
    return gsl_sf_laguerre_3(a,x);
}
double L(double a,double x,int n){
    return gsl_sf_laguerre_n(n,a,x);
}
//Lambert W Functions
double W0(double x){
    return gsl_sf_lambert_W0(x);
}
double Wm1(double x){
    return gsl_sf_lambert_Wm1(x);
}
//Legendre Polynomials
double legendre_P1(double x){
    return gsl_sf_legendre_P1(x);
}
double legendre_P2(double x){
    return gsl_sf_legendre_P2(x);
}
double legendre_P3(double x){
    return gsl_sf_legendre_P3(x);
}
double legendre_P(double x,int l){
    return gsl_sf_legendre_Pl(l,x);
}
double Q0(double x){
    return gsl_sf_legendre_Q0(x);
}
double Q1(double x){
    return gsl_sf_legendre_Q1(x);
}
double Q(double x,int l){
    return gsl_sf_legendre_Ql(l,x);
}
//Associated Legendre Polynomials and Spherical Harmonics
double Plm(double x,int l,int m){
    return gsl_sf_legendre_sphPlm(l,m,x);
}
double sphPlm(double x,int l,int m){
    return gsl_sf_legendre_sphPlm(l,m,x);
}
//Conical Functions
double conicalP_half(double x,double lambda){
    return gsl_sf_conicalP_half(lambda,x);
}
double conicalP_mhalf(double x,double lambda){
    return gsl_sf_conicalP_mhalf(lambda,x);
}
double conicalP0(double x,double lambda){
    return gsl_sf_conicalP_0(lambda,x);
}
double conicalP1(double x,double lambda){
    return gsl_sf_conicalP_1(lambda,x);
}
double conicalP_sph(double x,double lambda,int n){
    return gsl_sf_conicalP_sph_reg(n,lambda,x);
}
double conicalP_cyl(double x,double lambda,int n){
    return gsl_sf_conicalP_cyl_reg(n,lambda,x);
}
//Radial Functions for Hyperbolic Space
double H3d0(double lambda,double eta){
    return gsl_sf_legendre_H3d_0(lambda,eta);
}
double H3d1(double lambda,double eta){
    return gsl_sf_legendre_H3d_1(lambda,eta);
}
double H3d(double lambda,double eta,int n){
    return gsl_sf_legendre_H3d(n,lambda,eta);
}
//Psi Functions
double psi(double x){
    return gsl_sf_psi(x);
}
double psi1(double x){
    return gsl_sf_psi_1(x);
}
double psi_n(double x,int n){
    return gsl_sf_psi_n(n,x);
}
//Synchrotron Functions
double synchrotron1(double x){
    return gsl_sf_synchrotron_1(x);
}
double synchrotron2(double x){
    return gsl_sf_synchrotron_2(x);
}
//Transport Functions
double transport2(double x){
    return gsl_sf_transport_2(x);
}
double transport3(double x){
    return gsl_sf_transport_3(x);
}
double transport4(double x){
    return gsl_sf_transport_4(x);
}
double transport5(double x){
    return gsl_sf_transport_5(x);
}
//Zeta Functions
double zeta(double x){
    return gsl_sf_zeta(x);
}
double zetam1(double x){
    return gsl_sf_zetam1(x);
}
double hzeta(double x,double q){
    return gsl_sf_hzeta(x,q);
}
double eta(double x){
    return gsl_sf_eta(x);
}
}