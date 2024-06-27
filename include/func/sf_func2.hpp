//TODO:
//The tensor versions of the functions in the file func/sf_func1.hpp will be defined here
#pragma once
#include "../macros.hpp"
#include "../tensor.hpp"
namespace Ouroboros{
//For int and unsigned arguments the double value from tensor is casted to an int or unsigned int respectively
//Airy Functions
Tensor Ai(const Tensor& t,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor Bi(const Tensor& t,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor Ai_scaled(const Tensor& t,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor Bi_scaled(const Tensor& t,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor Ai_grad(const Tensor& t,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor Bi_grad(const Tensor& t,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor Ai_grad_scaled(const Tensor& t,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor Bi_grad_scaled(const Tensor& t,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor Ai_zero(const Tensor& t,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor Bi_zero(const Tensor& t,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor Ai_grad_zero(const Tensor& t,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor Bi_grad_zero(const Tensor& t,size_t min_count=__MIN__COUNT__FOR__THREAD__);
//Bessel Functions
Tensor J0(const Tensor& t,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor J1(const Tensor& t,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor Jn(const Tensor& x,int n,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor Jn(double x,const Tensor& n,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor Jn(const Tensor& x,const Tensor& n,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor J0_zero(const Tensor& t,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor J1_zero(const Tensor& t,size_t min_count=__MIN__COUNT__FOR__THREAD__);

Tensor Y0(const Tensor& t,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor Y1(const Tensor& t,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor Yn(const Tensor& x,int n,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor Yn(double x,const Tensor& n,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor Yn(const Tensor& x,const Tensor& n,size_t min_count=__MIN__COUNT__FOR__THREAD__);

Tensor I0(const Tensor& t,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor I1(const Tensor& t,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor In(const Tensor& x,int n,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor In(double x,const Tensor& n,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor In(const Tensor& x,const Tensor& n,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor I0_scaled(const Tensor& x,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor I1_scaled(const Tensor& x,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor In_scaled(const Tensor& x,int n,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor In_scaled(double x,const Tensor& n,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor In_scaled(const Tensor& x,const Tensor& n,size_t min_count=__MIN__COUNT__FOR__THREAD__);

Tensor K0(const Tensor& x,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor K1(const Tensor& x,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor Kn(const Tensor& x,int n,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor Kn(double x,const Tensor& n,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor Kn(const Tensor& x,const Tensor& n,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor K0_scaled(const Tensor& x,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor K1_scaled(const Tensor& x,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor Kn_scaled(const Tensor& x,int n,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor Kn_scaled(double x,const Tensor& n,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor Kn_scaled(const Tensor& x,const Tensor& n,size_t min_count=__MIN__COUNT__FOR__THREAD__);

Tensor Ynu(const Tensor& nu,double x,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor Ynu(double nu,const Tensor& x,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor Ynu(const Tensor& nu,const Tensor& x,size_t min_count=__MIN__COUNT__FOR__THREAD__);

Tensor Jnu(const Tensor& nu,double x,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor Jnu(double nu,const Tensor& x,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor Jnu(const Tensor& nu,const Tensor& x,size_t min_count=__MIN__COUNT__FOR__THREAD__);

Tensor Jnu_zero(const Tensor& nu,unsigned int n,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor Jnu_zero(double nu,const Tensor& n,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor Jnu_zero(const Tensor& nu,const Tensor& n,size_t min_count=__MIN__COUNT__FOR__THREAD__);

Tensor Inu(const Tensor& nu,double x,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor Inu(double nu,const Tensor& x,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor Inu(const Tensor& nu,const Tensor& x,size_t min_count=__MIN__COUNT__FOR__THREAD__);

Tensor Knu(const Tensor& nu,double x,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor Knu(double nu,const Tensor& x,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor Knu(const Tensor& nu,const Tensor& x,size_t min_count=__MIN__COUNT__FOR__THREAD__);

Tensor Knu_scaled(const Tensor& nu,double x,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor Knu_scaled(double nu,const Tensor& x,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor Knu_scaled(const Tensor& nu,const Tensor& x,size_t min_count=__MIN__COUNT__FOR__THREAD__);

Tensor lKnu(const Tensor& nu,double x,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor lKnu(double nu,const Tensor& x,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor lKnu(const Tensor& nu,const Tensor& x,size_t min_count=__MIN__COUNT__FOR__THREAD__);
//Clausen Functions
Tensor clausen(const Tensor& t,size_t min_count=__MIN__COUNT__FOR__THREAD__);
//Dawson Function
Tensor dawson(const Tensor& t,size_t min_count=__MIN__COUNT__FOR__THREAD__);
//Debye Functions
Tensor debye1(const Tensor& t,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor debye2(const Tensor& t,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor debye3(const Tensor& t,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor debye4(const Tensor& t,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor debye5(const Tensor& t,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor debye6(const Tensor& t,size_t min_count=__MIN__COUNT__FOR__THREAD__);
//Dilogarithm Function
Tensor dilog(const Tensor& t,size_t min_count=__MIN__COUNT__FOR__THREAD__);
//Elliptic Integrals
Tensor Kcomp(const Tensor& t,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor Ecomp(const Tensor& t,size_t min_count=__MIN__COUNT__FOR__THREAD__);

Tensor Pcomp(const Tensor& k,double n,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor Pcomp(double k,const Tensor& n,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor Pcomp(const Tensor& k,const Tensor& n,size_t min_count=__MIN__COUNT__FOR__THREAD__);

Tensor F(const Tensor& phi,double k,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor F(double phi,const Tensor& k,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor F(const Tensor& phi,const Tensor& k,size_t min_count=__MIN__COUNT__FOR__THREAD__);

Tensor E(const Tensor& phi,double k,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor E(double phi,const Tensor& k,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor E(const Tensor& phi,const Tensor& k,size_t min_count=__MIN__COUNT__FOR__THREAD__);

Tensor P(const Tensor& phi,double k,double n,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor P(double phi,const Tensor& k,double n,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor P(double phi,double k,const Tensor& n,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor P(const Tensor& phi,const Tensor& k,double n,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor P(double phi,const Tensor& k,const Tensor& n,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor P(const Tensor& phi,double k,const Tensor& n,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor P(const Tensor& phi,const Tensor& k,const Tensor& n,size_t min_count=__MIN__COUNT__FOR__THREAD__);

Tensor D(const Tensor& phi,double k,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor D(double phi,const Tensor& k,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor D(const Tensor& phi,const Tensor& k,size_t min_count=__MIN__COUNT__FOR__THREAD__);

Tensor RC(const Tensor& x,double y,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor RC(double x,const Tensor& y,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor RC(const Tensor& x,const Tensor& y,size_t min_count=__MIN__COUNT__FOR__THREAD__);

Tensor RD(const Tensor& x,double y,double z,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor RD(double x,const Tensor& y,double z,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor RD(double x,double y,const Tensor& z,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor RD(const Tensor& x,const Tensor& y,double z,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor RD(const Tensor& x,double y,const Tensor& z,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor RD(double x,const Tensor& y,const Tensor& z,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor RD(const Tensor& x,const Tensor& y,const Tensor& z,size_t min_count=__MIN__COUNT__FOR__THREAD__);

Tensor RF(const Tensor& x,double y,double z,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor RF(double x,const Tensor& y,double z,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor RF(double x,double y,const Tensor& z,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor RF(const Tensor& x,const Tensor& y,double z,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor RF(const Tensor& x,double y,const Tensor& z,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor RF(double x,const Tensor& y,const Tensor& z,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor RF(const Tensor& x,const Tensor& y,const Tensor& z,size_t min_count=__MIN__COUNT__FOR__THREAD__);

Tensor RJ(const Tensor& x,double y,double z,double p,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor RJ(double x,const Tensor& y,double z,double p,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor RJ(double x,double y,const Tensor& z,double p,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor RJ(double x,double y,double z,const Tensor& p,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor RJ(const Tensor& x,const Tensor& y,double z,double p,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor RJ(const Tensor& x,double y,const Tensor& z,double p,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor RJ(const Tensor& x,double y,double z,const Tensor& p,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor RJ(double x,const Tensor& y,const Tensor& z,double p,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor RJ(double x,const Tensor& y,double z,const Tensor& p,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor RJ(double x,double y,const Tensor& z,const Tensor& p,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor RJ(const Tensor& x,const Tensor& y,const Tensor& z,double p,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor RJ(const Tensor& x,const Tensor& y,double z,const Tensor& p,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor RJ(const Tensor& x,double y,const Tensor& z,const Tensor& p,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor RJ(double x,const Tensor& y,const Tensor& z,const Tensor& p,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor RJ(const Tensor& x,const Tensor& y,const Tensor& z,const Tensor& p,size_t min_count=__MIN__COUNT__FOR__THREAD__);
//Exponential Integrals
Tensor E1(const Tensor& t,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor E2(const Tensor& t,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor En(const Tensor& x,int n,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor En(double x,const Tensor& n,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor En(const Tensor& x,const Tensor& n,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor Ei(const Tensor& t,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor Shi(const Tensor& t,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor Chi(const Tensor& t,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor Ei_3(const Tensor& t,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor Si(const Tensor& t,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor Ci(const Tensor& t,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor atan_int(const Tensor& t,size_t min_count=__MIN__COUNT__FOR__THREAD__);
//Fermi-Dirac Functions
Tensor fermi_dirac_m1(const Tensor& t,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor fermi_dirac_0(const Tensor& t,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor fermi_dirac_1(const Tensor& t,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor fermi_dirac(const Tensor& x,int j,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor fermi_dirac(double x,const Tensor& j,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor fermi_dirac(const Tensor& x,const Tensor& j,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor fermi_dirac_half(const Tensor& t,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor fermi_dirac_mhalf(const Tensor& t,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor fermi_dirac_3half(const Tensor& t,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor fermi_dirac_inc_0(const Tensor& x,double b,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor fermi_dirac_inc_0(double x,const Tensor& b,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor fermi_dirac_inc_0(const Tensor& x,const Tensor& b,size_t min_count=__MIN__COUNT__FOR__THREAD__);
//Gamma and Related Functions
Tensor gamma(const Tensor& t,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor lgamma(const Tensor& t,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor gamma_star(const Tensor& t,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor gamma_inv(const Tensor& t,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor taylorcoeff(const Tensor& x,int n,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor taylorcoeff(double x,const Tensor& n,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor taylorcoeff(const Tensor& x,const Tensor& n,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor poch(const Tensor& a, double x,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor poch(double a, const Tensor& x,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor poch(const Tensor& a, const Tensor& x,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor lpoch(const Tensor& a, double x,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor lpoch(double a, const Tensor& x,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor lpoch(const Tensor& a, const Tensor& x,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor pochrel(const Tensor& a, double x,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor pochrel(double a, const Tensor& x,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor pochrel(const Tensor& a, const Tensor& x,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor gamma_inc(const Tensor& a, double x,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor gamma_inc(double a, const Tensor& x,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor gamma_inc(const Tensor& a, const Tensor& x,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor gamma_inc_Q(const Tensor& a, double x,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor gamma_inc_Q(double a, const Tensor& x,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor gamma_inc_Q(const Tensor& a, const Tensor& x,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor gamma_inc_P(const Tensor& a, double x,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor gamma_inc_P(double a, const Tensor& x,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor gamma_inc_P(const Tensor& a, const Tensor& x,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor beta(const Tensor& a, double x,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor beta(double a, const Tensor& x,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor beta(const Tensor& a, const Tensor& x,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor lbeta(const Tensor& a, double x,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor lbeta(double a, const Tensor& x,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor lbeta(const Tensor& a, const Tensor& x,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor beta_inc(const Tensor& a,double b,double x,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor beta_inc(double a,const Tensor& b,double x,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor beta_inc(double a,double b,const Tensor& x,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor beta_inc(const Tensor& a,const Tensor& b,double x,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor beta_inc(const Tensor& a,double b,const Tensor& x,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor beta_inc(double a,const Tensor& b,const Tensor& x,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor beta_inc(const Tensor& a,const Tensor& b,const Tensor& x,size_t min_count=__MIN__COUNT__FOR__THREAD__);
//Gegenbauer Functions
Tensor gegenpoly1(const Tensor& lambda,double x,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor gegenpoly1(double lambda,const Tensor& x,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor gegenpoly1(const Tensor& lambda,const Tensor& x,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor gegenpoly2(const Tensor& lambda,double x,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor gegenpoly2(double lambda,const Tensor& x,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor gegenpoly2(const Tensor& lambda,const Tensor& x,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor gegenpoly3(const Tensor& lambda,double x,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor gegenpoly3(double lambda,const Tensor& x,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor gegenpoly3(const Tensor& lambda,const Tensor& x,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor gegenpoly(const Tensor& lambda,double x,int n,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor gegenpoly(double lambda,const Tensor& x,int n,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor gegenpoly(double lambda,double x,const Tensor& n,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor gegenpoly(const Tensor& lambda,const Tensor& x,int n,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor gegenpoly(double lambda,const Tensor& x,const Tensor& n,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor gegenpoly(const Tensor& lambda,double x,const Tensor& n,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor gegenpoly(const Tensor& lambda,const Tensor& x,const Tensor& n,size_t min_count=__MIN__COUNT__FOR__THREAD__);
//Hermite Functions
Tensor hermite(const Tensor& x,int n,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor hermite(double x,const Tensor& n,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor hermite(const Tensor& x,const Tensor& n,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor hermite_prob(const Tensor& x,int n,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor hermite_prob(double x,const Tensor& n,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor hermite_prob(const Tensor& x,const Tensor& n,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor hermite_grad(const Tensor& x,int n,int order,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor hermite_grad(double x,const Tensor& n,int order,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor hermite_grad(double x,int n,const Tensor& order,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor hermite_grad(const Tensor& x,const Tensor& n,int order,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor hermite_grad(const Tensor& x,int n,const Tensor& order,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor hermite_grad(double x,const Tensor& n,const Tensor& order,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor hermite_grad(const Tensor& x,const Tensor& n,const Tensor& order,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor hermite_prob_grad(const Tensor& x,int n,int order,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor hermite_prob_grad(double x,const Tensor& n,int order,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor hermite_prob_grad(double x,int n,const Tensor& order,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor hermite_prob_grad(const Tensor& x,const Tensor& n,int order,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor hermite_prob_grad(const Tensor& x,int n,const Tensor& order,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor hermite_prob_grad(double x,const Tensor& n,const Tensor& order,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor hermite_prob_grad(const Tensor& x,const Tensor& n,const Tensor& order,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor hermite_func(const Tensor& x,int n,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor hermite_func(double x,const Tensor& n,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor hermite_func(const Tensor& x,const Tensor& n,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor hermite_func_fast(const Tensor& x,int n,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor hermite_func_fast(double x,const Tensor& n,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor hermite_func_fast(const Tensor& x,const Tensor& n,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor hermite_func_grad(const Tensor& x,int n,int order,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor hermite_func_grad(double x,const Tensor& n,int order,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor hermite_func_grad(double x,int n,const Tensor& order,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor hermite_func_grad(const Tensor& x,const Tensor& n,int order,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor hermite_func_grad(const Tensor& x,int n,const Tensor& order,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor hermite_func_grad(double x,const Tensor& n,const Tensor& order,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor hermite_func_grad(const Tensor& x,const Tensor& n,const Tensor& order,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor hermite_zero(const Tensor& n,int s,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor hermite_zero(int n,const Tensor& s,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor hermite_zero(const Tensor& n,const Tensor& s,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor hermite_prob_zero(const Tensor& n,int s,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor hermite_prob_zero(int n,const Tensor& s,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor hermite_prob_zero(const Tensor& n,const Tensor& s,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor hermite_func_zero(const Tensor& n,int s,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor hermite_func_zero(int n,const Tensor& s,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor hermite_func_zero(const Tensor& n,const Tensor& s,size_t min_count=__MIN__COUNT__FOR__THREAD__);
//Hypergeometric Functions
Tensor F01(const Tensor& a,double b,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor F01(double a,const Tensor& b,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor F01(const Tensor& a,const Tensor& b,size_t min_count=__MIN__COUNT__FOR__THREAD__);

Tensor F11(const Tensor& a,double b,double x,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor F11(double a,const Tensor& b,double x,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor F11(double a,double b,const Tensor& x,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor F11(const Tensor& a,const Tensor& b,double x,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor F11(const Tensor& a,double b,const Tensor& x,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor F11(double a,const Tensor& b,const Tensor& x,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor F11(const Tensor& a,const Tensor& b,const Tensor& x,size_t min_count=__MIN__COUNT__FOR__THREAD__);

Tensor U(const Tensor& a,double b,double x,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor U(double a,const Tensor& b,double x,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor U(double a,double b,const Tensor& x,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor U(const Tensor& a,const Tensor& b,double x,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor U(const Tensor& a,double b,const Tensor& x,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor U(double a,const Tensor& b,const Tensor& x,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor U(const Tensor& a,const Tensor& b,const Tensor& x,size_t min_count=__MIN__COUNT__FOR__THREAD__);

Tensor F21(const Tensor& a,double b,double c,double x,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor F21(double a,const Tensor& b,double c,double x,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor F21(double a,double b,const Tensor& c,double x,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor F21(double a,double b,double c,const Tensor& x,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor F21(const Tensor& a,const Tensor& b,double c,double x,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor F21(const Tensor& a,double b,const Tensor& c,double x,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor F21(const Tensor& a,double b,double c,const Tensor& x,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor F21(double a,const Tensor& b,const Tensor& c,double x,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor F21(double a,const Tensor& b,double c,const Tensor& x,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor F21(double a,double b,const Tensor& c,const Tensor& x,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor F21(const Tensor& a,const Tensor& b,const Tensor& c,double x,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor F21(const Tensor& a,const Tensor& b,double c,const Tensor& x,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor F21(const Tensor& a,double b,const Tensor& c,const Tensor& x,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor F21(double a,const Tensor& b,const Tensor& c,const Tensor& x,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor F21(const Tensor& a,const Tensor& b,const Tensor& c,const Tensor& x,size_t min_count=__MIN__COUNT__FOR__THREAD__);

Tensor F21_renorm(const Tensor& a,double b,double c,double x,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor F21_renorm(double a,const Tensor& b,double c,double x,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor F21_renorm(double a,double b,const Tensor& c,double x,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor F21_renorm(double a,double b,double c,const Tensor& x,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor F21_renorm(const Tensor& a,const Tensor& b,double c,double x,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor F21_renorm(const Tensor& a,double b,const Tensor& c,double x,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor F21_renorm(const Tensor& a,double b,double c,const Tensor& x,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor F21_renorm(double a,const Tensor& b,const Tensor& c,double x,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor F21_renorm(double a,const Tensor& b,double c,const Tensor& x,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor F21_renorm(double a,double b,const Tensor& c,const Tensor& x,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor F21_renorm(const Tensor& a,const Tensor& b,const Tensor& c,double x,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor F21_renorm(const Tensor& a,const Tensor& b,double c,const Tensor& x,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor F21_renorm(const Tensor& a,double b,const Tensor& c,const Tensor& x,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor F21_renorm(double a,const Tensor& b,const Tensor& c,const Tensor& x,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor F21_renorm(const Tensor& a,const Tensor& b,const Tensor& c,const Tensor& x,size_t min_count=__MIN__COUNT__FOR__THREAD__);

Tensor F20(const Tensor& a,double b,double x,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor F20(double a,const Tensor& b,double x,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor F20(double a,double b,const Tensor& x,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor F20(const Tensor& a,const Tensor& b,double x,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor F20(const Tensor& a,double b,const Tensor& x,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor F20(double a,const Tensor& b,const Tensor& x,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor F20(const Tensor& a,const Tensor& b,const Tensor& x,size_t min_count=__MIN__COUNT__FOR__THREAD__);
//Laguerre Functions
Tensor L1(const Tensor& a,double b,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor L1(double a,const Tensor& b,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor L1(const Tensor& a,const Tensor& b,size_t min_count=__MIN__COUNT__FOR__THREAD__);

Tensor L2(const Tensor& a,double b,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor L2(double a,const Tensor& b,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor L2(const Tensor& a,const Tensor& b,size_t min_count=__MIN__COUNT__FOR__THREAD__);

Tensor L3(const Tensor& a,double b,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor L3(double a,const Tensor& b,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor L3(const Tensor& a,const Tensor& b,size_t min_count=__MIN__COUNT__FOR__THREAD__);

Tensor L(const Tensor& a,double x,int n,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor L(double a,const Tensor& x,int n,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor L(double a,double x,const Tensor& n,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor L(const Tensor& a,const Tensor& x,int n,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor L(const Tensor& a,double x,const Tensor& n,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor L(double a,const Tensor& x,const Tensor& n,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor L(const Tensor& a,const Tensor& x,const Tensor& n,size_t min_count=__MIN__COUNT__FOR__THREAD__);
//Lambert W Functions
Tensor W0(const Tensor& x,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor Wm1(const Tensor& x,size_t min_count=__MIN__COUNT__FOR__THREAD__);
//Legendre Polynomials
Tensor legendre_P1(const Tensor& x,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor legendre_P2(const Tensor& x,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor legendre_P3(const Tensor& x,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor legendre_P(const Tensor& x,int l,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor legendre_P(double x,const Tensor& l,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor legendre_P(const Tensor& x,const Tensor& l,size_t min_count=__MIN__COUNT__FOR__THREAD__);

Tensor Q0(const Tensor& x,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor Q1(const Tensor& x,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor Q(const Tensor& x,int l,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor Q(double x,const Tensor& l,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor Q(const Tensor& x,const Tensor& l,size_t min_count=__MIN__COUNT__FOR__THREAD__);
//Associated Legendre Polynomials and Spherical Harmonics
Tensor Plm(const Tensor& x,int l,int m,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor Plm(double x,const Tensor& l,int m,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor Plm(double x,int l,const Tensor& m,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor Plm(const Tensor& x,const Tensor& l,int m,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor Plm(const Tensor& x,int l,const Tensor& m,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor Plm(double x,const Tensor& l,const Tensor& m,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor Plm(const Tensor& x,const Tensor& l,const Tensor& m,size_t min_count=__MIN__COUNT__FOR__THREAD__);

Tensor sphPlm(const Tensor& x,int l,int m,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor sphPlm(double x,const Tensor& l,int m,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor sphPlm(double x,int l,const Tensor& m,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor sphPlm(const Tensor& x,const Tensor& l,int m,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor sphPlm(const Tensor& x,int l,const Tensor& m,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor sphPlm(double x,const Tensor& l,const Tensor& m,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor sphPlm(const Tensor& x,const Tensor& l,const Tensor& m,size_t min_count=__MIN__COUNT__FOR__THREAD__);
//Conical Functions
Tensor conicalP_half(const Tensor& x,double lambda,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor conicalP_half(double x,const Tensor& lambda,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor conicalP_half(const Tensor& x,const Tensor& lambda,size_t min_count=__MIN__COUNT__FOR__THREAD__);

Tensor conicalP_mhalf(const Tensor& x,double lambda,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor conicalP_mhalf(double x,const Tensor& lambda,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor conicalP_mhalf(const Tensor& x,const Tensor& lambda,size_t min_count=__MIN__COUNT__FOR__THREAD__);

Tensor conicalP0(const Tensor& x,double lambda,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor conicalP0(double x,const Tensor& lambda,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor conicalP0(const Tensor& x,const Tensor& lambda,size_t min_count=__MIN__COUNT__FOR__THREAD__);

Tensor conicalP1(const Tensor& x,double lambda,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor conicalP1(double x,const Tensor& lambda,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor conicalP1(const Tensor& x,const Tensor& lambda,size_t min_count=__MIN__COUNT__FOR__THREAD__);

Tensor conicalP_sph(const Tensor& x,double lambda,int n,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor conicalP_sph(double x,const Tensor& lambda,int n,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor conicalP_sph(double x,double lambda,const Tensor& n,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor conicalP_sph(const Tensor& x,const Tensor& lambda,int n,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor conicalP_sph(const Tensor& x,double lambda,const Tensor& n,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor conicalP_sph(double x,const Tensor& lambda,const Tensor& n,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor conicalP_sph(const Tensor& x,const Tensor& lambda,const Tensor& n,size_t min_count=__MIN__COUNT__FOR__THREAD__);

Tensor conicalP_cyl(const Tensor& x,double lambda,int n,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor conicalP_cyl(double x,const Tensor& lambda,int n,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor conicalP_cyl(double x,double lambda,const Tensor& n,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor conicalP_cyl(const Tensor& x,const Tensor& lambda,int n,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor conicalP_cyl(const Tensor& x,double lambda,const Tensor& n,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor conicalP_cyl(double x,const Tensor& lambda,const Tensor& n,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor conicalP_cyl(const Tensor& x,const Tensor& lambda,const Tensor& n,size_t min_count=__MIN__COUNT__FOR__THREAD__);
//Radial Functions for Hyperbolic Space
Tensor H3d0(const Tensor& lambda,double eta,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor H3d0(double lambda,const Tensor& eta,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor H3d0(const Tensor& lambda,const Tensor& eta,size_t min_count=__MIN__COUNT__FOR__THREAD__);

Tensor H3d1(const Tensor& lambda,double eta,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor H3d1(double lambda,const Tensor& eta,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor H3d1(const Tensor& lambda,const Tensor& eta,size_t min_count=__MIN__COUNT__FOR__THREAD__);

Tensor H3d(const Tensor& lambda,double eta,int n,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor H3d(double lambda,const Tensor& eta,int n,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor H3d(double lambda,double eta,const Tensor& n,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor H3d(const Tensor& lambda,const Tensor& eta,int n,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor H3d(const Tensor& lambda,double eta,const Tensor& n,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor H3d(double lambda,const Tensor& eta,const Tensor& n,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor H3d(const Tensor& lambda,const Tensor& eta,const Tensor& n,size_t min_count=__MIN__COUNT__FOR__THREAD__);
}