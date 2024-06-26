#pragma once
#include "../macros.hpp"
namespace Ouroboros{
namespace Scalar{
//Airy Functions
double Ai(double x);
double Bi(double x);
double Ai_scaled(double x);
double Bi_scaled(double x);
double Ai_grad(double x);
double Bi_grad(double x);
double Ai_grad_scaled(double x);
double Bi_grad_scaled(double x);
double Ai_zero(unsigned int n);
double Bi_zero(unsigned int n);
double Ai_grad_zero(unsigned int n);
double Bi_grad_zero(unsigned int n);
//Bessel Functions
double J0(double x);
double J1(double x);
double Jn(double x,int n);
double J0_zero(unsigned int n);
double J1_zero(unsigned int n);

double Y0(double x);
double Y1(double x);
double Yn(double x,int n);

double I0(double x);
double I1(double x);
double In(double x,int n);
double I0_scaled(double x);
double I1_scaled(double x);
double In_scaled(double x,int n);

double K0(double x);
double K1(double x);
double Kn(double x,int n);
double K0_scaled(double x);
double K1_scaled(double x);
double Kn_scaled(double x,int n);

double Ynu(double nu,double x);
double Jnu(double nu,double x);
double Jnu_zero(double nu,unsigned int n);
double Inu(double nu,double x);
double Knu(double nu,double x);
double Knu_scaled(double nu,double x);
double lKnu(double nu,double x);
//Clausen Functions
double clausen(double x);
//Dawson Function
double dawson(double x);
//Debye Functions
double debye1(double x);
double debye2(double x);
double debye3(double x);
double debye4(double x);
double debye5(double x);
double debye6(double x);
//Dilogarithm Function
double dilog(double x);
//Elliptic Integrals
double Kcomp(double k);
double Ecomp(double k);
double Pcomp(double k,double n);
double F(double phi,double k);
double E(double phi,double k);
double P(double phi,double k,double n);
double D(double phi,double k);
double RC(double x,double y);
double RD(double x,double y,double z);
double RF(double x,double y,double z);
double RJ(double x,double y,double z,double p);
//Exponential Integrals
double E1(double x);
double E2(double x);
double En(double x,int n);
double Ei(double x);
double Shi(double x);
double Chi(double x);
double Ei_3(double x);
double Si(double x);
double Ci(double x);
double atan_int(double x);
//Fermi-Dirac Functions
double fermi_dirac_m1(double x);
double fermi_dirac_0(double x);
double fermi_dirac_1(double x);
double fermi_dirac(double x,int j);
double fermi_dirac_half(double x);
double fermi_dirac_mhalf(double x);
double fermi_dirac_3half(double x);
double fermi_dirac_inc_0(double x,double b);
//Gamma and Related Functions
double gamma(double x);
double lgamma(double x);
double gamma_star(double x);
double gamma_inv(double x);
double taylorcoeff(double x,int n);
double poch(double a, double x);
double lpoch(double a, double x);
double pochrel(double a, double x);
double gamma_inc(double a,double x);
double gamma_inc_Q(double a,double x);
double gamma_inc_P(double a,double x);
double beta(double a,double b);
double lbeta(double a,double b);
double beta_inc(double a,double b,double x);
//Gegenbauer Functions
double gegenpoly1(double lambda,double x);
double gegenpoly2(double lambda,double x);
double gegenpoly3(double lambda,double x);
double gegenpoly(double lambda,double x,int n);
//Hermite Functions
double hermite(double x,int n);
double hermite_prob(double x,int n);
double hermite_grad(double x,int n,int order);
double hermite_prob_grad(double x,int n,int order);
double hermite_func(double x,int n);
double hermite_func_fast(double x,int n);
double hermite_func_grad(double x,int n,int order);
double hermite_zero(int n,int s);
double hermite_prob_zero(int n,int s);
double hermite_func_zero(int n,int s);
//Hypergeometric Functions
double F01(double a,double b);
double F11(double a,double b,double x);
double U(double a,double b,double x);
double F21(double a,double b,double c,double x);
double F21_renorm(double a,double b,double c,double x);
double F20(double a,double b,double x);
//Laguerre Functions
double L1(double a,double x);
double L2(double a,double x);
double L3(double a,double x);
double L(double a,double x,int n);
//Lambert W Functions
double W0(double x);
double Wm1(double x);
//Legendre Polynomials
double legendre_P1(double x);
double legendre_P2(double x);
double legendre_P3(double x);
double legendre_P(double x,int l);
double Q0(double x);
double Q1(double x);
double Q(double x,int l);
//Associated Legendre Polynomials and Spherical Harmonics
double Plm(double x,int l,int m);
double sphPlm(double x,int l,int m);
//Conical Functions
double conicalP_half(double x,double lambda);
double conicalP_mhalf(double x,double lambda);
double conicalP0(double x,double lambda);
double conicalP1(double x,double lambda);
double conicalP_sph(double x,double lambda,int n);
double conicalP_cyl(double x,double lambda,int n);
//Radial Functions for Hyperbolic Space
double H3d0(double lambda,double eta);
double H3d1(double lambda,double eta);
double H3d(double lambda,double eta,int n);
//Psi Functions
double psi(double x);
double psi1(double x);
double psi_n(double x,int n);
//Synchrotron Functions
double synchrotron1(double x);
double synchrotron2(double x);
//Transport Functions
double transport2(double x);
double transport3(double x);
double transport4(double x);
double transport5(double x);
//Zeta Functions
double zeta(double x);
double zetam1(double x);
double hzeta(double x,double q);
double eta(double x);
//common activation function
double ELU(double x,double alpha=1.0);
double hardshrink(double x,double lambda=0.5);
double hardsigmoid(double x);
double hardtanh(double x,double min_val=-1.0,double max_val=1.0);
double hardswish(double x);
double leakyReLU(double x,double alpha=0.01);
double logsigmoid(double x);
double ReLU(double x);
double ReLU6(double x);
double RReLU(double x,double lower=0.125,double upper=0.333);
double SELU(double x);
double CELU(double x,double alpha);
double GELU(double x);
double GELU_fast(double x);
double sigmoid(double x);
double SiLU(double x);
double mish(double x);
double softplus(double x,double beta=1.0,double threshold=20.0);
double softshrink(double x,double lambda=0.5);
double softsign(double x);
double tanhshrink(double x);
double threshold(double x,double threshold=1.0,double value=0.0);
}
}