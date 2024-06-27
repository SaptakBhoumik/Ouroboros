#include "func/func.hpp"
namespace Ouroboros{
//Airy Functions
Tensor Ai(const Tensor& t,size_t min_count){
    return transform(Scalar::Ai,min_count,t);
}
Tensor Bi(const Tensor& t,size_t min_count){
    return transform(Scalar::Bi,min_count,t);
}
Tensor Ai_scaled(const Tensor& t,size_t min_count){
    return transform(Scalar::Ai_scaled,min_count,t);
}
Tensor Bi_scaled(const Tensor& t,size_t min_count){
    return transform(Scalar::Bi_scaled,min_count,t);
}
Tensor Ai_grad(const Tensor& t,size_t min_count){
    return transform(Scalar::Ai_grad,min_count,t);
}
Tensor Bi_grad(const Tensor& t,size_t min_count){
    return transform(Scalar::Bi_grad,min_count,t);
}
Tensor Ai_grad_scaled(const Tensor& t,size_t min_count){
    return transform(Scalar::Ai_grad_scaled,min_count,t);
}
Tensor Bi_grad_scaled(const Tensor& t,size_t min_count){
    return transform(Scalar::Bi_grad_scaled,min_count,t);
}
Tensor Ai_zero(const Tensor& t,size_t min_count){
    auto f=[](double x){return Scalar::Ai_zero((unsigned int)x);};
    return transform(f,min_count,t);
}
Tensor Bi_zero(const Tensor& t,size_t min_count){
    auto f=[](double x){return Scalar::Bi_zero((unsigned int)x);};
    return transform(f,min_count,t);
}
Tensor Ai_grad_zero(const Tensor& t,size_t min_count){
    auto f=[](double x){return Scalar::Ai_grad_zero((unsigned int)x);};
    return transform(f,min_count,t);
}
Tensor Bi_grad_zero(const Tensor& t,size_t min_count){
    auto f=[](double x){return Scalar::Bi_grad_zero((unsigned int)x);};
    return transform(f,min_count,t);
}
//Bessel Functions
Tensor J0(const Tensor& t,size_t min_count){
    return transform(Scalar::J0,min_count,t);
}
Tensor J1(const Tensor& t,size_t min_count){
    return transform(Scalar::J1,min_count,t);
}
Tensor Jn(const Tensor& x,int n,size_t min_count){
    auto func=[n](double x){return Scalar::Jn(x,n);};
    return transform(func,min_count,x);
}
Tensor Jn(double x,const Tensor& n,size_t min_count){
    auto func=[x](double n){return Scalar::Jn(x,(int)n);};
    return transform(func,min_count,n);
}
Tensor Jn(const Tensor& x,const Tensor& n,size_t min_count){
    auto func=[](double x,double n){return Scalar::Jn(x,(int)n);};
    return transform(func,min_count,x,n);
}
Tensor J0_zero(const Tensor& t,size_t min_count){
    return transform(Scalar::J0_zero,min_count,t);
}
Tensor J1_zero(const Tensor& t,size_t min_count){
    return transform(Scalar::J1_zero,min_count,t);
}

Tensor Y0(const Tensor& t,size_t min_count){
    return transform(Scalar::Y0,min_count,t);
}
Tensor Y1(const Tensor& t,size_t min_count){
    return transform(Scalar::Y1,min_count,t);
}
Tensor Yn(const Tensor& x,int n,size_t min_count){
    auto func=[n](double x){return Scalar::Yn(x,n);};
    return transform(func,min_count,x);
}
Tensor Yn(double x,const Tensor& n,size_t min_count){
    auto func=[x](double n){return Scalar::Yn(x,(int)n);};
    return transform(func,min_count,n);
}
Tensor Yn(const Tensor& x,const Tensor& n,size_t min_count){
    auto func=[](double x,double n){return Scalar::Yn(x,(int)n);};
    return transform(func,min_count,x,n);
}

Tensor I0(const Tensor& t,size_t min_count){
    return transform(Scalar::I0,min_count,t);
}
Tensor I1(const Tensor& t,size_t min_count){
    return transform(Scalar::I1,min_count,t);
}
Tensor In(const Tensor& x,int n,size_t min_count){
    auto func=[n](double x){return Scalar::In(x,n);};
    return transform(func,min_count,x);
}
Tensor In(double x,const Tensor& n,size_t min_count){
    auto func=[x](double n){return Scalar::In(x,(int)n);};
    return transform(func,min_count,n);
}
Tensor In(const Tensor& x,const Tensor& n,size_t min_count){
    auto func=[](double x,double n){return Scalar::In(x,(int)n);};
    return transform(func,min_count,x,n);
}
Tensor I0_scaled(const Tensor& x,size_t min_count){
    return transform(Scalar::I0_scaled,min_count,x);
}
Tensor I1_scaled(const Tensor& x,size_t min_count){
    return transform(Scalar::I1_scaled,min_count,x);
}
Tensor In_scaled(const Tensor& x,int n,size_t min_count){
    auto func=[n](double x){return Scalar::In_scaled(x,n);};
    return transform(func,min_count,x);
}
Tensor In_scaled(double x,const Tensor& n,size_t min_count){
    auto func=[x](double n){return Scalar::In_scaled(x,(int)n);};
    return transform(func,min_count,n);
}
Tensor In_scaled(const Tensor& x,const Tensor& n,size_t min_count){
    auto func=[](double x,double n){return Scalar::In_scaled(x,(int)n);};
    return transform(func,min_count,x,n);
}

Tensor K0(const Tensor& x,size_t min_count){
    return transform(Scalar::K0,min_count,x);
}
Tensor K1(const Tensor& x,size_t min_count){
    return transform(Scalar::K1,min_count,x);
}
Tensor Kn(const Tensor& x,int n,size_t min_count){
    auto func=[n](double x){return Scalar::Kn(x,n);};
    return transform(func,min_count,x);
}
Tensor Kn(double x,const Tensor& n,size_t min_count){
    auto func=[x](double n){return Scalar::Kn(x,(int)n);};
    return transform(func,min_count,n);
}
Tensor Kn(const Tensor& x,const Tensor& n,size_t min_count){
    auto func=[](double x,double n){return Scalar::Kn(x,(int)n);};
    return transform(func,min_count,x,n);
}
Tensor K0_scaled(const Tensor& x,size_t min_count){
    return transform(Scalar::K0_scaled,min_count,x);
}
Tensor K1_scaled(const Tensor& x,size_t min_count){
    return transform(Scalar::K1_scaled,min_count,x);
}
Tensor Kn_scaled(const Tensor& x,int n,size_t min_count){
    auto func=[n](double x){return Scalar::Kn_scaled(x,n);};
    return transform(func,min_count,x);
}
Tensor Kn_scaled(double x,const Tensor& n,size_t min_count){
    auto func=[x](double n){return Scalar::Kn_scaled(x,(int)n);};
    return transform(func,min_count,n);
}
Tensor Kn_scaled(const Tensor& x,const Tensor& n,size_t min_count){
    auto func=[](double x,double n){return Scalar::Kn_scaled(x,(int)n);};
    return transform(func,min_count,x,n);
}

Tensor Ynu(const Tensor& nu,double x,size_t min_count){
    auto func=[x](double nu){return Scalar::Ynu(nu,x);};
    return transform(func,min_count,nu);
}
Tensor Ynu(double nu,const Tensor& x,size_t min_count){
    auto func=[nu](double x){return Scalar::Ynu(nu,x);};
    return transform(func,min_count,x);
}
Tensor Ynu(const Tensor& nu,const Tensor& x,size_t min_count){
    auto func=[](double nu,double x){return Scalar::Ynu(nu,x);};
    return transform(func,min_count,nu,x);
}

Tensor Jnu(const Tensor& nu,double x,size_t min_count){
    auto func=[x](double nu){return Scalar::Jnu(nu,x);};
    return transform(func,min_count,nu);
}
Tensor Jnu(double nu,const Tensor& x,size_t min_count){
    auto func=[nu](double x){return Scalar::Jnu(nu,x);};
    return transform(func,min_count,x);
}
Tensor Jnu(const Tensor& nu,const Tensor& x,size_t min_count){
    auto func=[](double nu,double x){return Scalar::Jnu(nu,x);};
    return transform(func,min_count,nu,x);
}

Tensor Jnu_zero(const Tensor& nu,unsigned int n,size_t min_count){
    auto func=[n](double nu){return Scalar::Jnu_zero(nu,n);};
    return transform(func,min_count,nu);
}
Tensor Jnu_zero(double nu,const Tensor& n,size_t min_count){
    auto func=[nu](double n){return Scalar::Jnu_zero(nu,(unsigned int)n);};
    return transform(func,min_count,n);
}
Tensor Jnu_zero(const Tensor& nu,const Tensor& n,size_t min_count){
    auto func=[](double nu,double n){return Scalar::Jnu_zero(nu,(unsigned int)n);};
    return transform(func,min_count,nu,n);
}

Tensor Inu(const Tensor& nu,double x,size_t min_count){
    auto func=[x](double nu){return Scalar::Inu(nu,x);};
    return transform(func,min_count,nu);
}
Tensor Inu(double nu,const Tensor& x,size_t min_count){
    auto func=[nu](double x){return Scalar::Inu(nu,x);};
    return transform(func,min_count,x);
}
Tensor Inu(const Tensor& nu,const Tensor& x,size_t min_count){
    auto func=[](double nu,double x){return Scalar::Inu(nu,x);};
    return transform(func,min_count,nu,x);
}

Tensor Knu(const Tensor& nu,double x,size_t min_count){
    auto func=[x](double nu){return Scalar::Knu(nu,x);};
    return transform(func,min_count,nu);
}
Tensor Knu(double nu,const Tensor& x,size_t min_count){
    auto func=[nu](double x){return Scalar::Knu(nu,x);};
    return transform(func,min_count,x);
}
Tensor Knu(const Tensor& nu,const Tensor& x,size_t min_count){
    auto func=[](double nu,double x){return Scalar::Knu(nu,x);};
    return transform(func,min_count,nu,x);
}

Tensor Knu_scaled(const Tensor& nu,double x,size_t min_count){
    auto func=[x](double nu){return Scalar::Knu_scaled(nu,x);};
    return transform(func,min_count,nu);
}
Tensor Knu_scaled(double nu,const Tensor& x,size_t min_count){
    auto func=[nu](double x){return Scalar::Knu_scaled(nu,x);};
    return transform(func,min_count,x);
}
Tensor Knu_scaled(const Tensor& nu,const Tensor& x,size_t min_count){
    auto func=[](double nu,double x){return Scalar::Knu_scaled(nu,x);};
    return transform(func,min_count,nu,x);
}

Tensor lKnu(const Tensor& nu,double x,size_t min_count){
    auto func=[x](double nu){return Scalar::lKnu(nu,x);};
    return transform(func,min_count,nu);
}
Tensor lKnu(double nu,const Tensor& x,size_t min_count){
    auto func=[nu](double x){return Scalar::lKnu(nu,x);};
    return transform(func,min_count,x);
}
Tensor lKnu(const Tensor& nu,const Tensor& x,size_t min_count){
    auto func=[](double nu,double x){return Scalar::lKnu(nu,x);};
    return transform(func,min_count,nu,x);
}
//Clausen Functions
Tensor clausen(const Tensor& t,size_t min_count){
    return transform(Scalar::clausen,min_count,t);
}
//Dawson Function
Tensor dawson(const Tensor& t,size_t min_count){
    return transform(Scalar::dawson,min_count,t);
}
//Debye Functions
Tensor debye1(const Tensor& t,size_t min_count){
    return transform(Scalar::debye1,min_count,t);
}
Tensor debye2(const Tensor& t,size_t min_count){
    return transform(Scalar::debye2,min_count,t);
}
Tensor debye3(const Tensor& t,size_t min_count){
    return transform(Scalar::debye3,min_count,t);
}
Tensor debye4(const Tensor& t,size_t min_count){
    return transform(Scalar::debye4,min_count,t);
}
Tensor debye5(const Tensor& t,size_t min_count){
    return transform(Scalar::debye5,min_count,t);
}
Tensor debye6(const Tensor& t,size_t min_count){
    return transform(Scalar::debye6,min_count,t);
}
//Dilogarithm Function
Tensor dilog(const Tensor& t,size_t min_count){
    return transform(Scalar::dilog,min_count,t);
}
//Elliptic Integrals
Tensor Kcomp(const Tensor& t,size_t min_count){
    return transform(Scalar::Kcomp,min_count,t);
}
Tensor Ecomp(const Tensor& t,size_t min_count){
    return transform(Scalar::Ecomp,min_count,t);
}

Tensor Pcomp(const Tensor& k,double n,size_t min_count){
    auto func=[n](double k){return Scalar::Pcomp(k,n);};
    return transform(func,min_count,k);
}
Tensor Pcomp(double k,const Tensor& n,size_t min_count){
    auto func=[k](double n){return Scalar::Pcomp(k,n);};
    return transform(func,min_count,n);
}
Tensor Pcomp(const Tensor& k,const Tensor& n,size_t min_count){
    auto func=[](double k,double n){return Scalar::Pcomp(k,n);};
    return transform(func,min_count,k,n);
}

Tensor F(const Tensor& phi,double k,size_t min_count){
    auto func=[k](double phi){return Scalar::F(phi,k);};
    return transform(func,min_count,phi);
}
Tensor F(double phi,const Tensor& k,size_t min_count){
    auto func=[phi](double k){return Scalar::F(phi,k);};
    return transform(func,min_count,k);
}
Tensor F(const Tensor& phi,const Tensor& k,size_t min_count){
    auto func=[](double phi,double k){return Scalar::F(phi,k);};
    return transform(func,min_count,phi,k);
}

Tensor E(const Tensor& phi,double k,size_t min_count){
    auto func=[k](double phi){return Scalar::E(phi,k);};
    return transform(func,min_count,phi);
}
Tensor E(double phi,const Tensor& k,size_t min_count){
    auto func=[phi](double k){return Scalar::E(phi,k);};
    return transform(func,min_count,k);
}
Tensor E(const Tensor& phi,const Tensor& k,size_t min_count){
    auto func=[](double phi,double k){return Scalar::E(phi,k);};
    return transform(func,min_count,phi,k);

}

Tensor P(const Tensor& phi,double k,double n,size_t min_count){
    auto func=[k,n](double phi){return Scalar::P(phi,k,n);};
    return transform(func,min_count,phi);
}
Tensor P(double phi,const Tensor& k,double n,size_t min_count){
    auto func=[phi,n](double k){return Scalar::P(phi,k,n);};
    return transform(func,min_count,k);
}
Tensor P(double phi,double k,const Tensor& n,size_t min_count){
    auto func=[phi,k](double n){return Scalar::P(phi,k,n);};
    return transform(func,min_count,n);
}
Tensor P(const Tensor& phi,const Tensor& k,double n,size_t min_count){
    auto func=[n](double phi,double k){return Scalar::P(phi,k,n);};
    return transform(func,min_count,phi,k);
}
Tensor P(double phi,const Tensor& k,const Tensor& n,size_t min_count){
    auto func=[phi](double k,double n){return Scalar::P(phi,k,n);};
    return transform(func,min_count,k,n);
}
Tensor P(const Tensor& phi,double k,const Tensor& n,size_t min_count){
    auto func=[k](double phi,double n){return Scalar::P(phi,k,n);};
    return transform(func,min_count,phi,n);
}
Tensor P(const Tensor& phi,const Tensor& k,const Tensor& n,size_t min_count){
    auto func=[](double phi,double k,double n){return Scalar::P(phi,k,n);};
    return transform(func,min_count,phi,k,n);
}

Tensor D(const Tensor& phi,double k,size_t min_count){
    auto func=[k](double phi){return Scalar::D(phi,k);};
    return transform(func,min_count,phi);
}
Tensor D(double phi,const Tensor& k,size_t min_count){
    auto func=[phi](double k){return Scalar::D(phi,k);};
    return transform(func,min_count,k);
}
Tensor D(const Tensor& phi,const Tensor& k,size_t min_count){
    auto func=[](double phi,double k){return Scalar::D(phi,k);};
    return transform(func,min_count,phi,k);
}

Tensor RC(const Tensor& x,double y,size_t min_count){
    auto func=[y](double x){return Scalar::RC(x,y);};
    return transform(func,min_count,x);
}
Tensor RC(double x,const Tensor& y,size_t min_count){
    auto func=[x](double y){return Scalar::RC(x,y);};
    return transform(func,min_count,y);
}
Tensor RC(const Tensor& x,const Tensor& y,size_t min_count){
    auto func=[](double x,double y){return Scalar::RC(x,y);};
    return transform(func,min_count,x,y);
}

Tensor RD(const Tensor& x,double y,double z,size_t min_count){
    auto func=[y,z](double x){return Scalar::RD(x,y,z);};
    return transform(func,min_count,x);
}
Tensor RD(double x,const Tensor& y,double z,size_t min_count){
    auto func=[x,z](double y){return Scalar::RD(x,y,z);};
    return transform(func,min_count,y);
}
Tensor RD(double x,double y,const Tensor& z,size_t min_count){
    auto func=[x,y](double z){return Scalar::RD(x,y,z);};
    return transform(func,min_count,z);
}
Tensor RD(const Tensor& x,const Tensor& y,double z,size_t min_count){
    auto func=[z](double x,double y){return Scalar::RD(x,y,z);};
    return transform(func,min_count,x,y);
}
Tensor RD(const Tensor& x,double y,const Tensor& z,size_t min_count){
    auto func=[y](double x,double z){return Scalar::RD(x,y,z);};
    return transform(func,min_count,x,z);
}
Tensor RD(double x,const Tensor& y,const Tensor& z,size_t min_count){
    auto func=[x](double y,double z){return Scalar::RD(x,y,z);};
    return transform(func,min_count,y,z);
}
Tensor RD(const Tensor& x,const Tensor& y,const Tensor& z,size_t min_count){
    auto func=[](double x,double y,double z){return Scalar::RD(x,y,z);};
    return transform(func,min_count,x,y,z);
}

Tensor RF(const Tensor& x,double y,double z,size_t min_count){
    auto func=[y,z](double x){return Scalar::RF(x,y,z);};
    return transform(func,min_count,x);
}
Tensor RF(double x,const Tensor& y,double z,size_t min_count){
    auto func=[x,z](double y){return Scalar::RF(x,y,z);};
    return transform(func,min_count,y);
}
Tensor RF(double x,double y,const Tensor& z,size_t min_count){
    auto func=[x,y](double z){return Scalar::RF(x,y,z);};
    return transform(func,min_count,z);
}
Tensor RF(const Tensor& x,const Tensor& y,double z,size_t min_count){
    auto func=[z](double x,double y){return Scalar::RF(x,y,z);};
    return transform(func,min_count,x,y);
}
Tensor RF(const Tensor& x,double y,const Tensor& z,size_t min_count){
    auto func=[y](double x,double z){return Scalar::RF(x,y,z);};
    return transform(func,min_count,x,z);
}
Tensor RF(double x,const Tensor& y,const Tensor& z,size_t min_count){
    auto func=[x](double y,double z){return Scalar::RF(x,y,z);};
    return transform(func,min_count,y,z);
}
Tensor RF(const Tensor& x,const Tensor& y,const Tensor& z,size_t min_count){
    auto func=[](double x,double y,double z){return Scalar::RF(x,y,z);};
    return transform(func,min_count,x,y,z);
}

Tensor RJ(const Tensor& x,double y,double z,double p,size_t min_count){
    auto func=[y,z,p](double x){return Scalar::RJ(x,y,z,p);};
    return transform(func,min_count,x);
}
Tensor RJ(double x,const Tensor& y,double z,double p,size_t min_count){
    auto func=[x,z,p](double y){return Scalar::RJ(x,y,z,p);};
    return transform(func,min_count,y);
}
Tensor RJ(double x,double y,const Tensor& z,double p,size_t min_count){
    auto func=[x,y,p](double z){return Scalar::RJ(x,y,z,p);};
    return transform(func,min_count,z);
}
Tensor RJ(double x,double y,double z,const Tensor& p,size_t min_count){
    auto func=[x,y,z](double p){return Scalar::RJ(x,y,z,p);};
    return transform(func,min_count,p);
}
Tensor RJ(const Tensor& x,const Tensor& y,double z,double p,size_t min_count){
    auto func=[z,p](double x,double y){return Scalar::RJ(x,y,z,p);};
    return transform(func,min_count,x,y);
}
Tensor RJ(const Tensor& x,double y,const Tensor& z,double p,size_t min_count){
    auto func=[y,p](double x,double z){return Scalar::RJ(x,y,z,p);};
    return transform(func,min_count,x,z);
}
Tensor RJ(const Tensor& x,double y,double z,const Tensor& p,size_t min_count){
    auto func=[y,z](double x,double p){return Scalar::RJ(x,y,z,p);};
    return transform(func,min_count,x,p);
}
Tensor RJ(double x,const Tensor& y,const Tensor& z,double p,size_t min_count){
    auto func=[x,p](double y,double z){return Scalar::RJ(x,y,z,p);};
    return transform(func,min_count,y,z);
}
Tensor RJ(double x,const Tensor& y,double z,const Tensor& p,size_t min_count){
    auto func=[x,z](double y,double p){return Scalar::RJ(x,y,z,p);};
    return transform(func,min_count,y,p);
}
Tensor RJ(double x,double y,const Tensor& z,const Tensor& p,size_t min_count){
    auto func=[x,y](double z,double p){return Scalar::RJ(x,y,z,p);};
    return transform(func,min_count,z,p);
}
Tensor RJ(const Tensor& x,const Tensor& y,const Tensor& z,double p,size_t min_count){
    auto func=[p](double x,double y,double z){return Scalar::RJ(x,y,z,p);};
    return transform(func,min_count,x,y,z);
}
Tensor RJ(const Tensor& x,const Tensor& y,double z,const Tensor& p,size_t min_count){
    auto func=[z](double x,double y,double p){return Scalar::RJ(x,y,z,p);};
    return transform(func,min_count,x,y,p);
}
Tensor RJ(const Tensor& x,double y,const Tensor& z,const Tensor& p,size_t min_count){
    auto func=[y](double x,double z,double p){return Scalar::RJ(x,y,z,p);};
    return transform(func,min_count,x,z,p);
}
Tensor RJ(double x,const Tensor& y,const Tensor& z,const Tensor& p,size_t min_count){
    auto func=[x](double y,double z,double p){return Scalar::RJ(x,y,z,p);};
    return transform(func,min_count,y,z,p);
}
Tensor RJ(const Tensor& x,const Tensor& y,const Tensor& z,const Tensor& p,size_t min_count){
    return transform(Scalar::RJ,min_count,x,y,z,p);
}
Tensor E1(const Tensor& t,size_t min_count){
    return transform(Scalar::E1,min_count,t);
}
Tensor E2(const Tensor& t,size_t min_count){
    return transform(Scalar::E2,min_count,t);
}
Tensor En(const Tensor& x,int n,size_t min_count){
    auto func=[n](double x){return Scalar::En(x,(int)n);};
    return transform(func,min_count,x);
}
Tensor En(double x,const Tensor& n,size_t min_count){
    auto func=[x](double n){return Scalar::En(x,(int)n);};
    return transform(func,min_count,n);
}
Tensor En(const Tensor& x,const Tensor& n,size_t min_count){
    auto func=[](double x,double n){return Scalar::En(x,(int)n);};
    return transform(func,min_count,x,n);
}
Tensor Ei(const Tensor& t,size_t min_count){
    return transform(Scalar::Ei,min_count,t);
}
Tensor Shi(const Tensor& t,size_t min_count){
    return transform(Scalar::Shi,min_count,t);
}
Tensor Chi(const Tensor& t,size_t min_count){
    return transform(Scalar::Chi,min_count,t);
}
Tensor Ei_3(const Tensor& t,size_t min_count){
    return transform(Scalar::Ei_3,min_count,t);
}
Tensor Si(const Tensor& t,size_t min_count){
    return transform(Scalar::Si,min_count,t);
}
Tensor Ci(const Tensor& t,size_t min_count){
    return transform(Scalar::Ci,min_count,t);
}
Tensor atan_int(const Tensor& t,size_t min_count){
    return transform(Scalar::atan_int,min_count,t);
}
//Fermi-Dirac Functions
Tensor fermi_dirac_m1(const Tensor& t,size_t min_count){
    return transform(Scalar::fermi_dirac_m1,min_count,t);
}
Tensor fermi_dirac_0(const Tensor& t,size_t min_count){
    return transform(Scalar::fermi_dirac_0,min_count,t);
}
Tensor fermi_dirac_1(const Tensor& t,size_t min_count){
    return transform(Scalar::fermi_dirac_1,min_count,t);
}
Tensor fermi_dirac(const Tensor& x,int j,size_t min_count){
    auto func=[j](double x){return Scalar::fermi_dirac(x,j);};
    return transform(func,min_count,x);
}
Tensor fermi_dirac(double x,const Tensor& j,size_t min_count){
    auto func=[x](double j){return Scalar::fermi_dirac(x,(int)j);};
    return transform(func,min_count,j);
}
Tensor fermi_dirac(const Tensor& x,const Tensor& j,size_t min_count){
    auto func=[](double x,double j){return Scalar::fermi_dirac(x,(int)j);};
    return transform(func,min_count,x,j);
}
Tensor fermi_dirac_half(const Tensor& t,size_t min_count){
    return transform(Scalar::fermi_dirac_half,min_count,t);
}
Tensor fermi_dirac_mhalf(const Tensor& t,size_t min_count){
    return transform(Scalar::fermi_dirac_mhalf,min_count,t);
}
Tensor fermi_dirac_3half(const Tensor& t,size_t min_count){
    return transform(Scalar::fermi_dirac_3half,min_count,t);
}
Tensor fermi_dirac_inc_0(const Tensor& x,double b,size_t min_count){
    auto func=[b](double x){return Scalar::fermi_dirac_inc_0(x,b);};
    return transform(func,min_count,x);
}
Tensor fermi_dirac_inc_0(double x,const Tensor& b,size_t min_count){
    auto func=[x](double b){return Scalar::fermi_dirac_inc_0(x,b);};
    return transform(func,min_count,b);
}
Tensor fermi_dirac_inc_0(const Tensor& x,const Tensor& b,size_t min_count){
    auto func=[](double x,double b){return Scalar::fermi_dirac_inc_0(x,b);};
    return transform(func,min_count,x,b);
}
//Gamma and Related Functions
Tensor gamma(const Tensor& t,size_t min_count){
    return transform(Scalar::gamma,min_count,t);
}
Tensor lgamma(const Tensor& t,size_t min_count){
    return transform(Scalar::lgamma,min_count,t);
}
Tensor gamma_star(const Tensor& t,size_t min_count){
    return transform(Scalar::gamma_star,min_count,t);
}
Tensor gamma_inv(const Tensor& t,size_t min_count){
    return transform(Scalar::gamma_inv,min_count,t);
}
Tensor taylorcoeff(const Tensor& x,int n,size_t min_count){
    auto func=[n](double x){return Scalar::taylorcoeff(x,n);};
    return transform(func,min_count,x);
}
Tensor taylorcoeff(double x,const Tensor& n,size_t min_count){
    auto func=[x](double n){return Scalar::taylorcoeff(x,(int)n);};
    return transform(func,min_count,n);
}
Tensor taylorcoeff(const Tensor& x,const Tensor& n,size_t min_count){
    auto func=[](double x,double n){return Scalar::taylorcoeff(x,(int)n);};
    return transform(func,min_count,x,n);
}
Tensor poch(const Tensor& a, double x,size_t min_count){
    auto func=[x](double a){return Scalar::poch(a,x);};
    return transform(func,min_count,a);
}
Tensor poch(double a, const Tensor& x,size_t min_count){
    auto func=[a](double x){return Scalar::poch(a,x);};
    return transform(func,min_count,x);
}
Tensor poch(const Tensor& a, const Tensor& x,size_t min_count){
    return transform(Scalar::poch,min_count,a,x);
}
Tensor lpoch(const Tensor& a, double x,size_t min_count){
    auto func=[x](double a){return Scalar::lpoch(a,x);};
    return transform(func,min_count,a);
}
Tensor lpoch(double a, const Tensor& x,size_t min_count){
    auto func=[a](double x){return Scalar::lpoch(a,x);};
    return transform(func,min_count,x);
}
Tensor lpoch(const Tensor& a, const Tensor& x,size_t min_count){
    return transform(Scalar::lpoch,min_count,a,x);
}
Tensor pochrel(const Tensor& a, double x,size_t min_count){
    auto func=[x](double a){return Scalar::pochrel(a,x);};
    return transform(func,min_count,a);
}
Tensor pochrel(double a, const Tensor& x,size_t min_count){
    auto func=[a](double x){return Scalar::pochrel(a,x);};
    return transform(func,min_count,x);
}
Tensor pochrel(const Tensor& a, const Tensor& x,size_t min_count){
    return transform(Scalar::pochrel,min_count,a,x);
}
Tensor gamma_inc(const Tensor& a, double x,size_t min_count){
    auto func=[x](double a){return Scalar::gamma_inc(a,x);};
    return transform(func,min_count,a);
}
Tensor gamma_inc(double a, const Tensor& x,size_t min_count){
    auto func=[a](double x){return Scalar::gamma_inc(a,x);};
    return transform(func,min_count,x);
}
Tensor gamma_inc(const Tensor& a, const Tensor& x,size_t min_count){
    return transform(Scalar::gamma_inc,min_count,a,x);
}
Tensor gamma_inc_Q(const Tensor& a, double x,size_t min_count){
    auto func=[x](double a){return Scalar::gamma_inc_Q(a,x);};
    return transform(func,min_count,a);
}
Tensor gamma_inc_Q(double a, const Tensor& x,size_t min_count){
    auto func=[a](double x){return Scalar::gamma_inc_Q(a,x);};
    return transform(func,min_count,x);
}
Tensor gamma_inc_Q(const Tensor& a, const Tensor& x,size_t min_count){
    return transform(Scalar::gamma_inc_Q,min_count,a,x);
}
Tensor gamma_inc_P(const Tensor& a, double x,size_t min_count){
    auto func=[x](double a){return Scalar::gamma_inc_P(a,x);};
    return transform(func,min_count,a);
}
Tensor gamma_inc_P(double a, const Tensor& x,size_t min_count){
    auto func=[a](double x){return Scalar::gamma_inc_P(a,x);};
    return transform(func,min_count,x);
}
Tensor gamma_inc_P(const Tensor& a, const Tensor& x,size_t min_count){
    return transform(Scalar::gamma_inc_P,min_count,a,x);
}
Tensor beta(const Tensor& a, double x,size_t min_count){
    auto func=[x](double a){return Scalar::beta(a,x);};
    return transform(func,min_count,a);
}
Tensor beta(double a, const Tensor& x,size_t min_count){
    auto func=[a](double x){return Scalar::beta(a,x);};
    return transform(func,min_count,x);
}
Tensor beta(const Tensor& a, const Tensor& x,size_t min_count){
    return transform(Scalar::beta,min_count,a,x);
}
Tensor lbeta(const Tensor& a, double x,size_t min_count){
    auto func=[x](double a){return Scalar::lbeta(a,x);};
    return transform(func,min_count,a);
}
Tensor lbeta(double a, const Tensor& x,size_t min_count){
    auto func=[a](double x){return Scalar::lbeta(a,x);};
    return transform(func,min_count,x);
}
Tensor lbeta(const Tensor& a, const Tensor& x,size_t min_count){
    return transform(Scalar::lbeta,min_count,a,x);
}
Tensor beta_inc(const Tensor& a,double b,double x,size_t min_count){
    auto func=[b,x](double a){return Scalar::beta_inc(a,b,x);};
    return transform(func,min_count,a);
}
Tensor beta_inc(double a,const Tensor& b,double x,size_t min_count){
    auto func=[a,x](double b){return Scalar::beta_inc(a,b,x);};
    return transform(func,min_count,b);
}
Tensor beta_inc(double a,double b,const Tensor& x,size_t min_count){
    auto func=[a,b](double x){return Scalar::beta_inc(a,b,x);};
    return transform(func,min_count,x);
}
Tensor beta_inc(const Tensor& a,const Tensor& b,double x,size_t min_count){
    auto func=[x](double a,double b){return Scalar::beta_inc(a,b,x);};
    return transform(func,min_count,a,b);
}
Tensor beta_inc(const Tensor& a,double b,const Tensor& x,size_t min_count){
    auto func=[b](double a,double x){return Scalar::beta_inc(a,b,x);};
    return transform(func,min_count,a,x);
}
Tensor beta_inc(double a,const Tensor& b,const Tensor& x,size_t min_count){
    auto func=[a](double b,double x){return Scalar::beta_inc(a,b,x);};
    return transform(func,min_count,b,x);

}
Tensor beta_inc(const Tensor& a,const Tensor& b,const Tensor& x,size_t min_count){
    return transform(Scalar::beta_inc,min_count,a,b,x);
}
//Gegenbauer Functions
Tensor gegenpoly1(const Tensor& lambda,double x,size_t min_count){
    auto func=[x](double lambda){return Scalar::gegenpoly1(lambda,x);};
    return transform(func,min_count,lambda);
}
Tensor gegenpoly1(double lambda,const Tensor& x,size_t min_count){
    auto func=[lambda](double x){return Scalar::gegenpoly1(lambda,x);};
    return transform(func,min_count,x);
}
Tensor gegenpoly1(const Tensor& lambda,const Tensor& x,size_t min_count){
    return transform(Scalar::gegenpoly1,min_count,lambda,x);
}
Tensor gegenpoly2(const Tensor& lambda,double x,size_t min_count){
    auto func=[x](double lambda){return Scalar::gegenpoly2(lambda,x);};
    return transform(func,min_count,lambda);
}
Tensor gegenpoly2(double lambda,const Tensor& x,size_t min_count){
    auto func=[lambda](double x){return Scalar::gegenpoly2(lambda,x);};
    return transform(func,min_count,x);
}
Tensor gegenpoly2(const Tensor& lambda,const Tensor& x,size_t min_count){
    return transform(Scalar::gegenpoly2,min_count,lambda,x);
}
Tensor gegenpoly3(const Tensor& lambda,double x,size_t min_count){
    auto func=[x](double lambda){return Scalar::gegenpoly3(lambda,x);};
    return transform(func,min_count,lambda);
}
Tensor gegenpoly3(double lambda,const Tensor& x,size_t min_count){
    auto func=[lambda](double x){return Scalar::gegenpoly3(lambda,x);};
    return transform(func,min_count,x);
}
Tensor gegenpoly3(const Tensor& lambda,const Tensor& x,size_t min_count){
    return transform(Scalar::gegenpoly3,min_count,lambda,x);
}
Tensor gegenpoly(const Tensor& lambda,double x,int n,size_t min_count){
    auto func=[x,n](double lambda){return Scalar::gegenpoly(lambda,x,(int)n);};
    return transform(func,min_count,lambda);
}
Tensor gegenpoly(double lambda,const Tensor& x,int n,size_t min_count){
    auto func=[lambda,n](double x){return Scalar::gegenpoly(lambda,x,(int)n);};
    return transform(func,min_count,x);
}
Tensor gegenpoly(double lambda,double x,const Tensor& n,size_t min_count){
    auto func=[lambda,x](double n){return Scalar::gegenpoly(lambda,x,(int)n);};
    return transform(func,min_count,n);
}
Tensor gegenpoly(const Tensor& lambda,const Tensor& x,int n,size_t min_count){
    auto func=[n](double lambda,double x){return Scalar::gegenpoly(lambda,x,(int)n);};
    return transform(func,min_count,lambda,x);
}
Tensor gegenpoly(double lambda,const Tensor& x,const Tensor& n,size_t min_count){
    auto func=[lambda](double x,double n){return Scalar::gegenpoly(lambda,x,(int)n);};
    return transform(func,min_count,x,n);
}
Tensor gegenpoly(const Tensor& lambda,double x,const Tensor& n,size_t min_count){
    auto func=[x](double lambda,double n){return Scalar::gegenpoly(lambda,x,(int)n);};
    return transform(func,min_count,lambda,n);
}
Tensor gegenpoly(const Tensor& lambda,const Tensor& x,const Tensor& n,size_t min_count){
    auto func=[](double lambda,double x,double n){return Scalar::gegenpoly(lambda,x,(int)n);};
    return transform(func,min_count,lambda,x,n);
}
//Hermite Functions
Tensor hermite(const Tensor& x,int n,size_t min_count){
    auto func=[n](double x){return Scalar::hermite(x,n);};
    return transform(func,min_count,x);
}
Tensor hermite(double x,const Tensor& n,size_t min_count){
    auto func=[x](double n){return Scalar::hermite(x,(int)n);};
    return transform(func,min_count,n);
}
Tensor hermite(const Tensor& x,const Tensor& n,size_t min_count){
    auto func=[](double x,double n){return Scalar::hermite(x,(int)n);};
    return transform(func,min_count,x,n);
}
Tensor hermite_prob(const Tensor& x,int n,size_t min_count){
    auto func=[n](double x){return Scalar::hermite_prob(x,n);};
    return transform(func,min_count,x);
}
Tensor hermite_prob(double x,const Tensor& n,size_t min_count){
    auto func=[x](double n){return Scalar::hermite_prob(x,(int)n);};
    return transform(func,min_count,n);
}
Tensor hermite_prob(const Tensor& x,const Tensor& n,size_t min_count){
    auto func=[](double x,double n){return Scalar::hermite_prob(x,(int)n);};
    return transform(func,min_count,x,n);
}
Tensor hermite_grad(const Tensor& x,int n,int order,size_t min_count){
    auto func=[n,order](double x){return Scalar::hermite_grad(x,n,order);};
    return transform(func,min_count,x);
}
Tensor hermite_grad(double x,const Tensor& n,int order,size_t min_count){
    auto func=[x,order](double n){return Scalar::hermite_grad(x,(int)n,order);};
    return transform(func,min_count,n);
}
Tensor hermite_grad(double x,int n,const Tensor& order,size_t min_count){
    auto func=[x,n](double order){return Scalar::hermite_grad(x,n,(int)order);};
    return transform(func,min_count,order);
}
Tensor hermite_grad(const Tensor& x,const Tensor& n,int order,size_t min_count){
    auto func=[order](double x,double n){return Scalar::hermite_grad(x,(int)n,order);};
    return transform(func,min_count,x,n);
}
Tensor hermite_grad(const Tensor& x,int n,const Tensor& order,size_t min_count){
    auto func=[n](double x,double order){return Scalar::hermite_grad(x,n,(int)order);};
    return transform(func,min_count,x,order);
}
Tensor hermite_grad(double x,const Tensor& n,const Tensor& order,size_t min_count){
    auto func=[x](double n,double order){return Scalar::hermite_grad(x,(int)n,(int)order);};
    return transform(func,min_count,n,order);
}
Tensor hermite_grad(const Tensor& x,const Tensor& n,const Tensor& order,size_t min_count){
    auto func=[](double x,double n,double order){return Scalar::hermite_grad(x,(int)n,(int)order);};
    return transform(func,min_count,x,n,order);
}
Tensor hermite_prob_grad(const Tensor& x,int n,int order,size_t min_count){
    auto func=[n,order](double x){return Scalar::hermite_prob_grad(x,n,order);};
    return transform(func,min_count,x);
}
Tensor hermite_prob_grad(double x,const Tensor& n,int order,size_t min_count){
    auto func=[x,order](double n){return Scalar::hermite_prob_grad(x,(int)n,order);};
    return transform(func,min_count,n);
}
Tensor hermite_prob_grad(double x,int n,const Tensor& order,size_t min_count){
    auto func=[x,n](double order){return Scalar::hermite_prob_grad(x,n,(int)order);};
    return transform(func,min_count,order);
}
Tensor hermite_prob_grad(const Tensor& x,const Tensor& n,int order,size_t min_count){
    auto func=[order](double x,double n){return Scalar::hermite_prob_grad(x,(int)n,order);};
    return transform(func,min_count,x,n);
}
Tensor hermite_prob_grad(const Tensor& x,int n,const Tensor& order,size_t min_count){
    auto func=[n](double x,double order){return Scalar::hermite_prob_grad(x,n,(int)order);};
    return transform(func,min_count,x,order);
}
Tensor hermite_prob_grad(double x,const Tensor& n,const Tensor& order,size_t min_count){
    auto func=[x](double n,double order){return Scalar::hermite_prob_grad(x,(int)n,(int)order);};
    return transform(func,min_count,n,order);
}
Tensor hermite_prob_grad(const Tensor& x,const Tensor& n,const Tensor& order,size_t min_count){
    auto func=[](double x,double n,double order){return Scalar::hermite_prob_grad(x,(int)n,(int)order);};
    return transform(func,min_count,x,n,order);
}
Tensor hermite_func(const Tensor& x,int n,size_t min_count){
    auto func=[n](double x){return Scalar::hermite_func(x,n);};
    return transform(func,min_count,x);
}
Tensor hermite_func(double x,const Tensor& n,size_t min_count){
    auto func=[x](double n){return Scalar::hermite_func(x,(int)n);};
    return transform(func,min_count,n);
}
Tensor hermite_func(const Tensor& x,const Tensor& n,size_t min_count){
    auto func=[](double x,double n){return Scalar::hermite_func(x,(int)n);};
    return transform(func,min_count,x,n);
}
Tensor hermite_func_fast(const Tensor& x,int n,size_t min_count){
    auto func=[n](double x){return Scalar::hermite_func_fast(x,n);};
    return transform(func,min_count,x);
}
Tensor hermite_func_fast(double x,const Tensor& n,size_t min_count){
    auto func=[x](double n){return Scalar::hermite_func_fast(x,(int)n);};
    return transform(func,min_count,n);
}
Tensor hermite_func_fast(const Tensor& x,const Tensor& n,size_t min_count){
    auto func=[](double x,double n){return Scalar::hermite_func_fast(x,(int)n);};
    return transform(func,min_count,x,n);
}
Tensor hermite_func_grad(const Tensor& x,int n,int order,size_t min_count){
    auto func=[n,order](double x){return Scalar::hermite_func_grad(x,n,order);};
    return transform(func,min_count,x);
}
Tensor hermite_func_grad(double x,const Tensor& n,int order,size_t min_count){
    auto func=[x,order](double n){return Scalar::hermite_func_grad(x,(int)n,order);};
    return transform(func,min_count,n);
}
Tensor hermite_func_grad(double x,int n,const Tensor& order,size_t min_count){
    auto func=[x,n](double order){return Scalar::hermite_func_grad(x,n,(int)order);};
    return transform(func,min_count,order);
}
Tensor hermite_func_grad(const Tensor& x,const Tensor& n,int order,size_t min_count){
    auto func=[order](double x,double n){return Scalar::hermite_func_grad(x,n,order);};
    return transform(func,min_count,x,n);
}
Tensor hermite_func_grad(const Tensor& x,int n,const Tensor& order,size_t min_count){
    auto func=[n](double x,double order){return Scalar::hermite_func_grad(x,n,(int)order);};
    return transform(func,min_count,x,order);
}
Tensor hermite_func_grad(double x,const Tensor& n,const Tensor& order,size_t min_count){
    auto func=[x](double n,double order){return Scalar::hermite_func_grad(x,(int)n,(int)order);};
    return transform(func,min_count,n,order);
}
Tensor hermite_func_grad(const Tensor& x,const Tensor& n,const Tensor& order,size_t min_count){
    auto func=[](double x,double n,double order){return Scalar::hermite_func_grad(x,(int)n,(int)order);};
    return transform(func,min_count,x,n,order);
}
Tensor hermite_zero(const Tensor& n,int s,size_t min_count){
    auto func=[s](double n){return Scalar::hermite_zero(n,s);};
    return transform(func,min_count,n);
}
Tensor hermite_zero(int n,const Tensor& s,size_t min_count){
    auto func=[n](double s){return Scalar::hermite_zero(n,(int)s);};
    return transform(func,min_count,s);
}
Tensor hermite_zero(const Tensor& n,const Tensor& s,size_t min_count){
    auto func=[](double n,double s){return Scalar::hermite_zero(n,(int)s);};
    return transform(func,min_count,n,s);
}
Tensor hermite_prob_zero(const Tensor& n,int s,size_t min_count){
    auto func=[s](double n){return Scalar::hermite_prob_zero((int)n,s);};
    return transform(func,min_count,n);
}
Tensor hermite_prob_zero(int n,const Tensor& s,size_t min_count){
    auto func=[n](double s){return Scalar::hermite_prob_zero(n,(int)s);};
    return transform(func,min_count,s);
}
Tensor hermite_prob_zero(const Tensor& n,const Tensor& s,size_t min_count){
    auto func=[](double n,double s){return Scalar::hermite_prob_zero((int)n,(int)s);};
    return transform(func,min_count,n,s);
}
Tensor hermite_func_zero(const Tensor& n,int s,size_t min_count){
    auto func=[s](double n){return Scalar::hermite_func_zero((int)n,s);};
    return transform(func,min_count,n);
}
Tensor hermite_func_zero(int n,const Tensor& s,size_t min_count){
    auto func=[n](double s){return Scalar::hermite_func_zero(n,(int)s);};
    return transform(func,min_count,s);
}
Tensor hermite_func_zero(const Tensor& n,const Tensor& s,size_t min_count){
    auto func=[](double n,double s){return Scalar::hermite_func_zero((int)n,(int)s);};
    return transform(func,min_count,n,s);
}
//Hypergeometric Functions
Tensor F01(const Tensor& a,double b,size_t min_count){
    auto func=[b](double a){return Scalar::F01(a,b);};
    return transform(func,min_count,a);
}
Tensor F01(double a,const Tensor& b,size_t min_count){
    auto func=[a](double b){return Scalar::F01(a,b);};
    return transform(func,min_count,b);
}
Tensor F01(const Tensor& a,const Tensor& b,size_t min_count){
    return transform(Scalar::F01,min_count,a,b);
}

Tensor F11(const Tensor& a,double b,double x,size_t min_count){
    auto func=[b,x](double a){return Scalar::F11(a,b,x);};
    return transform(func,min_count,a);
}
Tensor F11(double a,const Tensor& b,double x,size_t min_count){
    auto func=[a,x](double b){return Scalar::F11(a,b,x);};
    return transform(func,min_count,b);
}
Tensor F11(double a,double b,const Tensor& x,size_t min_count){
    auto func=[a,b](double x){return Scalar::F11(a,b,x);};
    return transform(func,min_count,x);
}
Tensor F11(const Tensor& a,const Tensor& b,double x,size_t min_count){
    auto func=[x](double a,double b){return Scalar::F11(a,b,x);};
    return transform(func,min_count,a,b);
}
Tensor F11(const Tensor& a,double b,const Tensor& x,size_t min_count){
    auto func=[b](double a,double x){return Scalar::F11(a,b,x);};
    return transform(func,min_count,a,x);
}
Tensor F11(double a,const Tensor& b,const Tensor& x,size_t min_count){
    auto func=[a](double b,double x){return Scalar::F11(a,b,x);};
    return transform(func,min_count,b,x);
}
Tensor F11(const Tensor& a,const Tensor& b,const Tensor& x,size_t min_count){
    return transform(Scalar::F11,min_count,a,b,x);
}

Tensor U(const Tensor& a,double b,double x,size_t min_count){
    auto func=[b,x](double a){return Scalar::U(a,b,x);};
    return transform(func,min_count,a);
}
Tensor U(double a,const Tensor& b,double x,size_t min_count){
    auto func=[a,x](double b){return Scalar::U(a,b,x);};
    return transform(func,min_count,b);
}
Tensor U(double a,double b,const Tensor& x,size_t min_count){
    auto func=[a,b](double x){return Scalar::U(a,b,x);};
    return transform(func,min_count,x);
}
Tensor U(const Tensor& a,const Tensor& b,double x,size_t min_count){
    auto func=[x](double a,double b){return Scalar::U(a,b,x);};
    return transform(func,min_count,a,b);
}
Tensor U(const Tensor& a,double b,const Tensor& x,size_t min_count){
    auto func=[b](double a,double x){return Scalar::U(a,b,x);};
    return transform(func,min_count,a,x);
}
Tensor U(double a,const Tensor& b,const Tensor& x,size_t min_count){
    auto func=[a](double b,double x){return Scalar::U(a,b,x);};
    return transform(func,min_count,b,x);
}
Tensor U(const Tensor& a,const Tensor& b,const Tensor& x,size_t min_count){
    return transform(Scalar::U,min_count,a,b,x);
}

Tensor F21(const Tensor& a,double b,double c,double x,size_t min_count){
    auto func=[b,c,x](double a){return Scalar::F21(a,b,c,x);};
    return transform(func,min_count,a);
}
Tensor F21(double a,const Tensor& b,double c,double x,size_t min_count){
    auto func=[a,c,x](double b){return Scalar::F21(a,b,c,x);};
    return transform(func,min_count,b);
}
Tensor F21(double a,double b,const Tensor& c,double x,size_t min_count){
    auto func=[a,b,x](double c){return Scalar::F21(a,b,c,x);};
    return transform(func,min_count,c);
}
Tensor F21(double a,double b,double c,const Tensor& x,size_t min_count){
    auto func=[a,b,c](double x){return Scalar::F21(a,b,c,x);};
    return transform(func,min_count,x);
}
Tensor F21(const Tensor& a,const Tensor& b,double c,double x,size_t min_count){
    auto func=[c,x](double a,double b){return Scalar::F21(a,b,c,x);};
    return transform(func,min_count,a,b);
}
Tensor F21(const Tensor& a,double b,const Tensor& c,double x,size_t min_count){
    auto func=[b,x](double a,double c){return Scalar::F21(a,b,c,x);};
    return transform(func,min_count,a,c);
}
Tensor F21(const Tensor& a,double b,double c,const Tensor& x,size_t min_count){
    auto func=[b,c](double a,double x){return Scalar::F21(a,b,c,x);};
    return transform(func,min_count,a,x);
}
Tensor F21(double a,const Tensor& b,const Tensor& c,double x,size_t min_count){
    auto func=[a,x](double b,double c){return Scalar::F21(a,b,c,x);};
    return transform(func,min_count,b,c);
}
Tensor F21(double a,const Tensor& b,double c,const Tensor& x,size_t min_count){
    auto func=[a,c](double b,double x){return Scalar::F21(a,b,c,x);};
    return transform(func,min_count,b,x);
}
Tensor F21(double a,double b,const Tensor& c,const Tensor& x,size_t min_count){
    auto func=[a,b](double c,double x){return Scalar::F21(a,b,c,x);};
    return transform(func,min_count,c,x);
}
Tensor F21(const Tensor& a,const Tensor& b,const Tensor& c,double x,size_t min_count){
    auto func=[x](double a,double b,double c){return Scalar::F21(a,b,c,x);};
    return transform(func,min_count,a,b,c);
}
Tensor F21(const Tensor& a,const Tensor& b,double c,const Tensor& x,size_t min_count){
    auto func=[c](double a,double b,double x){return Scalar::F21(a,b,c,x);};
    return transform(func,min_count,a,b,x);
}
Tensor F21(const Tensor& a,double b,const Tensor& c,const Tensor& x,size_t min_count){
    auto func=[b](double a,double c,double x){return Scalar::F21(a,b,c,x);};
    return transform(func,min_count,a,c,x);
}
Tensor F21(double a,const Tensor& b,const Tensor& c,const Tensor& x,size_t min_count){
    auto func=[a](double b,double c,double x){return Scalar::F21(a,b,c,x);};
    return transform(func,min_count,b,c,x);
}
Tensor F21(const Tensor& a,const Tensor& b,const Tensor& c,const Tensor& x,size_t min_count){
    return transform(Scalar::F21,min_count,a,b,c,x);
}

Tensor F21_renorm(const Tensor& a,double b,double c,double x,size_t min_count){
    auto func=[b,c,x](double a){return Scalar::F21_renorm(a,b,c,x);};
    return transform(func,min_count,a);
}
Tensor F21_renorm(double a,const Tensor& b,double c,double x,size_t min_count){
    auto func=[a,c,x](double b){return Scalar::F21_renorm(a,b,c,x);};
    return transform(func,min_count,b);
}
Tensor F21_renorm(double a,double b,const Tensor& c,double x,size_t min_count){
    auto func=[a,b,x](double c){return Scalar::F21_renorm(a,b,c,x);};
    return transform(func,min_count,c);
}
Tensor F21_renorm(double a,double b,double c,const Tensor& x,size_t min_count){
    auto func=[a,b,c](double x){return Scalar::F21_renorm(a,b,c,x);};
    return transform(func,min_count,x);
}
Tensor F21_renorm(const Tensor& a,const Tensor& b,double c,double x,size_t min_count){
    auto func=[c,x](double a,double b){return Scalar::F21_renorm(a,b,c,x);};
    return transform(func,min_count,a,b);
}
Tensor F21_renorm(const Tensor& a,double b,const Tensor& c,double x,size_t min_count){
    auto func=[b,x](double a,double c){return Scalar::F21_renorm(a,b,c,x);};
    return transform(func,min_count,a,c);
}
Tensor F21_renorm(const Tensor& a,double b,double c,const Tensor& x,size_t min_count){
    auto func=[b,c](double a,double x){return Scalar::F21_renorm(a,b,c,x);};
    return transform(func,min_count,a,x);
}
Tensor F21_renorm(double a,const Tensor& b,const Tensor& c,double x,size_t min_count){
    auto func=[a,x](double b,double c){return Scalar::F21_renorm(a,b,c,x);};
    return transform(func,min_count,b,c);
}
Tensor F21_renorm(double a,const Tensor& b,double c,const Tensor& x,size_t min_count){
    auto func=[a,c](double b,double x){return Scalar::F21_renorm(a,b,c,x);};
    return transform(func,min_count,b,x);
}
Tensor F21_renorm(double a,double b,const Tensor& c,const Tensor& x,size_t min_count){
    auto func=[a,b](double c,double x){return Scalar::F21_renorm(a,b,c,x);};
    return transform(func,min_count,c,x);
}
Tensor F21_renorm(const Tensor& a,const Tensor& b,const Tensor& c,double x,size_t min_count){
    auto func=[x](double a,double b,double c){return Scalar::F21_renorm(a,b,c,x);};
    return transform(func,min_count,a,b,c);
}
Tensor F21_renorm(const Tensor& a,const Tensor& b,double c,const Tensor& x,size_t min_count){
    auto func=[c](double a,double b,double x){return Scalar::F21_renorm(a,b,c,x);};
    return transform(func,min_count,a,b,x);
}
Tensor F21_renorm(const Tensor& a,double b,const Tensor& c,const Tensor& x,size_t min_count){
    auto func=[b](double a,double c,double x){return Scalar::F21_renorm(a,b,c,x);};
    return transform(func,min_count,a,c,x);
}
Tensor F21_renorm(double a,const Tensor& b,const Tensor& c,const Tensor& x,size_t min_count){
    auto func=[a](double b,double c,double x){return Scalar::F21_renorm(a,b,c,x);};
    return transform(func,min_count,b,c,x);
}
Tensor F21_renorm(const Tensor& a,const Tensor& b,const Tensor& c,const Tensor& x,size_t min_count){
    return transform(Scalar::F21_renorm,min_count,a,b,c,x);

}

Tensor F20(const Tensor& a,double b,double x,size_t min_count){
    auto func=[b,x](double a){return Scalar::F20(a,b,x);};
    return transform(func,min_count,a);

}
Tensor F20(double a,const Tensor& b,double x,size_t min_count){
    auto func=[a,x](double b){return Scalar::F20(a,b,x);};
    return transform(func,min_count,b);

}
Tensor F20(double a,double b,const Tensor& x,size_t min_count){
    auto func=[a,b](double x){return Scalar::F20(a,b,x);};
    return transform(func,min_count,x);
}
Tensor F20(const Tensor& a,const Tensor& b,double x,size_t min_count){
    auto func=[x](double a,double b){return Scalar::F20(a,b,x);};
    return transform(func,min_count,a,b);
}
Tensor F20(const Tensor& a,double b,const Tensor& x,size_t min_count){
    auto func=[b](double a,double x){return Scalar::F20(a,b,x);};
    return transform(func,min_count,a,x);
}
Tensor F20(double a,const Tensor& b,const Tensor& x,size_t min_count){
    auto func=[a](double b,double x){return Scalar::F20(a,b,x);};
    return transform(func,min_count,b,x);
}
Tensor F20(const Tensor& a,const Tensor& b,const Tensor& x,size_t min_count){
    return transform(Scalar::F20,min_count,a,b,x);
}
//Laguerre Functions
Tensor L1(const Tensor& a,double b,size_t min_count){
    auto func=[b](double a){return Scalar::L1(a,b);};
    return transform(func,min_count,a);
}
Tensor L1(double a,const Tensor& b,size_t min_count){
    auto func=[a](double b){return Scalar::L1(a,b);};
    return transform(func,min_count,b);
}
Tensor L1(const Tensor& a,const Tensor& b,size_t min_count){
    return transform(Scalar::L1,min_count,a,b);
}

Tensor L2(const Tensor& a,double b,size_t min_count){
    auto func=[b](double a){return Scalar::L2(a,b);};
    return transform(func,min_count,a);
}
Tensor L2(double a,const Tensor& b,size_t min_count){
    auto func=[a](double b){return Scalar::L2(a,b);};
    return transform(func,min_count,b);
}
Tensor L2(const Tensor& a,const Tensor& b,size_t min_count){
    return transform(Scalar::L2,min_count,a,b);
}

Tensor L3(const Tensor& a,double b,size_t min_count){
    auto func=[b](double a){return Scalar::L3(a,b);};
    return transform(func,min_count,a);
}
Tensor L3(double a,const Tensor& b,size_t min_count){
    auto func=[a](double b){return Scalar::L3(a,b);};
    return transform(func,min_count,b);
}
Tensor L3(const Tensor& a,const Tensor& b,size_t min_count){
    return transform(Scalar::L3,min_count,a,b);
}

Tensor L(const Tensor& a,double x,int n,size_t min_count){
    auto func=[x,n](double a){return Scalar::L(a,x,n);};
    return transform(func,min_count,a);
}
Tensor L(double a,const Tensor& x,int n,size_t min_count){
    auto func=[a,n](double x){return Scalar::L(a,x,n);};
    return transform(func,min_count,x);
}
Tensor L(double a,double x,const Tensor& n,size_t min_count){
    auto func=[a,x](double n){return Scalar::L(a,x,(int)n);};
    return transform(func,min_count,n);
}
Tensor L(const Tensor& a,const Tensor& x,int n,size_t min_count){
    auto func=[n](double a,double x){return Scalar::L(a,x,n);};
    return transform(func,min_count,a,x);
}
Tensor L(const Tensor& a,double x,const Tensor& n,size_t min_count){
    auto func=[x](double a,double n){return Scalar::L(a,x,(int)n);};
    return transform(func,min_count,a,n);
}
Tensor L(double a,const Tensor& x,const Tensor& n,size_t min_count){
    auto func=[a](double x,double n){return Scalar::L(a,x,(int)n);};
    return transform(func,min_count,x,n);
}
Tensor L(const Tensor& a,const Tensor& x,const Tensor& n,size_t min_count){
    auto func=[](double a,double x,double n){return Scalar::L(a,x,(int)n);};
    return transform(func,min_count,a,x,n);
}
//Lambert W Functions
Tensor W0(const Tensor& x,size_t min_count){
    return transform(Scalar::W0,min_count,x);
}
Tensor Wm1(const Tensor& x,size_t min_count){
    return transform(Scalar::Wm1,min_count,x);
}
//Legendre Polynomials
Tensor legendre_P1(const Tensor& x,size_t min_count){
    return transform(Scalar::legendre_P1,min_count,x);
}
Tensor legendre_P2(const Tensor& x,size_t min_count){
    return transform(Scalar::legendre_P2,min_count,x);
}
Tensor legendre_P3(const Tensor& x,size_t min_count){
    return transform(Scalar::legendre_P3,min_count,x);
}
Tensor legendre_P(const Tensor& x,int l,size_t min_count){
    auto func=[l](double x){return Scalar::legendre_P(x,l);};
    return transform(func,min_count,x);
}
Tensor legendre_P(double x,const Tensor& l,size_t min_count){
    auto func=[x](double l){return Scalar::legendre_P(x,(int)l);};
    return transform(func,min_count,l);
}
Tensor legendre_P(const Tensor& x,const Tensor& l,size_t min_count){
    auto func=[](double x,double l){return Scalar::legendre_P(x,(int)l);};
    return transform(func,min_count,x,l);
}

Tensor Q0(const Tensor& x,size_t min_count){
    return transform(Scalar::Q0,min_count,x);
}
Tensor Q1(const Tensor& x,size_t min_count){
    return transform(Scalar::Q1,min_count,x);
}
Tensor Q(const Tensor& x,int l,size_t min_count){
    auto func=[l](double x){return Scalar::Q(x,l);};
    return transform(func,min_count,x);
}
Tensor Q(double x,const Tensor& l,size_t min_count){
    auto func=[x](double l){return Scalar::Q(x,(int)l);};
    return transform(func,min_count,l);
}
Tensor Q(const Tensor& x,const Tensor& l,size_t min_count){
    auto func=[](double x,double l){return Scalar::Q(x,(int)l);};
    return transform(func,min_count,x,l);
}
//Associated Legendre Polynomials and Spherical Harmonics
Tensor Plm(const Tensor& x,int l,int m,size_t min_count){
    auto func=[l,m](double x){return Scalar::Plm(x,l,m);};
    return transform(func,min_count,x);
}
Tensor Plm(double x,const Tensor& l,int m,size_t min_count){
    auto func=[x,m](double l){return Scalar::Plm(x,(int)l,m);};
    return transform(func,min_count,l);
}
Tensor Plm(double x,int l,const Tensor& m,size_t min_count){
    auto func=[x,l](double m){return Scalar::Plm(x,l,(int)m);};
    return transform(func,min_count,m);
}
Tensor Plm(const Tensor& x,const Tensor& l,int m,size_t min_count){
    auto func=[m](double x,double l){return Scalar::Plm(x,(int)l,m);};
    return transform(func,min_count,x,l);
}
Tensor Plm(const Tensor& x,int l,const Tensor& m,size_t min_count){
    auto func=[l](double x,double m){return Scalar::Plm(x,l,(int)m);};
    return transform(func,min_count,x,m);
}
Tensor Plm(double x,const Tensor& l,const Tensor& m,size_t min_count){
    auto func=[x](double l,double m){return Scalar::Plm(x,(int)l,(int)m);};
    return transform(func,min_count,l,m);
}
Tensor Plm(const Tensor& x,const Tensor& l,const Tensor& m,size_t min_count){
    auto func=[](double x,double l,double m){return Scalar::Plm(x,(int)l,(int)m);};
    return transform(func,min_count,x,l,m);
}

Tensor sphPlm(const Tensor& x,int l,int m,size_t min_count){
    auto func=[l,m](double x){return Scalar::sphPlm(x,l,m);};
    return transform(func,min_count,x);
}
Tensor sphPlm(double x,const Tensor& l,int m,size_t min_count){
    auto func=[x,m](double l){return Scalar::sphPlm(x,(int)l,m);};
    return transform(func,min_count,l);
}
Tensor sphPlm(double x,int l,const Tensor& m,size_t min_count){
    auto func=[x,l](double m){return Scalar::sphPlm(x,l,(int)m);};
    return transform(func,min_count,m);
}
Tensor sphPlm(const Tensor& x,const Tensor& l,int m,size_t min_count){
    auto func=[m](double x,double l){return Scalar::sphPlm(x,(int)l,m);};
    return transform(func,min_count,x,l);
}
Tensor sphPlm(const Tensor& x,int l,const Tensor& m,size_t min_count){
    auto func=[l](double x,double m){return Scalar::sphPlm(x,l,(int)m);};
    return transform(func,min_count,x,m);
}
Tensor sphPlm(double x,const Tensor& l,const Tensor& m,size_t min_count){
    auto func=[x](double l,double m){return Scalar::sphPlm(x,(int)l,(int)m);};
    return transform(func,min_count,l,m);
}
Tensor sphPlm(const Tensor& x,const Tensor& l,const Tensor& m,size_t min_count){
    auto func=[](double x,double l,double m){return Scalar::sphPlm(x,(int)l,(int)m);};
    return transform(func,min_count,x,l,m);
}
//Conical Functions
Tensor conicalP_half(const Tensor& x,double lambda,size_t min_count){
    auto func=[lambda](double x){return Scalar::conicalP_half(x,lambda);};
    return transform(func,min_count,x);
}
Tensor conicalP_half(double x,const Tensor& lambda,size_t min_count){
    auto func=[x](double lambda){return Scalar::conicalP_half(x,lambda);};
    return transform(func,min_count,lambda);
}
Tensor conicalP_half(const Tensor& x,const Tensor& lambda,size_t min_count){
    return transform(Scalar::conicalP_half,min_count,x,lambda);
}

Tensor conicalP_mhalf(const Tensor& x,double lambda,size_t min_count){
    auto func=[lambda](double x){return Scalar::conicalP_mhalf(x,lambda);};
    return transform(func,min_count,x);
}
Tensor conicalP_mhalf(double x,const Tensor& lambda,size_t min_count){
    auto func=[x](double lambda){return Scalar::conicalP_mhalf(x,lambda);};
    return transform(func,min_count,lambda);
}
Tensor conicalP_mhalf(const Tensor& x,const Tensor& lambda,size_t min_count){
    return transform(Scalar::conicalP_mhalf,min_count,x,lambda);
}

Tensor conicalP0(const Tensor& x,double lambda,size_t min_count){
    auto func=[lambda](double x){return Scalar::conicalP0(x,lambda);};
    return transform(func,min_count,x);
}
Tensor conicalP0(double x,const Tensor& lambda,size_t min_count){
    auto func=[x](double lambda){return Scalar::conicalP0(x,lambda);};
    return transform(func,min_count,lambda);
}
Tensor conicalP0(const Tensor& x,const Tensor& lambda,size_t min_count){
    return transform(Scalar::conicalP0,min_count,x,lambda);
}

Tensor conicalP1(const Tensor& x,double lambda,size_t min_count){
    auto func=[lambda](double x){return Scalar::conicalP1(x,lambda);};
    return transform(func,min_count,x);
}
Tensor conicalP1(double x,const Tensor& lambda,size_t min_count){
    auto func=[x](double lambda){return Scalar::conicalP1(x,lambda);};
    return transform(func,min_count,lambda);
}
Tensor conicalP1(const Tensor& x,const Tensor& lambda,size_t min_count){
    return transform(Scalar::conicalP1,min_count,x,lambda);
}

Tensor conicalP_sph(const Tensor& x,double lambda,int n,size_t min_count){
    auto func=[lambda,n](double x){return Scalar::conicalP_sph(x,lambda,n);};
    return transform(func,min_count,x);
}
Tensor conicalP_sph(double x,const Tensor& lambda,int n,size_t min_count){
    auto func=[x,n](double lambda){return Scalar::conicalP_sph(x,lambda,n);};
    return transform(func,min_count,lambda);
}
Tensor conicalP_sph(double x,double lambda,const Tensor& n,size_t min_count){
    auto func=[x,lambda](double n){return Scalar::conicalP_sph(x,lambda,(int)n);};
    return transform(func,min_count,n);
}
Tensor conicalP_sph(const Tensor& x,const Tensor& lambda,int n,size_t min_count){
    auto func=[n](double x,double lambda){return Scalar::conicalP_sph(x,lambda,n);};
    return transform(func,min_count,x,lambda);
}
Tensor conicalP_sph(const Tensor& x,double lambda,const Tensor& n,size_t min_count){
    auto func=[lambda](double x,double n){return Scalar::conicalP_sph(x,lambda,(int)n);};
    return transform(func,min_count,x,n);
}
Tensor conicalP_sph(double x,const Tensor& lambda,const Tensor& n,size_t min_count){
    auto func=[x](double lambda,double n){return Scalar::conicalP_sph(x,lambda,(int)n);};
    return transform(func,min_count,lambda,n);
}
Tensor conicalP_sph(const Tensor& x,const Tensor& lambda,const Tensor& n,size_t min_count){
    auto func=[](double x,double lambda,double n){return Scalar::conicalP_sph(x,lambda,(int)n);};
    return transform(func,min_count,x,lambda,n);
}

Tensor conicalP_cyl(const Tensor& x,double lambda,int n,size_t min_count){
    auto func=[lambda,n](double x){return Scalar::conicalP_cyl(x,lambda,n);};
    return transform(func,min_count,x);
}
Tensor conicalP_cyl(double x,const Tensor& lambda,int n,size_t min_count){
    auto func=[x,n](double lambda){return Scalar::conicalP_cyl(x,lambda,n);};
    return transform(func,min_count,lambda);
}
Tensor conicalP_cyl(double x,double lambda,const Tensor& n,size_t min_count){
    auto func=[x,lambda](double n){return Scalar::conicalP_cyl(x,lambda,(int)n);};
    return transform(func,min_count,n);
}
Tensor conicalP_cyl(const Tensor& x,const Tensor& lambda,int n,size_t min_count){
    auto func=[n](double x,double lambda){return Scalar::conicalP_cyl(x,lambda,n);};
    return transform(func,min_count,x,lambda);
}
Tensor conicalP_cyl(const Tensor& x,double lambda,const Tensor& n,size_t min_count){
    auto func=[lambda](double x,double n){return Scalar::conicalP_cyl(x,lambda,(int)n);};
    return transform(func,min_count,x,n);
}
Tensor conicalP_cyl(double x,const Tensor& lambda,const Tensor& n,size_t min_count){
    auto func=[x](double lambda,double n){return Scalar::conicalP_cyl(x,lambda,(int)n);};
    return transform(func,min_count,lambda,n);
}
Tensor conicalP_cyl(const Tensor& x,const Tensor& lambda,const Tensor& n,size_t min_count){
    auto func=[](double x,double lambda,double n){return Scalar::conicalP_cyl(x,lambda,(int)n);};
    return transform(func,min_count,x,lambda,n);
}
//Radial Functions for Hyperbolic Space
Tensor H3d0(const Tensor& lambda,double eta,size_t min_count){
    auto func=[eta](double lambda){return Scalar::H3d0(lambda,eta);};
    return transform(func,min_count,lambda);
}
Tensor H3d0(double lambda,const Tensor& eta,size_t min_count){
    auto func=[lambda](double eta){return Scalar::H3d0(lambda,eta);};
    return transform(func,min_count,eta);
}
Tensor H3d0(const Tensor& lambda,const Tensor& eta,size_t min_count){
    return transform(Scalar::H3d0,min_count,lambda,eta);
}

Tensor H3d1(const Tensor& lambda,double eta,size_t min_count){
    auto func=[eta](double lambda){return Scalar::H3d1(lambda,eta);};
    return transform(func,min_count,lambda);
}
Tensor H3d1(double lambda,const Tensor& eta,size_t min_count){
    auto func=[lambda](double eta){return Scalar::H3d1(lambda,eta);};
    return transform(func,min_count,eta);
}
Tensor H3d1(const Tensor& lambda,const Tensor& eta,size_t min_count){
    return transform(Scalar::H3d1,min_count,lambda,eta);
}

Tensor H3d(const Tensor& lambda,double eta,int n,size_t min_count){
    auto func=[eta,n](double lambda){return Scalar::H3d(lambda,eta,n);};
    return transform(func,min_count,lambda);
}
Tensor H3d(double lambda,const Tensor& eta,int n,size_t min_count){
    auto func=[lambda,n](double eta){return Scalar::H3d(lambda,eta,n);};
    return transform(func,min_count,eta);
}
Tensor H3d(double lambda,double eta,const Tensor& n,size_t min_count){
    auto func=[lambda,eta](double n){return Scalar::H3d(lambda,eta,(int)n);};
    return transform(func,min_count,n);
}
Tensor H3d(const Tensor& lambda,const Tensor& eta,int n,size_t min_count){
    auto func=[n](double lambda,double eta){return Scalar::H3d(lambda,eta,n);};
    return transform(func,min_count,lambda,eta);
}
Tensor H3d(const Tensor& lambda,double eta,const Tensor& n,size_t min_count){
    auto func=[eta](double lambda,double n){return Scalar::H3d(lambda,eta,(int)n);};
    return transform(func,min_count,lambda,n);
}
Tensor H3d(double lambda,const Tensor& eta,const Tensor& n,size_t min_count){
    auto func=[lambda](double eta,double n){return Scalar::H3d(lambda,eta,(int)n);};
    return transform(func,min_count,eta,n);
}
Tensor H3d(const Tensor& lambda,const Tensor& eta,const Tensor& n,size_t min_count){
    auto func=[](double lambda,double eta,double n){return Scalar::H3d(lambda,eta,(int)n);};
    return transform(func,min_count,lambda,eta,n);
}
//Psi Functions
Tensor psi(const Tensor& x,size_t min_count){
    return transform(Scalar::psi,min_count,x);
}
Tensor psi1(const Tensor& x,size_t min_count){
    return transform(Scalar::psi1,min_count,x);
}
Tensor psi_n(const Tensor& x,int n,size_t min_count){
    auto func=[n](double x){return Scalar::psi_n(x,n);};
    return transform(func,min_count,x);
}
Tensor psi_n(double x,const Tensor& n,size_t min_count){
    auto func=[x](double n){return Scalar::psi_n(x,(int)n);};
    return transform(func,min_count,n);
}
Tensor psi_n(const Tensor& x,const Tensor& n,size_t min_count){
    auto func=[](double x,double n){return Scalar::psi_n(x,(int)n);};
    return transform(func,min_count,x,n);
}
//Synchrotron Functions
Tensor synchrotron1(const Tensor& x,size_t min_count){
    return transform(Scalar::synchrotron1,min_count,x);
}
Tensor synchrotron2(const Tensor& x,size_t min_count){
    return transform(Scalar::synchrotron2,min_count,x);
}
//Transport Functions
Tensor transport2(const Tensor& x,size_t min_count){
    return transform(Scalar::transport2,min_count,x);
}
Tensor transport3(const Tensor& x,size_t min_count){
    return transform(Scalar::transport3,min_count,x);
}
Tensor transport4(const Tensor& x,size_t min_count){
    return transform(Scalar::transport4,min_count,x);
}
Tensor transport5(const Tensor& x,size_t min_count){
    return transform(Scalar::transport5,min_count,x);
}
//Zeta Functions
Tensor zeta(const Tensor& x,size_t min_count){
    return transform(Scalar::zeta,min_count,x);
}
Tensor zetam1(const Tensor& x,size_t min_count){
    return transform(Scalar::zetam1,min_count,x);
}
Tensor hzeta(const Tensor& x,double q,size_t min_count){
    auto func=[q](double x){return Scalar::hzeta(x,q);};
    return transform(func,min_count,x);
}
Tensor hzeta(double x,const Tensor& q,size_t min_count){
    auto func=[x](double q){return Scalar::hzeta(x,q);};
    return transform(func,min_count,q);
}
Tensor hzeta(const Tensor& x,const Tensor& q,size_t min_count){
    return transform(Scalar::hzeta,min_count,x,q);
}
Tensor eta(const Tensor& x,size_t min_count){
    return transform(Scalar::eta,min_count,x);
}
//common activation function
Tensor ELU(const Tensor& x,double alpha,size_t min_count){
    auto func=[alpha](double x){return Scalar::ELU(x,alpha);};
    return transform(func,min_count,x);
}
Tensor ELU(double x,const Tensor& alpha,size_t min_count){
    auto func=[x](double alpha){return Scalar::ELU(x,alpha);};
    return transform(func,min_count,alpha);
}
Tensor ELU(const Tensor& x,const Tensor& alpha,size_t min_count){
    return transform(Scalar::ELU,min_count,x,alpha);
}

Tensor hardshrink(const Tensor& x,double lambda,size_t min_count){
    auto func=[lambda](double x){return Scalar::hardshrink(x,lambda);};
    return transform(func,min_count,x);
}
Tensor hardshrink(double x,const Tensor& lambda,size_t min_count){
    auto func=[x](double lambda){return Scalar::hardshrink(x,lambda);};
    return transform(func,min_count,lambda);
}
Tensor hardshrink(const Tensor& x,const Tensor& lambda,size_t min_count){
    return transform(Scalar::hardshrink,min_count,x,lambda);
}

Tensor hardsigmoid(const Tensor& x,size_t min_count){
    return transform(Scalar::hardsigmoid,min_count,x);
}

Tensor hardtanh(const Tensor& x,double min_val,double max_val,size_t min_count){
    auto func=[min_val,max_val](double x){return Scalar::hardtanh(x,min_val,max_val);};
    return transform(func,min_count,x);
}
Tensor hardtanh(double x,const Tensor& min_val,double max_val,size_t min_count){
    auto func=[x,max_val](double min_val){return Scalar::hardtanh(x,min_val,max_val);};
    return transform(func,min_count,min_val);
}
Tensor hardtanh(double x, double min_val, const Tensor& max_val,size_t min_count){
    auto func=[x,min_val](double max_val){return Scalar::hardtanh(x,min_val,max_val);};
    return transform(func,min_count,max_val);
}
Tensor hardtanh(const Tensor& x,const Tensor& min_val,double max_val,size_t min_count){
    auto func=[max_val](double x,double min_val){return Scalar::hardtanh(x,min_val,max_val);};
    return transform(func,min_count,x,min_val);
}
Tensor hardtanh(const Tensor& x,double min_val,const Tensor& max_val,size_t min_count){
    auto func=[min_val](double x,double max_val){return Scalar::hardtanh(x,min_val,max_val);};
    return transform(func,min_count,x,max_val);
}
Tensor hardtanh(double x,const Tensor& min_val,const Tensor& max_val,size_t min_count){
    auto func=[x](double min_val,double max_val){return Scalar::hardtanh(x,min_val,max_val);};
    return transform(func,min_count,min_val,max_val);
}
Tensor hardtanh(const Tensor& x,const Tensor& min_val,const Tensor& max_val,size_t min_count){
    return transform(Scalar::hardtanh,min_count,x,min_val,max_val);
}

Tensor hardswish(const Tensor& x,size_t min_count){
    return transform(Scalar::hardswish,min_count,x);
}

Tensor leakyReLU(const Tensor& x,double alpha,size_t min_count){
    auto func=[alpha](double x){return Scalar::leakyReLU(x,alpha);};
    return transform(func,min_count,x);
}
Tensor leakyReLU(double x,const Tensor& alpha,size_t min_count){
    auto func=[x](double alpha){return Scalar::leakyReLU(x,alpha);};
    return transform(func,min_count,alpha);
}
Tensor leakyReLU(const Tensor& x,const Tensor& alpha,size_t min_count){
    return transform(Scalar::leakyReLU,min_count,x,alpha);
}

Tensor logsigmoid(const Tensor& x,size_t min_count){
    return transform(Scalar::logsigmoid,min_count,x);
}

Tensor ReLU(const Tensor& x,size_t min_count){
    return transform(Scalar::ReLU,min_count,x);
}

Tensor ReLU6(const Tensor& x,size_t min_count){
    return transform(Scalar::ReLU6,min_count,x);
}

Tensor RReLU(const Tensor& x,double lower,double upper,size_t min_count){
    auto func=[lower,upper](double x){return Scalar::RReLU(x,lower,upper);};
    return transform(func,min_count,x);
}
Tensor RReLU(double x,const Tensor& lower,double upper,size_t min_count){
    auto func=[x,upper](double lower){return Scalar::RReLU(x,lower,upper);};
    return transform(func,min_count,lower);
}
Tensor RReLU(double x,double lower,const Tensor& upper,size_t min_count){
    auto func=[x,lower](double upper){return Scalar::RReLU(x,lower,upper);};
    return transform(func,min_count,upper);
}
Tensor RReLU(const Tensor& x,const Tensor& lower,double upper,size_t min_count){
    auto func=[upper](double x,double lower){return Scalar::RReLU(x,lower,upper);};
    return transform(func,min_count,x,lower);
}
Tensor RReLU(const Tensor& x,double lower,const Tensor& upper,size_t min_count){
    auto func=[lower](double x,double upper){return Scalar::RReLU(x,lower,upper);};
    return transform(func,min_count,x,upper);
}
Tensor RReLU(double x,const Tensor& lower,const Tensor& upper,size_t min_count){
    auto func=[x](double lower,double upper){return Scalar::RReLU(x,lower,upper);};
    return transform(func,min_count,lower,upper);
}
Tensor RReLU(const Tensor& x,const Tensor& lower,const Tensor& upper,size_t min_count){
    return transform(Scalar::RReLU,min_count,x,lower,upper);
}

Tensor SELU(const Tensor& x,size_t min_count){
    return transform(Scalar::SELU,min_count,x);
}

Tensor CELU(const Tensor& x,double alpha,size_t min_count){
    auto func=[alpha](double x){return Scalar::CELU(x,alpha);};
    return transform(func,min_count,x);
}
Tensor CELU(double x,const Tensor& alpha,size_t min_count){
    auto func=[x](double alpha){return Scalar::CELU(x,alpha);};
    return transform(func,min_count,alpha);
}
Tensor CELU(const Tensor& x,const Tensor& alpha,size_t min_count){
    return transform(Scalar::CELU,min_count,x,alpha);
}

Tensor GELU(const Tensor& x,size_t min_count){
    return transform(Scalar::GELU,min_count,x);
}
Tensor GELU_fast(const Tensor& x,size_t min_count){
    return transform(Scalar::GELU_fast,min_count,x);
}

Tensor sigmoid(const Tensor& x,size_t min_count){
    return transform(Scalar::sigmoid,min_count,x);
}
Tensor SiLU(const Tensor& x,size_t min_count){
    return transform(Scalar::SiLU,min_count,x);
}
Tensor mish(const Tensor& x,size_t min_count){
    return transform(Scalar::mish,min_count,x);
}

Tensor softplus(const Tensor& x,double beta,double threshold,size_t min_count){
    auto func=[beta,threshold](double x){return Scalar::softplus(x,beta,threshold);};
    return transform(func,min_count,x);
}
Tensor softplus(double x,const Tensor& beta,double threshold,size_t min_count){
    auto func=[x,threshold](double beta){return Scalar::softplus(x,beta,threshold);};
    return transform(func,min_count,beta);
}
Tensor softplus(double x,double beta,const Tensor& threshold,size_t min_count){
    auto func=[x,beta](double threshold){return Scalar::softplus(x,beta,threshold);};
    return transform(func,min_count,threshold);
}
Tensor softplus(const Tensor& x,const Tensor& beta,double threshold,size_t min_count){
    auto func=[threshold](double x,double beta){return Scalar::softplus(x,beta,threshold);};
    return transform(func,min_count,x,beta);
}
Tensor softplus(const Tensor& x,double beta,const Tensor& threshold,size_t min_count){
    auto func=[beta](double x,double threshold){return Scalar::softplus(x,beta,threshold);};
    return transform(func,min_count,x,threshold);
}
Tensor softplus(double x,const Tensor& beta,const Tensor& threshold,size_t min_count){
    auto func=[x](double beta,double threshold){return Scalar::softplus(x,beta,threshold);};
    return transform(func,min_count,beta,threshold);
}
Tensor softplus(const Tensor& x,const Tensor& beta,const Tensor& threshold,size_t min_count){
    return transform(Scalar::softplus,min_count,x,beta,threshold);
}

Tensor softshrink(const Tensor& x,double lambda,size_t min_count){
    auto func=[lambda](double x){return Scalar::softshrink(x,lambda);};
    return transform(func,min_count,x);
}
Tensor softshrink(double x,const Tensor& lambda,size_t min_count){
    auto func=[x](double lambda){return Scalar::softshrink(x,lambda);};
    return transform(func,min_count,lambda);
}
Tensor softshrink(const Tensor& x,const Tensor& lambda,size_t min_count){
    return transform(Scalar::softshrink,min_count,x,lambda);
}

Tensor softsign(const Tensor& x,size_t min_count){
    return transform(Scalar::softsign,min_count,x);
}
Tensor tanhshrink(const Tensor& x,size_t min_count){
    return transform(Scalar::tanhshrink,min_count,x);
}

Tensor threshold(const Tensor& x,double threshold,double value,size_t min_count){
    auto func=[threshold,value](double x){return Scalar::threshold(x,threshold,value);};
    return transform(func,min_count,x);
}
Tensor threshold(double x,const Tensor& threshold,double value,size_t min_count){
    auto func=[x,value](double threshold){return Scalar::threshold(x,threshold,value);};
    return transform(func,min_count,threshold);
}
Tensor threshold(double x,double threshold,const Tensor& value,size_t min_count){
    auto func=[x,threshold](double value){return Scalar::threshold(x,threshold,value);};
    return transform(func,min_count,value);
}
Tensor threshold(const Tensor& x,const Tensor& threshold,double value,size_t min_count){
    auto func=[value](double x,double threshold){return Scalar::threshold(x,threshold,value);};
    return transform(func,min_count,x,threshold);
}
Tensor threshold(const Tensor& x,double threshold,const Tensor& value,size_t min_count){
    auto func=[threshold](double x,double value){return Scalar::threshold(x,threshold,value);};
    return transform(func,min_count,x,value);
}
Tensor threshold(double x,const Tensor& threshold,const Tensor& value,size_t min_count){
    auto func=[x](double threshold,double value){return Scalar::threshold(x,threshold,value);};
    return transform(func,min_count,threshold,value);
}
Tensor threshold(const Tensor& x,const Tensor& threshold,const Tensor& value,size_t min_count){
    return transform(Scalar::threshold,min_count,x,threshold,value);
} 
}