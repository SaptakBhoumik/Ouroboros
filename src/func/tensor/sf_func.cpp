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

}