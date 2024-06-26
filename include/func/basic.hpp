#pragma once
#include "../tensor.hpp"
#include "../macros.hpp"
namespace Ouroboros{
namespace Scalar{
double abs(double x);

double exp(double x);

double ln(double x);
double log10(double x);
double log2(double x);
double log(double x,double base);


double cbrt(double x);
double sqrt(double x);
double pow(double x,double y);

double hypot2(double x,double y);

double hypot3(double x,double y,double z);

double ceil(double x);
double floor(double x);
double trunc(double x);
double nearbyint(double x);
double rint(double x);
double round(double x);
double fmod(double x,double y);

double min(double x,double y);
double max(double x,double y);

double clamp(double x,double min,double max);

double clamp(double x,double min,double max,double c);

double sign(double x);
double fdim(double x,double y);
}
Tensor abs(const Tensor& t,size_t min_count=__MIN__COUNT__FOR__THREAD__);

Tensor exp(const Tensor& t,size_t min_count=__MIN__COUNT__FOR__THREAD__);

Tensor ln(const Tensor& t,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor log10(const Tensor& t,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor log2(const Tensor& t,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor log(const Tensor& t,double base,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor log(double x,const Tensor& base,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor log(const Tensor& t,const Tensor& base,size_t min_count=__MIN__COUNT__FOR__THREAD__);

Tensor cbrt(const Tensor& t,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor sqrt(const Tensor& t,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor pow(const Tensor& t,double y,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor pow(double x,const Tensor& y,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor pow(const Tensor& t,const Tensor& y,size_t min_count=__MIN__COUNT__FOR__THREAD__);

Tensor hypot2(const Tensor& x,double y,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor hypot2(double x,const Tensor& y,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor hypot2(const Tensor& x,const Tensor& y,size_t min_count=__MIN__COUNT__FOR__THREAD__);

Tensor hypot3(const Tensor& x,double y,double z,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor hypot3(double x,const Tensor& y,double z,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor hypot3(double x,double y,const Tensor& z,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor hypot3(const Tensor& x,const Tensor& y,double z,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor hypot3(const Tensor& x,double y,const Tensor& z,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor hypot3(double x,const Tensor& y,const Tensor& z,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor hypot3(const Tensor& x,const Tensor& y,const Tensor& z,size_t min_count=__MIN__COUNT__FOR__THREAD__);

Tensor ceil(const Tensor& t,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor floor(const Tensor& t,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor trunc(const Tensor& t,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor nearbyint(const Tensor& t,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor rint(const Tensor& t,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor round(const Tensor& t,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor fmod(const Tensor& x,double y,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor fmod(double x,const Tensor& y,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor fmod(const Tensor& x,const Tensor& y,size_t min_count=__MIN__COUNT__FOR__THREAD__);

Tensor min(const Tensor& x,double y,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor min(double x,const Tensor& y,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor min(const Tensor& x,const Tensor& y,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor max(const Tensor& x,double y,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor max(double x,const Tensor& y,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor max(const Tensor& x,const Tensor& y,size_t min_count=__MIN__COUNT__FOR__THREAD__);

Tensor clamp(const Tensor& x,double min,double max,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor clamp(double x,const Tensor& min,double max,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor clamp(double x,double min,const Tensor& max,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor clamp(const Tensor& x,const Tensor& min,double max,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor clamp(const Tensor& x,double min,const Tensor& max,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor clamp(double x,const Tensor& min,const Tensor& max,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor clamp(const Tensor& x,const Tensor& min,const Tensor& max,size_t min_count=__MIN__COUNT__FOR__THREAD__);

Tensor clamp(const Tensor& x,double min,double max,double c,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor clamp(double x,const Tensor& min,double max,double c,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor clamp(double x,double min,const Tensor& max,double c,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor clamp(double x,double min,double max,const Tensor& c,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor clamp(const Tensor& x,const Tensor& min,double max,double c,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor clamp(const Tensor& x,double min,const Tensor& max,double c,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor clamp(const Tensor& x,double min,double max,const Tensor& c,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor clamp(double x,const Tensor& min,const Tensor& max,double c,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor clamp(double x,const Tensor& min,double max,const Tensor& c,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor clamp(double x,double min,const Tensor& max,const Tensor& c,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor clamp(const Tensor& x,const Tensor& min,const Tensor& max,double c,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor clamp(const Tensor& x,const Tensor& min,double max,const Tensor& c,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor clamp(const Tensor& x,double min,const Tensor& max,const Tensor& c,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor clamp(double x,const Tensor& min,const Tensor& max,const Tensor& c,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor clamp(const Tensor& x,const Tensor& min,const Tensor& max,const Tensor& c,size_t min_count=__MIN__COUNT__FOR__THREAD__);

Tensor sign(const Tensor& t,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor fdim(const Tensor& x,double y,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor fdim(double x,const Tensor& y,size_t min_count=__MIN__COUNT__FOR__THREAD__);
Tensor fdim(const Tensor& x,const Tensor& y,size_t min_count=__MIN__COUNT__FOR__THREAD__);
}