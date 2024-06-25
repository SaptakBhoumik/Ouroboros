#pragma once
namespace Ouroboros{
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