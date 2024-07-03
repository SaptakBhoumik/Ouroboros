#pragma once
#include <cstdint>
#include <cstddef>
#include "macros.hpp"
namespace Ouroboros{
namespace CPU{
double* neg_ptr(const double* a,size_t size);//-a

double* add_ptr(const double* a,const double* b,size_t size);//a+b
double* sub_ptr(const double* a,const double* b,size_t size);//a-b
double* mul_ptr(const double* a,const double* b,size_t size);//a*b
double* div_ptr(const double* a,const double* b,size_t size);//a/b

double* add_ptr(const double* a,double b,size_t size);//a+b
double* sub_ptr(const double* a,double b,size_t size);//a-b
double* mul_ptr(const double* a,double b,size_t size);//a*b
double* div_ptr(const double* a,double b,size_t size);//a/b

double* sub_ptr(double a,const double* b,size_t size);//a-b
double* div_ptr(double a,const double* b,size_t size);//a*b

void add_ptr_self(double* a,const double* b,size_t size);//a+=b
void sub_ptr_self(double* a,const double* b,size_t size);//a-=b
void mul_ptr_self(double* a,const double* b,size_t size);//a*=b
void div_ptr_self(double* a,const double* b,size_t size);//a/=b

void add_ptr_self(double* a,double b,size_t size);//a+=b
void sub_ptr_self(double* a,double b,size_t size);//a-=b
void mul_ptr_self(double* a,double b,size_t size);//a*=b
void div_ptr_self(double* a,double b,size_t size);//a/=b

bool* eq_ptr(const double* a,const double* b,size_t size);//a==b
bool* neq_ptr(const double* a,const double* b,size_t size);//a!=b
bool* lt_ptr(const double* a,const double* b,size_t size);//a<b
bool* gt_ptr(const double* a,const double* b,size_t size);//a>b
bool* lteq_ptr(const double* a,const double* b,size_t size);//a<=b
bool* gteq_ptr(const double* a,const double* b,size_t size);//a>=b


bool* eq_ptr(const double* a,double b,size_t size);//a==b
bool* neq_ptr(const double* a,double b,size_t size);//a!=b
bool* lt_ptr(const double* a,double b,size_t size);//a<b
bool* gt_ptr(const double* a,double b,size_t size);//a>b
bool* lteq_ptr(const double* a,double b,size_t size);//a<=b
bool* gteq_ptr(const double* a,double b,size_t size);//a>=b
}
}