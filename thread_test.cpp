#include <omp.h>
#include <iostream>
#include <chrono>
#include <cmath>
#define __CHUNK_SIZE__  (65536)

double* neg_ptr_par1(const double* a,size_t size){
    auto result=new double[size];
    size_t loop_c=size/__CHUNK_SIZE__;
    size_t end=0;
    if(loop_c<=1){
        goto single_thread;
    }
    #pragma omp parallel for
    for(size_t i=0;i<loop_c;i++){
        size_t start=i*__CHUNK_SIZE__;
        end=(i+1)*__CHUNK_SIZE__;
        for(size_t j=start;j<end;++j){
            result[j]=std::sin(a[j]);
        }
    }
    single_thread:{}
    for(size_t i=end;i<size;i++){
        result[i]=std::sin(a[i]);
    }
    return result;
}
double* neg_ptr_par2(const double* a,size_t size){
    auto result=new double[size];
    size_t loop_c=size/__CHUNK_SIZE__;
    size_t end=0;
    if(loop_c<=1){
        goto single_thread;
    }
    #pragma omp parallel for
    for(size_t i=0;i<loop_c;i++){
        size_t start=i*__CHUNK_SIZE__;
        end=(i+1)*__CHUNK_SIZE__;
        #pragma omp simd
        for(size_t j=start;j<end;++j){
            result[j]=std::sin(a[j]);
        }
    }
    single_thread:{}
    for(size_t i=end;i<size;i++){
        result[i]=std::sin(a[i]);
    }
    return result;
}
double* neg_ptr_par3(const double* a,size_t size){
    auto result=new double[size];
    #pragma omp parallel for
    for(size_t i=0;i<size;i++){
        result[i]=std::sin(a[i]);
    }
    return result;
}
double* neg_ptr_simd(const double* a,size_t size){
    auto result=new double[size];
    #pragma omp simd
    for(size_t i=0;i<size;i++){
        result[i]=std::sin(a[i]);
    }
    return result;
}
double* neg_ptr_sin(const double* a,size_t size){
    auto result=new double[size];
    for(size_t i=0;i<size;i++){
        result[i]=std::sin(a[i]);
    }
    return result;
}
int main(){
    omp_set_num_threads(16);
    std::cout<<"No of threads: "<<8<<std::endl;
    std::cout<<"Enter the size of the array: ";
    size_t size;
    std::cin>>size;
    double* a=new double[size];
    
    auto start=std::chrono::high_resolution_clock::now();
    double* result=neg_ptr_par1(a,size);
    auto end=std::chrono::high_resolution_clock::now();
    std::cout<<"Time for parallel1: "<<std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count()<<"milliseconds"<<std::endl;
    delete[] result;

    start=std::chrono::high_resolution_clock::now();
    result=neg_ptr_par2(a,size);
    end=std::chrono::high_resolution_clock::now();
    std::cout<<"Time for parallel2: "<<std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count()<<"milliseconds"<<std::endl;
    delete[] result;

    start=std::chrono::high_resolution_clock::now();
    result=neg_ptr_par3(a,size);
    end=std::chrono::high_resolution_clock::now();
    std::cout<<"Time for parallel3: "<<std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count()<<"milliseconds"<<std::endl;
    delete[] result;

    start=std::chrono::high_resolution_clock::now();
    result=neg_ptr_simd(a,size);
    end=std::chrono::high_resolution_clock::now();
    std::cout<<"Time for simd: "<<std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count()<<"milliseconds"<<std::endl;
    std::cout<<result[0]<<std::endl;
    delete[] result;

    start=std::chrono::high_resolution_clock::now();
    result=neg_ptr_sin(a,size);
    end=std::chrono::high_resolution_clock::now();
    std::cout<<"Time for single thread: "<<std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count()<<"milliseconds"<<std::endl;
    std::cout<<result[0]<<std::endl;
    delete[] result;
}