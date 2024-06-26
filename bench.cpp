#include "func/func.hpp"
#include <chrono>

Ouroboros::Tensor my_impl(const Ouroboros::Tensor& A){
    return Ouroboros::transform<Ouroboros::sin>(A);
}
Ouroboros::Tensor threaded_impl(const Ouroboros::Tensor& A){
    Ouroboros::Tensor res(A.shape());
    size_t count=A.shape().count();
    double* data=res.data();
    #pragma omp parallel for
    for(size_t i=0;i<count;i++){
        data[i]=Ouroboros::sin(A.data()[i]);
    }
    return res;
}
Ouroboros::Tensor non_threaded_impl(const Ouroboros::Tensor& A){
    Ouroboros::Tensor res(A.shape());
    size_t count=A.shape().count();
    double* data=res.data();
    for(size_t i=0;i<count;i++){
        data[i]=Ouroboros::sin(A.data()[i]);
    }
    return res;
}
int main(){
    Ouroboros::Tensor A({100000000});
    Ouroboros::Tensor res({100000000});
    auto start = std::chrono::high_resolution_clock::now();
    res=my_impl(A);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "My implementation: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    start = std::chrono::high_resolution_clock::now();
    res=threaded_impl(A);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Threaded implementation: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    start = std::chrono::high_resolution_clock::now();
    res=non_threaded_impl(A);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Non-threaded implementation: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl; 
    A=Ouroboros::Tensor({200000});
    res=Ouroboros::Tensor({200000});
    start = std::chrono::high_resolution_clock::now();
    res=my_impl(A);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "My implementation: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    start = std::chrono::high_resolution_clock::now();
    res=threaded_impl(A);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Threaded implementation: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    start = std::chrono::high_resolution_clock::now();
    res=non_threaded_impl(A);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Non-threaded implementation: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl; 
}