#include "func/func.hpp"
double func(double a,double b,double c){
    return a*b+c;
}
int main(){
    Ouroboros::Tensor A({2, 2});
    Ouroboros::Tensor B({2, 2});

    A[{0, 0}] = 0;
    A[{0, 1}] = 1;
    A[{1, 0}] = 2;
    A[{1, 1}] = 3;

    B[0] = 0;
    B[1] = 1;
    B[2] = 2;
    B[3] = 3;

    Ouroboros::Tensor C = A + B;

    std::cout << A << std::endl;
    std::cout << B << std::endl;
    std::cout << C << std::endl;

    C.reshape({1, 2, 2});
    std::cout << C << std::endl;
    C.reshape({2, 2, 1});
    std::cout << C << std::endl;
    C.reshape({2, 1, 2});
    std::cout << C << std::endl;

    Ouroboros::Tensor t1({2, 3});
    Ouroboros::Tensor t2({3, 1});
    t1[{0, 0}] = 1;
    t1[{0, 1}] = 2;
    t1[{0, 2}] = 3;

    t1[{1, 0}] = 4;
    t1[{1, 1}] = 5;
    t1[{1, 2}] = 6;

    t2[{0, 0}] = 7;
    t2[{1, 0}] = 8;
    t2[{2, 0}] = 9;

    std::cout << t1 << std::endl;
    std::cout << t2 << std::endl;
    auto D = Ouroboros::matmul(t1, t2);
    std::cout << D << std::endl;
    D.reshape({2});
    std::cout << D << std::endl;
    t2.reshape({3});
    std::cout << matvecmul(t1, t2) << std::endl;

    D = Ouroboros::CreateTensor::rand({2, 2}, -1, 1);
    std::cout << D << std::endl;
    D = Ouroboros::CreateTensor::scalar_matrix(3, 2);
    std::cout << D << std::endl;
    D = Ouroboros::CreateTensor::diagonal_matrix({2, 1, 2});
    std::cout << D << std::endl;
    //Transform is used to apply a function to each element of a tensor
    D = Ouroboros::transform(Ouroboros::Scalar::sin,D);
    std::cout << D << std::endl;
    auto E = Ouroboros::CreateTensor::fill({2, 2}, 1);
    std::cout<<E<<std::endl;
    auto F = Ouroboros::CreateTensor::fill({2, 2}, 2);
    std::cout<<F<<std::endl;
    auto G = Ouroboros::CreateTensor::fill({2, 2}, 3);
    std::cout<<G<<std::endl;
    auto H = Ouroboros::transform(func,E,F,G);
    std::cout<<H<<std::endl;

    H=Ouroboros::sin(H);
    std::cout<<H<<std::endl;
    H=Ouroboros::pow(F,2);
    std::cout<<H<<std::endl;
    H=Ouroboros::pow(3,F);
    std::cout<<H<<std::endl;
    H=Ouroboros::pow(F,2.5*E);
    std::cout<<H<<std::endl;
    std::cout<<L1(E,F)<<std::endl;
    std::cout<<L(E,F,1)<<std::endl;
    std::cout<<L2(E,F)<<std::endl;
    std::cout<<L(E,F,2)<<std::endl;
    std::cout<<L3(E,F)<<std::endl;
    std::cout<<L(E,F,3)<<std::endl;
    std::cout<<L(E,F,4)<<std::endl;
    std::cout<<L(E,F,5)<<std::endl;

    //TODO:Test bool tensor
    A=Ouroboros::cos(A);
    std::cout << A << std::endl;
    std::cout << B << std::endl;
    std::cout << (A > B)<< std::endl;
    std::cout<<Ouroboros::max(A,B)<<std::endl;
    auto new_Tensor=Ouroboros::CreateTensor::where(A > B, A, B);//Basically max function
    std::cout << new_Tensor << std::endl;
    new_Tensor=Ouroboros::CreateTensor::where(A > B, 1, 0);
    std::cout << new_Tensor << std::endl;
}