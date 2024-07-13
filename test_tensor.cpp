#include "func.hpp"
#include "utils.hpp"
double func(double a,double b,double c){
    return a*b+c;
}
double acc(Ouroboros::Utils::Iterator<double> a){
    double sum=0;
    for(auto x:a){
        sum+=x;
    }
    return sum;
}
double cum_sum(double a,double b){
    return a+b;
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
    auto sin=[](double x)->double{return std::sin(x);};
    D = Ouroboros::transform<sin>(D);
    std::cout << D << std::endl;
    auto bool_test=[](double x)->bool{return x>0.9;};
    auto B__=Ouroboros::transform<bool_test>(D);
    std::cout << B__ << std::endl;
    auto E = Ouroboros::CreateTensor::fill({2, 2}, 1);
    std::cout<<E<<std::endl;
    auto F = Ouroboros::CreateTensor::fill({2, 2}, 2);
    std::cout<<F<<std::endl;
    auto G = Ouroboros::CreateTensor::fill({2, 2}, 3);
    std::cout<<G<<std::endl;
    auto H = Ouroboros::transform<func>(E,F,G);
    std::cout<<H<<std::endl;

    auto new_Tensor=Ouroboros::CreateTensor::where(A > B, A, B);//Basically max function
    std::cout << new_Tensor << std::endl;
    new_Tensor=Ouroboros::CreateTensor::where(A > B, 1, 0);
    std::cout << new_Tensor << std::endl;
    auto test=Ouroboros::CreateTensor::linspace({2,3,4},0,23);
    std::cout << test << std::endl;
    std::cout << test.slice({0, 1, 1},{2,3,4}); 
    std::fstream file;
    file.open("test.bin",std::ios::out|std::ios::binary);
    std::cout<<"Writing to file\n";
    Ouroboros::Utils::write_bin_tensor(file,test,Ouroboros::Utils::TensorType::FLOAT);
    std::cout << B__ << std::endl;
    Ouroboros::Utils::write_bin_bool_tensor(file,B__);
    file.close();
    std::cout<<"Reading from file\n";
    file.open("test.bin",std::ios::in|std::ios::binary);
    auto test2=Ouroboros::Utils::read_bin_tensor(file);
    std::cout << test2 << std::endl;
    auto B__2=Ouroboros::Utils::read_bin_bool_tensor(file);
    std::cout << B__2 << std::endl;
    file.close();
    auto t_t=Ouroboros::reduce<acc>(test,2);
    std::cout << t_t << std::endl;
    t_t=Ouroboros::reduce<acc>(test,1);
    std::cout << t_t << std::endl;
    t_t=Ouroboros::reduce<acc>(test,0);
    std::cout << t_t << std::endl;
    auto func=[](double a,bool b)->double{return a;};
    auto t_t2=Ouroboros::transform<func>(test,test==test);
    std::cout << t_t2 << std::endl;
    auto t_t3=Ouroboros::accumulate<cum_sum>(test,2);
    std::cout<<t_t3<<std::endl;
    t_t3=Ouroboros::accumulate<cum_sum>(test,1);
    std::cout<<t_t3<<std::endl;
    t_t3=Ouroboros::accumulate<cum_sum>(test,0);
    std::cout<<t_t3<<std::endl;
    t1=Ouroboros::CreateTensor::linspace({2,3},1,6);
    t2=Ouroboros::CreateTensor::linspace({4},1,4);
    auto dot=[](double a,double b)->double{return a*b;};
    auto t3=Ouroboros::outer<dot>(t1,t2);
    std::cout<<t3<<std::endl;
    t1=Ouroboros::CreateTensor::linspace({4,1},1,4);
    std::cout<<t1<<std::endl;
    t2=Ouroboros::CreateTensor::linspace({2,1},1,2);
    std::cout<<t2<<std::endl;
    t3=Ouroboros::at<dot>(t1,{0,0},{2,1},t2);
    std::cout<<t3<<std::endl;
    auto test_f=[](double a)->double{return a*a;};
    t3=Ouroboros::at<test_f>(t1,{1,0},{3,1});
    std::cout<<t3<<std::endl;

    t1=Ouroboros::CreateTensor::linspace({2,2},1,4);
    std::cout<<t1<<std::endl;
    t2=Ouroboros::CreateTensor::linspace({2,1},1,2);
    std::cout<<t2<<std::endl;
    auto func2=[](double a,double b)->double{return a/b;};
    t3=Ouroboros::broadcast<func2>(t1,t2);
    std::cout<<t3<<std::endl;
    t3=Ouroboros::broadcast<func2>(t2,t1);
    std::cout<<t3<<std::endl;
    {
        std::size_t g=0;
        auto func3=[g](double a,double b)->double{return a+b+g;};
        auto t=Ouroboros::CreateTensor::linspace({2,2},1,4);
        auto t3=Ouroboros::transform(func3,t,t);
        t3=Ouroboros::reduce(acc,t,{0,1});
        std::cout<<t3<<std::endl;
    }
    t1=Ouroboros::CreateTensor::linspace({2,2},1,4);
    std::cout<<t1<<std::endl;
    t2=Ouroboros::CreateTensor::linspace({2,1},1,2);
    std::cout<<t2<<std::endl;
    t3=Ouroboros::concat(1,t1,t2);
    std::cout<<t3<<std::endl;
    t3=Ouroboros::flip(t1,1);
    std::cout<<t3<<std::endl;
    t3=Ouroboros::flip(t1,0);
    std::cout<<t3<<std::endl;
    std::cout<<Ouroboros::transpose(t1)<<std::endl;
    t1=Ouroboros::CreateTensor::linspace({2,2,2},1,8);
    std::cout<<t1<<std::endl;
    std::cout<<Ouroboros::transpose(t1,1,2)<<std::endl;
}
