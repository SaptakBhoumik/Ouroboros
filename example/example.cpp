#include <ouroboros/ouroboros.hpp>

int main(){
    Ouroboros::Tensor t1=Ouroboros::CreateTensor::linspace({3,3},1.0,9.0);
    std::cout<<"t2:\n"<<t1<<"\n";
    Ouroboros::Tensor t2=Ouroboros::CreateTensor::linspace({3,3},9.0,1.0);
    std::cout<<"t2:\n"<<t2<<"\n";

    auto func=[](double x,double y)->double{
        return x*y;
    };
    Ouroboros::Tensor t3=Ouroboros::outer<func>(t1,t2);//Outer product of 2 tensors
    std::cout<<"t3:\n"<<t3<<"\n";

    auto sin=[](double x)->double{
        return std::sin(x);
    };
    t3=Ouroboros::transform<sin>(t3);//Apply sin to all elements of t1
    std::cout<<"sin(t3):\n"<<t3<<"\n";
    
    std::cout<<"t1+t2:\n"<<t1+t2<<"\n";//Element wise addition

    //There's a lot more you can do with tensors, check the documentation for more information
}