#include <cmath>
#include <ouroboros/tensor.hpp>
#include <iostream>
int main(){
    //Transform i.e apply a function to every element of 1 or more tensors
    {
        Ouroboros::Tensor<double> t({2,2},2.0);
        std::cout<<"t:\n"<<t<<"\n";
        auto func=[](double x)->double{return x*x;};
        Ouroboros::Tensor<double> t2=Ouroboros::transform<func>(t);//Since func returns double, we get a Tensor
                                                           //If func returns bool, then we get a BoolTensor
        std::cout<<"t2:\n"<<t2<<"\n";
        auto func2=[](double x,double y)->double{return std::pow(x,y);};
        Ouroboros::Tensor<double> t3=Ouroboros::transform<func2>(t,t2);//Since func2 returns double, we get a Tensor
        std::cout<<"t3:\n"<<t3<<"\n";
        /*
        If func takes in n arguments then transform takes in n arguments(Tensor or BoolTensor or combination of both)
        If func is not a const expression then use Ouroboros::transform(func,t) 
        Like the following is not a const expression function so we use Ouroboros::transform(func,t) 
        auto func=[a](double x)->double{return a*x;};
        This method applies for transform,reduce,accumulate,outer,at,broadcast,
        2 method are provided because I prefer to use transform<func>(t) instead of transform(func,t) but it is not always possible for every function
        */
    }
    // Reduce i.e apply a function along an axis of a tensor and reduce the dimension of the tensor
    {
        Ouroboros::Tensor<double> t({2,2});
        for(std::size_t i=0;i<t.size();i++){
            t[i]=i+1;
        }
        std::cout<<"t:\n"<<t<<"\n";
        auto func=[](Ouroboros::IdxIterator it,const double* tensor)->double{
            //Used for sum reduction
            double result=0;
            for(auto it1=it.begin();it1!=it.end();++it1){
                result+=tensor[*it1];
            }
            return result;
        };
        Ouroboros::Tensor<double> t2=Ouroboros::reduce<func>(0,t);//Reduce along the 0th axis
        std::cout<<"t2:\n"<<t2<<"\n";
        Ouroboros::Tensor<double> t3=Ouroboros::reduce<func>(1,t);//Reduce along the 1st axis
        std::cout<<"t3:\n"<<t3<<"\n";
    }
    //Accumulate
    {
        Ouroboros::Tensor<double> t({2,2});
        for(std::size_t i=0;i<t.size();i++){
            t[i]=i+1;
        }
        std::cout<<"t:\n"<<t<<"\n";
        auto func=[](double innitial,double x)->double{
            //Used for sum accumulation
            return innitial+x;
        };//Innitial is the previous accumulated value
        Ouroboros::Tensor<double> t2=Ouroboros::accumulate<func>(t);//Accumulate along the 0th axis and innitial is 0
        std::cout<<"t2:\n"<<t2<<"\n";
        Ouroboros::Tensor<double> t3=Ouroboros::accumulate<func>(t,1);//Accumulate along the 1st axis and innitial is 0
        std::cout<<"t3:\n"<<t3<<"\n";
        auto func2=[](double innitial,double x)->double{
            //Used for product accumulation
            return innitial*x;
        };
        Ouroboros::Tensor<double> t4=Ouroboros::accumulate<func2>(t,0,1.0);//Accumulate along 0 and then 1 axis and innitial is 1
        std::cout<<"t4:\n"<<t4<<"\n";
    }

    //Outer
    {
        /*
        Maths behind outer function:
        R(i0,i1,...,in,j0,j1,...,jm)=func(A(i0,i1,...,in),B(j0,j1,...,jm))
        Where (i0,i1,...,in) is every possible index of A and (j0,j1,...,jm) is every possible index of B
        A,B are input tensors
        R is the output tensor
        */

        Ouroboros::Tensor<double> t({2,2});
        Ouroboros::Tensor<double> t2({3,5});
        for(std::size_t i=0;i<t.size();i++){
            t[i]=i+1;
        }
        for(std::size_t i=0;i<t2.size();i++){
            t2[i]=i+1;
        }
        std::cout<<"t:\n"<<t<<"\n";
        std::cout<<"t2:\n"<<t2<<"\n";
        auto func=[](double x,double y)->double{
            //Used for outer product of 2 tensors
            return x*y;
        };
        Ouroboros::Tensor<double> t3=Ouroboros::outer<func>(t,t2);

        std::cout<<"t3:\n"<<t3<<"\n";
    }
    // //At i.e apply a function to a slice of a tensor
    // {
    //     Ouroboros::Tensor<double> t1({2,3});
    //     for(std::size_t i=0;i<t1.size();i++){
    //         t1[i]=i+1;
    //     }
    //     std::cout<<"t1:\n"<<t1<<"\n";
    //     auto func=[](double x)->double{
    //         return std::sin(x);
    //     };
    //     std::vector<size_t> from={0,0};
    //     std::vector<size_t> to={1,2};
    //     Ouroboros::Tensor<double> t2=Ouroboros::at<func>(from,to,{1,1},t1);
    //     std::cout<<"t2:\n"<<t2<<"\n";
    //     /*
    //     If `func` takes in n arguments then `at` takes in n arguments(Tensor or BoolTensor or combination of both) excluding `to` and `from`
    //     If `T` is the type of the first argument of `func` then the return type of `at` is `T`
    //     The `func` is applied to the slice of t1 from `from` to `to` 
    //     The shape of the tensor(if any) other than t1 must be same as from-to
    //     */
    // }
    //broadcast:-If shapes mismatch but u still want to apply a function elementwise. Similar to how numpy broadcasts
    {
        Ouroboros::Tensor<double> t1({2,2});
        Ouroboros::Tensor<double> t2({2,1});
        for(std::size_t i=0;i<t1.size();i++){
            t1[i]=i+1;
        }
        for(std::size_t i=0;i<t2.size();i++){
            t2[i]=i+1;
        }
        std::cout<<"t1:\n"<<t1<<"\n";
        std::cout<<"t2:\n"<<t2<<"\n";
        auto func=[](double x,double y)->double{
            return x/y;
        };
        Ouroboros::Tensor<double> t3=Ouroboros::broadcast<func>(t1,t2);
        std::cout<<"t3:\n"<<t3<<"\n";
        Ouroboros::Tensor<double> t4=Ouroboros::broadcast<func>(t2,t1);
        std::cout<<"t4:\n"<<t4<<"\n";
    }
    //Concatenation of tensor
    {
        Ouroboros::Tensor<double> t1({2,2});
        Ouroboros::Tensor<double> t2({1,2});
        Ouroboros::Tensor<double> t3({3,2});
        for(std::size_t i=0;i<t1.size();i++){
            t1[i]=i+1;
        }
        for(std::size_t i=0;i<t2.size();i++){
            t2[i]=i+1;
        }
        for(std::size_t i=0;i<t3.size();i++){
            t3[i]=i+1;
        }
        std::cout<<"t1:\n"<<t1<<"\n";
        std::cout<<"t2:\n"<<t2<<"\n";
        std::cout<<"t3:\n"<<t3<<"\n";
        Ouroboros::Tensor<double> t4=Ouroboros::concat(0,t1,t2,t3);//Concatenate along 0th axis
        std::cout<<"t3:\n"<<t4<<"\n";
        /*
        You can pass as many tensors as you want
        They will be concatenated along the axis you specify
        The shapes of the tensors must be same except along the axis you are concatenating(it may or may not be same)
        */
    }
    //Flip:-If it takes a Tensor then it returns a Tensor else it returns a BoolTensor
    {
        Ouroboros::Tensor<double> t1({2,2});
        for(std::size_t i=0;i<t1.size();i++){
            t1[i]=i+1;
        }
        std::cout<<"t1:\n"<<t1<<"\n";
        Ouroboros::Tensor<double> t2=Ouroboros::flip(t1,0);//Flip along 0th axis
        std::cout<<"t2:\n"<<t2<<"\n";
        Ouroboros::Tensor<double> t3=Ouroboros::flip(t1,1);//Flip along 1st axis
        std::cout<<"t3:\n"<<t3<<"\n";
    }
    //Transpose i.e swap the axes of a tensor
    {
        Ouroboros::Tensor<double> t1({2,2,2});
        for(std::size_t i=0;i<t1.size();i++){
            t1[i]=i+1;
        }
        std::cout<<"t1:\n"<<t1<<"\n";
        Ouroboros::Tensor<double> t2=Ouroboros::transpose(t1,1,2);//Default is 0,1
        std::cout<<"t2:\n"<<t2<<"\n";
    }
}