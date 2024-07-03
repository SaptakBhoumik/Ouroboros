#include <ouroboros/ouroboros.hpp>
#include <iostream>
int main(){
    //Transform i.e apply a function to every element of 1 or more tensors
    {
        Ouroboros::Tensor t=Ouroboros::CreateTensor::fill({2,2},2.0);
        std::cout<<"t:\n"<<t<<"\n";
        auto func=[](double x)->double{return x*x;};
        Ouroboros::Tensor t2=Ouroboros::transform<func>(t);//Since func returns double, we get a Tensor
                                                           //If func returns bool, then we get a BoolTensor
        std::cout<<"t2:\n"<<t2<<"\n";
        auto func2=[](double x,double y)->double{return std::pow(x,y);};
        Ouroboros::Tensor t3=Ouroboros::transform<func2>(t,t2);//Since func2 returns double, we get a Tensor
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
    //Reduce i.e apply a function along an axis of a tensor and reduce the dimension of the tensor
    {
        Ouroboros::Tensor t=Ouroboros::CreateTensor::linspace({2,2},1.0,4.0);
        std::cout<<"t:\n"<<t<<"\n";
        auto func=[](Ouroboros::Utils::Iterator<double> it){
            //Used for sum reduction
            double sum=0;
            for(auto x:it){
                sum+=x;
            }
            return sum;
        };
        Ouroboros::Tensor t2=Ouroboros::reduce<func>(t);//Reduce along the 0th axis
        std::cout<<"t2:\n"<<t2<<"\n";
        Ouroboros::Tensor t3=Ouroboros::reduce<func>(t,1);//Reduce along the 1st axis
        std::cout<<"t3:\n"<<t3<<"\n";
        Ouroboros::Tensor t4=Ouroboros::reduce<func>(t,std::vector<size_t>{});//Reduce the whole tensor to get tensor with single element 
                                                                              //Shape is {1}
        std::cout<<"t4:\n"<<t4<<"\n";
        Ouroboros::Tensor t5=Ouroboros::reduce<func>(t,{0,1});//Reduce along 0 and then 1 axis
        std::cout<<"t5:\n"<<t5<<"\n";
    }
    //Accumulate
    {
        Ouroboros::Tensor t=Ouroboros::CreateTensor::linspace({2,2},1.0,4.0);
        std::cout<<"t:\n"<<t<<"\n";
        auto func=[](double innitial,double x)->double{
            //Used for sum accumulation
            return innitial+x;
        };//Innitial is the previous accumulated value
        Ouroboros::Tensor t2=Ouroboros::accumulate<func>(t);//Accumulate along the 0th axis and innitial is 0
        std::cout<<"t2:\n"<<t2<<"\n";
        Ouroboros::Tensor t3=Ouroboros::accumulate<func>(t,1);//Accumulate along the 1st axis and innitial is 0
        std::cout<<"t3:\n"<<t3<<"\n";
        auto func2=[](double innitial,double x)->double{
            //Used for product accumulation
            return innitial*x;
        };
        Ouroboros::Tensor t4=Ouroboros::accumulate<func2>(t,0,1.0);//Accumulate along 0 and then 1 axis and innitial is 1
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

        Ouroboros::Tensor t=Ouroboros::CreateTensor::linspace({2,2},1.0,4.0);
        Ouroboros::Tensor t2=Ouroboros::CreateTensor::linspace({3,5},1.0,15.0);
        std::cout<<"t:\n"<<t<<"\n";
        std::cout<<"t2:\n"<<t2<<"\n";
        auto func=[](double x,double y)->double{
            //Used for outer product of 2 tensors
            return x*y;
        };
        Ouroboros::Tensor t3=Ouroboros::outer<func>(t,t2);

        std::cout<<"t3:\n"<<t3<<"\n";
    }
    //At i.e apply a function to a slice of a tensor
    {
        Ouroboros::Tensor t1=Ouroboros::CreateTensor::linspace({2,2},1.0,4.0);
        Ouroboros::Tensor t2=Ouroboros::CreateTensor::linspace({1,2},1.0,2.0);
        std::cout<<"t1:\n"<<t1<<"\n";
        std::cout<<"t2:\n"<<t2<<"\n";
        auto func=[](double x,double y)->double{
            return x*y;
        };
        Ouroboros::Shape from={0,0};
        Ouroboros::Shape to={1,2};
        Ouroboros::Tensor t3=Ouroboros::at<func>(t1,from,to,t2);
        std::cout<<"t3:\n"<<t3<<"\n";
        /*
        If `func` takes in n arguments then `at` takes in n arguments(Tensor or BoolTensor or combination of both) excluding `to` and `from`
        If `T` is the type of the first argument of `func` then the return type of `at` is `T`
        The `func` is applied to the slice of t1 from `from` to `to` 
        The shape of the tensor(if any) other than t1 must be same as from-to
        */
    }
    //broadcast:-If shapes mismatch but u still want to apply a function elementwise. Similar to how numpy broadcasts
    {
        Ouroboros::Tensor t1=Ouroboros::CreateTensor::linspace({2,2},1.0,4.0);
        Ouroboros::Tensor t2=Ouroboros::CreateTensor::linspace({2,1},1.0,2.0);
        std::cout<<"t1:\n"<<t1<<"\n";
        std::cout<<"t2:\n"<<t2<<"\n";
        auto func=[](double x,double y)->double{
            return x/y;
        };
        Ouroboros::Tensor t3=Ouroboros::broadcast<func>(t1,t2);
        std::cout<<"t3:\n"<<t3<<"\n";
        Ouroboros::Tensor t4=Ouroboros::broadcast<func>(t2,t1);
        std::cout<<"t4:\n"<<t4<<"\n";
    }
    //Concatenation of tensor
    {
        Ouroboros::Tensor t1=Ouroboros::CreateTensor::linspace({2,2},1.0,4.0);
        Ouroboros::Tensor t2=Ouroboros::CreateTensor::linspace({1,2},2.0,4.0);
        Ouroboros::Tensor t3=Ouroboros::CreateTensor::linspace({3,2},1.0,6.0);
        std::cout<<"t1:\n"<<t1<<"\n";
        std::cout<<"t2:\n"<<t2<<"\n";
        std::cout<<"t3:\n"<<t3<<"\n";
        Ouroboros::Tensor t4=Ouroboros::concat(0,t1,t2,t3);//Concatenate along 0th axis
        std::cout<<"t3:\n"<<t4<<"\n";
        /*
        You can pass as many tensors as you want
        They will be concatenated along the axis you specify
        The shapes of the tensors must be same except along the axis you are concatenating(it may or may not be same)
        */
    }
    //Flip:-If it takes a Tensor then it returns a Tensor else it returns a BoolTensor
    {
        Ouroboros::Tensor t1=Ouroboros::CreateTensor::linspace({2,2},1.0,4.0);
        std::cout<<"t1:\n"<<t1<<"\n";
        Ouroboros::Tensor t2=Ouroboros::flip(t1,0);//Flip along 0th axis
        std::cout<<"t2:\n"<<t2<<"\n";
        Ouroboros::Tensor t3=Ouroboros::flip(t1,1);//Flip along 1st axis
        std::cout<<"t3:\n"<<t3<<"\n";
        Ouroboros::Tensor t4=Ouroboros::flip(t1,{0,1});//Flip along 0st and then 1th axis
        std::cout<<"t4:\n"<<t4<<"\n";
        Ouroboros::Tensor t5=Ouroboros::flip(t1,std::vector<size_t>{});//Flip the whole data otr
        std::cout<<"t5:\n"<<t5<<"\n";
    }
    //Transpose i.e swap the axes of a tensor
    {
        Ouroboros::Tensor t1=Ouroboros::CreateTensor::linspace({2,2,2},1.0,8.0);
        std::cout<<"t1:\n"<<t1<<"\n";
        Ouroboros::Tensor t2=Ouroboros::transpose(t1,1,2);//Default is 0,1
        std::cout<<"t2:\n"<<t2<<"\n";
    }
    //Norm
    {
        Ouroboros::Tensor t1=Ouroboros::CreateTensor::linspace({2,2,2},1.0,8.0);
        Ouroboros::Tensor t2=Ouroboros::norm(t1,2);//Calculate norm along axis 2
        std::cout<<"t2:\n"<<t2<<"\n";
        Ouroboros::Tensor t3=Ouroboros::norm(t1,{2,1});//Calculate norm along axis 2 and then 1
        std::cout<<"t3:\n"<<t3<<"\n";
        /*
        norm2 is just the square of norm
        */
        t2=Ouroboros::norm2(t1,2);//Calculate norm 2along axis 2
        std::cout<<"t2:\n"<<t2<<"\n";
        t3=Ouroboros::norm2(t1,{2,1});//Calculate norm2 along axis 2 and then 1
        std::cout<<"t3:\n"<<t3<<"\n";

        Ouroboros::Tensor t4=Ouroboros::normalize(t1,2);//Normalize along axis 2
        std::cout<<"t4:\n"<<t4<<"\n";
        Ouroboros::Tensor t5=Ouroboros::normalize(t1,{2,1});//Normalize along axis 2 and then 1
        std::cout<<"t5:\n"<<t5<<"\n";
        Ouroboros::Tensor t6=Ouroboros::normalize(t1);//Normalize along all axis
        std::cout<<"t6:\n"<<t6<<"\n";
    }
}