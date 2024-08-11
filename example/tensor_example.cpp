#include <ouroboros/ouroboros.hpp>
#include <iostream>
int main(){
    Ouroboros::Shape shape={2,3,4};
    //Innitialize a Tensor of shape 2x3x4 with all values as 1
    Ouroboros::Tensor t1(shape);//We dont specify the value so only the data is allocated but not initialized with any value
    Ouroboros::Tensor t2(shape,1.0);//We specify the value so the data is allocated and initialized with the value 1
    std::cout<<"t1="<<t1<<std::endl;
    std::cout<<"t2="<<t2<<std::endl;
    double* data=new double[shape.count()];//It has to be heap allocated
    for(std::size_t i=0;i<shape.count();i++){
        data[i]=i;
    }
    //This method is useful when you have a preallocated array and you want to use it as the data for the tensor
    //But in general u should avoid it
    Ouroboros::Tensor t3(shape,data);//Note that the data is shared and not copied so the user should not use the data afterwards cuz we take ownership of the data
    std::cout<<"t3="<<t3<<std::endl;
    //Copy constructor
    Ouroboros::Tensor t4(t3);
    std::cout<<"t4="<<t4<<std::endl;
    //Move constructor
    Ouroboros::Tensor t5(std::move(t4));
    std::cout<<"t5="<<t5<<std::endl;
    //Other metho to create a tensor
    t5=Ouroboros::CreateTensor::ones(shape);//All the item are 1
    std::cout<<"t5="<<t5<<std::endl;
    t5=Ouroboros::CreateTensor::zeros(shape);//All the item are 0
    std::cout<<"t5="<<t5<<std::endl;
    t5=Ouroboros::CreateTensor::rand(shape);//All the item are random between 0 and 1
    std::cout<<"t5="<<t5<<std::endl;
    t5=Ouroboros::CreateTensor::rand(shape,-2,2);//All the item are random between -2 and 2
    std::cout<<"t5="<<t5<<std::endl;
    t5=Ouroboros::CreateTensor::rand(shape,-2);//All the item are random between -2 and 1
    std::cout<<"t5="<<t5<<std::endl;
    t5=Ouroboros::CreateTensor::fill(shape,4);//All the item are 4
    std::cout<<"t5="<<t5<<std::endl;
    t5=Ouroboros::CreateTensor::fill(shape,[]()->double{return 0;});//Use a lambda function to fill the tensor
    std::cout<<"t5="<<t5<<std::endl;
    t5=Ouroboros::CreateTensor::linspace(shape,0,10);//Create a tensor with with shape and values from 0 to 10
    std::cout<<"t5="<<t5<<std::endl;
    t5=Ouroboros::CreateTensor::logspace(shape,0,10);//Create a tensor with with shape and values from 10^0,10^1,10^2,...,10^10
    std::cout<<"t5="<<t5<<std::endl;
    t5=Ouroboros::CreateTensor::logspace(shape,0,10,2);//Create a tensor with with shape and values from 2^0,2^1,2^2,...,2^10	
    std::cout<<"t5="<<t5<<std::endl;
    t5=Ouroboros::CreateTensor::scalar_matrix(7);//Create a scalar matrix with shape 7x7 with all values 0 except the diagonal which are 1
    std::cout<<"t5="<<t5<<std::endl;
    t5=Ouroboros::CreateTensor::scalar_matrix(7,2);//Create a scalar matrix with shape 7x7 with all values 2 except the diagonal which are 1
    std::cout<<"t5="<<t5<<std::endl;
    t5=Ouroboros::CreateTensor::diagonal_matrix({1,2,3,4});//Create a diagonal matrix with shape 4x4 with diagonal values 1,2,3,4
    std::cout<<"t5="<<t5<<std::endl;
    //If condition is true then x else y
    Ouroboros::BoolTensor condition(shape,true);
    condition[{0,0,0}]=false;
    //Create a tensor with shape 2x3x4 with all values 1 if the condition is true and 2 if the condition is false at the same index
    t5=Ouroboros::CreateTensor::where(condition,1.0,2.0);
    std::cout<<"5="<<t5<<std::endl;
    /*
    Note:- The following methods are also available
    Ouroboros::CreateTensor::where(condition,tensor1,value)
    Ouroboros::CreateTensor::where(condition,value,tensor1)
    Ouroboros::CreateTensor::where(condition,tensor1,tensor2)
    Similar to numpy.where in python
    */
    //Assignment operator
    t5=t1;
    std::cout<<"t5="<<t5<<std::endl;
    //Move assignment operator
    t5=std::move(t1);
    std::cout<<"t5="<<t5<<std::endl;
    //Reshape the tensor
    t5.reshape({1,2,3,2,2});
    std::cout<<"t5="<<t5<<std::endl;
    t5.flatten();
    std::cout<<"t5="<<t5<<std::endl; 
    //Modify the tensor
    t5.fill(5);//Fill the tensor with 5
    std::cout<<"t5="<<t5<<std::endl;  
    t5.fill([]()->double{return 50;});//Use a lambda function to fill the tensor
    std::cout<<"t5="<<t5<<std::endl;

    t5.zeros();//Fill the tensor with 0
    std::cout<<"t5="<<t5<<std::endl;
    t5.ones();//Fill the tensor with 1
    std::cout<<"t5="<<t5<<std::endl;
    
    t5.rand();//Fill the tensor with random values between 0 and 1
    std::cout<<"t5="<<t5<<std::endl;                                    
    t5.rand(-2,2);//Fill the tensor with random values between -2 and 2
    std::cout<<"t5="<<t5<<std::endl;
    t5.rand(-2);//Fill the tensor with random values between -2 and 1
    std::cout<<"t5="<<t5<<std::endl;

    t5.clean();//Fills it with 0 if the value is less than 0
    std::cout<<"t5="<<t5<<std::endl;
    t5.clean(2);//Fills it with 0 if the value is less than 2
    std::cout<<"t5="<<t5<<std::endl;
    t5.clean(4,1);//Fills it with 1 if the value is less than 4

    t5.rand(-10,10);

    t5.clamp(2,4);//Clips the values between 2 and 4 i.e if the value is less than 2 then it is set to 2 and if the value is greater than 4 then it is set to 4
    std::cout<<"t5="<<t5<<std::endl;
    t5.clamp(1,3,5);//If the value is less than 1 then it is set to 1 and if the value is greater than 3 then it is set to 3 else it is set to 5
    std::cout<<"t5="<<t5<<std::endl;

    t5.rand(-10,10);
    t5.threshold(2);//If the value is less than 2 then it is set to 0 
    std::cout<<"t5="<<t5<<std::endl;
    t5.threshold(4,5);//If the value is less than 4 then it is set to 5
    std::cout<<"t5="<<t5<<std::endl;

    t5.replace(5,10);//Replace all the values 5 with 10
    std::cout<<"t5="<<t5<<std::endl;

    t5.reshape({2,3,4});
    /*
    Few other useful methods of the tensor class:-
    void fill_nan(double value=0.0);//Fill all the nan values with the value
    void fill_inf(double value=0.0);//Fill all the inf values with the value
    void fill_neg_inf(double value=0.0);//Fill all the -inf values with the value

    void fill_nan_inf(double value=0.0);//Fill all the nan and inf values with the value
    void fill_nan_neg_inf(double value=0.0);//Fill all the nan and -inf values with the value
    void fill_inf_neg_inf(double value=0.0);//Fill all the inf and -inf values with the value

    void fill_nan_inf_neg_inf(double value=0.0);//Fill all the nan,inf and -inf values with the value

    bool is_zero()const;//Returns true if all the values are 0
    bool is_finite()const;//Returns true if all the values are finite
    bool has_nan()const;//Returns true if there is any nan value
    */

    //Indexing
    std::cout<<"t5[{0,0,0}]="<<t5[{0,0,0}]<<std::endl;
    t5[{0,0,0}]=10;//Set the value at index 0,0,0 to 10
    std::cout<<"t5[{0,0,0}]="<<t5[{0,0,0}]<<std::endl;

    std::cout<<"t5[1]="<<t5[1]<<std::endl;
    t5[1]=20;//Set the value at offset 1 of the tensor data to 20
    std::cout<<"t5[1]="<<t5[1]<<std::endl;

    std::cout<<"Offset of {1,0,0} in data of t5="<<t5.offset({1,0,0})<<std::endl;
    
    //Slicing
    t5=Ouroboros::CreateTensor::linspace({2,3,4},0,23);
    std::cout<<"t5="<<t5<<std::endl;
    std::vector<size_t> start={0,0,0};
    std::vector<size_t> end={1,2,3};
    std::vector<size_t> step={1,2,1};
    std::cout<<"Slice from start to end with step="<<t5.slice(start,end,step)<<std::endl;
    /*
    t5.slice(start,end) is same as t5.slice(start,end,{1,1,..})
    t5.slice(start,end,2) is same as t5.slice(start,end,{2,2,..})
    */

    //Get the shape of the tensor
    std::cout<<"Shape of t5="<<t5.shape()<<std::endl;
    //Get the strides of the tensor
    std::cout<<"Strides of t5="<<t5.strides()<<std::endl;
    //Get the count of the tensor i.e. the number of elements in the tensor
    std::cout<<"Count of t5="<<t5.count()<<std::endl;
    //Get the dimension of the tensor
    std::cout<<"Dimension of t5="<<t5.dim()<<std::endl;
    //Get the tensor data ptr
    double* ptr=t5.data();

    //Norm of the tensor
    std::cout<<"Norm of t5="<<t5.norm()<<std::endl;//sqrt(sum(xi^2)) where xi is the ith element of the tensor data ptr
    //Norm2 of the tensor
    std::cout<<"Norm2 of t5="<<t5.norm2()<<std::endl;//sum(xi^2) where xi is the ith element of the tensor data ptr
    /*
    if tensor=[1,2,3] then norm = 1*1+2*2+3*3=14 and norm2=sqrt(14)=3.74
    */
    //Sum of the tensor elements
    std::cout<<"Sum of t5="<<t5.sum()<<std::endl;//(x1+x2+x3....) where xi is the ith element of the tensor data ptr
    //Product of the tensor elements
    std::cout<<"Product of t5="<<t5.prod()<<std::endl;//(x1*x2*x3...) where xi is the ith element of the tensor data ptr
    //Mean of the tensor elements
    std::cout<<"Mean of t5="<<t5.mean()<<std::endl;//(x1+x2+x3....)/count where xi is the ith element of the tensor data ptr

    //Max of the tensor elements
    std::cout<<"Max of t5="<<t5.max()<<std::endl;//max(x1,x2,x3...) where xi is the ith element of the tensor data ptr
    //Min of the tensor elements
    std::cout<<"Min of t5="<<t5.min()<<std::endl;//min(x1,x2,x3...)  where xi is the ith element of the tensor data ptr

    //Max index of the tensor elements
    std::pair<double,std::size_t> max_index=t5.max_index();
    std::cout<<"Max of t5="<<max_index.first<<" Offset where it was first found="<<max_index.second<<std::endl;//max(x1,x2,x3...) where xi is the ith element of the tensor data ptr
    //Min index of the tensor elements
    std::pair<double,std::size_t> min_index=t5.min_index();
    std::cout<<"Min of t5="<<min_index.first<<" Offset where it was first found="<<min_index.second<<std::endl;//min(x1,x2,x3...)  where xi is the ith element of the tensor data ptr

    //Tensor operations
    Ouroboros::Tensor t6=Ouroboros::CreateTensor::linspace({2,3,4},0,23);
    Ouroboros::Tensor t7=Ouroboros::CreateTensor::linspace({2,3,4},-23,0);
    Ouroboros::Tensor t8=Ouroboros::CreateTensor::linspace({2,3,4},23,0);

    std::cout<<"t6="<<t6<<std::endl;
    std::cout<<"t7="<<t7<<std::endl;    
    std::cout<<"t8="<<t8<<std::endl;    

    std::cout<<"t6+t7="<<t6+t7<<std::endl;
    std::cout<<"t6-t7="<<t6-t7<<std::endl;
    std::cout<<"t6*t7="<<t6*t7<<std::endl;
    std::cout<<"t6/t7="<<t6/t7<<std::endl;

    /*
    You can also do operations with a scalar i.e Tensor+double and double+Tensor are both valid
    Goes for all the operations
    */
    t6+=t7;//-=,*=,/= are also valid and += double, -= double, *= double, /= double are also valid
    std::cout<<"t6="<<t6<<std::endl;
    //Element wise comparison
    Ouroboros::BoolTensor bool_tensor=t6>t8;
    std::cout<<"t6>t8="<<bool_tensor<<std::endl;
    /*
    ==,!=,<,>,<=,>= are also valid
    tensor == double,tensor != double,tensor < double,tensor > double,tensor <= double,tensor >= double are also also valid
    double == tensor,double != tensor,double < tensor,double > tensor,double <= tensor,double >= tensor are also also valid
    */

    //Matrix
    Ouroboros::Tensor mat1=Ouroboros::CreateTensor::linspace({2,3},0,6);
    Ouroboros::Tensor mat2=Ouroboros::CreateTensor::linspace({3,4},0,12);
    std::cout<<"mat1="<<mat1<<std::endl;
    std::cout<<"mat2="<<mat2<<std::endl;
    std::cout<<"matmul(mat1,mat2)="<<Ouroboros::matmul(mat1,mat2)<<std::endl;//Matrix multiplication
    Ouroboros::Tensor vec1=Ouroboros::CreateTensor::linspace({3},0,2);
    std::cout<<"vec1="<<vec1<<std::endl;
    std::cout<<"matvecmul(mat1,vec1)="<<Ouroboros::matvecmul(mat1,vec1)<<std::endl;//Matrix vector multiplication
    
    //Determinants
    Ouroboros::Tensor mat3=Ouroboros::CreateTensor::linspace({3,3},0,8);
    mat3[{0,0}]=1;
    std::cout<<"mat3="<<mat3<<std::endl;
    std::cout<<"Determinant of mat3="<<Ouroboros::determinant(mat3)<<std::endl;
    //Cofactor
    std::cout<<"Cofactor(0,0) of mat3="<<Ouroboros::cofactor(mat3,0,0)<<std::endl;
    //Minor
    std::cout<<"Minor(0,0) of mat3="<<Ouroboros::minor(mat3,0,0)<<std::endl;
    //Adjoint
    std::cout<<"Adjoint of mat3="<<Ouroboros::adjoint(mat3)<<std::endl;
    return 0;
}