#include <ouroboros/ouroboros.hpp>
#include <iostream>
int main(){
    {
        std::cout<<"Writing to a binary file\n";
        std::fstream file("example.bin",std::ios::out|std::ios::binary);
        Ouroboros::Utils::write_bin_double(file,3.14);
        Ouroboros::Tensor t=Ouroboros::CreateTensor::ones({2,2});
        std::cout<<"t:\n"<<t<<"\n";
        Ouroboros::Utils::write_bin_tensor(file,t);//By default it writes the tensor as double i.e every element take 8 bytes
        Ouroboros::Utils::write_bin_tensor(file,t,Ouroboros::Utils::TensorType::FLOAT);//Now every element takes 4 bytes
        file.close();
        /*
        enum class TensorType{
            //Done to reduce the size of the tensor in the file
            INT8,//Save the tensor as std::int8_t
            INT16,//Save the tensor as std::int16_t
            INT32,//Save the tensor as std::int32_t
            INT64,//Save the tensor as std::int64_t
            UINT8,//Save the tensor as std::uint8_t
            UINT16,//Save the tensor as std::uint16_t
            UINT32,//Save the tensor as std::uint32_t
            UINT64,//Save the tensor as std::uint64_t
            FLOAT,//Save the tensor as float
            DOUBLE,//Save the tensor as double
        };
        */
    }
    {
        std::cout<<"Reading from a binary file\n";
        std::fstream file("example.bin",std::ios::in|std::ios::binary);
        std::cout<<"Double value: "<<Ouroboros::Utils::read_bin_double(file)<<"\n";
        std::cout<<"Tensor as double:\n"<<Ouroboros::Utils::read_bin_tensor(file)<<"\n";
        std::cout<<"Tensor as float:\n"<<Ouroboros::Utils::read_bin_tensor(file)<<"\n";
        file.close();
    }
}