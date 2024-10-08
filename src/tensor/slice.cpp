#include "tensor.hpp"
namespace Ouroboros{
void BoolTensor::sliceRecursive(BoolTensor& output, const std::vector<size_t>& start, const std::vector<size_t>& step, std::vector<size_t>& indices, 
                        std::vector<size_t>& output_indices, std::size_t dimension){
    if (dimension == this->m_shape.dim()) {
        output.m_data[output.offset(output_indices)] = this->m_data[this->offset(indices)];
    } 
    else {
        for (std::size_t i = start[dimension], out_idx = 0; 
                    i < start[dimension] + step[dimension] * output.m_shape[dimension]; i += step[dimension],
                     ++out_idx) {
            indices[dimension] = i;
            output_indices[dimension]=out_idx;
            sliceRecursive(output, start, step, indices, output_indices, dimension + 1);
        }
    }
}
void Tensor::sliceRecursive(Tensor& output, const std::vector<size_t>& start, const std::vector<size_t>& step, std::vector<size_t>& indices, 
                        std::vector<size_t>& output_indices, std::size_t dimension){
    if (dimension == this->m_shape.dim()) {
        output.m_data[output.offset(output_indices)] = this->m_data[this->offset(indices)];
    } 
    else {
        for (std::size_t i = start[dimension], out_idx = 0; 
                    i < start[dimension] + step[dimension] * output.m_shape[dimension]; i += step[dimension],
                     ++out_idx) {
            indices[dimension] = i;
            output_indices[dimension]=out_idx;
            sliceRecursive(output, start, step, indices, output_indices, dimension + 1);
        }
    }
}
}