

project('Ouroboros', 'cpp', version: '0.1.0')

include = include_directories('include/')
omp=dependency('openmp')
cblas=dependency('cblas')

add_project_arguments('-std=c++2a', language: ['cpp','c'])

subdir('src')

install_headers(
    [
        'include/op.hpp',
        'include/shape.hpp',
        'include/tensor.hpp',
        'include/macros.hpp',
        'include/func.hpp',
    ],
subdir : 'ouroboros')

ouroboros=static_library(
    'ouroboros',
    include_directories:include,
    sources:tensor_src+op_src+utils_src,
    dependencies:[omp,cblas],
    install:true
)

executable(
    'test_tensor.elf',
    sources: ['test_tensor.cpp'], 
    dependencies:[omp,cblas],
    include_directories:include,
    link_with: [ouroboros]
)
