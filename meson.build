

project('Ouroboros', 'cpp', version: '0.1.0')

include = include_directories('include/')
omp=dependency('openmp')
cblas=dependency('cblas')

add_project_arguments('-std=c++2a', language: ['cpp','c'])

subdir('src')

install_headers(
    [
        'include/func.hpp',
        'include/macros.hpp',
        'include/op.hpp',
        'include/ouroboros.hpp',
        'include/private_impl.hpp',
        'include/shape.hpp',
        'include/tensor.hpp',
        'include/utils.hpp',
    ],
subdir : 'ouroboros')

ouroboros=static_library(
    'ouroboros',
    include_directories:include,
    sources:tensor_src+op_src+utils_src,
    dependencies:[omp,cblas],
    install:true
)

pkg = import('pkgconfig')
headers = ['.'] # subdirectories of ${prefix}/${includedir} to add to header path
pkg.generate(libraries : ouroboros,
             subdirs : headers,
             version : '0.1.0',
             name : 'Ouroboros',
             filebase : 'ouroboros',
             description : 'A simple c++ tensor library')
             
executable(
    'test_tensor.elf',
    sources: ['test_tensor.cpp'], 
    dependencies:[omp,cblas],
    include_directories:include,
    link_with: [ouroboros]
)
