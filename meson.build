

project('Ouroboros', 'cpp', version: '0.1.0')

include = include_directories('include/')
omp=dependency('openmp')
cblas=dependency('cblas')
gsl=dependency('gsl')

add_project_arguments('-std=c++2a', language: ['cpp','c'])

subdir('src')

install_headers(
    [
        'include/op.hpp',
        'include/shape.hpp',
        'include/tensor.hpp',
        'include/macros.hpp',
    ],
subdir : 'ouroboros')

install_headers(
    [
        'include/func/basic.hpp',
        'include/func/err_func.hpp',
        'include/func/func.hpp',
        'include/func/h_trig.hpp',
        'include/func/sf_func1.hpp',
        'include/func/sf_func2.hpp',
        'include/func/trig.hpp',
    ],
subdir : 'ouroboros/func')

ouroboros=static_library(
    'ouroboros',
    include_directories:include,
    sources:tensor_src+scalar_func_src+tensor_func_src,
    dependencies:[omp,cblas,gsl],
    install:true
)

executable(
    'test_tensor.elf',
    sources: ['test_tensor.cpp'], 
    dependencies:[omp,cblas,gsl],
    include_directories:include,
    link_with: [ouroboros]
)
executable(
    'bench.elf',
    sources: ['bench.cpp'], 
    dependencies:[omp,cblas,gsl],
    include_directories:include,
    link_with: [ouroboros]
)
