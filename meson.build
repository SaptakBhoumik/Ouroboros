

project('Ouroboros', 'cpp', version: '0.1.0')

include = include_directories('include/')
omp=dependency('openmp')
cblas=dependency('cblas')
gsl=dependency('gsl')

add_project_arguments('-std=c++2a', language: ['cpp','c'])

subdir('src')

ouroboros=static_library(
    'ouroboros',
    include_directories:include,
    sources:tensor_src+scalar_func_src,
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
