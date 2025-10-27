

project('Ouroboros', 'cpp', version: '0.2.0')

include = include_directories('include/')
# omp=dependency('openmp')
cblas=dependency('cblas')

add_project_arguments('-std=c++2a', language: ['cpp','c'])

subdir('src')

install_headers(
    [
        'include/func0.hpp',

        'include/op0.hpp',
        'include/op1.hpp',
        'include/op2.hpp',
        'include/op3.hpp',

        'include/tensor.hpp',
    ],
subdir : 'ouroboros')

ouroboros=static_library(
    'ouroboros',
    include_directories:include,
    sources:src,
    # dependencies:[omp,cblas],
    dependencies:[cblas],
    install:true
)

pkg = import('pkgconfig')
headers = ['.'] # subdirectories of ${prefix}/${includedir} to add to header path
pkg.generate(libraries : ouroboros,
             subdirs : headers,
             version : '0.2.0',
             name : 'Ouroboros',
             filebase : 'ouroboros',
             description : 'A simple c++ tensor library')
