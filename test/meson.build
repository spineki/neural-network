matrix_test = executable(
    'matrix_test', # test executable name 
    'Matrix_test.cpp', # tests source files to be compiled,
    '../src/Matrix.cpp',
    include_directories : [include_dirs, src_dirs]  # need to add src to help to find headers
    ) # link test executable with previously declared shared library :code:`libuuid`

# test execution 
test('Matrix test', matrix_test)


network_test = executable(
    'network_test', # test executable name 
    'Network_test.cpp', # tests source files to be compiled,
    '../src/Network.cpp',
    '../src/Matrix.cpp',
    include_directories : [include_dirs, src_dirs]  # need to add src to help to find headers
    ) # link test executable with previously declared shared library :code:`libuuid`

# # test execution 
test('Network test', network_test)