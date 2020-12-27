testexe = executable(
    'testexe', # test executable name 
    'Matrix_test.cpp', # tests source files to be compiled,
    '../src/Matrix.cpp',
    include_directories : [include_dirs, src_dirs]  # need to add src to help to find headers
    ) # link test executable with previously declared shared library :code:`libuuid`

# test execution 
test('Matrix test', testexe)