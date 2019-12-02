cmake_minimum_required(VERSION 3.12)

project(RA_CPP_tt)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)

conan_basic_setup()

find_package(OpenMP REQUIRED)
                 
add_executable(Matching matching.cpp)
target_compile_features(Matching PRIVATE cxx_std_17)

target_link_libraries(Matching
    PRIVATE
        ${CONAN_LIBS}
	OpenMP::OpenMP_CXX
)


