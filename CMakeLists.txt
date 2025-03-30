cmake_minimum_required(VERSION 3.22)
project(uncertainty_propagation)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
#ADD_DEFINITIONS(-DEIGEN_NO_MALLOC -DUKF_DOUBLE_PRECISION)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

find_package(Boost REQUIRED)

# --- Eigen ---
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS})

# --- Python (Anaconda) ---
set(Python3_ROOT_DIR "C:/Users/dbezu/AppData/Local/Programs/Python/Python311/")
find_package(Python3 COMPONENTS Development NumPy REQUIRED)

# --- GoogleTest ---
find_package(GTest REQUIRED)
include(GoogleTest)

# --- Потоки (Threads) ---
find_package(Threads REQUIRED)

include_directories(include)

add_library(lib
            src/model/simple_vehicle_model.cpp
            src/model/normal_vehicle_model.cpp
            src/model/kinematic_vehicle_model.cpp
            src/model/mobile_robot_model.cpp
            src/utilities.cpp
            src/filter/simple_vehicle_nkf.cpp
            src/filter/simple_vehicle_ukf.cpp
            src/filter/simple_vehicle_ekf.cpp
            src/filter/normal_vehicle_nkf.cpp
            src/filter/normal_vehicle_ukf.cpp
            src/filter/normal_vehicle_ekf.cpp
            src/filter/kinematic_vehicle_nkf.cpp
            src/filter/kinematic_vehicle_ukf.cpp
            src/filter/kinematic_vehicle_ekf.cpp
            src/filter/mobile_robot_nkf.cpp
            src/filter/mobile_robot_ekf.cpp
            src/filter/mobile_robot_ukf.cpp
            src/filter/paper_example_2d_ukf.cpp
            src/filter/paper_example_3d_ukf.cpp
            src/distribution/base_distribution.cpp
            src/distribution/uniform_distribution.cpp
            src/distribution/normal_distribution.cpp
            src/distribution/exponential_distribution.cpp
            src/distribution/two_dimensional_normal_distribution.cpp
            src/distribution/three_dimensional_normal_distribution.cpp
            src/distribution/four_dimensional_normal_distribution.cpp)

target_link_libraries(lib PRIVATE Eigen3::Eigen)

add_executable(main main.cpp)
target_include_directories(main PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(main Python3::Python Python3::NumPy lib)

add_executable(main_non_gaussian main_non_gaussian.cpp)
target_include_directories(main_non_gaussian PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(main_non_gaussian Python3::Python Python3::NumPy lib)

add_executable(normal_vehicle_scenario normal_vehicle_scenario.cpp)
target_include_directories(normal_vehicle_scenario PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(normal_vehicle_scenario Python3::Python Python3::NumPy lib)

add_executable(kinematic_vehicle_scenario kinematic_vehicle_scenario.cpp)
target_include_directories(kinematic_vehicle_scenario PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(kinematic_vehicle_scenario Python3::Python Python3::NumPy lib)

add_executable(mobile_robot_scenario mobile_robot_scenario.cpp)
target_include_directories(mobile_robot_scenario PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(mobile_robot_scenario Python3::Python Python3::NumPy lib)

add_executable(paper_example paper_example.cpp)
target_include_directories(paper_example PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(paper_example Python3::Python Python3::NumPy lib)

add_executable(data_preprocessor data/data_preprocessor.cpp)
add_executable(data_analyzer data/data_analyzer.cpp)
target_include_directories(data_analyzer PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(data_analyzer Python3::Python Python3::NumPy lib)

# --- Тесты (GoogleTest) ---
# Добавляем флаг -fno-rtti для тестовых целей, чтобы отключить RTTI и устранить ошибку линковки
add_executable(test_normal_distribution test/distribution/test_normal_distribution.cpp)
target_link_libraries(test_normal_distribution PRIVATE GTest::gtest GTest::gtest_main lib Threads::Threads)
target_compile_options(test_normal_distribution PRIVATE -fno-rtti)
gtest_add_tests(TARGET test_normal_distribution)

add_executable(test_uniform_distribution test/distribution/test_uniform_distribution.cpp)
target_link_libraries(test_uniform_distribution PRIVATE GTest::gtest GTest::gtest_main lib Threads::Threads)
target_compile_options(test_uniform_distribution PRIVATE -fno-rtti)
gtest_add_tests(TARGET test_uniform_distribution)

add_executable(test_exponential_distribution test/distribution/test_exponential_distribution.cpp)
target_link_libraries(test_exponential_distribution PRIVATE GTest::gtest GTest::gtest_main lib Threads::Threads)
target_compile_options(test_exponential_distribution PRIVATE -fno-rtti)
gtest_add_tests(TARGET test_exponential_distribution)

add_executable(test_two_dimensional_normal_distribution test/distribution/test_two_dimensional_normal_distribution.cpp)
target_link_libraries(test_two_dimensional_normal_distribution PRIVATE GTest::gtest GTest::gtest_main lib Threads::Threads)
target_compile_options(test_two_dimensional_normal_distribution PRIVATE -fno-rtti)
gtest_add_tests(TARGET test_two_dimensional_normal_distribution)

add_executable(test_three_dimensional_normal_distribution test/distribution/test_three_dimensional_normal_distribution.cpp)
target_link_libraries(test_three_dimensional_normal_distribution PRIVATE GTest::gtest GTest::gtest_main lib Threads::Threads)
target_compile_options(test_three_dimensional_normal_distribution PRIVATE -fno-rtti)
gtest_add_tests(TARGET test_three_dimensional_normal_distribution)

add_executable(test_simple_vehicle_nkf test/filter/test_simple_vehicle_nkf.cpp)
target_link_libraries(test_simple_vehicle_nkf PRIVATE GTest::gtest GTest::gtest_main lib Threads::Threads)
target_compile_options(test_simple_vehicle_nkf PRIVATE -fno-rtti)
gtest_add_tests(TARGET test_simple_vehicle_nkf)

add_executable(test_kinematic_vehicle_model test/model/test_kinematic_vehicle_model.cpp)
target_link_libraries(test_kinematic_vehicle_model PRIVATE GTest::gtest GTest::gtest_main lib Threads::Threads)
target_compile_options(test_kinematic_vehicle_model PRIVATE -fno-rtti)
gtest_add_tests(TARGET test_kinematic_vehicle_model)

add_executable(test_mobile_robot_model test/model/test_mobile_robot_model.cpp)
target_link_libraries(test_mobile_robot_model PRIVATE GTest::gtest GTest::gtest_main lib Threads::Threads)
target_compile_options(test_mobile_robot_model PRIVATE -fno-rtti)
gtest_add_tests(TARGET test_mobile_robot_model)

add_executable(test_simple_vehicle_model test/model/test_simple_vehicle_model.cpp)
target_link_libraries(test_simple_vehicle_model PRIVATE GTest::gtest GTest::gtest_main lib Threads::Threads)
target_compile_options(test_simple_vehicle_model PRIVATE -fno-rtti)
gtest_add_tests(TARGET test_simple_vehicle_model)