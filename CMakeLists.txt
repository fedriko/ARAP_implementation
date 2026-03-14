cmake_minimum_required(VERSION 3.14)
project(MonProjetARAP)

include(FetchContent)
FetchContent_Declare(
  libigl
  GIT_REPOSITORY https://github.com/libigl/libigl.git
  GIT_TAG v2.5.0
)
FetchContent_MakeAvailable(libigl)

add_executable(${PROJECT_NAME} main.cpp)

# Liez à la fois igl::core ET Eigen3::Eigen
target_link_libraries(${PROJECT_NAME} PUBLIC igl::core Eigen3::Eigen)
