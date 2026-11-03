cmake_minimum_required(VERSION 3.28)
project(gomoku CXX)

if (NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message(FATAL_ERROR "Only Linux with CUDA 13.0 is supported")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(CUDAToolkit 13.0 REQUIRED)

set(TORCH_VERSION "2.11.0%2Bcu130")
set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cu130/libtorch-shared-with-deps-${TORCH_VERSION}.zip")

include(FetchContent)
FetchContent_Declare(
  libtorch
  URL ${LIBTORCH_URL}
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_GetProperties(libtorch)
if (NOT libtorch_POPULATED)
  message(STATUS "LibTorch is not available. Downloading from: ${LIBTORCH_URL}")
endif()
FetchContent_MakeAvailable(libtorch)
find_package(Torch REQUIRED PATHS "${libtorch_SOURCE_DIR}/share/cmake/Torch" NO_DEFAULT_PATH)
if (NOT CAFFE2_USE_CUDA)
  message(FATAL_ERROR "LibTorch with CUDA support is required")
endif()

add_subdirectory(third_party)

find_package(OpenGL REQUIRED)

file(GLOB_RECURSE SOURCES "src/*.cpp")

add_executable(gomoku ${SOURCES})
target_include_directories(gomoku PRIVATE include)

target_compile_options(gomoku PRIVATE -Wall -Wextra)

target_link_libraries(gomoku PRIVATE imgui glfw OpenGL::GL "${TORCH_LIBRARIES}")
