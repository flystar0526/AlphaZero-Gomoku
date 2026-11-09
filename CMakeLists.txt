cmake_minimum_required(VERSION 3.28)
project(gomoku CXX)

# Require Linux and CUDA 13.0 environment
if (NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message(FATAL_ERROR "Only Linux with CUDA 13.0 is supported")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(CUDAToolkit 13.0 REQUIRED)

# Download and configure LibTorch (CUDA 13.0 build) automatically if not cached
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

# Build bundled third-party dependencies (GLFW and Dear ImGui)
add_subdirectory(third_party)

find_package(OpenGL REQUIRED)

# Pure C++ core library (rules, game state, file I/O, MCTS, AI move selection).
# Zero GUI or LibTorch dependencies so core unit tests compile and run fast.
add_library(gomoku_core STATIC
  src/ai_player.cpp
  src/board.cpp
  src/configuration.cpp
  src/file_manager.cpp
  src/game.cpp
  src/game_position.cpp
  src/game_record.cpp
  src/mcts.cpp
  src/replay_controller.cpp
  src/simulation.cpp
)
target_include_directories(gomoku_core PUBLIC include)
target_compile_options(gomoku_core PRIVATE -Wall -Wextra)

# Neural network backend library encapsulating LibTorch / TorchScript.
add_library(gomoku_nn STATIC
  src/policy_value_network.cpp
)
target_include_directories(gomoku_nn PUBLIC include)
target_compile_options(gomoku_nn PRIVATE -Wall -Wextra)
target_link_libraries(gomoku_nn PUBLIC gomoku_core PRIVATE "${TORCH_LIBRARIES}")

# Main executable (GUI by default, headless with --simulation).
add_executable(gomoku
  src/main.cpp
  src/app.cpp
  src/grid.cpp
  src/info_panel.cpp
)
target_include_directories(gomoku PRIVATE include)
target_compile_options(gomoku PRIVATE -Wall -Wextra)
target_link_libraries(gomoku PRIVATE gomoku_nn imgui glfw OpenGL::GL)

# Automated tests (enabled by default; disable with -DBUILD_TESTING=OFF)
include(CTest)
if (BUILD_TESTING)
  # Core logic tests (board rules, game state, undo/replay, MCTS)
  add_executable(gomoku_core_tests tests/core_tests.cpp)
  target_compile_options(gomoku_core_tests PRIVATE -Wall -Wextra)
  target_link_libraries(gomoku_core_tests PRIVATE gomoku_core)
  add_test(NAME core COMMAND gomoku_core_tests)

  add_executable(gomoku_simulation_tests tests/simulation_tests.cpp)
  target_compile_options(gomoku_simulation_tests PRIVATE -Wall -Wextra)
  target_link_libraries(gomoku_simulation_tests PRIVATE gomoku_core)
  add_test(NAME simulation COMMAND gomoku_simulation_tests)
  add_test(NAME cli COMMAND ${CMAKE_COMMAND}
    "-DGOMOKU_EXECUTABLE=$<TARGET_FILE:gomoku>"
    -P "${PROJECT_SOURCE_DIR}/tests/cli_tests.cmake")

  # TorchScript model contract (CPU) and inference (CUDA) tests
  add_executable(gomoku_model_tests tests/model_tests.cpp)
  target_compile_options(gomoku_model_tests PRIVATE -Wall -Wextra)
  target_link_libraries(gomoku_model_tests PRIVATE gomoku_nn "${TORCH_LIBRARIES}")
  add_test(NAME model_contract COMMAND gomoku_model_tests
    "${PROJECT_SOURCE_DIR}/model/alpha-zero.pt")
  add_test(NAME model_cuda COMMAND gomoku_model_tests
    "${PROJECT_SOURCE_DIR}/model/alpha-zero.pt" --cuda)
  add_test(NAME model_simulation COMMAND gomoku_model_tests
    "${PROJECT_SOURCE_DIR}/model/alpha-zero.pt" --simulation)
  set_tests_properties(model_cuda PROPERTIES SKIP_RETURN_CODE 77)
endif()
