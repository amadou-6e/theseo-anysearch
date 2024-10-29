cmake_minimum_required(VERSION 3.10)
project(Project)

# Specify C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include FetchContent for downloading dependencies
include(FetchContent)

# Fetch cpp-httplib (header-only)
FetchContent_Declare(
    httplib
    GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
    GIT_TAG v0.12.0
)
FetchContent_MakeAvailable(httplib)

# Fetch stb (header-only)
FetchContent_Declare(
    stb
    GIT_REPOSITORY https://github.com/nothings/stb.git
    GIT_TAG master
)
FetchContent_MakeAvailable(stb)

# Find packages with specific configs
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(glfw3 3.3 REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(Threads REQUIRED)

# Add executable (removed stb_impl.cpp)
add_executable(${PROJECT_NAME} 
    main.cpp
)

# Link libraries
target_link_libraries(${PROJECT_NAME} 
    PRIVATE
    ${OPENGL_LIBRARIES}
    GLEW::GLEW
    glfw
    OpenSSL::SSL
    OpenSSL::Crypto
    Threads::Threads
)

# Add include directories
target_include_directories(${PROJECT_NAME}
    PRIVATE
    ${OPENGL_INCLUDE_DIRS}
    ${GLEW_INCLUDE_DIRS}
    ${httplib_SOURCE_DIR}
    ${stb_SOURCE_DIR}
)

# Print debug information
message(STATUS "OpenGL libraries: ${OPENGL_LIBRARIES}")
message(STATUS "GLEW libraries: ${GLEW_LIBRARIES}")
message(STATUS "GLFW libraries: ${glfw3_LIBRARIES}")
message(STATUS "OpenSSL libraries: ${OPENSSL_LIBRARIES}")