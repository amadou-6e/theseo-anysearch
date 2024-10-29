cmake_minimum_required(VERSION 3.10)
project(Project)

# Specify C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set OpenGL preference to GLVND
set(OpenGL_GL_PREFERENCE GLVND)

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

# Fetch tinyobjloader
FetchContent_Declare(
    tinyobjloader
    GIT_REPOSITORY https://github.com/tinyobjloader/tinyobjloader.git
    GIT_TAG v2.0.0rc9
)
FetchContent_MakeAvailable(tinyobjloader)

# Find all required packages
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(glfw3 3.3 REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(Threads REQUIRED)
find_package(assimp REQUIRED)
find_package(TBB REQUIRED)
find_package(Boost REQUIRED)

# Manual OpenVDB configuration for system installation
set(OPENVDB_INCLUDE_DIRS "/usr/include")
set(OPENVDB_LIBRARIES "/usr/lib/x86_64-linux-gnu/libopenvdb.so")

# Add Imath library
set(IMATH_LIBRARIES "/usr/lib/x86_64-linux-gnu/libImath-3_1.so")
set(IMATH_INCLUDE_DIRS "/usr/include/Imath")

# First declare the executable, then link libraries
add_executable(${PROJECT_NAME} 
    main.cpp
    cpp/src/voxelization.cpp
)

# Add include directories
target_include_directories(${PROJECT_NAME}
    PRIVATE
    ${OPENGL_INCLUDE_DIRS}
    ${GLEW_INCLUDE_DIRS}
    ${httplib_SOURCE_DIR}
    ${stb_SOURCE_DIR}
    ${OPENVDB_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
    ${tinyobjloader_SOURCE_DIR}
    ${IMATH_INCLUDE_DIRS}
    cpp/include
)

# Link all libraries
target_link_libraries(${PROJECT_NAME} 
    PRIVATE
    ${OPENGL_LIBRARIES}
    GLEW::GLEW
    glfw
    OpenSSL::SSL
    OpenSSL::Crypto
    Threads::Threads
    ${OPENVDB_LIBRARIES}
    assimp::assimp
    TBB::tbb
    ${Boost_LIBRARIES}
    tinyobjloader
    ${IMATH_LIBRARIES}
)

# Print debug information
message(STATUS "OpenGL libraries: ${OPENGL_LIBRARIES}")
message(STATUS "GLEW libraries: ${GLEW_LIBRARIES}")
message(STATUS "GLFW libraries: ${glfw3_LIBRARIES}")
message(STATUS "OpenSSL libraries: ${OPENSSL_LIBRARIES}")
message(STATUS "OpenVDB include dirs: ${OPENVDB_INCLUDE_DIRS}")
message(STATUS "OpenVDB libraries: ${OPENVDB_LIBRARIES}")
message(STATUS "Boost libraries: ${Boost_LIBRARIES}")
message(STATUS "tinyobjloader source dir: ${tinyobjloader_SOURCE_DIR}")
message(STATUS "Imath libraries: ${IMATH_LIBRARIES}")