cmake_minimum_required (VERSION 3.5)
project(ransac)

set(CMAKE_CXX_STANDARD 14)
set(COMPILE_FLAGS "${COMPILE_FLAGS} -03")

# Add Boost library
find_package(Boost COMPONENTS program_options REQUIRED)
if(Boost_FOUND)
    message(STATUS "Boost_INCLUDE_DIRS: ${Boost_INCLUDE_DIRS}")
    message(STATUS "Boost_LIBRARIES: ${Boost_LIBRARIES}")
    message(STATUS "Boost_VERSION: ${Boost_VERSION}")

    include_directories(${Boost_INCLUDE_DIRS})
    link_libraries(${Boost_LIBRARIES})
endif()

# Add Eigen library
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

find_package(VTK REQUIRED)
include_directories(${VTK_INCLUDE_DIRS})
link_libraries(${VTK_LIBRARIES})
message(STATUS "VTK_INCLUDE_DIRS: ${VTK_INCLUDE_DIRS}")
message(STATUS "VTK_LIBRARIES: ${VTK_LIBRARIES}")

set(COMPILE_FLAGS "${COMPILE_FLAGS} -O3 -Wpointer-arith -Wold-style-cast")

# Add source to this project's executable.
set(SOURCES "src/main.cpp"
    "src/stl_reader.h"
    "src/ransac.h"
    "src/ransac.cpp"
)
add_executable (${PROJECT_NAME} ${SOURCES})

target_link_libraries(${PROJECT_NAME} Eigen3::Eigen)
