cmake_minimum_required(VERSION 2.6)
project(roytracker)

# Options
option(ROYTRACKER_BUILD_TESTS "Build tests (requires GTest)" ON)
option(ROYTRACKER_BUILD_SAMPLES "Build samples" ON)
set(ROYTRACKER_TEST_OUTPUT_PATH "" CACHE STRING "Folder where to put results from write tests")

# Default build type: Release
if(NOT CMAKE_BUILD_TYPE AND NOT MSVC)
  set(CMAKE_BUILD_TYPE "Release")
endif()

# Load OpenCV dependency
find_package(OpenCV COMPONENTS calib3d video features2d core REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Use OpenMP for multithreading
find_package(OpenMP QUIET)
if(OPENMP_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# Build project
include_directories("${CMAKE_CURRENT_SOURCE_DIR}")
file(GLOB_RECURSE ROYTRACKER_SOURCES_LIST "${CMAKE_CURRENT_SOURCE_DIR}/roytracker/*.cpp")
add_library(roytracker ${ROYTRACKER_SOURCES_LIST} ${RPLY_SOURCES_LIST})
target_link_libraries(roytracker ${OpenCV_LIBS})
set(ROYTRACKER_LIBS roytracker ${OpenCV_LIBS})

# When tests are enabled, GTest must be installed
if(ROYTRACKER_BUILD_TESTS)
	enable_testing()
	find_package(GTest QUIET)
	if(GTEST_FOUND)
		message(STATUS "GTestIncludes: ${GTEST_INCLUDE_DIRS}")
		message(STATUS "GTestLibraries: ${GTEST_LIBRARIES}")
		add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/tests")
	else()
    message(STATUS "GTest could not be found")
	endif()
endif()

if(ROYTRACKER_BUILD_SAMPLES)
  message(STATUS "Including Samples")
  add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/samples")
endif()
