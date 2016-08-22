project(roytracker_samples)

# Setup path for test data (interested in images)
set(DATA_PATH "${CMAKE_SOURCE_DIR}/tests/data")

if("${ROYTRACKER_TEST_OUTPUT_PATH}" STREQUAL "")
  set(ROYTRACKER_TEST_OUTPUT_PATH ".")
endif()

include_directories("${CMAKE_CURRENT_SOURCE_DIR}")

# Rule: Each file will contain a single sample
file(GLOB ROYTRACKER_SAMPLE_SOURCES *.cpp)

# Create an executable for each sample
foreach(SAMPLE_SOURCE_FILE ${ROYTRACKER_SAMPLE_SOURCES})
  # Get the filename part (no path)
  get_filename_component(SAMPLE_SOURCE_FILENAME ${SAMPLE_SOURCE_FILE} NAME)
  # Remove the extension
  string(REPLACE ".cpp" "" SAMPLE_NAME ${SAMPLE_SOURCE_FILENAME})
  add_executable(${SAMPLE_NAME} ${SAMPLE_SOURCE_FILE})
  target_link_libraries(${SAMPLE_NAME} ${ROYTRACKER_LIBS} )
endforeach()
