project(roytracker_tests)
add_definitions(-std=c++11)

# Files used by tests
set(DATA_PATH "${CMAKE_CURRENT_SOURCE_DIR}/data")

if("${ROYTRACKER_TEST_OUTPUT_PATH}" STREQUAL "")
  set(ROYTRACKER_TEST_OUTPUT_PATH ".")
endif()

add_definitions(
  -DROYTRACKER_SAMPLE_IMAGE_00="${DATA_PATH}/image-00.jpg"
  -DROYTRACKER_SAMPLE_IMAGE_01="${DATA_PATH}/image-01.jpg"
  -DROYTRACKER_SAMPLE_TRACK_RESULT="${ROYTRACKER_TEST_OUTPUT_PATH}/track-result.jpg"
)

# Rule: Each file will contain a single test
file(GLOB ROYTRACKER_TEST_SOURCES *.cpp)

# Create an executable for each test
foreach(TEST_SOURCE_FILE ${ROYTRACKER_TEST_SOURCES})
  # Get the filename part (no path)
  get_filename_component(TEST_SOURCE_FILENAME ${TEST_SOURCE_FILE} NAME)
  # Remove the extension
  string(REPLACE ".cpp" "" TEST_NAME ${TEST_SOURCE_FILENAME})
  add_executable(${TEST_NAME} ${TEST_SOURCE_FILE})
  target_link_libraries(${TEST_NAME} ${ROYTRACKER_LIBS} ${GTEST_BOTH_LIBRARIES})
  # Add to GTest
  GTEST_ADD_TESTS(${TEST_NAME} "${TEST_ARGS}" AUTO)
endforeach()
