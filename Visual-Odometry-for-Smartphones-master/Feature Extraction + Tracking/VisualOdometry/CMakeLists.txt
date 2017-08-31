project(VisualOdometry)
cmake_minimum_required(VERSION 2.8)
find_package( OpenCV REQUIRED )
FIND_PACKAGE(Boost COMPONENTS system filesystem program_options REQUIRED)

SET(NEEDED_LIBS
        ${OpenCV_LIBS}
    	${Boost_CHRONO_LIBRARY}
	${Boost_FILESYSTEM_LIBRARY}
	${Boost_DATE_TIME_LIBRARY}
	${Boost_IOSTREAMS_LIBRARY}
	${Boost_SERIALIZATION_LIBRARY}
	${Boost_SYSTEM_LIBRARY}
	${Boost_THREAD_LIBRARY}
        ${Boost_PROGRAM_OPTIONS_LIBRARY}
)

INCLUDE_DIRECTORIES(
        ${PROJECT_SOURCE_DIR}/include
        include/
)

file(GLOB SOURCE_FILES
        ${PROJECT_SOURCE_DIR}/src/main.cpp
)

file(GLOB HEADER_FILES
        ${PROJECT_SOURCE_DIR}/include/VO.h
)

SET(HEADER_FILES
        include/VO.h
)

SET(SOURCE_FILES
        src/main.cpp
        src/VO.cpp
)

ADD_EXECUTABLE(${PROJECT_NAME} ${HEADER_FILES} ${SOURCE_FILES})
TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${NEEDED_LIBS})

