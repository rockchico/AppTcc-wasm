project(FrameExtractor)
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

file(GLOB SOURCE_FILES
        ${PROJECT_SOURCE_DIR}/src/main.cpp
)

SET(SOURCE_FILES
        src/main.cpp
)

ADD_EXECUTABLE(${PROJECT_NAME} ${SOURCE_FILES})
TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${NEEDED_LIBS})


