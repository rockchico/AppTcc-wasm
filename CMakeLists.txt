project (AppTccWasm)
cmake_minimum_required(VERSION 3.0)

set(OpenCV_STATIC ON)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS} .)

# francisco
add_library(app_tcc_lib VO.cpp VO.h) # !!!
#add_library(app_tcc_lib VisualOdometry/src/VO.cpp VisualOdometry/include/VO.h)

if (EMSCRIPTEN)
	
    # francisco
    add_executable (app_tcc_wasm web/app_tcc_js.cpp) # !!!
    #add_executable (app_tcc_wasm web/app_tcc_js-vo.cpp)

    target_link_libraries(app_tcc_wasm ${OpenCV_LIBS} app_tcc_lib) # !!!
    #target_link_libraries(app_tcc_wasm ${OpenCV_LIBS})

    add_custom_command(TARGET app_tcc_wasm POST_BUILD              # Adds a post-build event to app_tcc_wasm
                      COMMAND ${CMAKE_COMMAND} -E copy_if_different  # which executes "cmake - E copy_if_different..."
                      ${CMAKE_CURRENT_BINARY_DIR}/app_tcc_wasm.js # <--this is in-file
                      /var/www/html/AppTcc/www/testes/teste-wasm/app_tcc_wasm.js)   # <--this is out-file path

    add_custom_command(TARGET app_tcc_wasm POST_BUILD              # Adds a post-build event to app_tcc_wasm
                      COMMAND ${CMAKE_COMMAND} -E copy_if_different  # which executes "cmake - E copy_if_different..."
                      ${CMAKE_CURRENT_BINARY_DIR}/app_tcc_wasm.wasm # <--this is in-file
                      /var/www/html/AppTcc/www/testes/teste-wasm/app_tcc_wasm.wasm)   # <--this is out-file path

else()
	add_executable (app_tcc main.cpp)
	target_link_libraries(app_tcc ${OpenCV_LIBS} app_tcc_lib)
endif()

if(UNIX)
    
    # -s ALLOW_MEMORY_GROWTH=1
    # Cannot enlarge memory arrays. Either (1) compile with  -s TOTAL_MEMORY=X  with X higher than the current value 16777216, (2) compile with  -s ALLOW_MEMORY_GROWTH=1  which allows increasing the size at runtime, or (3) if you want malloc to return NULL (0) instead of this abort, compile with  -s ABORTING_MALLOC=0 
    
    
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++1z --llvm-lto 1 --bind -s WASM=1 -s ALLOW_MEMORY_GROWTH=1 -s DISABLE_EXCEPTION_CATCHING=0 -s ASSERTIONS=2 --memory-init-file 0 -O3") # -Oz --profiling / -s WASM=1
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -s DISABLE_EXCEPTION_CATCHING=0 -s NO_FILESYSTEM=1 -s ELIMINATE_DUPLICATE_FUNCTIONS=1 -s NO_EXIT_RUNTIME=1") # reduce file size
	
	if(DEFINED CMAKE_BUILD_TYPE)
		SET(CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE})
	else()
	    SET(CMAKE_BUILD_TYPE Release)
    endif()
	
elseif(MSVC)
   add_definitions(/D_CRT_SECURE_NO_WARNINGS)
   add_definitions(/D_SCL_SECURE_NO_WARNINGS)
   add_definitions(/DNOMINMAX)
   set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
   set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd")
endif()
