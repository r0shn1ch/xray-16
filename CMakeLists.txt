cmake_minimum_required(VERSION 3.23)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(PreProjectInit)

calculate_xray_build_id(XRAY_BUILD_ID)
project(OpenXRay
    DESCRIPTION "OpenXRay is an improved version of the X-Ray Engine, \
                the game engine used in the world-famous S.T.A.L.K.E.R. game series by GSC Game World."
    VERSION 1.6.02.${XRAY_BUILD_ID}
    HOMEPAGE_URL "https://github.com/OpenXRay/xray-16"
    LANGUAGES CXX C
)

# Installation directory defaults are needed by all toolchains, including MSVC.
include(GNUInstallDirs)

# Unix toolchains discover SDL2 in XRay.Compiler.GNULike.cmake. The MSVC path
# uses vcpkg in CMake builds, so provide the imported target before engine and
# renderer subdirectories reference SDL2::SDL2.
if (WIN32)
    find_package(SDL2 2.0.18 CONFIG REQUIRED)
    # The MSVC toolchain does not run XRay.Compiler.GNULike.cmake, so resolve
    # the libraries required by the engine targets here for Windows CMake
    # builds as well. They are also required at CMake generation time when
    # building only xrRenderVK because the top-level project defines all game
    # targets in the same configure step.
    find_package(OpenAL CONFIG QUIET)
    if (TARGET OpenAL::OpenAL)
        set(OpenAL_FOUND TRUE)
    else()
        find_package(OpenAL REQUIRED)
    endif()
    find_package(JPEG)
    find_package(Ogg REQUIRED)
    find_package(Vorbis REQUIRED)
    find_package(Theora REQUIRED)
    find_package(LZO REQUIRED)
endif()

# Keep a stable engine root when OpenXRay is embedded with add_subdirectory().
# CMAKE_SOURCE_DIR belongs to the outer application in that configuration.
set(XRAY_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

option(BUILD_SHARED_LIBS "Build shared libraries"     ${BUILD_SHARED_LIBS_DEFAULT_VALUE})

option(XRAY_USE_LUAJIT   "Use LuaJIT"                 ON)
option(XRAY_USE_ASAN     "Use AddressSanitizer"       OFF)
option(XRAY_ENABLE_TRACY "Enable tracy profiler"      OFF)
option(XRAY_BUILD_VULKAN_TESTS "Build focused Vulkan renderer host tests" OFF)

if (XRAY_BUILD_VULKAN_TESTS)
    enable_testing()
endif()

# The Android SDL activity loads the engine as libmain.so. Static archives
# linked into that shared object must therefore retain position-independent
# code even though the normal desktop build keeps BUILD_SHARED_LIBS disabled.
set(XRAY_STATIC_PIC ${BUILD_SHARED_LIBS})
if (ANDROID AND XRAY_ANDROID_SHARED)
    set(XRAY_STATIC_PIC ON)
endif()

include(XRay.Build)
include(XRay.Packaging)

function(xray_add_externals)
    if (ANDROID AND CMAKE_HOST_WIN32 AND XRAY_USE_LUAJIT)
        if (NOT LUAJIT_HOST_C_COMPILER)
            find_program(xray_luajit_host_cc NAMES gcc clang REQUIRED NO_CACHE)
            set(LUAJIT_HOST_C_COMPILER "${xray_luajit_host_cc}")
        endif()
        # buildvm runs on Windows but must generate frames for the Android ABI.
        set(CMAKE_POLICY_DEFAULT_CMP0126 NEW)
        set(lua_target_os "-DLUAJIT_OS=LUAJIT_OS_LINUX")
        set(LUAJIT_HOST_C_COMPILER "${LUAJIT_HOST_C_COMPILER}\;${lua_target_os}")

        set(lua_build_dir "${CMAKE_CURRENT_BINARY_DIR}/Externals/LuaJIT-proj")
        set(lua_buildvm_dir "${lua_build_dir}/HostBuildTools/buildvm")
        if (EXISTS "${lua_buildvm_dir}/compile_commands.json")
            file(STRINGS "${lua_buildvm_dir}/compile_commands.json" buildvm_target
                REGEX "${lua_target_os}" LIMIT_COUNT 1)
            if (NOT buildvm_target)
                file(GLOB lua_generated_headers "${lua_build_dir}/lj_*def.h")
                file(REMOVE "${lua_buildvm_dir}/CMakeCache.txt" "${lua_buildvm_dir}/buildvm.exe"
                    "${lua_buildvm_dir}/Release/buildvm.exe" "${lua_build_dir}/lj_vm.S"
                    "${lua_build_dir}/jit/vmdef.lua" ${lua_generated_headers})
            endif()
        endif()
    endif()
    add_subdirectory(Externals)
endfunction()
xray_add_externals()
if (MSVC AND TARGET xrLuaJIT)
    # The bundled LuaJIT build files also supply MinGW options on Windows.
    get_target_property(lua_compile_options xrLuaJIT COMPILE_OPTIONS)
    list(REMOVE_ITEM lua_compile_options -Wno-comment -O3 -fomit-frame-pointer -fno-stack-protector)
    set_property(TARGET xrLuaJIT PROPERTY COMPILE_OPTIONS "${lua_compile_options}")
    get_target_property(lua_link_options xrLuaJIT LINK_OPTIONS)
    list(FILTER lua_link_options EXCLUDE REGEX "^ *-shared -Wl,")
    set_property(TARGET xrLuaJIT PROPERTY LINK_OPTIONS "${lua_link_options}")
    if (BUILD_SHARED_LIBS)
        target_compile_definitions(xrLuaJIT PUBLIC LUA_BUILD_AS_DLL)
    endif()
endif()
target_include_directories(xrLuaFix PRIVATE
    "${XRAY_SOURCE_DIR}/Externals"
    "${XRAY_SOURCE_DIR}/Externals/luabind")

add_subdirectory("${XRAY_SOURCE_DIR}/Externals/Vulkan-Headers" EXCLUDE_FROM_ALL)
include(NvrhiDependencies)
include(VulkanDependencies)
add_subdirectory(src)
add_subdirectory(res)
add_subdirectory(misc)
