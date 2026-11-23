include(FetchContent)

# The renderer uses the header directly. The dependency's CMake project builds its demo.
FetchContent_Declare(xray_bcdec
    GIT_REPOSITORY https://github.com/iOrange/bcdec.git
    GIT_TAG 80859ed3b7afb1c527a2a99d70c61457bea72d0c
    SOURCE_SUBDIR include
)
FetchContent_MakeAvailable(xray_bcdec)

set(MESHOPT_BUILD_DEMO OFF CACHE BOOL "Build the meshoptimizer demo" FORCE)
set(MESHOPT_BUILD_GLTFPACK OFF CACHE BOOL "Build meshoptimizer gltfpack" FORCE)
set(MESHOPT_BUILD_SHARED_LIBS OFF CACHE BOOL "Build meshoptimizer as a static library" FORCE)
set(MESHOPT_INSTALL OFF CACHE BOOL "Install meshoptimizer" FORCE)
FetchContent_Declare(xray_meshoptimizer
    GIT_REPOSITORY https://github.com/zeux/meshoptimizer.git
    GIT_TAG f105bb570c4fa16a687e869f1bcf9d4b84158e23
)
FetchContent_MakeAvailable(xray_meshoptimizer)

if (ANDROID AND ANDROID_ABI STREQUAL "arm64-v8a")
    set(FFX_API_BACKEND VK_ANDROID_ARM64 CACHE STRING "FidelityFX Vulkan Android backend" FORCE)
    set(FFX_FSR3 ON CACHE BOOL "Build FidelityFX FSR3" FORCE)
    set(FFX_AUTO_COMPILE_SHADERS ON CACHE BOOL "Compile FidelityFX shaders" FORCE)
    set(FFX_BUILD_AS_DLL OFF CACHE BOOL "Build FidelityFX as a shared library" FORCE)
    FetchContent_Declare(xray_fsr3_android
        GIT_REPOSITORY https://github.com/Snowapril/fsr3-android.git
        GIT_TAG 3a369b3d2d3734de0f6e7aa84084879fe7f70535
        SOURCE_SUBDIR sdk
    )
    FetchContent_MakeAvailable(xray_fsr3_android)

    # API 26's loader does not export these core Vulkan commands.
    target_compile_definitions(ffx_backend_vk_android_arm64 PRIVATE
        vkGetPhysicalDeviceProperties2=xrayFsrGetPhysicalDeviceProperties2
        vkGetPhysicalDeviceFeatures2=xrayFsrGetPhysicalDeviceFeatures2
        vkWaitSemaphores=xrayFsrWaitSemaphores
        vkGetSemaphoreCounterValue=xrayFsrGetSemaphoreCounterValue
    )
endif()
