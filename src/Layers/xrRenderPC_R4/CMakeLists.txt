add_library(xrRenderPC_R4)
find_package(directxtex CONFIG REQUIRED)

target_sources(xrRenderPC_R4 PRIVATE
    "${XRAY_SOURCE_DIR}/Externals/imgui/backends/imgui_impl_dx11.cpp"
    packages.config
    r2_test_hw.cpp
    r4_rendertarget.h
    r4_rendertarget_accum_direct.cpp
    r4_rendertarget_build_textures.cpp
    r4_rendertarget_phase_combine.cpp
    r4_rendertarget_phase_hdao.cpp
    r4_rendertarget_u_set_rt.cpp
    r4_shaders.cpp
    R_Backend_LOD.cpp
    R_Backend_LOD.h
    stdafx.cpp
    stdafx.h
    xrRender_R4.cpp
)

include(R4Sources.cmake)

target_include_directories(xrRenderPC_R4
    PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${XRAY_SOURCE_DIR}/src"
    "${XRAY_SOURCE_DIR}/sdk/include"
    "${XRAY_SOURCE_DIR}/Externals/imgui"
    "${XRAY_SOURCE_DIR}/src/Layers/xrRender_R2"
    "${XRAY_SOURCE_DIR}/sdk/include/nvapi"
    "${XRAY_SOURCE_DIR}/sdk/include/DirectXMesh"
    "${XRAY_SOURCE_DIR}/Externals/AGS_SDK"
)

target_link_libraries(xrRenderPC_R4
    PRIVATE
    xrCore
    xrCDB
    xrEngine
    xrParticles
    xrScriptEngine
    xrAPI
    xrMiscMath
    xrImGui
    xrMaterialSystem
    Microsoft::DirectXTex
    d3dcompiler
    dxguid
)

if (CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(XRAY_DX_ARCH x64)
    set(XRAY_NVAPI_NAME nvapi64)
else()
    set(XRAY_DX_ARCH x86)
    set(XRAY_NVAPI_NAME nvapi32)
endif()
target_link_libraries(xrRenderPC_R4 PRIVATE
    "${XRAY_SOURCE_DIR}/sdk/libraries/${XRAY_DX_ARCH}/${XRAY_NVAPI_NAME}.lib"
    "${XRAY_SOURCE_DIR}/Externals/AGS_SDK/ags_lib/lib/amd_ags_${XRAY_DX_ARCH}_2019_MD.lib")

target_compile_definitions(xrRenderPC_R4
    PRIVATE
    XRRENDER_R4_EXPORTS
    USE_DX11
    RENDER_NAMESPACE=render_r4
)

set_target_properties(xrRenderPC_R4 PROPERTIES
    PREFIX ""
)

target_precompile_headers(xrRenderPC_R4
    PRIVATE
    stdafx.h
)

if (BUILD_SHARED_LIBS)
    install(TARGETS xrRenderPC_R4 LIBRARY
        DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    )
endif()
