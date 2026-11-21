if (ANDROID AND XRAY_ANDROID_SHARED)
    add_library(xr_3da SHARED)
else()
    add_executable(xr_3da)
endif()

target_sources(xr_3da PRIVATE
    entry_point.cpp
    stdafx.h

    resource.h
    resource.rc

    splash_oxr.bmp
    stalker_cop.ico
    stalker_cs.ico
    stalker_soc.ico
)

set_property(
    DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    PROPERTY
    VS_STARTUP_PROJECT xr_3da
)

target_include_directories(xr_3da
    PRIVATE
    "${XRAY_SOURCE_DIR}/src"
    "${XRAY_SOURCE_DIR}/sdk/include"
)

target_link_libraries(xr_3da
    PRIVATE
    xrCore
    xrAPI
    xrEngine
    xrRender_GL
    xrRenderVK
    xrGame
    xrMaterialSystem
    xrParticles
)

if (WIN32)
    set_target_properties(xr_3da PROPERTIES WIN32_EXECUTABLE ON)
    target_link_libraries(xr_3da PRIVATE xrRenderPC_R4)
endif()

if (ANDROID AND XRAY_ANDROID_SHARED)
    # SDLActivity loads libmain.so and resolves the C-linkage SDL_main symbol.
    target_compile_definitions(xr_3da PRIVATE XRAY_ANDROID_SHARED)
endif()

set_target_properties(xr_3da PROPERTIES
    PREFIX ""
)

if (ANDROID AND XRAY_ANDROID_SHARED)
    set_target_properties(xr_3da PROPERTIES
        PREFIX "lib"
        OUTPUT_NAME main
    )
    if (CMAKE_HOST_WIN32)
        # Keep Ninja's final freshness check stable when the Windows-hosted
        # Android linker leaves libmain.so timestamped just before a linked
        # static archive.
        add_custom_command(TARGET xr_3da POST_BUILD
            COMMAND "${CMAKE_COMMAND}" -E touch "$<TARGET_FILE:xr_3da>"
            VERBATIM
        )
    endif()
endif()

install(TARGETS xr_3da
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
)
