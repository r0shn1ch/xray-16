add_compile_definitions(
    $<$<CONFIG:ReleaseMasterGold>:MASTER_GOLD>
    $<$<NOT:$<BOOL:${BUILD_SHARED_LIBS}>>:XRAY_STATIC_BUILD>
    $<$<CONFIG:ReleaseMasterGold>:XRAY_EXCEPTIONS=0>
    $<$<NOT:$<CONFIG:ReleaseMasterGold>>:XRAY_EXCEPTIONS=1>
)

add_compile_options(${XRAY_ENABLE_WARNINGS})

# Common engine headers validate that the luabind submodule is present. Make
# its public include root available to every engine target that includes them.
include_directories("${XRAY_SOURCE_DIR}/Externals/luabind")
include_directories("${XRAY_SOURCE_DIR}/Externals")

add_subdirectory(utils)
add_subdirectory(Layers)
add_subdirectory(xr_3da)
add_subdirectory(xrAICore)
add_subdirectory(xrUICore)
add_subdirectory(xrCDB)
add_subdirectory(xrCore)
add_subdirectory(xrEngine)
add_subdirectory(xrGame)
add_subdirectory(xrGameSpy)
add_subdirectory(xrMaterialSystem)
add_subdirectory(xrNetServer)
add_subdirectory(xrParticles)
add_subdirectory(xrPhysics)
add_subdirectory(xrScriptEngine)
add_subdirectory(xrSound)

if (XRAY_BUILD_VULKAN_TESTS)
    add_executable(android_renderer_choice_test "${XRAY_SOURCE_DIR}/tests/android_renderer_choice.cpp")
    target_include_directories(android_renderer_choice_test PRIVATE "${XRAY_SOURCE_DIR}")
    target_compile_options(android_renderer_choice_test PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>:/UNDEBUG>
        $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-UNDEBUG>)
    add_test(NAME android_renderer_choice COMMAND android_renderer_choice_test)
    function(xray_add_vulkan_state_test target source)
        add_executable(${target} "${XRAY_SOURCE_DIR}/${source}")
        target_include_directories(${target} PRIVATE
            "${XRAY_SOURCE_DIR}"
            "${XRAY_SOURCE_DIR}/src"
            "${XRAY_NVRHI_INCLUDE_DIRS}")
        target_compile_definitions(${target} PRIVATE DEBUG)
        target_compile_options(${target} PRIVATE
            $<$<CXX_COMPILER_ID:MSVC>:/UNDEBUG>
            $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-UNDEBUG>
        )
        add_test(NAME ${target} COMMAND ${target})
    endfunction()

    xray_add_vulkan_state_test(vulkan_device_resource_state_test tests/vulkan_device_resource_state.cpp)
    xray_add_vulkan_state_test(vulkan_probe_test tests/vulkan_probe.cpp)
    target_sources(vulkan_probe_test PRIVATE
        "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/VulkanProbe.cpp"
        "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/VulkanHardware.cpp")
    target_include_directories(vulkan_probe_test PRIVATE "${XRAY_SOURCE_DIR}/Externals/Vulkan-Headers/include")
    target_link_libraries(vulkan_probe_test PRIVATE SDL2::SDL2)
    xray_add_vulkan_state_test(vulkan_frame_phase_test tests/vulkan_frame_phase.cpp)
    xray_add_vulkan_state_test(vulkan_game_shader_resources_test tests/vulkan_game_shader_resources.cpp)
    target_sources(vulkan_game_shader_resources_test PRIVATE
        "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/GameShaderResources.cpp"
        "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/GameComputePipeline.cpp"
        "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/ComputePipeline.cpp"
        "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/RuntimeSlangCompiler.cpp"
        "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/ShaderReflection.cpp"
        "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/ShaderModule.cpp")
    target_include_directories(vulkan_game_shader_resources_test PRIVATE
        "${XRAY_SOURCE_DIR}/Externals/Vulkan-Headers/include"
        "${CMAKE_CURRENT_BINARY_DIR}/Layers/xrRenderVK/generated")
    xray_add_vulkan_state_test(legacy_filename_test tests/legacy_filename.cpp)
    target_compile_options(legacy_filename_test PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/utf-8>)
    xray_add_vulkan_state_test(vulkan_render_regressions_test tests/vulkan_render_regressions.cpp)
    xray_add_vulkan_state_test(vulkan_fxaa_scene_test tests/vulkan_fxaa_scene.cpp)
    target_include_directories(vulkan_fxaa_scene_test PRIVATE "${XRAY_SOURCE_DIR}/Externals/gli/external")
    xray_add_vulkan_state_test(vulkan_bloom_scene_test tests/vulkan_bloom_scene.cpp)
    target_include_directories(vulkan_bloom_scene_test PRIVATE "${XRAY_SOURCE_DIR}/Externals/gli/external")
    xray_add_vulkan_state_test(vulkan_texture_upload_test tests/vulkan_texture_upload.cpp)
    target_sources(vulkan_texture_upload_test PRIVATE "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/TextureUpload.cpp"
        "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/ImageStateTracker.cpp")
    target_include_directories(vulkan_texture_upload_test PRIVATE
        "${XRAY_SOURCE_DIR}/Externals/Vulkan-Headers/include"
        "${XRAY_NVRHI_INCLUDE_DIRS}")
    target_link_libraries(vulkan_texture_upload_test PRIVATE ${XRAY_NVRHI_LIBRARIES})
    target_compile_definitions(vulkan_texture_upload_test PRIVATE NOMINMAX)
    xray_add_vulkan_state_test(vulkan_dds_test tests/vulkan_dds.cpp)
    target_sources(vulkan_dds_test PRIVATE
        "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/Bc6Decoder.cpp"
        "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/Bc7Decoder.cpp"
        "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/DdsTexture.cpp")
    target_include_directories(vulkan_dds_test PRIVATE "${XRAY_SOURCE_DIR}/Externals/Vulkan-Headers/include"
        "${XRAY_SOURCE_DIR}/Externals/gli" "${XRAY_SOURCE_DIR}/Externals/gli/external"
        "${xray_bcdec_SOURCE_DIR}")
    xray_add_vulkan_state_test(vulkan_geometry_command_cache_test tests/vulkan_geometry_command_cache.cpp)
    target_include_directories(vulkan_geometry_command_cache_test PRIVATE
        "${XRAY_SOURCE_DIR}/Externals/Vulkan-Headers/include")
    xray_add_vulkan_state_test(vulkan_camera_state_test tests/vulkan_camera_state.cpp)
    xray_add_vulkan_state_test(vulkan_temporal_jitter_test tests/vulkan_temporal_jitter.cpp)
    target_include_directories(vulkan_temporal_jitter_test PRIVATE
        "${XRAY_SOURCE_DIR}/Externals/Vulkan-Headers/include")
    xray_add_vulkan_state_test(vulkan_object_motion_history_test tests/vulkan_object_motion_history.cpp)
    xray_add_vulkan_state_test(vulkan_pipeline_cache_test tests/vulkan_pipeline_cache.cpp)
    target_sources(vulkan_pipeline_cache_test PRIVATE
        "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/EmbeddedShaderFingerprint.cpp")
    target_include_directories(vulkan_pipeline_cache_test PRIVATE "${XRAY_SOURCE_DIR}/Externals/Vulkan-Headers/include")
    xray_add_vulkan_state_test(vulkan_game_lighting_test tests/vulkan_game_lighting.cpp)
    target_link_libraries(vulkan_game_lighting_test PRIVATE xrCore)
    xray_add_vulkan_state_test(vulkan_graphics_shader_stages_test tests/vulkan_graphics_shader_stages.cpp)
    target_include_directories(vulkan_graphics_shader_stages_test PRIVATE
        "${XRAY_SOURCE_DIR}/Externals/Vulkan-Headers/include")
    xray_add_vulkan_state_test(vulkan_shader_reflection_test tests/vulkan_shader_reflection.cpp)
    target_sources(vulkan_shader_reflection_test PRIVATE
        "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/ShaderReflection.cpp"
        "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/ShaderModule.cpp")
    target_include_directories(vulkan_shader_reflection_test PRIVATE
        "${XRAY_SOURCE_DIR}/Externals/Vulkan-Headers/include")
    xray_add_vulkan_state_test(vulkan_game_shader_interface_test tests/vulkan_game_shader_interface.cpp)
    target_sources(vulkan_game_shader_interface_test PRIVATE
        "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/ShaderReflection.cpp"
        "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/ShaderModule.cpp")
    target_include_directories(vulkan_game_shader_interface_test PRIVATE
        "${XRAY_SOURCE_DIR}/Externals/Vulkan-Headers/include"
        "${XRAY_NVRHI_INCLUDE_DIRS}")
    target_sources(vulkan_game_lighting_test PRIVATE
        "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/VulkanGameLightingMath.cpp")
    xray_add_vulkan_state_test(vulkan_slide_windows_test tests/vulkan_slide_windows.cpp)
    xray_add_vulkan_state_test(vulkan_skeleton_bones_test tests/vulkan_skeleton_bones.cpp)
    xray_add_vulkan_state_test(vulkan_skeleton_motions_test tests/vulkan_skeleton_motions.cpp)
    xray_add_vulkan_state_test(vulkan_motion_playback_test tests/vulkan_motion_playback.cpp)
    xray_add_vulkan_state_test(vulkan_particle_catalog_test tests/vulkan_particle_catalog.cpp)
    xray_add_vulkan_state_test(vulkan_gpu_particle_collision_test tests/vulkan_gpu_particle_collision.cpp)
    target_sources(vulkan_gpu_particle_collision_test PRIVATE
        "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/GpuParticleStaticCollision.cpp")
    xray_add_vulkan_state_test(vulkan_wallmark_geometry_test tests/vulkan_wallmark_geometry.cpp)
    xray_add_vulkan_state_test(vulkan_screen_copy_pass_test tests/vulkan_screen_copy_pass.cpp)
    target_sources(vulkan_screen_copy_pass_test PRIVATE
        "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/ScreenCopyPass.cpp")
    target_include_directories(vulkan_screen_copy_pass_test PRIVATE
        "${XRAY_SOURCE_DIR}/Externals/Vulkan-Headers/include")
    xray_add_vulkan_state_test(vulkan_frame_lifecycle_test tests/vulkan_frame_lifecycle.cpp)
    target_sources(vulkan_frame_lifecycle_test PRIVATE
        "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/FrameContext.cpp"
        "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/FrameImagePool.cpp"
        "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/FrameImageGraph.cpp"
        "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/ImageStateTracker.cpp")
    target_include_directories(vulkan_frame_lifecycle_test PRIVATE
        "${XRAY_SOURCE_DIR}/Externals/Vulkan-Headers/include"
        "${XRAY_NVRHI_INCLUDE_DIRS}")
    target_compile_definitions(vulkan_frame_lifecycle_test PRIVATE NOMINMAX)
    target_link_libraries(vulkan_frame_lifecycle_test PRIVATE ${XRAY_NVRHI_LIBRARIES})
    xray_add_vulkan_state_test(vulkan_frame_image_graph_test tests/vulkan_frame_image_graph.cpp)
    target_sources(vulkan_frame_image_graph_test PRIVATE
        "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/FrameImageGraph.cpp"
        "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/ImageStateTracker.cpp")
    target_include_directories(vulkan_frame_image_graph_test PRIVATE
        "${XRAY_SOURCE_DIR}/Externals/Vulkan-Headers/include")
    xray_add_vulkan_state_test(vulkan_frame_submission_graph_test tests/vulkan_frame_submission_graph.cpp)
    target_include_directories(vulkan_frame_submission_graph_test PRIVATE
        "${XRAY_SOURCE_DIR}/Externals/Vulkan-Headers/include")
    xray_add_vulkan_state_test(vulkan_frame_image_pool_test tests/vulkan_frame_image_pool.cpp)
    target_sources(vulkan_frame_image_pool_test PRIVATE
        "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/FrameImagePool.cpp")
    target_include_directories(vulkan_frame_image_pool_test PRIVATE
        "${XRAY_SOURCE_DIR}/Externals/Vulkan-Headers/include"
        "${XRAY_NVRHI_INCLUDE_DIRS}")
    target_link_libraries(vulkan_frame_image_pool_test PRIVATE ${XRAY_NVRHI_LIBRARIES})
    xray_add_vulkan_state_test(vulkan_buffer_upload_test tests/vulkan_buffer_upload.cpp)
    target_sources(vulkan_buffer_upload_test PRIVATE
        "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/BufferUpload.cpp"
        "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/BufferResource.cpp")
    target_include_directories(vulkan_buffer_upload_test PRIVATE
        "${XRAY_SOURCE_DIR}/Externals/Vulkan-Headers/include")
    target_link_libraries(vulkan_buffer_upload_test PRIVATE nvrhi)
    xray_add_vulkan_state_test(vulkan_game_capabilities_test tests/vulkan_game_capabilities.cpp)
    target_sources(vulkan_game_capabilities_test PRIVATE
        "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/VulkanHardware.cpp")
    target_include_directories(vulkan_game_capabilities_test PRIVATE
        "${XRAY_SOURCE_DIR}/Externals/Vulkan-Headers/include")
    xray_add_vulkan_state_test(vulkan_surface_lifecycle_test tests/vulkan_surface_lifecycle.cpp)
    xray_add_vulkan_state_test(vulkan_frame_metrics_test tests/vulkan_frame_metrics.cpp)
    xray_add_vulkan_state_test(vulkan_frame_benchmark_test tests/vulkan_frame_benchmark.cpp)
    target_sources(vulkan_wallmark_geometry_test PRIVATE
        "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/WallmarkGeometry.cpp")
    target_sources(vulkan_particle_catalog_test PRIVATE
        "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/ParticleCatalog.cpp"
        "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/EngineParticleSource.cpp")
    target_include_directories(vulkan_particle_catalog_test PRIVATE
        "${XRAY_SOURCE_DIR}/src" $<TARGET_PROPERTY:xrRenderVK,INCLUDE_DIRECTORIES>)
    target_compile_definitions(vulkan_particle_catalog_test PRIVATE SDL_MAIN_HANDLED)
    target_link_libraries(vulkan_particle_catalog_test PRIVATE xrCore)
    target_sources(vulkan_motion_playback_test PRIVATE
        "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/MotionPlayback.cpp")
    target_sources(vulkan_skeleton_motions_test PRIVATE
        "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/SkeletonMotions.cpp")
    target_sources(vulkan_skeleton_bones_test PRIVATE
        "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/SkeletonBones.cpp")
    target_include_directories(vulkan_skeleton_bones_test PRIVATE
        "${XRAY_SOURCE_DIR}/Externals/Vulkan-Headers/include")
    target_sources(vulkan_slide_windows_test PRIVATE
        "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/SlidingWindows.cpp"
        "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/ModelGeometry.cpp"
        "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/LevelModels.cpp"
        "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/SpecialVisuals.cpp")

    add_executable(vulkan_deferred_pass_test
        "${XRAY_SOURCE_DIR}/tests/vulkan_deferred_pass.cpp"
    )
    target_include_directories(vulkan_deferred_pass_test PRIVATE
        "${XRAY_SOURCE_DIR}"
        "${XRAY_SOURCE_DIR}/src"
        "${XRAY_SOURCE_DIR}/Externals/Vulkan-Headers/include"
    )
    target_link_libraries(vulkan_deferred_pass_test PRIVATE
        xrRenderVK xrEngine xrAPI xrScriptEngine xrCDB xrCore xrMaterialSystem)
    target_compile_definitions(vulkan_deferred_pass_test PRIVATE DEBUG)
    target_compile_options(vulkan_deferred_pass_test PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>:/UNDEBUG>
        $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-UNDEBUG>
    )
    add_test(NAME vulkan_deferred_pass COMMAND vulkan_deferred_pass_test)
    # Executes shipped SPIR-V and reads back results from a headless Vulkan
    # device. Use XRAY_VK_GPU_TEST_DEVICE_INDEX to pin selectable tests to a
    # physical device (Vulkan enumeration order), for example 1 on a hybrid
    # graphics laptop with the discrete adapter listed second.
    set(XRAY_VK_GPU_TEST_DEVICE_INDEX "" CACHE STRING
        "Physical Vulkan device index for GPU tests; unset selects each test's default")
    set(_xray_vk_gpu_device_args)
    set(_xray_vk_gpu_device_option_args)
    if (NOT XRAY_VK_GPU_TEST_DEVICE_INDEX STREQUAL "")
        if (NOT XRAY_VK_GPU_TEST_DEVICE_INDEX MATCHES "^-?[0-9]+$")
            message(FATAL_ERROR "XRAY_VK_GPU_TEST_DEVICE_INDEX must be an integer")
        endif()
        set(_xray_vk_gpu_device_args "${XRAY_VK_GPU_TEST_DEVICE_INDEX}")
        set(_xray_vk_gpu_device_option_args --device-index "${XRAY_VK_GPU_TEST_DEVICE_INDEX}")
    endif()
    set(_xray_vk_motion_device_index -1)
    if (NOT XRAY_VK_GPU_TEST_DEVICE_INDEX STREQUAL "")
        set(_xray_vk_motion_device_index "${XRAY_VK_GPU_TEST_DEVICE_INDEX}")
    endif()
    find_package(Vulkan QUIET)
    if (Vulkan_FOUND AND NOT ANDROID)
        add_executable(vulkan_bloom_gpu_test
            "${XRAY_SOURCE_DIR}/tests/vulkan_bloom_gpu.cpp"
            "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/FrameImageGraph.cpp"
            "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/ImageStateTracker.cpp"
            "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/ScreenCopyPass.cpp"
            "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/ShaderModule.cpp")
        target_include_directories(vulkan_bloom_gpu_test PRIVATE
            "${XRAY_SOURCE_DIR}"
            "${XRAY_SOURCE_DIR}/Externals/Vulkan-Headers/include"
            "${XRAY_NVRHI_INCLUDE_DIRS}")
        target_link_libraries(vulkan_bloom_gpu_test PRIVATE Vulkan::Vulkan)
        target_compile_options(vulkan_bloom_gpu_test PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/EHsc>)
        add_test(NAME vulkan_bloom_gpu COMMAND vulkan_bloom_gpu_test ${_xray_vk_gpu_device_args})
        set_tests_properties(vulkan_bloom_gpu PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 30)
        add_executable(vulkan_gbuffer_depth_gpu_test
            "${XRAY_SOURCE_DIR}/tests/vulkan_gbuffer_depth_gpu.cpp"
            "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/GBufferTargets.cpp"
            "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/GBufferRenderPass.cpp"
            "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/FrameImagePool.cpp"
            "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/FrameImageGraph.cpp"
            "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/ImageStateTracker.cpp")
        target_include_directories(vulkan_gbuffer_depth_gpu_test PRIVATE
            "${XRAY_SOURCE_DIR}" "${XRAY_SOURCE_DIR}/Externals/Vulkan-Headers/include"
            "${XRAY_NVRHI_INCLUDE_DIRS}")
        target_link_libraries(vulkan_gbuffer_depth_gpu_test PRIVATE Vulkan::Vulkan ${XRAY_NVRHI_LIBRARIES})
        target_compile_definitions(vulkan_gbuffer_depth_gpu_test PRIVATE NOMINMAX)
        add_test(NAME vulkan_gbuffer_depth_gpu COMMAND vulkan_gbuffer_depth_gpu_test ${_xray_vk_gpu_device_args})
        set_tests_properties(vulkan_gbuffer_depth_gpu PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 30)
        add_executable(vulkan_hiz_gpu_test
            "${XRAY_SOURCE_DIR}/tests/vulkan_hiz_gpu.cpp"
            "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/HiZPyramid.cpp"
            "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/FrameImagePool.cpp"
            "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/FrameImageGraph.cpp"
            "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/ImageStateTracker.cpp"
            "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/ShaderModule.cpp"
            "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/ComputePipeline.cpp")
        target_include_directories(vulkan_hiz_gpu_test PRIVATE
            "${XRAY_SOURCE_DIR}" "${XRAY_SOURCE_DIR}/Externals/Vulkan-Headers/include"
            "${XRAY_NVRHI_INCLUDE_DIRS}")
        target_link_libraries(vulkan_hiz_gpu_test PRIVATE Vulkan::Vulkan ${XRAY_NVRHI_LIBRARIES})
        target_compile_definitions(vulkan_hiz_gpu_test PRIVATE NOMINMAX)
        target_compile_options(vulkan_hiz_gpu_test PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/EHsc>)
        add_test(NAME vulkan_hiz_gpu COMMAND vulkan_hiz_gpu_test ${_xray_vk_gpu_device_args})
        set_tests_properties(vulkan_hiz_gpu PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 30)
        add_executable(vulkan_motion_gpu_test
            "${XRAY_SOURCE_DIR}/tests/vulkan_motion_gpu.cpp"
            "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/MotionVectors.cpp"
            "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/ObjectVelocityPass.cpp"
            "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/FrameImagePool.cpp"
            "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/FrameImageGraph.cpp"
            "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/ImageStateTracker.cpp"
            "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/ShaderModule.cpp"
            "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/ComputePipeline.cpp")
        target_include_directories(vulkan_motion_gpu_test PRIVATE
            "${XRAY_SOURCE_DIR}" "${XRAY_SOURCE_DIR}/Externals/Vulkan-Headers/include"
            "${XRAY_NVRHI_INCLUDE_DIRS}")
        target_link_libraries(vulkan_motion_gpu_test PRIVATE Vulkan::Vulkan ${XRAY_NVRHI_LIBRARIES})
        target_compile_definitions(vulkan_motion_gpu_test PRIVATE NOMINMAX)
        target_compile_options(vulkan_motion_gpu_test PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/EHsc>)
        add_executable(vulkan_ray_capabilities_gpu_test
            "${XRAY_SOURCE_DIR}/tests/vulkan_ray_capabilities_gpu.cpp"
            "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/VulkanHardware.cpp"
            "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/AccelerationStructure.cpp"
            "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/LevelRayScene.cpp"
            "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/RayQueryBatch.cpp"
            "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/RaySurface.cpp"
            "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/IndirectRayFrames.cpp"
            "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/IndirectSunPass.cpp"
            "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/ReferencePathTracePass.cpp"
            "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/StorageComputePass.cpp"
            "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/GpuParticleCompute.cpp"
            "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/GpuParticleUpload.cpp"
            "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/GpuParticleUniforms.cpp"
            "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/VsmPageAllocator.cpp"
            "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/VsmDemandPass.cpp"
            "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/VsmStaticCache.cpp"
            "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/VsmFrameResources.cpp"
            "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/VsmClipmap.cpp"
            "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/VsmAtlas.cpp"
            "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/VsmPageRender.cpp"
            "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/VsmResolvePass.cpp"
            "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/VsmCasterBinning.cpp"
            "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/VsmInvalidation.cpp"
            "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/FrameImagePool.cpp"
            "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/FrameImageGraph.cpp"
            "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/ImageStateTracker.cpp"
            "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/RayGeneratePass.cpp"
            "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/RayVisibilityPass.cpp"
            "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/BindlessTextureTable.cpp"
            "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/ComputePipeline.cpp"
            "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/ShaderModule.cpp"
            "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/BufferUpload.cpp"
            "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/BufferResource.cpp")
        target_include_directories(vulkan_ray_capabilities_gpu_test PRIVATE
            "${XRAY_SOURCE_DIR}" "${XRAY_SOURCE_DIR}/src"
            "${XRAY_SOURCE_DIR}/Externals/Vulkan-Headers/include")
        target_link_libraries(vulkan_ray_capabilities_gpu_test PRIVATE Vulkan::Vulkan ${XRAY_NVRHI_LIBRARIES} xrCore)
        target_compile_definitions(vulkan_ray_capabilities_gpu_test PRIVATE NOMINMAX)
        add_test(NAME vulkan_ray_capabilities_gpu COMMAND vulkan_ray_capabilities_gpu_test
            ${_xray_vk_gpu_device_option_args})
        set_tests_properties(vulkan_ray_capabilities_gpu PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 120)
        add_test(NAME vulkan_motion_gpu COMMAND vulkan_motion_gpu_test ${_xray_vk_motion_device_index})
        set_tests_properties(vulkan_motion_gpu PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 30)
        add_test(NAME vulkan_motion_gpu_fallback COMMAND vulkan_motion_gpu_test ${_xray_vk_motion_device_index} fallback)
        set_tests_properties(vulkan_motion_gpu_fallback PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 30)
        add_test(NAME vulkan_motion_gpu_reprojected COMMAND vulkan_motion_gpu_test ${_xray_vk_motion_device_index} reprojected)
        set_tests_properties(vulkan_motion_gpu_reprojected PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 30)
        add_test(NAME vulkan_motion_gpu_object COMMAND vulkan_motion_gpu_test ${_xray_vk_motion_device_index} object)
        set_tests_properties(vulkan_motion_gpu_object PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 30)
        add_test(NAME vulkan_motion_gpu_object_indirect COMMAND vulkan_motion_gpu_test
            ${_xray_vk_motion_device_index} indirect)
        set_tests_properties(vulkan_motion_gpu_object_indirect PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 30)
        add_test(NAME vulkan_motion_gpu_deform COMMAND vulkan_motion_gpu_test ${_xray_vk_motion_device_index} deform)
        set_tests_properties(vulkan_motion_gpu_deform PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 30)
        add_test(NAME vulkan_motion_gpu_wind COMMAND vulkan_motion_gpu_test ${_xray_vk_motion_device_index} wind)
        set_tests_properties(vulkan_motion_gpu_wind PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 30)
        add_test(NAME vulkan_motion_gpu_offset COMMAND vulkan_motion_gpu_test ${_xray_vk_motion_device_index} offset)
        set_tests_properties(vulkan_motion_gpu_offset PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 30)
        add_test(NAME vulkan_motion_gpu_cutout COMMAND vulkan_motion_gpu_test ${_xray_vk_motion_device_index} cutout)
        set_tests_properties(vulkan_motion_gpu_cutout PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 30)
        add_test(NAME vulkan_motion_gpu_cutout_solid COMMAND vulkan_motion_gpu_test ${_xray_vk_motion_device_index} cutout_solid)
        set_tests_properties(vulkan_motion_gpu_cutout_solid PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 30)
        add_test(NAME vulkan_motion_gpu_skinned COMMAND vulkan_motion_gpu_test ${_xray_vk_motion_device_index} skinned)
        set_tests_properties(vulkan_motion_gpu_skinned PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 30)
        add_test(NAME vulkan_motion_gpu_skinned_blend COMMAND vulkan_motion_gpu_test ${_xray_vk_motion_device_index} skinned_blend)
        set_tests_properties(vulkan_motion_gpu_skinned_blend PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 30)
        add_test(NAME vulkan_motion_gpu_skinned_cutout COMMAND vulkan_motion_gpu_test ${_xray_vk_motion_device_index} skinned_cutout)
        set_tests_properties(vulkan_motion_gpu_skinned_cutout PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 30)
        add_executable(vulkan_temporal_resolve_gpu_test
            "${XRAY_SOURCE_DIR}/tests/vulkan_temporal_resolve_gpu.cpp"
            "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/TemporalResolve.cpp"
            "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/FrameImagePool.cpp"
            "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/ShaderModule.cpp"
            "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/ComputePipeline.cpp")
        target_include_directories(vulkan_temporal_resolve_gpu_test PRIVATE
            "${XRAY_SOURCE_DIR}" "${XRAY_SOURCE_DIR}/Externals/Vulkan-Headers/include")
        target_link_libraries(vulkan_temporal_resolve_gpu_test PRIVATE Vulkan::Vulkan ${XRAY_NVRHI_LIBRARIES})
        add_test(NAME vulkan_temporal_resolve_gpu COMMAND vulkan_temporal_resolve_gpu_test ${_xray_vk_gpu_device_args})
        set_tests_properties(vulkan_temporal_resolve_gpu PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 30)
        add_executable(vulkan_temporal_exposure_gpu_test
            "${XRAY_SOURCE_DIR}/tests/vulkan_temporal_exposure_gpu.cpp"
            "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/TemporalExposure.cpp"
            "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/FrameImagePool.cpp"
            "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/ShaderModule.cpp"
            "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/ComputePipeline.cpp")
        target_include_directories(vulkan_temporal_exposure_gpu_test PRIVATE
            "${XRAY_SOURCE_DIR}" "${XRAY_SOURCE_DIR}/Externals/Vulkan-Headers/include"
            "${XRAY_NVRHI_INCLUDE_DIRS}")
        target_link_libraries(vulkan_temporal_exposure_gpu_test PRIVATE Vulkan::Vulkan ${XRAY_NVRHI_LIBRARIES})
        target_compile_definitions(vulkan_temporal_exposure_gpu_test PRIVATE NOMINMAX)
        target_compile_options(vulkan_temporal_exposure_gpu_test PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/EHsc>)
        add_test(NAME vulkan_temporal_exposure_gpu COMMAND vulkan_temporal_exposure_gpu_test ${_xray_vk_gpu_device_args})
        set_tests_properties(vulkan_temporal_exposure_gpu PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 30)
        add_executable(vulkan_ao_gpu_test "${XRAY_SOURCE_DIR}/tests/vulkan_ao_gpu.cpp")
        target_include_directories(vulkan_ao_gpu_test PRIVATE
            "${XRAY_SOURCE_DIR}" "${XRAY_SOURCE_DIR}/Externals/Vulkan-Headers/include")
        target_link_libraries(vulkan_ao_gpu_test PRIVATE Vulkan::Vulkan)
        add_test(NAME vulkan_ao_gpu COMMAND vulkan_ao_gpu_test ${_xray_vk_gpu_device_args})
        set_tests_properties(vulkan_ao_gpu PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 30)
        add_executable(vulkan_bc7_gpu_test "${XRAY_SOURCE_DIR}/tests/vulkan_bc7_gpu.cpp"
            "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/Bc7Decoder.cpp")
        target_include_directories(vulkan_bc7_gpu_test PRIVATE "${XRAY_SOURCE_DIR}/Externals/Vulkan-Headers/include")
        target_link_libraries(vulkan_bc7_gpu_test PRIVATE Vulkan::Vulkan)
        target_compile_options(vulkan_bc7_gpu_test PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/EHsc>)
        add_test(NAME vulkan_bc7_gpu COMMAND vulkan_bc7_gpu_test ${_xray_vk_gpu_device_args})
        set_tests_properties(vulkan_bc7_gpu PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 30)
        add_executable(vulkan_light_grid_gpu_test
            "${XRAY_SOURCE_DIR}/tests/vulkan_light_grid_gpu.cpp"
            "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/LightGrid.cpp"
            "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/FrameImageGraph.cpp"
            "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/ImageStateTracker.cpp"
            "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/ComputePipeline.cpp"
            "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/BufferResource.cpp"
            "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/ShaderModule.cpp"
            "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/ScenePass.cpp")
        target_include_directories(vulkan_light_grid_gpu_test PRIVATE "${XRAY_SOURCE_DIR}" "${XRAY_SOURCE_DIR}/Externals/Vulkan-Headers/include")
        target_link_libraries(vulkan_light_grid_gpu_test PRIVATE Vulkan::Vulkan nvrhi)
        target_compile_options(vulkan_light_grid_gpu_test PRIVATE
            $<$<CXX_COMPILER_ID:MSVC>:/UNDEBUG>
            $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-UNDEBUG>
            $<$<CXX_COMPILER_ID:MSVC>:/EHsc>)
        add_test(NAME vulkan_light_grid_gpu COMMAND vulkan_light_grid_gpu_test ${_xray_vk_gpu_device_args})
        add_executable(vulkan_frame_buffer_gpu_test
            "${XRAY_SOURCE_DIR}/tests/vulkan_frame_buffer_gpu.cpp"
            "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/FrameImageGraph.cpp"
            "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/ImageStateTracker.cpp")
        target_include_directories(vulkan_frame_buffer_gpu_test PRIVATE
            "${XRAY_SOURCE_DIR}" "${XRAY_SOURCE_DIR}/Externals/Vulkan-Headers/include")
        target_link_libraries(vulkan_frame_buffer_gpu_test PRIVATE Vulkan::Vulkan)
        add_test(NAME vulkan_frame_buffer_gpu COMMAND vulkan_frame_buffer_gpu_test ${_xray_vk_gpu_device_args})
        set_tests_properties(vulkan_frame_buffer_gpu PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 30)
        add_executable(vulkan_local_shadow_gpu_test
            "${XRAY_SOURCE_DIR}/tests/vulkan_local_shadow_gpu.cpp"
            "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/LocalShadowTargets.cpp"
            "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/FrameImagePool.cpp"
            "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/BufferResource.cpp")
        target_include_directories(vulkan_local_shadow_gpu_test PRIVATE
            "${XRAY_SOURCE_DIR}" "${XRAY_SOURCE_DIR}/src"
            "${XRAY_SOURCE_DIR}/Externals/Vulkan-Headers/include"
            "${XRAY_NVRHI_INCLUDE_DIRS}"
            $<TARGET_PROPERTY:xrRenderVK,INCLUDE_DIRECTORIES>)
        target_link_libraries(vulkan_local_shadow_gpu_test PRIVATE Vulkan::Vulkan ${XRAY_NVRHI_LIBRARIES})
        target_compile_definitions(vulkan_local_shadow_gpu_test PRIVATE NOMINMAX)
        target_compile_options(vulkan_local_shadow_gpu_test PRIVATE
            $<$<CXX_COMPILER_ID:MSVC>:/UNDEBUG>
            $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-UNDEBUG>
            $<$<CXX_COMPILER_ID:MSVC>:/EHsc>)
        add_test(NAME vulkan_local_shadow_gpu COMMAND vulkan_local_shadow_gpu_test ${_xray_vk_gpu_device_args})
        add_executable(vulkan_surface_channels_test
            "${XRAY_SOURCE_DIR}/tests/vulkan_surface_channels.cpp"
            "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/VulkanGameLightingMath.cpp")
        target_include_directories(vulkan_surface_channels_test PRIVATE
            "${XRAY_SOURCE_DIR}/src"
            "${XRAY_SOURCE_DIR}/Externals/Vulkan-Headers/include")
        target_link_libraries(vulkan_surface_channels_test PRIVATE Vulkan::Vulkan nvrhi xrCore)
        target_link_options(vulkan_surface_channels_test PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/STACK:8388608>)
        target_compile_options(vulkan_surface_channels_test PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/EHsc>)
        add_test(NAME vulkan_surface_channels COMMAND vulkan_surface_channels_test
            "${XRAY_SOURCE_DIR}/res/gamedata/shaders/vk" ${_xray_vk_gpu_device_option_args})
        # Validation reports errors without changing the executable's exit code.
        # A numerical image match must not hide invalid Vulkan synchronization.
        set_tests_properties(vulkan_bloom_gpu vulkan_hiz_gpu vulkan_motion_gpu vulkan_motion_gpu_fallback vulkan_motion_gpu_reprojected vulkan_motion_gpu_object vulkan_motion_gpu_object_indirect vulkan_motion_gpu_deform vulkan_motion_gpu_offset vulkan_motion_gpu_cutout vulkan_motion_gpu_cutout_solid vulkan_motion_gpu_skinned vulkan_motion_gpu_skinned_blend vulkan_motion_gpu_skinned_cutout vulkan_ray_capabilities_gpu vulkan_temporal_resolve_gpu vulkan_temporal_exposure_gpu vulkan_ao_gpu vulkan_bc7_gpu vulkan_light_grid_gpu vulkan_frame_buffer_gpu vulkan_local_shadow_gpu vulkan_surface_channels
            PROPERTIES FAIL_REGULAR_EXPRESSION "Validation Error;SYNC-HAZARD")
    endif()
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if (Python3_Interpreter_FOUND)
        add_test(NAME gl_depth_only COMMAND "${Python3_EXECUTABLE}"
            "${XRAY_SOURCE_DIR}/tests/test_gl_depth_only.py")
        set_tests_properties(gl_depth_only PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 30)
        add_test(NAME vulkan_detail_spirv COMMAND "${Python3_EXECUTABLE}"
            "${XRAY_SOURCE_DIR}/tests/vulkan_detail_spirv.py"
            "${XRAY_SOURCE_DIR}/res/gamedata/shaders")
        add_test(NAME vulkan_game_shader_assets COMMAND "${Python3_EXECUTABLE}"
            -m unittest tests.test_vulkan_game_shaders)
        set_tests_properties(vulkan_game_shader_assets PROPERTIES WORKING_DIRECTORY "${XRAY_SOURCE_DIR}")
    endif()

    add_executable(vulkan_scene_pass_test
        "${XRAY_SOURCE_DIR}/tests/vulkan_scene_pass.cpp"
        "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/ScenePass.cpp")
    target_include_directories(vulkan_scene_pass_test PRIVATE
        "${XRAY_SOURCE_DIR}" "${XRAY_SOURCE_DIR}/Externals/Vulkan-Headers/include"
        "${XRAY_NVRHI_INCLUDE_DIRS}")
    target_compile_definitions(vulkan_scene_pass_test PRIVATE NOMINMAX)
    target_compile_options(vulkan_scene_pass_test PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>:/UNDEBUG>
        $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-UNDEBUG>)
    add_test(NAME vulkan_scene_pass COMMAND vulkan_scene_pass_test)

    add_executable(vulkan_level_visibility_test
        "${XRAY_SOURCE_DIR}/tests/vulkan_level_visibility.cpp"
    )
    target_include_directories(vulkan_level_visibility_test PRIVATE
        "${XRAY_SOURCE_DIR}"
        "${XRAY_SOURCE_DIR}/src"
        $<TARGET_PROPERTY:xrRenderVK,INCLUDE_DIRECTORIES>
    )
    target_compile_definitions(vulkan_level_visibility_test PRIVATE DEBUG)
    target_compile_options(vulkan_level_visibility_test PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>:/UNDEBUG>
        $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-UNDEBUG>
    )
    target_link_libraries(vulkan_level_visibility_test PRIVATE xrRenderVK xrCDB xrCore xrMiscMath)
    add_test(NAME vulkan_level_visibility COMMAND vulkan_level_visibility_test)

    add_executable(vulkan_level_models_test
        "${XRAY_SOURCE_DIR}/tests/vulkan_level_models.cpp"
    )
    target_include_directories(vulkan_level_models_test PRIVATE
        "${XRAY_SOURCE_DIR}"
        "${XRAY_SOURCE_DIR}/src"
        $<TARGET_PROPERTY:xrRenderVK,INCLUDE_DIRECTORIES>
    )
    target_compile_definitions(vulkan_level_models_test PRIVATE DEBUG)
    target_compile_options(vulkan_level_models_test PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>:/UNDEBUG>
        $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-UNDEBUG>
    )
    target_link_libraries(vulkan_level_models_test PRIVATE xrRenderVK xrCDB xrCore)
    add_test(NAME vulkan_level_models COMMAND vulkan_level_models_test)

    add_executable(vulkan_cluster_lod_test
        "${XRAY_SOURCE_DIR}/tests/vulkan_cluster_lod.cpp"
    )
    target_include_directories(vulkan_cluster_lod_test PRIVATE
        "${XRAY_SOURCE_DIR}"
        "${XRAY_SOURCE_DIR}/src"
        "${XRAY_SOURCE_DIR}/Externals/Vulkan-Headers/include"
        $<TARGET_PROPERTY:xrRenderVK,INCLUDE_DIRECTORIES>
    )
    target_compile_definitions(vulkan_cluster_lod_test PRIVATE DEBUG NOMINMAX)
    target_compile_options(vulkan_cluster_lod_test PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>:/UNDEBUG>
        $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-UNDEBUG>
    )
    target_link_libraries(vulkan_cluster_lod_test PRIVATE xrRenderVK xrCDB xrCore)
    add_test(NAME vulkan_cluster_lod COMMAND vulkan_cluster_lod_test)

    add_executable(vulkan_cluster_visibility_gpu_test
        "${XRAY_SOURCE_DIR}/tests/vulkan_cluster_visibility_gpu.cpp")
    target_include_directories(vulkan_cluster_visibility_gpu_test PRIVATE
        "${XRAY_SOURCE_DIR}"
        "${XRAY_SOURCE_DIR}/src"
        "${XRAY_SOURCE_DIR}/Externals/Vulkan-Headers/include"
        $<TARGET_PROPERTY:xrRenderVK,INCLUDE_DIRECTORIES>)
    target_compile_definitions(vulkan_cluster_visibility_gpu_test PRIVATE DEBUG NOMINMAX)
    target_compile_options(vulkan_cluster_visibility_gpu_test PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>:/UNDEBUG>
        $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-UNDEBUG>
    )
    target_link_libraries(vulkan_cluster_visibility_gpu_test PRIVATE xrRenderVK xrCDB xrCore Vulkan::Vulkan)
    add_test(NAME vulkan_cluster_visibility_gpu
        COMMAND vulkan_cluster_visibility_gpu_test ${_xray_vk_gpu_device_option_args})
    set_tests_properties(vulkan_cluster_visibility_gpu PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 30)

    add_executable(vulkan_shader_material_library_test
        "${XRAY_SOURCE_DIR}/tests/vulkan_shader_material_library.cpp")
    target_include_directories(vulkan_shader_material_library_test PRIVATE
        "${XRAY_SOURCE_DIR}" "${XRAY_SOURCE_DIR}/src"
        $<TARGET_PROPERTY:xrRenderVK,INCLUDE_DIRECTORIES>)
    target_compile_definitions(vulkan_shader_material_library_test PRIVATE SDL_MAIN_HANDLED)
    target_link_libraries(vulkan_shader_material_library_test PRIVATE xrRenderVK xrEngine xrScriptEngine xrCDB xrCore)
    add_test(NAME vulkan_shader_material_library COMMAND vulkan_shader_material_library_test)

    add_executable(vulkan_level_pass_graph_test
        "${XRAY_SOURCE_DIR}/tests/vulkan_level_pass_graph.cpp")
    target_include_directories(vulkan_level_pass_graph_test PRIVATE "${XRAY_SOURCE_DIR}")
    target_compile_options(vulkan_level_pass_graph_test PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>:/UNDEBUG>
        $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-UNDEBUG>)
    add_test(NAME vulkan_level_pass_graph COMMAND vulkan_level_pass_graph_test)

    add_executable(vulkan_detail_assets_test
        "${XRAY_SOURCE_DIR}/tests/vulkan_detail_assets.cpp"
        "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/DetailAssets.cpp")
    target_include_directories(vulkan_detail_assets_test PRIVATE "${XRAY_SOURCE_DIR}")
    target_compile_options(vulkan_detail_assets_test PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>:/UNDEBUG>
        $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-UNDEBUG>)
    add_test(NAME vulkan_detail_assets COMMAND vulkan_detail_assets_test)

    add_executable(vulkan_fog_parameters_test "${XRAY_SOURCE_DIR}/tests/vulkan_fog_parameters.cpp")
    target_include_directories(vulkan_fog_parameters_test PRIVATE "${XRAY_SOURCE_DIR}")
    target_compile_options(vulkan_fog_parameters_test PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>:/UNDEBUG>
        $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-UNDEBUG>)
    add_test(NAME vulkan_fog_parameters COMMAND vulkan_fog_parameters_test)

    add_executable(vulkan_local_shadow_targets_test
        "${XRAY_SOURCE_DIR}/tests/vulkan_local_shadow_targets.cpp"
        "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/LocalShadowTargets.cpp"
        "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/FrameImagePool.cpp"
        "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/BufferResource.cpp")
    target_include_directories(vulkan_local_shadow_targets_test PRIVATE
        "${XRAY_SOURCE_DIR}" "${XRAY_SOURCE_DIR}/src"
        "${XRAY_SOURCE_DIR}/Externals/Vulkan-Headers/include"
        "${XRAY_NVRHI_INCLUDE_DIRS}"
        $<TARGET_PROPERTY:xrRenderVK,INCLUDE_DIRECTORIES>)
    target_link_libraries(vulkan_local_shadow_targets_test PRIVATE Vulkan::Vulkan ${XRAY_NVRHI_LIBRARIES})
    target_compile_definitions(vulkan_local_shadow_targets_test PRIVATE NOMINMAX)
    target_compile_options(vulkan_local_shadow_targets_test PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>:/UNDEBUG>
        $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-UNDEBUG>)
    add_test(NAME vulkan_local_shadow_targets COMMAND vulkan_local_shadow_targets_test)

    add_executable(vulkan_water_targets_test
        "${XRAY_SOURCE_DIR}/tests/vulkan_water_targets.cpp"
        "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/WaterTargets.cpp"
        "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/FrameImagePool.cpp"
        "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/FrameImageGraph.cpp"
        "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/ImageStateTracker.cpp"
        "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/BufferResource.cpp")
    target_include_directories(vulkan_water_targets_test PRIVATE
        "${XRAY_SOURCE_DIR}" "${XRAY_SOURCE_DIR}/src"
        "${XRAY_SOURCE_DIR}/Externals/Vulkan-Headers/include"
        "${XRAY_NVRHI_INCLUDE_DIRS}"
        $<TARGET_PROPERTY:xrRenderVK,INCLUDE_DIRECTORIES>)
    target_link_libraries(vulkan_water_targets_test PRIVATE Vulkan::Vulkan ${XRAY_NVRHI_LIBRARIES})
    target_compile_definitions(vulkan_water_targets_test PRIVATE NOMINMAX)
    target_compile_options(vulkan_water_targets_test PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>:/UNDEBUG>
        $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-UNDEBUG>)
    add_test(NAME vulkan_water_targets COMMAND vulkan_water_targets_test)

    add_executable(vulkan_gbuffer_targets_test
        "${XRAY_SOURCE_DIR}/tests/vulkan_gbuffer_targets.cpp"
    )
    target_link_libraries(vulkan_gbuffer_targets_test PRIVATE
        xrRenderVK xrEngine xrAPI xrScriptEngine xrCDB xrCore xrMaterialSystem)
    target_include_directories(vulkan_gbuffer_targets_test PRIVATE
        "${XRAY_SOURCE_DIR}" "${XRAY_SOURCE_DIR}/Externals/Vulkan-Headers/include")
    target_compile_options(vulkan_gbuffer_targets_test PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>:/UNDEBUG>
        $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-UNDEBUG>)
    add_test(NAME vulkan_gbuffer_targets COMMAND vulkan_gbuffer_targets_test)

    add_executable(vulkan_water_local_pipeline_test
        "${XRAY_SOURCE_DIR}/tests/vulkan_water_local_pipeline.cpp")
    target_link_libraries(vulkan_water_local_pipeline_test PRIVATE
        xrRenderVK xrEngine xrAPI xrScriptEngine xrCDB xrCore xrMaterialSystem)
    target_include_directories(vulkan_water_local_pipeline_test PRIVATE
        "${XRAY_SOURCE_DIR}" "${XRAY_SOURCE_DIR}/Externals/Vulkan-Headers/include")
    target_compile_options(vulkan_water_local_pipeline_test PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>:/UNDEBUG>
        $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-UNDEBUG>)
    add_test(NAME vulkan_water_local_pipeline COMMAND vulkan_water_local_pipeline_test)

    add_executable(vulkan_visual_catalog_test
        "${XRAY_SOURCE_DIR}/tests/vulkan_visual_catalog.cpp"
    )
    target_include_directories(vulkan_visual_catalog_test PRIVATE
        "${XRAY_SOURCE_DIR}"
        "${XRAY_SOURCE_DIR}/src"
        $<TARGET_PROPERTY:xrRenderVK,INCLUDE_DIRECTORIES>
    )
    target_compile_definitions(vulkan_visual_catalog_test PRIVATE DEBUG)
    target_compile_options(vulkan_visual_catalog_test PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>:/UNDEBUG>
        $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-UNDEBUG>
    )
    target_link_libraries(vulkan_visual_catalog_test PRIVATE xrRenderVK xrCDB xrCore)
    add_test(NAME vulkan_visual_catalog COMMAND vulkan_visual_catalog_test)

    xray_add_vulkan_state_test(game_news_serialization_test tests/game_news_serialization.cpp)
    target_sources(game_news_serialization_test PRIVATE
        "${XRAY_SOURCE_DIR}/src/xrGame/game_news_serialization.cpp")
    target_link_libraries(game_news_serialization_test PRIVATE xrCore)
    add_test(NAME game_news_serialization COMMAND game_news_serialization_test)

    xray_add_vulkan_state_test(map_location_serialization tests/map_location_serialization.cpp)

    add_executable(vulkan_model_geometry_test
        "${XRAY_SOURCE_DIR}/tests/vulkan_model_geometry.cpp"
    )
    target_include_directories(vulkan_model_geometry_test PRIVATE
        "${XRAY_SOURCE_DIR}"
        "${XRAY_SOURCE_DIR}/src"
        $<TARGET_PROPERTY:xrRenderVK,INCLUDE_DIRECTORIES>
    )
    target_compile_definitions(vulkan_model_geometry_test PRIVATE SDL_MAIN_HANDLED)
    target_link_libraries(vulkan_model_geometry_test PRIVATE xrRenderVK xrEngine xrAPI xrScriptEngine xrCDB xrCore xrMaterialSystem)
    add_test(NAME vulkan_model_geometry COMMAND vulkan_model_geometry_test)

    add_executable(vulkan_kinematics_test
        "${XRAY_SOURCE_DIR}/tests/vulkan_kinematics.cpp"
    )
    target_include_directories(vulkan_kinematics_test PRIVATE
        "${XRAY_SOURCE_DIR}"
        "${XRAY_SOURCE_DIR}/src"
        $<TARGET_PROPERTY:xrRenderVK,INCLUDE_DIRECTORIES>
    )
    target_compile_definitions(vulkan_kinematics_test PRIVATE SDL_MAIN_HANDLED PURE_DYNAMIC_CAST)
    target_link_libraries(vulkan_kinematics_test PRIVATE xrRenderVK xrEngine xrAPI xrScriptEngine xrCDB xrCore xrMaterialSystem)
    add_test(NAME vulkan_kinematics COMMAND vulkan_kinematics_test)

    add_executable(vulkan_particle_visual_test
        "${XRAY_SOURCE_DIR}/tests/vulkan_particle_visual.cpp"
    )
    target_include_directories(vulkan_particle_visual_test PRIVATE
        "${XRAY_SOURCE_DIR}"
        "${XRAY_SOURCE_DIR}/src"
        $<TARGET_PROPERTY:xrRenderVK,INCLUDE_DIRECTORIES>
    )
    target_compile_definitions(vulkan_particle_visual_test PRIVATE SDL_MAIN_HANDLED)
    target_link_libraries(vulkan_particle_visual_test PRIVATE xrRenderVK xrParticles xrEngine xrAPI xrScriptEngine xrCDB xrCore xrMaterialSystem)
    add_test(NAME vulkan_particle_visual COMMAND vulkan_particle_visual_test)

    if (Vulkan_FOUND AND NOT ANDROID)
        add_executable(vulkan_particle_frame_gpu_test "${XRAY_SOURCE_DIR}/tests/vulkan_particle_frame_gpu.cpp")
        target_include_directories(vulkan_particle_frame_gpu_test PRIVATE
            "${XRAY_SOURCE_DIR}" "${XRAY_SOURCE_DIR}/src" $<TARGET_PROPERTY:xrRenderVK,INCLUDE_DIRECTORIES>)
        target_compile_definitions(vulkan_particle_frame_gpu_test PRIVATE SDL_MAIN_HANDLED)
        target_link_libraries(vulkan_particle_frame_gpu_test PRIVATE xrRenderVK xrParticles xrEngine xrAPI xrScriptEngine xrCDB xrCore xrMaterialSystem Vulkan::Vulkan)
        add_test(NAME vulkan_particle_frame_gpu COMMAND vulkan_particle_frame_gpu_test
            "${XRAY_SOURCE_DIR}/res/gamedata/shaders" ${_xray_vk_gpu_device_option_args})
        set_tests_properties(vulkan_particle_frame_gpu PROPERTIES
            SKIP_RETURN_CODE 77 TIMEOUT 120 WORKING_DIRECTORY "${XRAY_SOURCE_DIR}")
    endif()

    add_executable(vulkan_renderer_module_test
        "${XRAY_SOURCE_DIR}/tests/vulkan_renderer_module.cpp"
    )
    target_include_directories(vulkan_renderer_module_test PRIVATE
        "${XRAY_SOURCE_DIR}"
        "${XRAY_SOURCE_DIR}/src"
        $<TARGET_PROPERTY:xrRenderVK,INCLUDE_DIRECTORIES>
    )
    target_compile_definitions(vulkan_renderer_module_test PRIVATE SDL_MAIN_HANDLED)
    target_link_libraries(vulkan_renderer_module_test PRIVATE xrRenderVK xrParticles xrEngine xrAPI xrScriptEngine xrCDB xrCore xrMaterialSystem)
    add_test(NAME vulkan_renderer_module COMMAND vulkan_renderer_module_test)
    add_executable(vulkan_ui_flow_test "${XRAY_SOURCE_DIR}/tests/vulkan_ui_flow.cpp")
    target_include_directories(vulkan_ui_flow_test PRIVATE "${XRAY_SOURCE_DIR}" "${XRAY_SOURCE_DIR}/src"
        $<TARGET_PROPERTY:xrRenderVK,INCLUDE_DIRECTORIES>)
    target_compile_definitions(vulkan_ui_flow_test PRIVATE DEBUG SDL_MAIN_HANDLED)
    target_compile_options(vulkan_ui_flow_test PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>:/UNDEBUG>
        $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-UNDEBUG>)
    target_link_libraries(vulkan_ui_flow_test PRIVATE xrRenderVK xrParticles xrEngine xrAPI xrScriptEngine xrCDB xrCore xrMaterialSystem)
    add_test(NAME vulkan_ui_flow COMMAND vulkan_ui_flow_test)
    add_executable(vulkan_asset_flow_test "${XRAY_SOURCE_DIR}/tests/vulkan_asset_flow.cpp")
    set_target_properties(vulkan_asset_flow_test PROPERTIES OUTPUT_NAME vulkan_asset_flow)
    target_include_directories(vulkan_asset_flow_test PRIVATE "${XRAY_SOURCE_DIR}" "${XRAY_SOURCE_DIR}/src"
        $<TARGET_PROPERTY:xrRenderVK,INCLUDE_DIRECTORIES>)
    target_compile_definitions(vulkan_asset_flow_test PRIVATE SDL_MAIN_HANDLED)
    target_link_libraries(vulkan_asset_flow_test PRIVATE xrRenderVK xrParticles xrEngine xrAPI xrScriptEngine xrCDB xrCore xrMaterialSystem)
    add_test(NAME vulkan_asset_flow COMMAND vulkan_asset_flow_test)

    if (NOT ANDROID AND UNIX)
        add_executable(vulkan_validation_host_test "${XRAY_SOURCE_DIR}/tests/vulkan_validation_host.cpp")
        target_include_directories(vulkan_validation_host_test PRIVATE "${XRAY_SOURCE_DIR}"
            $<TARGET_PROPERTY:xrRenderVK,INCLUDE_DIRECTORIES>)
        target_link_libraries(vulkan_validation_host_test PRIVATE xrRenderVK xrParticles xrEngine xrAPI xrScriptEngine xrCDB xrCore xrMaterialSystem SDL2::SDL2)
        # Requires an X11/Wayland Vulkan surface and an installed validation layer.
        # Invoke explicitly under xvfb-run when those are available.
    endif()

    add_executable(vulkan_shader_module_test
        "${XRAY_SOURCE_DIR}/tests/vulkan_shader_module.cpp"
        "${XRAY_SOURCE_DIR}/src/Layers/xrRenderVK/ShaderModule.cpp"
    )
    target_include_directories(vulkan_shader_module_test PRIVATE
        "${XRAY_SOURCE_DIR}"
        "${XRAY_SOURCE_DIR}/Externals/Vulkan-Headers/include"
    )
    target_compile_definitions(vulkan_shader_module_test PRIVATE DEBUG)
    target_compile_options(vulkan_shader_module_test PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>:/UNDEBUG>
        $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-UNDEBUG>
    )
    add_test(NAME vulkan_shader_module COMMAND vulkan_shader_module_test)

    if (WIN32)
        # ReleaseMasterGold engine DLLs may depend on SDL2d.dll from the Debug
        # output directory. Make that runtime dependency available to CTest
        # without relying on the shell used to invoke ctest.
        get_property(xray_vulkan_tests DIRECTORY PROPERTY TESTS)
        set_tests_properties(${xray_vulkan_tests} PROPERTIES
            ENVIRONMENT_MODIFICATION
                "PATH=path_list_prepend:${XRAY_SOURCE_DIR}/bin/${CMAKE_SYSTEM_PROCESSOR}/Debug")
    endif()
endif()
