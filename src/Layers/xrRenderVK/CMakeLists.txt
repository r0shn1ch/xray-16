add_library(xrRenderVK STATIC)

include("${XRAY_SOURCE_DIR}/cmake/GenerateRuntimeGameShaderVariants.cmake")
set(XRAY_RUNTIME_GAME_SHADER_VARIANTS "${CMAKE_CURRENT_BINARY_DIR}/generated/RuntimeGameShaderVariants.h")
xray_generate_runtime_game_shader_variants(
    "${XRAY_SOURCE_DIR}/res/gamedata/shaders/vk/opaque-variants.json"
    "${XRAY_RUNTIME_GAME_SHADER_VARIANTS}"
)
target_sources(xrRenderVK PRIVATE "${XRAY_RUNTIME_GAME_SHADER_VARIANTS}")
target_include_directories(xrRenderVK PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")
target_include_directories(xrRenderVK PRIVATE "${xray_bcdec_SOURCE_DIR}")

if (ANDROID AND CMAKE_HOST_WIN32)
    # llvm-ar on Windows can round the archive's timestamp below the newest
    # Android object file. Keep Ninja's post-link freshness check stable.
    add_custom_command(TARGET xrRenderVK POST_BUILD
        COMMAND "${CMAKE_COMMAND}" -E touch "$<TARGET_FILE:xrRenderVK>"
        VERBATIM
    )
endif()

target_sources(xrRenderVK PRIVATE
    PipelineCache.h
    VulkanRendererModule.cpp
    VulkanProbe.cpp
    VulkanProbe.h
    SlidingWindows.cpp
    SlidingWindows.h
    SpecialVisuals.cpp
    SpecialVisuals.h
    BufferResource.cpp
    BufferResource.h
    AccelerationStructure.cpp
    AccelerationStructure.h
    LevelRayScene.cpp
    LevelRayScene.h
    RayQueryBatch.cpp
    ReferencePathTracePass.cpp
    ReferencePathTracePass.h
    RaySurface.cpp
    IndirectRayFrames.cpp
    IndirectRayFrames.h
    IndirectSunPass.cpp
    StorageComputePass.cpp
    VsmPageAllocator.cpp
    VsmPageAllocator.h
    VsmDemandPass.cpp
    VsmDemandPass.h
    VsmStaticCache.cpp
    VsmStaticCache.h
    VsmFrameResources.cpp
    VsmFrameResources.h
    VsmClipmap.cpp
    VsmClipmap.h
    VsmAtlas.cpp
    VsmAtlas.h
    VsmPageRender.cpp
    VsmPageRender.h
    VsmLevelPageRender.cpp
    VsmResolvePass.cpp
    VsmResolvePass.h
    VsmCasterBinning.cpp
    VsmCasterBinning.h
    VsmInvalidation.cpp
    VsmInvalidation.h
    GpuParticleCollisionData.h
    GpuParticleStaticCollision.cpp
    GpuParticleStaticCollision.h
    GpuParticleLevelCollision.cpp
    GpuParticleLevelCollision.h
    GpuParticleDynamicCollision.cpp
    GpuParticleDynamicCollision.h
    GpuParticleBuffers.cpp
    GpuParticleBuffers.h
    GpuParticleUpload.cpp
    GpuParticleUpload.h
    GpuParticleUniforms.cpp
    GpuParticleUniforms.h
    GpuParticleFrame.cpp
    GpuParticleFrame.h
    GpuParticleFrameInput.cpp
    GpuParticleFrameInput.h
    GpuParticleReadback.cpp
    GpuParticleReadback.h
    GpuParticleRuntime.cpp
    GpuParticleRuntime.h
    GpuParticleDrawBindings.cpp
    GpuParticleDrawBindings.h
    GpuParticleDrawPass.cpp
    GpuParticleDrawPass.h
    GpuParticleGraphics.cpp
    GpuParticleGraphics.h
    SmokeTrailCompute.cpp
    SmokeTrailCompute.h
    SmokeTrailDrawPass.cpp
    SmokeTrailDrawPass.h
    SmokeTrailNoise.cpp
    SmokeTrailNoise.h
    SmokeTrailController.h
    TrailState.h
    GpuParticleDrawBuckets.h
    GpuParticleNoise.cpp
    GpuParticleNoise.h
    GpuParticleCommands.cpp
    GpuParticleCommands.h
    GpuParticleEmitters.cpp
    GpuParticleEmitters.h
    GpuParticlePrograms.cpp
    GpuParticlePrograms.h
    GpuParticleMaterials.cpp
    GpuParticleMaterials.h
    GpuParticleCompute.cpp
    GpuParticleCompute.h
    GpuParticleTranslate.cpp
    GpuParticleTranslate.h
    RayQueryBatch.h
    RayQueryBatchShaders.h
    SkyProbeReducePass.h
    SkyProbeBakePass.h
    RayGenerateShaders.h
    IndirectRadianceShaders.h
    WorldRadianceCacheShaders.h
    ReferencePathTraceShaders.h
    RestirGIShaders.h
    RayGeneratePass.cpp
    RayGeneratePass.h
    RayVisibilityShaders.h
    RayVisibilityPass.cpp
    RayVisibilityPass.h
    RayFrameResources.cpp
    RayFrameResources.h
    smoke/ray_visibility.comp
    smoke/sky_probe_reduce.comp
    smoke/detail_cull.comp
    smoke/detail_generate.comp
    smoke/detail_cell_cull.comp
    smoke/visibility_common.glsl
    smoke/visibility_geometry.glsl
    smoke/visibility_material_resolve.comp
    smoke/visibility_software_raster.comp
    smoke/visibility_gbuffer.vert
    smoke/visibility_gbuffer.frag
    smoke/hdr10_output.frag
    VisibilityPasses.cpp
    VisibilityPasses.h
    DetailCullPass.h
    DetailGeneratePass.h
    DetailCellCullPass.h
    smoke/ray_generate.comp
    smoke/indirect_radiance.comp
    smoke/world_radiance_cache.glsl
    smoke/world_radiance_cache_update.comp
    smoke/reference_path_trace.comp
    smoke/restir_gi_spatial.comp
    BindlessTextureTable.cpp
    BindlessTextureTable.h
    BufferUpload.cpp
    BufferUpload.h
    Bc7Decoder.cpp
    Bc7Decoder.h
    Bc6Decoder.cpp
    Bc6Decoder.h
    DdsTexture.cpp
    DdsTexture.h
    DetailAssets.cpp
    DetailAssets.h
    DeferredPass.cpp
    DeferredPass.h
    ClusterVisibility.cpp
    ClusterVisibility.h
    ClusterVisibilityShaders.h
    smoke/cluster_visibility.comp
    GBufferRenderPass.cpp
    LightGrid.cpp
    ComputePipeline.cpp
    ComputePipeline.h
    LightingLimits.h
    LightGrid.h
    LightGridShaders.h
    HiZPyramid.cpp
    HiZPyramid.h
    HiZShaders.h
    smoke/hiz_build.comp
    MotionVectors.cpp
    MotionVectors.h
    MotionVectorShaders.h
    smoke/motion_vectors.comp
    ObjectMotionHistory.h
    ObjectVelocityPass.cpp
    ObjectVelocityPass.h
    ObjectVelocityShaders.h
    smoke/object_velocity.vert
    smoke/object_velocity_skinned.vert
    smoke/object_velocity.frag
    smoke/object_velocity_particle.vert
    smoke/object_velocity_particle.frag
    TemporalExposure.cpp
    TemporalExposure.h
    TemporalExposureShaders.h
    smoke/temporal_exposure_adapt.comp
    smoke/temporal_exposure_histogram.comp
    TemporalResolve.cpp
    TemporalResolve.h
    TemporalResolveShaders.h
    Fsr3Upscaler.cpp
    Fsr3Upscaler.h
    smoke/temporal_resolve.comp
    TemporalJitter.h
    DeferredFrame.cpp
    DeferredFrame.h
    ForwardLighting.h
    DeferredAlphaTestShaders.h
    VulkanGameShaders.h
    DeferredShaderFactory.cpp
    DeferredShaderFactory.h
    GameShaderResources.cpp
    GameShaderResources.h
    RuntimeSlangCompiler.cpp
    RuntimeSlangCompiler.h
    GameShaderPipeline.cpp
    GameShaderInterface.h
    GraphicsShaderStages.h
    GameComputePipeline.cpp
    ShaderMaterialLibrary.cpp
    ShaderMaterialLibrary.h
    DeferredShaders.h
    WeatherShaders.h
    smoke/weather.frag
    smoke/deferred.frag
    smoke/local_light.frag
    smoke/rain_wetness.glsl
    EngineTextureSource.cpp
    EmbeddedShaderFingerprint.cpp
    EmbeddedShaderFingerprint.h
    EngineTextureSource.h
    EngineLevelModels.cpp
    EngineLevelModels.h
    FrameContext.cpp
    FrameContext.h
    FrameGraphAllocation.cpp
    FrameImageGraph.cpp
    FrameImageGraph.h
    FrameBufferPool.cpp
    FrameBufferPool.h
    FrameSubmissionGraph.h
    FrameImagePool.cpp
    FrameImagePool.h
    GameTextureFactory.cpp
    GameTextureFactory.h
    GBufferTargets.cpp
    GBufferTargets.h
    GBufferLightingBindings.cpp
    SunShadowTargets.cpp
    SunShadowTargets.h
    LocalShadowTargets.cpp
    LocalShadowTargets.h
    LocalLightShaders.h
    WaterTargets.cpp
    WaterTargets.h
    WaterShaders.h
    ShadowShaders.h
    GpuLevel.cpp
    GpuLevel.h
    ClusterLodImplementation.cpp
    ClusterLodBuilder.cpp
    ClusterLodBuilder.h
    clusterlod.h
    ClusterBasis.cpp
    ClusterBasis.h
    ClusterDeviation.cpp
    ClusterDeviation.h
    GpuModel.cpp
    GpuModel.h
    ImageStateTracker.cpp
    ImageStateTracker.h
    LevelModels.cpp
    LevelModels.h
    LevelVisibility.cpp
    LevelVisibility.h
    ModelGeometry.cpp
    ModelGeometry.h
    ModelTessellation.cpp
    ModelTessellation.h
    SkeletonBones.cpp
    SkeletonBones.h
    SkeletonMotions.cpp
    SkeletonMotions.h
    MotionPlayback.cpp
    MotionPlayback.h
    ParticleCatalog.cpp
    ParticleCatalog.h
    ParticleVisual.cpp
    ParticleVisual.h
    VulkanKinematics.cpp
    VulkanKinematics.h
    ScreenCopyPass.cpp
    ScreenCopyPass.h
    PostProcessShaders.h
    smoke/postprocess.frag
    smoke/bloom_bright.frag
    smoke/bloom_upsample.frag
    smoke/sun_shafts.frag
    smoke/bloom.glsl
    smoke/fxaa.glsl
    ScenePass.cpp
    ScenePass.h
    ShaderModule.cpp
    ShaderModule.h
    ShaderReflection.cpp
    ShaderReflection.h
    SmokeShaders.h
    SmokeTrianglePass.cpp
    SmokeTrianglePass.h
    TextureUpload.cpp
    TextureUpload.h
    VulkanHardware.cpp
    VulkanHardware.h
    VulkanWindowDevice.cpp
    VulkanWindowDevice.h
    VisualCatalog.cpp
    VisualCatalog.h
    VulkanVisual.cpp
    VulkanVisual.h
    VulkanModelVisual.h
    VulkanUIShader.cpp
    VulkanUIShader.h
    VulkanRenderFactory.cpp
    VulkanRenderFactory.h
    VulkanImGuiRender.cpp
    VulkanImGuiRender.h
    VulkanVideoTexture.cpp
    VulkanVideoTexture.h
    AndroidMediaVideoDecoder.cpp
    AndroidMediaVideoDecoder.h
    VulkanUISequenceVideoItem.cpp
    VulkanUISequenceVideoItem.h
    VulkanFontRender.cpp
    VulkanFontRender.h
    VulkanStatGraphRender.cpp
    VulkanStatGraphRender.h
    VulkanWallMarkArray.cpp
    VulkanWallMarkArray.h
    VulkanEnvDescriptorRender.cpp
    VulkanEnvDescriptorRender.h
    VulkanEnvironmentRender.cpp
    VulkanEnvironmentRender.h
    VulkanRainRender.cpp
    VulkanRainRender.h
    VulkanLensFlareRender.cpp
    VulkanLensFlareRender.h
    VulkanThunderboltRender.cpp
    VulkanThunderboltRender.h
    VulkanDrawUtils.cpp
    VulkanDrawUtils.h
    VulkanDebugRender.cpp
    VulkanDebugRender.h
    EngineParticleSource.cpp
    EngineParticleSource.h
    WallmarkGeometry.cpp
    WallmarkGeometry.h
    VulkanLevelRender.cpp
    VulkanLevelRender.h
    VulkanCameraState.h
    VulkanDeviceResourceState.h
    VulkanFramePhaseState.h
    VulkanRenderContextState.h
    VulkanGameLighting.cpp
    VulkanGameLighting.h
    VulkanGameLightingMath.cpp
    VulkanGameLightingMath.h
    VulkanUIRender.cpp
    VulkanUIRender.h
    Light_DB.cpp
    Light_DB.h
    r__render_lights.cpp
    r__render_draw.cpp
    r__model_pool.cpp
    VulkanGameDevice.cpp
    VulkanGameDevice.h
)

include("${CMAKE_SOURCE_DIR}/cmake/XRay.Gli.cmake")
xray_gli_include_dir(XRAY_GLI_INCLUDE_DIR)

target_include_directories(xrRenderVK
    PUBLIC
    "${CMAKE_SOURCE_DIR}/Externals/Vulkan-Headers/include"
    PRIVATE
    "${CMAKE_SOURCE_DIR}/src"
    "${CMAKE_SOURCE_DIR}/sdk/include"
    "${CMAKE_SOURCE_DIR}/Externals"
    "${CMAKE_SOURCE_DIR}/Externals/luabind"
    "${CMAKE_SOURCE_DIR}/Externals/tracy/public"
    "${CMAKE_SOURCE_DIR}/Externals/imgui"
    "${XRAY_GLI_INCLUDE_DIR}"
    "${CMAKE_SOURCE_DIR}/Externals/gli/external"
)

# Public renderer headers expose NVRHI types to consumers such as xrEngine's
# Android Vulkan smoke path, so propagate NVRHI's include directories as well
# as its static link dependencies.
target_link_libraries(xrRenderVK
    PUBLIC xrRenderGraph ${XRAY_NVRHI_LIBRARIES}
    PRIVATE SDL2::SDL2 xrCDB xrImGui meshoptimizer
)

if (TARGET nvrhi_vk)
    target_sources(xrRenderVK PRIVATE NvrhiVulkanDispatch.cpp)
endif()

if (MSVC)
    # ReleaseMasterGold disables exceptions project-wide, but the Vulkan
    # renderer uses catch handlers to recover from NVRHI/resource failures.
    target_compile_options(xrRenderVK PRIVATE /EHsc)
endif()

if (ANDROID AND ANDROID_ABI STREQUAL "arm64-v8a")
    target_link_libraries(xrRenderVK PRIVATE
        ffx_backend_vk_android_arm64
        ffx_fsr3_android_arm64
        ffx_fsr3upscaler_android_arm64
        ffx_frameinterpolation_android_arm64
        ffx_opticalflow_android_arm64
    )
    target_compile_definitions(xrRenderVK PRIVATE XRAY_HAS_FSR3_ANDROID=1)
endif()

# Slang provides runtime HLSL compilation for 64-bit Android Vulkan builds.
# Its Android binaries are built from the pinned source release by
# android/build-slang.sh; the checked-in renderer remains usable on other
# platforms with its precompiled SPIR-V assets.
if (ANDROID AND (ANDROID_ABI STREQUAL "arm64-v8a" OR ANDROID_ABI STREQUAL "x86_64"))
    if (NOT XRAY_SLANG_ROOT)
        message(FATAL_ERROR
            "XRAY_SLANG_ROOT is required for Android ${ANDROID_ABI}. Run android/build-android-native.sh "
            "or set XRAY_SLANG_ROOT to a Slang Android install prefix.")
    endif()
    set(slang_DIR "${XRAY_SLANG_ROOT}/lib/cmake/slang")
    find_package(slang CONFIG REQUIRED NO_DEFAULT_PATH)
    if (NOT TARGET slang::slang)
        message(FATAL_ERROR "Slang package at XRAY_SLANG_ROOT does not define slang::slang")
    endif()
    target_link_libraries(xrRenderVK PRIVATE slang::slang)
    target_compile_definitions(xrRenderVK PRIVATE XRAY_HAS_SLANG=1)
elseif (NOT ANDROID AND XRAY_SLANG_ROOT)
    set(slang_DIR "${XRAY_SLANG_ROOT}/lib/cmake/slang")
    find_package(slang CONFIG QUIET NO_DEFAULT_PATH)
    if (TARGET slang::slang)
        target_link_libraries(xrRenderVK PRIVATE slang::slang)
        target_compile_definitions(xrRenderVK PRIVATE XRAY_HAS_SLANG=1)
    endif()
endif()

if (WIN32)
    # NVRHI's Vulkan backend publishes VK_USE_PLATFORM_WIN32_KHR; keep the
    # Windows min/max macros out of renderer translation units using std::min/max.
    target_compile_definitions(xrRenderVK PRIVATE NOMINMAX)
endif()

set_target_properties(xrRenderVK PROPERTIES
    PREFIX ""
    UNITY_BUILD OFF
)
