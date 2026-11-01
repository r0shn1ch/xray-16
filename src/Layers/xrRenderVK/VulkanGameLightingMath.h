#pragma once

#include <array>
#include <cstdint>
#include <vector>
#include <string>

namespace xray::render::vulkan
{
enum class VulkanLightType : uint32_t
{
    Direct,
    Point,
    Spot,
    OmniPart,
    Reflected
};

struct VulkanLightSnapshot
{
    VulkanLightType type{VulkanLightType::Point};
    bool active{};
    bool shadow{};
    std::array<float, 3> position{};
    std::array<float, 3> direction{0.f, -1.f, 0.f};
    std::array<float, 3> color{1.f, 1.f, 1.f};
    float range{1.f};
    float cone{0.785398163f};
    std::array<float, 3> right{1, 0, 0};
    std::string texture;
    uint64_t id{};
    float visibility{1.f};
    bool hemi_only{};
    std::array<float, 4> attenuation{1.f, 0.f, 0.f, 0.f}; // constant, linear, quadratic, falloff
};

struct VulkanObjectLighting
{
    float luminocity{};
    float hemi_luminocity{};
    // IRender_ObjectSpecific / CROS_impl order: +X, +Y, +Z, -X, -Y, -Z.
    std::array<float, 6> hemi_cube{};
};

// Vertex shaders consume the positive faces as xyz, followed by negative xyz.
constexpr std::array<unsigned, 6> VulkanShaderHemiFaceOrder{0, 1, 2, 3, 4, 5};

enum VulkanObjectLightingMode : uint32_t
{
    VulkanTraceLights = 1u << 0,
    VulkanTraceSun = 1u << 1,
    VulkanTraceHemi = 1u << 2,
    VulkanTraceAll = VulkanTraceLights | VulkanTraceSun | VulkanTraceHemi
};

VulkanObjectLighting evaluate_object_lighting(const std::array<float, 3> &position, const float *ambient,
                                              const float *hemi, const float *sun, uint32_t mode,
                                              const std::vector<VulkanLightSnapshot> &lights,
                                              const std::array<float, 6> *sky_faces = nullptr, float light_scale = .2f,
                                              float light_flow = .1f);
} // namespace xray::render::vulkan
