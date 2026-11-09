#include "xrEngine/stdafx.h"
#include "Include/xrRender/xrRender.h"
#include "src/Layers/xrRenderVK/VulkanLevelRender.h"
#include "src/Layers/xrRenderVK/VulkanRenderFactory.h"
#include "src/Layers/xrRenderVK/VulkanUIRender.h"
#include "src/Layers/xrRenderVK/VulkanDrawUtils.h"
#include "src/Layers/xrRenderVK/VulkanDebugRender.h"
#include "src/Layers/xrRenderVK/VulkanEnvironmentRender.h"
#include "src/Layers/xrRenderVK/VulkanImGuiRender.h"
#include "src/Layers/xrRenderVK/ResourceIssueCounter.h"
#include "Include/xrRender/RenderFactory.h"
#include "xrEngine/XR_IOConsole.h"

#include <cassert>
#include <array>
#include <new>

static void check_camera_frustum()
{
    using xray::render::vulkan::VulkanLevelRender;
    // Heap contents must not decide whether a tracer can query the camera
    // before its first update. Static renderer instances can conceal this.
    alignas(VulkanLevelRender) std::array<unsigned char, sizeof(VulkanLevelRender)> storage;
    storage.fill(0xa5);
    auto* renderer = new (storage.data()) VulkanLevelRender;
    IRender& render = *renderer;
    assert(render.ViewBase.p_count == 0);
    Fvector inside, outside, far;
    inside.set(0.f, 0.f, .5f);
    outside.set(2.f, 0.f, .5f);
    far.set(0.f, 0.f, 2.f);
    assert(render.ViewBase.testSphere_dirty(outside, .05f));

    const Fmatrix saved_transform = Device.mFullTransform;
    Device.mFullTransform.identity();
    render.OnCameraUpdated();
    assert(render.ViewBase.p_count == 5);
    assert(render.ViewBase.testSphere_dirty(inside, .05f));
    assert(!render.ViewBase.testSphere_dirty(outside, .05f));
    assert(!render.ViewBase.testSphere_dirty(far, .05f));

    Fmatrix view, project;
    view.translate(-10.f, 0.f, 0.f);
    project.identity();
    render.SetCacheXform(view, project);
    assert(!render.ViewBase.testSphere_dirty(inside, .05f));
    Fvector moved;
    moved.set(10.f, 0.f, .5f);
    assert(render.ViewBase.testSphere_dirty(moved, .05f));

    Fbox box;
    box.vMin.set(9.9f, -.1f, .4f);
    box.vMax.set(10.1f, .1f, .6f);
    assert(render.occ_visible(box));
    box.vMin.set(-.1f, -.1f, .4f);
    box.vMax.set(.1f, .1f, .6f);
    assert(!render.occ_visible(box));

    // A camera callback must replace a previously cached transform.
    render.OnCameraUpdated();
    assert(render.ViewBase.testSphere_dirty(inside, .05f));
    assert(!render.ViewBase.testSphere_dirty(moved, .05f));
    Device.mFullTransform = saved_transform;
    renderer->~VulkanLevelRender();
}

namespace xray::render::vulkan
{
RendererModule* GetRendererModule();
}

int main()
{
    Core.Initialize("vulkan_renderer_module_test", nullptr, false);
    check_camera_frustum();
    // SetupEnv registers renderer commands in the engine-owned console.
    CConsole console;
    Console = &console;
    {
        xray::render::vulkan::ResourceIssueCounter issues;
        const std::string source = "file=$level$/level.geom visual=2 material=7 shader='mod\\plain' reason=missing TEXCOORD1";
        assert(issues.record(source));
        assert(!issues.record(source) && !issues.record(source));
        assert(issues.record("file=other.ogf reason=unsupported vertex format"));
        unsigned reports = 0;
        issues.report_repeats([&](const std::string& context, uint64_t count)
        {
            assert(context == source && count == 3);
            ++reports;
        });
        assert(reports == 1);
        issues.clear(); // Asset reload starts a new diagnostic generation.
        assert(issues.record(source));
    }
    {
        xray::render::vulkan::VulkanEnvironmentRender::Library groups;
        assert(groups.particles_group_begin() == groups.particles_group_end());
        xray::render::vulkan::ParticleCatalog catalog;
        catalog.groups.push_back({"mod\\arbitrary_one", 0, 1.f, {}});
        catalog.groups.push_back({"mod\\arbitrary_two", 0, 2.f, {}});
        groups.assign(std::move(catalog));
        auto cursor = groups.particles_group_begin();
        assert(cursor && cursor != groups.particles_group_end());
        assert(groups.particles_group_id(**cursor) == "mod\\arbitrary_one");
        groups.particles_group_next(cursor);
        assert(cursor != groups.particles_group_end());
        assert(groups.particles_group_id(**cursor) == "mod\\arbitrary_two");
        groups.particles_group_next(cursor);
        assert(cursor == groups.particles_group_end());
        groups.particles_group_next(cursor); // One-past-end stays one-past-end.
        assert(cursor == groups.particles_group_end());
        const particles_systems::PS::CPGDef* foreign_entry = nullptr;
        const particles_systems::PS::CPGDef* const* foreign = &foreign_entry;
        groups.particles_group_next(foreign);
        assert(foreign == &foreign_entry); // A foreign iterator is never advanced.
        xr_vector<shared_str> ids;
        groups.particles_group_ids(ids);
        assert(ids.size() == 2 && ids[0] == "mod\\arbitrary_one" && ids[1] == "mod\\arbitrary_two");
        xray::render::vulkan::ParticleCatalog replacement;
        replacement.groups.push_back({"mod\\replacement", 0, 3.f, {}});
        groups.assign(std::move(replacement));
        ids.clear();
        groups.particles_group_ids(ids);
        assert(ids.size() == 1 && ids[0] == "mod\\replacement");
        groups.clear();
        assert(groups.particles_group_begin() == groups.particles_group_end());
        groups.particles_group_next(cursor); // A former generation cannot be dereferenced.
        assert(cursor != groups.particles_group_begin());
    }
    auto* module = xray::render::vulkan::GetRendererModule();
    for (int i = 0; i < 2; ++i)
    {
        module->SetupEnv("renderer_vulkan");
        assert(dynamic_cast<xray::render::vulkan::VulkanLevelRender*>(GEnv.Render));
        assert(dynamic_cast<xray::render::vulkan::VulkanRenderFactory *>(GEnv.RenderFactory));
        auto* environment = GEnv.RenderFactory->CreateEnvironmentRender();
        auto& groups = environment->particles_systems_library();
        assert(groups.particles_group_begin() == groups.particles_group_end());
        xr_vector<shared_str> ids;
        groups.particles_group_ids(ids);
        assert(ids.empty());
        GEnv.RenderFactory->DestroyEnvironmentRender(environment);
        auto* imgui = GEnv.RenderFactory->CreateImGuiRender();
        ImGuiContext* context = ImGui::CreateContext();
        imgui->OnDeviceCreate(context);
        imgui->OnDeviceResetBegin();
        imgui->OnDeviceResetEnd();
        assert(ImGui::GetCurrentContext() == context);
        assert((ImGui::GetIO().BackendFlags & ImGuiBackendFlags_RendererHasTextures) != 0);
        imgui->OnDeviceDestroy();
        GEnv.RenderFactory->DestroyImGuiRender(imgui);
        ImGui::DestroyContext(context);
        assert(dynamic_cast<xray::render::vulkan::VulkanDrawUtils *>(GEnv.DU));
        assert(dynamic_cast<xray::render::vulkan::VulkanUIRender *>(GEnv.UIRender));
        u32 flags = SDL_WINDOW_HIDDEN | SDL_WINDOW_RESIZABLE;
        GEnv.Render->ObtainRequiredWindowFlags(flags);
        assert((flags & SDL_WINDOW_VULKAN) && !(flags & SDL_WINDOW_OPENGL));
#ifdef DEBUG
        assert(dynamic_cast<xray::render::vulkan::VulkanDebugRender *>(GEnv.DRender));
#endif
        module->ClearEnv();
        assert(!GEnv.Render && !GEnv.RenderFactory && !GEnv.DU && !GEnv.UIRender && !GEnv.DRender);
    }
    Console = nullptr;
}
