#pragma once

#if defined(XR_PLATFORM_ANDROID)

#include <string>

namespace AndroidVulkanSmoke
{
// Validates the Android Vulkan surface/device/swapchain/present path without
// changing the gameplay renderer. The caller keeps GLES as the portable
// fallback until the full xrRenderVK backend is ready.
bool Run(std::string& reason);
}

#endif
