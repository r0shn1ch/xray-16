#pragma once

namespace xray::render::vulkan
{
class PipelineCacheFingerprint;
void append_embedded_shader_fingerprint(PipelineCacheFingerprint& fingerprint);
}
