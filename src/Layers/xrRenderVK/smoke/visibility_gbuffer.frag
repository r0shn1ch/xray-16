#version 450

layout(binding = 0) uniform usampler2D visibilityDepth;
layout(binding = 1) uniform sampler2D resolvedBaseColor;
layout(binding = 2) uniform sampler2D resolvedNormal;

layout(location = 0) out vec4 albedo;
layout(location = 1) out vec4 normalBuffer;
layout(location = 2) out vec4 surfaceBuffer;

void main()
{
    ivec2 pixel = ivec2(gl_FragCoord.xy);
    if (any(greaterThanEqual(pixel, textureSize(resolvedNormal, 0))))
        discard;
    vec4 normal = texelFetch(resolvedNormal, pixel, 0);
    if (all(equal(normal.xyz, vec3(0.0))))
        discard;

    gl_FragDepth = uintBitsToFloat(texelFetch(visibilityDepth, pixel, 0).r);
    albedo = texelFetch(resolvedBaseColor, pixel, 0);
    normalBuffer = vec4(normal.xyz * 0.5 + 0.5, normal.a);
    // The flat opaque/lightmapped materials admitted to visibility raster use
    // the default material class (1), encoded as class / 3 in opaque.ps.
    surfaceBuffer = vec4(0.0, 1.0 / 3.0, 0.0, 0.0);
}
