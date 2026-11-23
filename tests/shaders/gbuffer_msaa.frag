#version 450
layout(location = 0) out vec4 albedo;
layout(location = 1) out vec4 normal;
layout(location = 2) out vec4 material;
void main() { albedo = vec4(1, 0, 0, 1); normal = vec4(.5, .5, 1, 1); material = vec4(1); }
