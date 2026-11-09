#version 450
layout(push_constant) uniform Camera { mat4 mvp; } camera;
layout(set=1,binding=0,std430) readonly buffer Pose { mat4 bones[]; } pose;
layout(location=0) in vec3 position;
layout(location=2) in vec2 uv;
layout(location=3) in uvec4 indices;
layout(location=4) in vec4 weights;
layout(location=0) out vec2 texcoord;
void main()
{
    vec4 local = vec4(position, 1);
    vec4 p = vec4(0);
    if (weights.x > 0) p += pose.bones[indices.x] * local * weights.x;
    if (weights.y > 0) p += pose.bones[indices.y] * local * weights.y;
    if (weights.z > 0) p += pose.bones[indices.z] * local * weights.z;
    if (weights.w > 0) p += pose.bones[indices.w] * local * weights.w;
    gl_Position = camera.mvp * p;
    texcoord = uv;
}
