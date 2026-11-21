#version 450
layout(location = 0) in vec3 position;
layout(location = 2) in vec2 uv;
layout(location = 3) in uvec4 bone_indices;
layout(location = 4) in vec4 weights;
layout(location = 0) out vec4 previous_clip;
layout(location = 1) out vec4 current_clip;
layout(location = 2) out vec2 texcoord;
layout(set = 1, binding = 0, std430) readonly buffer CurrentPose
{
    mat4 bones[];
} current_pose;
layout(set = 2, binding = 0, std430) readonly buffer PreviousPose
{
    mat4 bones[];
} previous_pose;
layout(push_constant) uniform Matrices
{
    mat4 current_mvp;
    mat4 previous_mvp;
} transforms;
void main()
{
    vec4 rest = vec4(position, 1.0);
    vec4 current_position = vec4(0.0);
    vec4 previous_position = vec4(0.0);
    for (int i = 0; i < 4; ++i)
    {
        if (weights[i] <= 0.0) continue;
        current_position += current_pose.bones[bone_indices[i]] * rest * weights[i];
        previous_position += previous_pose.bones[bone_indices[i]] * rest * weights[i];
    }
    gl_Position = transforms.current_mvp * current_position;
    gl_Position.y = -gl_Position.y;
    current_clip = gl_Position;
    previous_clip = transforms.previous_mvp * previous_position;
    previous_clip.y = -previous_clip.y;
    texcoord = uv;
}
