#version 460 core

layout (location = 0) in vec2 position;

uniform mat4 view;
uniform mat4 projection;

void main() {
    gl_Position = projection * view * vec4(position, 0.0, 1.0);
}
