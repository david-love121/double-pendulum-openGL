#version 460 core

in vec2 TexCoord;
uniform vec4 color;

out vec4 FragColor;

void main() {
    vec2 center = vec2(0.5, 0.5);
    float dist = distance(TexCoord, center);
    
    if (dist > 0.5) {
        discard;
    }
    
    FragColor = color;
}
