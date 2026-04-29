#version 330

uniform sampler2D cameraTex;
uniform sampler2D maskTex;

in vec2 v_uv;
out vec4 fragColor;

void main()
{
    vec3 cam = texture(cameraTex, v_uv).rgb;
    float m = texture(maskTex, v_uv).r;

    fragColor = vec4(m, m, m, 1.0);
}