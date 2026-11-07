#include <glm/glm.hpp>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace
{
using glm::vec2;
using glm::vec3;
using glm::dot;
using glm::min;
using glm::max;
using glm::abs;
using glm::clamp;
constexpr int Width = 96, Height = 64;
std::vector<vec3> source(Width * Height);
unsigned samples;

vec3 aa_sample(vec2 uv)
{
    ++samples;
    const float x = std::clamp(uv.x * Width - .5f, 0.f, float(Width - 1));
    const float y = std::clamp(uv.y * Height - .5f, 0.f, float(Height - 1));
    const int ix = int(x), iy = int(y);
    const int nx = std::min(ix + 1, Width - 1), ny = std::min(iy + 1, Height - 1);
    return glm::mix(glm::mix(source[iy * Width + ix], source[iy * Width + nx], x - ix),
                    glm::mix(source[ny * Width + ix], source[ny * Width + nx], x - ix), y - iy);
}

// Execute the production shader's filter as C++ with a bilinear software
// texture sampler. This is not a second implementation of the algorithm.
#include "src/Layers/xrRenderVK/smoke/fxaa.glsl"

vec2 uv(int x, int y) { return vec2((x + .5f) / Width, (y + .5f) / Height); }
const vec2 pixel(1.f / Width, 1.f / Height);
void write_scene(const std::string& filename, const std::vector<vec3>& filtered)
{
    std::ofstream out(filename, std::ios::binary);
    out << "P6\n" << Width * 2 << ' ' << Height << "\n255\n";
    for (int y = 0; y < Height; ++y)
        for (const auto* image : {static_cast<const std::vector<vec3>*>(&source), &filtered})
            for (int x = 0; x < Width; ++x)
                for (unsigned c = 0; c < 3; ++c)
                    out.put(char(std::lround(255.f * std::clamp((*image)[y * Width + x][c], 0.f, 1.f))));
    assert(out.good());
}
}

int main(int argc, char** argv)
{
    // Flat colors and disabled AA must retain color. No green-only luma:
    // independently test all three channels, including pure red and blue.
    for (const vec3 color : {vec3(0), vec3(1), vec3(.2f, .5f, .9f)})
    {
        std::fill(source.begin(), source.end(), color);
        for (const vec2 pos : {vec2(0), vec2(.5f), vec2(1)})
        {
            samples = 0;
            assert(glm::length(fxaa(pos, pixel, 1.f) - color) < 1e-6f);
            assert(samples == 5); // flat-region early exit
            samples = 0;
            assert(glm::length(fxaa(pos, pixel, 0.f) - color) < 1e-6f);
            assert(samples == 1);
        }
    }
    // Move two slanted silhouettes through subpixel phases and compare with
    // analytic pixel coverage (supersampled reference). A no-op filter fails
    // the strict improvement assertion, while a global blur fails flat tests.
    for (unsigned channel = 0; channel < 3; ++channel)
    {
        double original_error = 0, filtered_error = 0;
        unsigned changed = 0;
        for (unsigned frame = 0; frame < 16; ++frame)
        {
            const float offset = 15.f + frame / 16.f;
            const float slope = .37f;
            vec3 ink(0.f); ink[channel] = 1.f;
            for (int y = 0; y < Height; ++y)
                for (int x = 0; x < Width; ++x)
                    source[y * Width + x] = x + .5f > offset + slope * (y + .5f) ? ink : vec3(0.f);
            std::vector<vec3> filtered(source.size());
            for (int y = 0; y < Height; ++y)
                for (int x = 0; x < Width; ++x)
                {
                    const vec3 value = fxaa(uv(x, y), pixel, 1.f);
                    filtered[y * Width + x] = value;
                    assert(std::isfinite(value[channel]) && value[channel] >= 0.f && value[channel] <= 1.f);
                    const auto unchanged = fxaa(uv(x, y), pixel, 0.f);
                    assert(glm::length(unchanged - source[y * Width + x]) < 1e-5f);
                    if (glm::length(value - unchanged) > .001f) ++changed;
                    if (y < 8 || y >= Height - 8) continue;
                    double coverage = 0;
                    for (unsigned sy = 0; sy < 64; ++sy)
                        coverage += std::clamp(double(x + 1) - offset - slope * (y + (sy + .5) / 64), 0., 1.);
                    coverage /= 64;
                    original_error += std::pow(source[y * Width + x][channel] - coverage, 2);
                    filtered_error += std::pow(value[channel] - coverage, 2);
                    if (std::abs(x + .5f - offset - slope * (y + .5f)) > 5.f)
                        assert(glm::length(value - unchanged) < 1e-5f);
                }
            if (argc == 2 && frame % 4 == 0)
                write_scene(std::string(argv[1]) + "-" + std::to_string(channel) + "-" + std::to_string(frame) + ".ppm", filtered);
        }
        std::cout << "channel=" << channel << " raw_error=" << original_error
                  << " fxaa_error=" << filtered_error << " changed=" << changed << '\n';
        assert(changed > 100);
        assert(filtered_error < original_error * .9);
    }
}
