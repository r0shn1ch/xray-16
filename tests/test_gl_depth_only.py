"""Render depth through the production GL link helpers, including null PS passes."""
from pathlib import Path
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class DepthOnlyPassTest(unittest.TestCase):
    def test_intentional_null_fragment_writes_depth_but_failed_shader_is_rejected(self):
        if not sys.platform.startswith("linux") or not shutil.which("g++") or not shutil.which("pkg-config"):
            self.skipTest("requires Linux EGL/OpenGL development libraries and g++")
        flags = subprocess.run(["pkg-config", "--cflags", "--libs", "egl", "gl"], capture_output=True, text=True)
        if flags.returncode:
            self.skipTest("EGL/OpenGL development libraries unavailable")
        source = (ROOT / "src/Layers/xrRender/ShaderResourceTraits.h").read_text()
        source = source[source.index("static GLuint GLLinkMonolithicProgram"):source.index("\n#endif", source.index("static GLuint GLGeneratePipeline"))]
        resource = (ROOT / "src/Layers/xrRenderGL/glResourceManager_Resources.cpp").read_text()
        resource = resource[resource.index("bool CResourceManager::_LinkPP"):resource.index("void CResourceManager::_DeletePP")]
        fixture = r'''
#define GL_GLEXT_PROTOTYPES 1
#include <GL/glcorearb.h>
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <type_traits>
#include <string>
#include <cstring>
#include <strings.h>
#include <vector>
using pcstr = const char*;
template<class T> using xr_vector = std::vector<T>;
#define R_ASSERT(x) assert(x)
#define CHK_GL(x) do { x; assert(glGetError() == GL_NO_ERROR); } while (false)
void Msg(const char*, ...) {}
void bind_fragment_outputs(GLuint) {}
void show_compile_errors(pcstr, GLuint, GLuint) { assert(false && "unexpected shader link failure"); }
''' + source + r'''
bool GLAD_GL_ARB_separate_shader_objects = false;
int xr_stricmp(const char* a, const char* b) { return strcasecmp(a, b); }
constexpr unsigned RC_dest_all = 1;
struct Constants { void parse(GLuint*, unsigned) {} };
struct Stage { GLuint sh; std::string cName; };
struct SPP { GLuint pp{}; std::string cName{"test pass"}; Constants constants; };
struct SPass { SPP* pp; Stage* ps; Stage* vs; Stage* gs; };
struct CResourceManager { bool _LinkPP(SPass&); };
''' + resource + r'''

// Support the pre-fix signature too, so the same GPU oracle can demonstrate
// the original runtime failure, rather than merely failing to compile it.
template<class Link> GLuint link_pass(Link fn, const char* name, GLuint ps, GLuint vs, bool depth_only) {
    if constexpr (std::is_invocable_r_v<GLuint, Link, pcstr, GLuint, GLuint, GLuint, bool>)
        return fn(name, ps, vs, 0, depth_only);
    else
        return fn(name, ps, vs, 0);
}
GLuint shader(GLenum kind, const char* text) {
    GLuint result = glCreateShader(kind);
    glShaderSource(result, 1, &text, nullptr); glCompileShader(result);
    GLint compiled = 0; glGetShaderiv(result, GL_COMPILE_STATUS, &compiled); assert(compiled);
    return result;
}
GLuint separable(GLuint shader) {
    GLuint result = glCreateProgram(); glProgramParameteri(result, GL_PROGRAM_SEPARABLE, GL_TRUE);
    glAttachShader(result, shader); glLinkProgram(result); glDetachShader(result, shader);
    GLint linked = 0; glGetProgramiv(result, GL_LINK_STATUS, &linked); assert(linked);
    return result;
}
void check_depth() {
    glClearDepth(1.); glClear(GL_DEPTH_BUFFER_BIT);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    float depth[32 * 32]{}; glReadPixels(0, 0, 32, 32, GL_DEPTH_COMPONENT, GL_FLOAT, depth);
    assert(glGetError() == GL_NO_ERROR);
    for (float value : depth) assert(std::abs(value - .5f) < .00001f);
}
int main() {
    auto get_display = reinterpret_cast<PFNEGLGETPLATFORMDISPLAYEXTPROC>(eglGetProcAddress("eglGetPlatformDisplayEXT"));
    if (!get_display) return 77;
    EGLDisplay display = get_display(EGL_PLATFORM_SURFACELESS_MESA, EGL_DEFAULT_DISPLAY, nullptr);
    if (!eglInitialize(display, nullptr, nullptr) || !eglBindAPI(EGL_OPENGL_API)) return 77;
    const EGLint attributes[]{EGL_SURFACE_TYPE, EGL_PBUFFER_BIT, EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT, EGL_NONE};
    EGLConfig config{}; EGLint count = 0;
    if (!eglChooseConfig(display, attributes, &config, 1, &count) || !count) return 77;
    const EGLint context_attributes[]{EGL_CONTEXT_MAJOR_VERSION, 4, EGL_CONTEXT_MINOR_VERSION, 3,
        EGL_CONTEXT_OPENGL_PROFILE_MASK, EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT, EGL_NONE};
    EGLContext context = eglCreateContext(display, config, EGL_NO_CONTEXT, context_attributes);
    if (context == EGL_NO_CONTEXT || !eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, context)) return 77;
    std::printf("GL depth oracle: %s\n", glGetString(GL_RENDERER));
    const GLuint vs = shader(GL_VERTEX_SHADER, R"(#version 430 core
void main() { vec2 p[3] = vec2[](vec2(-1,-1),vec2(3,-1),vec2(-1,3)); gl_Position=vec4(p[gl_VertexID],0,1); })");
    const GLuint ps = shader(GL_FRAGMENT_SHADER, R"(#version 430 core
layout(location=0) out vec4 color; void main() { color=vec4(1); })");
    GLuint depth{}, fbo{}, vao{};
    glGenTextures(1, &depth); glBindTexture(GL_TEXTURE_2D, depth);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32F, 32, 32, 0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
    glGenFramebuffers(1, &fbo); glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depth, 0);
    glDrawBuffer(GL_NONE); glReadBuffer(GL_NONE);
    assert(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);
    glGenVertexArrays(1, &vao); glBindVertexArray(vao);
    glViewport(0, 0, 32, 32); glEnable(GL_DEPTH_TEST); glDepthFunc(GL_LESS); glDepthMask(GL_TRUE);
    const GLuint program = link_pass(GLLinkMonolithicProgram, "shadow|null|null", 0, vs, true);
    assert(program && "intentional null PS must link a monolithic depth pass");
    glUseProgram(program); check_depth(); glUseProgram(0); glDeleteProgram(program);
    assert(!link_pass(GLLinkMonolithicProgram, "material|failed|null", 0, vs, false));
    assert(!link_pass(GLLinkMonolithicProgram, "null|null|null", 0, 0, true));
    const GLuint color_program = link_pass(GLLinkMonolithicProgram, "material|color|null", ps, vs, false);
    assert(color_program); glUseProgram(color_program); check_depth(); glUseProgram(0); glDeleteProgram(color_program);
    const GLuint vs_program = separable(vs), ps_program = separable(ps);
    const GLuint pipeline = link_pass(GLGeneratePipeline, "shadow|null|null", 0, vs_program, true);
    assert(pipeline && "intentional null PS must link a separable depth pass");
    glBindProgramPipeline(pipeline); check_depth(); glBindProgramPipeline(0); glDeleteProgramPipelines(1, &pipeline);
    assert(!link_pass(GLGeneratePipeline, "material|failed|null", 0, vs_program, false));
    assert(!link_pass(GLGeneratePipeline, "null|null|null", 0, 0, true));
    const GLuint color_pipeline = link_pass(GLGeneratePipeline, "material|color|null", ps_program, vs_program, false);
    assert(color_pipeline); glBindProgramPipeline(color_pipeline); check_depth();
    glBindProgramPipeline(0); glDeleteProgramPipelines(1, &color_pipeline);
    // Exercise the real resource-layer decision too: an intentional sentinel
    // and a failed named shader both have handle zero, but only one is valid.
    CResourceManager resources;
    for (bool separate : {false, true}) {
        GLAD_GL_ARB_separate_shader_objects = separate;
        Stage vertex{separate ? vs_program : vs, "shadow"}, fragment{0, "null"}, geometry{0, "null"};
        SPP depth_pass; SPass pass{&depth_pass, &fragment, &vertex, &geometry};
        assert(resources._LinkPP(pass) && depth_pass.pp);
        if (separate) glBindProgramPipeline(depth_pass.pp); else glUseProgram(depth_pass.pp);
        check_depth(); glUseProgram(0); glBindProgramPipeline(0);
        if (separate) glDeleteProgramPipelines(1, &depth_pass.pp); else glDeleteProgram(depth_pass.pp);
        fragment.cName = "failed_compile";
        SPP failed_pass; SPass failed{&failed_pass, &fragment, &vertex, &geometry};
        assert(!resources._LinkPP(failed) && !failed_pass.pp);
    }
    glDeleteProgram(vs_program); glDeleteProgram(ps_program); glDeleteShader(vs); glDeleteShader(ps);
    glDeleteVertexArrays(1, &vao); glDeleteFramebuffers(1, &fbo); glDeleteTextures(1, &depth);
    assert(glGetError() == GL_NO_ERROR);
    eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
    eglDestroyContext(display, context); eglTerminate(display);
    std::puts("Both production link paths wrote all 1024 depth texels; failed stages remained rejected.");
}
'''
        with tempfile.TemporaryDirectory() as directory:
            cpp = Path(directory) / "depth.cpp"
            executable = Path(directory) / "depth"
            cpp.write_text(fixture)
            subprocess.run(["g++", "-std=c++17", str(cpp), "-o", str(executable), *flags.stdout.split()], check=True)
            environment = os.environ.copy()
            mesa_vendor = Path("/usr/share/glvnd/egl_vendor.d/50_mesa.json")
            if mesa_vendor.is_file():
                environment.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", str(mesa_vendor))
            environment.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
            result = subprocess.run([str(executable)], env=environment, capture_output=True, text=True)
            if result.returncode == 77:
                self.skipTest("no surfaceless EGL OpenGL 4.3 device")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            print(result.stdout, end="")


if __name__ == "__main__":
    result = unittest.main(exit=False).result
    sys.exit(77 if result.skipped and result.testsRun == len(result.skipped) else not result.wasSuccessful())
