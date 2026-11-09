"""Exercise production Android window sizing before/after Vulkan creation."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def fixture(source):
    helpers = source.split('namespace\n{\n', 1)[1].split('\n}\n#endif', 1)[0]
    sizing = source[source.index('    dwWidth = psDeviceMode.Width;'):
                    source.index('\nSDL_Window* CRenderDevice::GetApplicationWindow()')]
    return r'''
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstring>
#define XR_PLATFORM_ANDROID
using u32 = unsigned;
using pcstr = const char*;
auto xr_strlen(const char* s) { return std::strlen(s); }
template<class T> void clamp(T& v, T a, T b) { v = std::clamp(v,a,b); }
template<class T> T _min(T a, T b) { return std::min(a,b); }
template<class T> T _max(T a, T b) { return std::max(a,b); }
void Msg(const char*, ...) {}
struct { const char* Params; } Core;
constexpr unsigned SDL_WINDOW_VULKAN = 1;
struct Window { unsigned flags; } window;
unsigned SDL_GetWindowFlags(Window* w) { return w->flags; }
struct { u32 Width=2400, Height=1080; } psDeviceMode;
struct Device {
 bool b_is_Ready = false;
 Window* m_sdlWnd = &window;
 u32 dwWidth=0, dwHeight=0;
 void update();
};
''' + helpers + '\nvoid Device::update() {\n' + sizing + r'''
int main() {
 Device device;
 // Auto selection is Vulkan too: command-line spelling is not a renderer ID.
 Core.Params = "-renderer-auto -android-render-width 960 -android-render-height 432";
 window.flags = SDL_WINDOW_VULKAN;
 device.update();
 assert(device.dwWidth==960 && device.dwHeight==432);
 // The renderer supplies native logical dimensions when creation completes.
 device.dwWidth=2400; device.dwHeight=1080; device.b_is_Ready=true;
 for (unsigned i=0; i<20; ++i) {
  device.update();
  assert(device.dwWidth==2400 && device.dwHeight==1080);
 }
 // A later native surface size change also updates the UI coordinate system.
 psDeviceMode.Width=1920; psDeviceMode.Height=1080;
 device.update(); assert(device.dwWidth==1920 && device.dwHeight==1080);
 // GLES still renders the complete frame at the requested size.
 window.flags=0; device.update();
 assert(device.dwWidth==960 && device.dwHeight==432);
 Core.Params="-android-render-scale 75";
 device.update(); assert(device.dwWidth==1440 && device.dwHeight==810);
 // Before Vulkan creation the scale continues to select its scene target.
 window.flags=SDL_WINDOW_VULKAN; device.b_is_Ready=false;
 device.update(); assert(device.dwWidth==1440 && device.dwHeight==810);
 device.b_is_Ready=true; device.update();
 assert(device.dwWidth==1920 && device.dwHeight==1080);
}
'''


class AndroidRenderSizeTest(unittest.TestCase):
    def test_live_vulkan_window_updates_keep_native_ui_dimensions(self):
        compiler = shutil.which('g++')
        if not compiler:
            self.skipTest('g++ is required for the production sizing regression')
        source = (ROOT / 'src/xrEngine/Device_mode.cpp').read_text()
        with tempfile.TemporaryDirectory() as directory:
            cpp, exe = Path(directory) / 'sizing.cpp', Path(directory) / 'sizing.exe'
            cpp.write_text(fixture(source))
            subprocess.run([compiler, '-std=c++17', str(cpp), '-o', str(exe)], check=True)
            subprocess.run([str(exe)], check=True)
            # The old implementation repeatedly replaced native UI dimensions
            # with scene dimensions; this must fail the same runtime assertions.
            old = source.replace(
                '    if (b_is_Ready && (SDL_GetWindowFlags(m_sdlWnd) & SDL_WINDOW_VULKAN))\n'
                '        return;\n', '', 1)
            self.assertNotEqual(old, source)
            cpp.write_text(fixture(old))
            subprocess.run([compiler, '-std=c++17', str(cpp), '-o', str(exe)], check=True)
            self.assertNotEqual(subprocess.run([str(exe)], capture_output=True).returncode, 0)


if __name__ == '__main__':
    unittest.main()
