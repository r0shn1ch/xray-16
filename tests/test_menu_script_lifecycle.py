"""Exercise production menu event callbacks through script-engine teardown/restart."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def fixture(callbacks):
    return r'''
#include <cassert>
#include <memory>
#include <utility>
using CID = unsigned;
struct CEventNotifierCallbackWithCid {
 explicit CEventNotifierCallbackWithCid(CID) {}
 virtual ~CEventNotifierCallbackWithCid() = default;
 virtual void ProcessEvent() = 0;
};
struct CMainMenu {
 bool active = false, script_ready = true;
 bool m_reloadUIAfterScriptEngineStart = false;
 unsigned destroyed = 0, cleaned = 0, reloaded = 0;
 bool IsActive() const { return active; }
 void DestroyInternal(bool force) { assert(force); ++destroyed; }
 void CleanInternals() { ++cleaned; }
 void ReloadUI() { assert(script_ready); ++reloaded; }
};
auto callbacks_for(CMainMenu* menu) {
''' + callbacks + r'''
 return std::make_pair(std::make_unique<CResetEventCb>(1,menu),std::make_unique<CStartedEventCb>(2,menu));
}
int main() {
 CMainMenu menu; auto [reset, started] = callbacks_for(&menu);
 // Inactive menu: destroy its old script-backed dialog but don't create one.
 menu.script_ready = false; reset->ProcessEvent();
 assert(menu.destroyed == 1 && menu.reloaded == 0);
 menu.script_ready = true; started->ProcessEvent();
 assert(menu.reloaded == 0);
 // The active menu must defer creation until the new script factory exists.
 menu.active = true;
 for (unsigned i=0; i<3; ++i) {
  menu.script_ready = false; reset->ProcessEvent();
  assert(menu.reloaded == i && menu.m_reloadUIAfterScriptEngineStart);
  menu.script_ready = true; started->ProcessEvent();
  assert(menu.reloaded == i+1 && !menu.m_reloadUIAfterScriptEngineStart);
  started->ProcessEvent(); assert(menu.reloaded == i+1);
 }
 // Closing the menu while reset is pending must cancel reconstruction.
 menu.script_ready = false; reset->ProcessEvent(); menu.active = false;
 menu.script_ready = true; started->ProcessEvent();
 assert(menu.reloaded == 3 && !menu.m_reloadUIAfterScriptEngineStart);
}
'''


class MenuScriptLifecycleTest(unittest.TestCase):
    def test_menu_factory_is_not_called_during_script_engine_reset(self):
        compiler = shutil.which('g++')
        if not compiler:
            self.skipTest('g++ required to execute production event callbacks')
        source = (ROOT/'src/xrGame/MainMenu.cpp').read_text()
        callbacks = source[source.index('    class CResetEventCb'):source.index('    m_script_reset_event_cid =')]
        with tempfile.TemporaryDirectory() as directory:
            cpp = Path(directory)/'menu.cpp'
            exe = Path(directory)/'menu.exe'
            def run(code):
                cpp.write_text(fixture(code))
                subprocess.run([compiler,'-std=c++17',str(cpp),'-o',str(exe)],check=True)
                return subprocess.run([str(exe)],capture_output=True).returncode
            self.assertEqual(run(callbacks),0)
            # The original reset-time recreation must fail this same fixture.
            old = callbacks.replace('m_mainmenu->m_reloadUIAfterScriptEngineStart = true;',
                                    'm_mainmenu->ReloadUI();',1)
            self.assertNotEqual(old,callbacks)
            self.assertNotEqual(run(old),0)


if __name__ == '__main__':
    unittest.main()
