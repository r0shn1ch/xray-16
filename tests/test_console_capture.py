"""Reproduce buried console captures using production input/console methods."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def regression_source(console_source):
    inputs = (ROOT / 'src/xrEngine/xr_input.cpp').read_text()
    capture = inputs[inputs.index('void CInput::iCapture('):inputs.index('void CInput::OnAppActivate(')]
    frame = console_source[console_source.index('void CConsole::OnFrame()'):console_source.index('    bool bGame = false;')]+ '}\n'
    show_hide = console_source[console_source.index('void CConsole::Show()'):console_source.index('void CConsole::SelectCommand()')]
    prefix = r'''
#include <vector>
#include <cassert>
#include <iterator>
#define VERIFY assert
#define ZoneScoped
using u32 = unsigned;
template<class T> using xr_vector = std::vector<T>;
struct IInputReceiver {
 virtual ~IInputReceiver() = default;
 virtual void IR_OnActivate() {}
 virtual void IR_OnDeactivate() {}
 void IR_Capture(); void IR_Release();
};
struct CInput {
 xr_vector<IInputReceiver*> cbStack;
 struct {} controllerState;
 void iCapture(IInputReceiver*); void iRelease(IInputReceiver*);
 IInputReceiver* CurrentIR() { return cbStack.back(); }
};
CInput input; CInput* pInput = &input;
void IInputReceiver::IR_Capture() { pInput->iCapture(this); }
void IInputReceiver::IR_Release() { pInput->iRelease(this); }
struct Editor {
 bool active = false;
 bool IsActiveState() { return active; }
 void UpdateTextInput(bool = false) {}
};
struct { unsigned dwFrame = 1; Editor ed;
 Editor& editor() { return ed; }
 struct { void Add(void*) {} void Remove(void*) {} } seqFrame;
} Device;
void* g_pGamePersistent = nullptr;
struct { bool isDedicatedServer = false; } GEnv;
struct CConsole : IInputReceiver {
 bool bVisible = false;
 char m_edit_string[1024]{};
 void reset_cmd_history_idx() {} void reset_selected_tip() {} void update_tips() {}
 void Show(); void Hide(); void OnFrame();
};
'''
    check = r'''
int main() {
 IInputReceiver dummy, level, modal;
 input.cbStack.push_back(&dummy);
 level.IR_Capture();
 CConsole console;
 for (unsigned i=0; i<8; ++i) {
  console.Show();
  // Loading/modal UI captures above an already visible console.
  modal.IR_Capture();
  console.OnFrame();
  assert(input.CurrentIR() == &console);
  console.OnFrame(); // another frame must not add another capture
  console.Hide();
  assert(input.CurrentIR() == &modal);
  modal.IR_Release();
  assert(input.CurrentIR() == &level);
  assert(input.cbStack.size() == 2);
 }
 // A console that was not capturing while the editor was active also works.
 Device.ed.active = true; console.Show();
 Device.ed.active = false; console.OnFrame(); console.Hide();
 assert(input.CurrentIR() == &level && input.cbStack.size() == 2);
}
'''
    return prefix + capture + frame + show_hide + check


class ConsoleCaptureTest(unittest.TestCase):
    def test_modal_recapture_and_hide_leave_gameplay_in_control(self):
        compiler = shutil.which('g++')
        if not compiler:
            self.skipTest('g++ is required for the production console regression')
        console = (ROOT / 'src/xrEngine/XR_IOConsole.cpp').read_text()
        with tempfile.TemporaryDirectory() as directory:
            cpp = Path(directory) / 'console.cpp'
            exe = Path(directory) / 'console.exe'
            cpp.write_text(regression_source(console))
            subprocess.run([compiler, '-std=c++17', str(cpp), '-o', str(exe)], check=True)
            subprocess.run([str(exe)], check=True)
            # Prove this fixture catches the old hidden-capture failure.
            old = console.replace('            IR_Release();\n            IR_Capture();', '            IR_Capture();', 1)
            self.assertNotEqual(old, console)
            cpp.write_text(regression_source(old))
            subprocess.run([compiler, '-std=c++17', str(cpp), '-o', str(exe)], check=True)
            result = subprocess.run([str(exe)], capture_output=True)
            self.assertNotEqual(result.returncode, 0)


if __name__ == '__main__':
    unittest.main()
