package org.openxray.app;

import org.libsdl.app.SDLActivity;

/**
 * SDL entry point for the first Android bring-up APK.
 *
 * The initial APK deliberately starts the native engine in headless smoke
 * mode. This validates the Android ABI, JNI loader and xrCore filesystem
 * bootstrap without bundling proprietary game data.
 */
public final class XRayActivity extends SDLActivity {
    @Override
    protected String[] getLibraries() {
        // SDL2 is linked statically into libmain.so for this APK.
        return new String[] { "main" };
    }

    @Override
    protected String[] getArguments() {
        return new String[] { "-headless-smoke", "-no_gamepad", "-nosplash" };
    }
}
