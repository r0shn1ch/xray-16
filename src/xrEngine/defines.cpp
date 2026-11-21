#include "stdafx.h"

#ifdef DEBUG
ECORE_API bool bDebug = false;

#endif

// Video
DeviceMode psDeviceMode = {0, rsWindowedBorderless, 0, 0, 0, 32};

// release version always has "mt_*" enabled
Flags32 psDeviceFlags =
{
   rsDrawStatic | rsDrawDynamic | rsDrawDetails | rsDrawParticles | mtSound | mtNetwork
};

// textures
int psTextureLOD = 1;
