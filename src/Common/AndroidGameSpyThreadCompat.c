#include "AndroidGameSpyThreadCompat.h"

#if defined(__ANDROID__)
int xrGameSpyAndroidCancel(pthread_t thread)
{
    // Bionic does not expose pthread_cancel. Detaching preserves the
    // ownership contract for callers without sending a non-portable signal
    // to a thread that may be running engine code.
    return pthread_detach(thread);
}
#endif
