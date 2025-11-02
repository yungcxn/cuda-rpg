#ifndef VULKAN_H
#define VULKAN_H

#include <vulkan/vulkan.h>
#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif
#include "util/ccuda.h"

#ifndef VK_USE_PLATFORM_XLIB_KHR
        #define VK_USE_PLATFORM_XLIB_KHR
#endif
        #include <X11/Xlib.h>


typedef struct {
        Display* x_display;
        Window x_window;
} vulkan_setupdata_t;

vulkan_setupdata_t vulkan_setup(void);
void vulkan_cleanup(void);
void vulkan_pre_render(uint32_t* image_index, ccuda_surfaceobj_t *outsurf);
void vulkan_post_render(uint32_t image_index);

Display* vulkan_get_xdisplay(void);
Window vulkan_get_xwindow(void);

#endif // VULKAN_H