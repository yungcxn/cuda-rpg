
#include "render.cuh"

#include <X11/Xlib.h>
#define VK_USE_PLATFORM_XLIB_KHR
#include <vulkan/vulkan.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <unistd.h>

#include "../def.h"
#include "kernel.cuh"
#include "spriteinfo.cuh"
#include "tileinfo.cuh"
#include "../types/vec.h"

#define RENDER_WIDTH 256
#define RENDER_HEIGHT 144
#define RENDER_WINDOW_WIDTH 1920
#define RENDER_WINDOW_HEIGHT 1080
#define RENDER_TITLE "RPG"

static Display *x_display = 0;
static Window  x_window  = 0;

static VkInstance instance = VK_NULL_HANDLE;
static VkSurfaceKHR surface = VK_NULL_HANDLE;
static VkPhysicalDevice phys_dev = VK_NULL_HANDLE;
static VkDevice device = VK_NULL_HANDLE;
static VkQueue graphics_queue = VK_NULL_HANDLE;
static uint32_t graphics_queue_family = UINT32_MAX;

static VkSwapchainKHR swapchain = VK_NULL_HANDLE;
static VkImage* swapchain_images = 0;
static uint32_t swapchain_image_count = 0;

static VkImage shared_image = VK_NULL_HANDLE;
static VkDeviceMemory shared_memory = VK_NULL_HANDLE;
static cudaExternalMemory_t cuda_ext_mem;
static cudaMipmappedArray_t cuda_mipmap;
static cudaSurfaceObject_t cuda_surface;

static VkImage debug_overlay_image = VK_NULL_HANDLE;
static VkDeviceMemory debug_overlay_memory = VK_NULL_HANDLE;
static cudaSurfaceObject_t debug_overlay_surface = 0;
static cudaMipmappedArray_t debug_overlay_mipmap = 0;
static cudaExternalMemory_t debug_overlay_ext_mem;

static VkExtent2D current_extent = {RENDER_WINDOW_WIDTH, RENDER_WINDOW_HEIGHT};
static VkOffset2D render_offset = {0, 0};
static VkExtent2D render_extent = {RENDER_WIDTH, RENDER_HEIGHT};

static inline void _calculate_viewport(void) {
        float window_aspect = (float)current_extent.width / (float)current_extent.height;
        float canvas_aspect = (float)RENDER_WIDTH / (float)RENDER_HEIGHT;

        if (window_aspect > canvas_aspect) {
                render_extent.height = current_extent.height;
                render_extent.width = (uint32_t)((float)current_extent.height * canvas_aspect);
                render_offset.x = (current_extent.width - render_extent.width) / 2;
                render_offset.y = 0;
        } else {
                render_extent.width = current_extent.width;
                render_extent.height = (uint32_t)((float)current_extent.width / canvas_aspect);
                render_offset.x = 0;
                render_offset.y = (current_extent.height - render_extent.height) / 2;
        }
}

static inline void _create_swapchain_and_shared_image(void) {
        VkSurfaceCapabilitiesKHR caps;
        if (vkGetPhysicalDeviceSurfaceCapabilitiesKHR(phys_dev, surface, &caps) != VK_SUCCESS) THROW("Failed to get surface capabilities");

        uint32_t fmt_count = 0;
        if (vkGetPhysicalDeviceSurfaceFormatsKHR(phys_dev, surface, &fmt_count, 0) != VK_SUCCESS) THROW("Failed to get surface format count");
        if (fmt_count == 0) THROW("No surface formats found");

        VkSurfaceFormatKHR* formats = (VkSurfaceFormatKHR*) malloc(sizeof(VkSurfaceFormatKHR) * fmt_count);
        if (vkGetPhysicalDeviceSurfaceFormatsKHR(phys_dev, surface, &fmt_count, formats) != VK_SUCCESS) 
                THROW("Failed to get surface formats");
        VkSurfaceFormatKHR surface_format = formats[0];
        free(formats);

        VkFormatProperties fmt_props;
        vkGetPhysicalDeviceFormatProperties(phys_dev, surface_format.format, &fmt_props);
        if (!(fmt_props.optimalTilingFeatures & (VK_FORMAT_FEATURE_BLIT_SRC_BIT | VK_FORMAT_FEATURE_BLIT_DST_BIT))) {
                printf("Format %d lacks optimal blit support (features 0x%x). Use shader/copy fallback.", 
                       (int)surface_format.format, fmt_props.optimalTilingFeatures);
                THROW("Surface format does not support vkCmdBlitImage; adapt code to use shader copy or supported format");
        }

        Window root_return;
        int x, y;
        unsigned int width, height, border, depth;
        XGetGeometry(x_display, x_window, &root_return, &x, &y, &width, &height, &border, &depth);
        
        VkExtent2D extent;
        if (caps.currentExtent.width != UINT32_MAX) {
                extent = caps.currentExtent;
        } else {
                extent.width = width;
                extent.height = height;
                if (extent.width < caps.minImageExtent.width) extent.width = caps.minImageExtent.width;
                else if (extent.width > caps.maxImageExtent.width) extent.width = caps.maxImageExtent.width;
                if (extent.height < caps.minImageExtent.height) extent.height = caps.minImageExtent.height;
                else if (extent.height > caps.maxImageExtent.height) extent.height = caps.maxImageExtent.height;
        }
        current_extent = extent;
        _calculate_viewport();

        uint32_t img_count = caps.minImageCount + 1;
        if (caps.maxImageCount > 0 && img_count > caps.maxImageCount) img_count = caps.maxImageCount;

        VkSwapchainCreateInfoKHR sci = {
                .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
                .pNext = 0,
                .flags = 0,
                .surface = surface,
                .minImageCount = img_count,
                .imageFormat = surface_format.format,
                .imageColorSpace = surface_format.colorSpace,
                .imageExtent = extent,
                .imageArrayLayers = 1,
                .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                .imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
                .queueFamilyIndexCount = 0,
                .pQueueFamilyIndices = 0,
                .preTransform = caps.currentTransform,
                .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
                .presentMode = VK_PRESENT_MODE_FIFO_KHR,
                .clipped = VK_TRUE,
                .oldSwapchain = VK_NULL_HANDLE
        };

        if (vkCreateSwapchainKHR(device, &sci, 0, &swapchain) != VK_SUCCESS) 
                THROW("Failed to create swapchain");

        vkGetSwapchainImagesKHR(device, swapchain, &swapchain_image_count, 0);
        if (swapchain_images) free(swapchain_images);
        swapchain_images = (VkImage*) malloc(sizeof(VkImage) * swapchain_image_count);
        vkGetSwapchainImagesKHR(device, swapchain, &swapchain_image_count, swapchain_images);

        VkCommandPool cmd_pool = VK_NULL_HANDLE;
        VkCommandPoolCreateInfo cpi = {
                .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                .pNext = 0,
                .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
                .queueFamilyIndex = graphics_queue_family,
        };
        if (vkCreateCommandPool(device, &cpi, 0, &cmd_pool) != VK_SUCCESS) 
                THROW("Failed to create command pool");

        VkCommandBufferAllocateInfo cbi = {
                .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                .commandPool = cmd_pool,
                .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                .commandBufferCount = 1
        };
        VkCommandBuffer cmd;
        if (vkAllocateCommandBuffers(device, &cbi, &cmd) != VK_SUCCESS) 
                THROW("Failed to allocate command buffer");

        VkCommandBufferBeginInfo bi = {
                .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
        };
        if (vkBeginCommandBuffer(cmd, &bi) != VK_SUCCESS) 
                THROW("Failed to begin command buffer");

        for (uint32_t i = 0; i < swapchain_image_count; ++i) {
                VkImageMemoryBarrier barrier = {
                        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                        .pNext = 0,
                        .srcAccessMask = 0,
                        .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
                        .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                        .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                        .image = swapchain_images[i],
                        .subresourceRange = {
                                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                .baseMipLevel = 0,
                                .levelCount = 1,
                                .baseArrayLayer = 0,
                                .layerCount = 1
                        }
                };
                vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, 0, 0, 0, 1, &barrier);
        }

        VkImageMemoryBarrier shared_barrier = {
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                .pNext = 0,
                .srcAccessMask = 0,
                .dstAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
                .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = shared_image,
                .subresourceRange = {
                        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                        .baseMipLevel = 0,
                        .levelCount = 1,
                        .baseArrayLayer = 0,
                        .layerCount = 1
                }
        };
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, 0, 0, 0, 1, &shared_barrier);

        VkImageMemoryBarrier debug_init_barrier = {
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                .pNext = 0,
                .srcAccessMask = 0,
                .dstAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
                .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = debug_overlay_image,
                .subresourceRange = {
                        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                        .baseMipLevel = 0,
                        .levelCount = 1,
                        .baseArrayLayer = 0,
                        .layerCount = 1
                }
        };
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, 0, 0, 0, 1, &debug_init_barrier);

        if (vkEndCommandBuffer(cmd) != VK_SUCCESS) THROW("Failed to end command buffer");

        VkSubmitInfo si = {
                .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                .commandBufferCount = 1,
                .pCommandBuffers = &cmd
        };
        VkResult submit_r = vkQueueSubmit(graphics_queue, 1, &si, VK_NULL_HANDLE);
        if (submit_r != VK_SUCCESS) THROW("Failed to submit command buffer: %d", submit_r);
        vkQueueWaitIdle(graphics_queue);

        vkFreeCommandBuffers(device, cmd_pool, 1, &cmd);
        vkDestroyCommandPool(device, cmd_pool, 0);

        VkExternalMemoryImageCreateInfo ext_mem_img_info = {
                .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO,
                .pNext = 0,
                .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR
        };

        VkImageCreateInfo img_info = {
                .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                .pNext = &ext_mem_img_info,
                .flags = 0,
                .imageType = VK_IMAGE_TYPE_2D,
                .format = surface_format.format,
                .extent = { RENDER_WIDTH, RENDER_HEIGHT, 1 },
                .mipLevels = 1,
                .arrayLayers = 1,
                .samples = VK_SAMPLE_COUNT_1_BIT,
                .tiling = VK_IMAGE_TILING_OPTIMAL,
                .usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                .queueFamilyIndexCount = 0,
                .pQueueFamilyIndices = 0,
                .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED
        };

        if (vkCreateImage(device, &img_info, 0, &shared_image) != VK_SUCCESS) 
                THROW("Failed to create shared image");

        VkImageCreateInfo debug_img_info = {
                .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                .pNext = &ext_mem_img_info,
                .flags = 0,
                .imageType = VK_IMAGE_TYPE_2D,
                .format = surface_format.format,
                .extent = { current_extent.width, current_extent.height, 1 },
                .mipLevels = 1,
                .arrayLayers = 1,
                .samples = VK_SAMPLE_COUNT_1_BIT,
                .tiling = VK_IMAGE_TILING_OPTIMAL,
                .usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                .queueFamilyIndexCount = 0,
                .pQueueFamilyIndices = 0,
                .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED
        };

        if (vkCreateImage(device, &debug_img_info, 0, &debug_overlay_image) != VK_SUCCESS) THROW("Failed to create debug overlay image");

        VkMemoryRequirements debug_mem_reqs;
        vkGetImageMemoryRequirements(device, debug_overlay_image, &debug_mem_reqs);

        VkPhysicalDeviceMemoryProperties debug_mem_props;
        vkGetPhysicalDeviceMemoryProperties(phys_dev, &debug_mem_props);

        uint32_t debug_mem_type_idx = UINT32_MAX;
        for (uint32_t i = 0; i < debug_mem_props.memoryTypeCount; ++i) {
                if ((debug_mem_reqs.memoryTypeBits & (1 << i)) && (debug_mem_props.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
                        debug_mem_type_idx = i;
                        break;
                }
        }

        if (debug_mem_type_idx == UINT32_MAX) THROW("Failed to find suitable memory type for debug overlay image");

        VkMemoryRequirements mem_reqs;
        vkGetImageMemoryRequirements(device, shared_image, &mem_reqs);

        VkPhysicalDeviceMemoryProperties mem_props;
        vkGetPhysicalDeviceMemoryProperties(phys_dev, &mem_props);
        uint32_t mem_type_idx = UINT32_MAX;
        for (uint32_t i = 0; i < mem_props.memoryTypeCount; ++i) {
                if ((mem_reqs.memoryTypeBits & (1 << i)) && (mem_props.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
                        mem_type_idx = i;
                        break;
                }
        }
        if (mem_type_idx == UINT32_MAX) THROW("Failed to find suitable memory type");

        VkExportMemoryAllocateInfo exp_mem_info = {
                .sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO,
                .pNext = 0,
                .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR
        };

        VkMemoryAllocateInfo alloc_info = {
                .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                .pNext = &exp_mem_info,
                .allocationSize = mem_reqs.size,
                .memoryTypeIndex = mem_type_idx
        };

        if (vkAllocateMemory(device, &alloc_info, 0, &shared_memory) != VK_SUCCESS) 
                THROW("Failed to allocate shared memory");
        if (vkBindImageMemory(device, shared_image, shared_memory, 0) != VK_SUCCESS) 
                THROW("Failed to bind image memory");

        VkMemoryGetFdInfoKHR get_fd_info = {
                .sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR,
                .pNext = 0,
                .memory = shared_memory,
                .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR
        };

        PFN_vkGetMemoryFdKHR vkGetMemoryFdKHR = 
                (PFN_vkGetMemoryFdKHR) vkGetDeviceProcAddr(device, "vkGetMemoryFdKHR");
        if (!vkGetMemoryFdKHR) THROW("Failed to get vkGetMemoryFdKHR");

        int fd = -1;
        if (vkGetMemoryFdKHR(device, &get_fd_info, &fd) != VK_SUCCESS) THROW("Failed to get memory FD");

        cudaExternalMemoryHandleDesc cuda_ext_mem_desc = {};
        cuda_ext_mem_desc.type = cudaExternalMemoryHandleTypeOpaqueFd;
        cuda_ext_mem_desc.handle.fd = fd;
        cuda_ext_mem_desc.size = mem_reqs.size;

        if (cudaImportExternalMemory(&cuda_ext_mem, &cuda_ext_mem_desc) != cudaSuccess) 
                THROW("Failed to import external memory to CUDA");

        close(fd);

        cudaExternalMemoryMipmappedArrayDesc mipmap_desc = {};
        mipmap_desc.extent = make_cudaExtent(RENDER_WIDTH, RENDER_HEIGHT, 1);
        mipmap_desc.formatDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
        mipmap_desc.numLevels = 1;
        mipmap_desc.flags = cudaArraySurfaceLoadStore;

        if (cudaExternalMemoryGetMappedMipmappedArray(&cuda_mipmap, cuda_ext_mem, &mipmap_desc) 
                != cudaSuccess) THROW("Failed to get CUDA mipmapped array");

        cudaArray_t cuda_array = 0;
        if (cudaGetMipmappedArrayLevel(&cuda_array, cuda_mipmap, 0) != cudaSuccess) THROW("Failed to get CUDA array level");

        cudaResourceDesc res_desc = {};
        res_desc.resType = cudaResourceTypeArray;
        res_desc.res.array.array = cuda_array;

        if (cudaCreateSurfaceObject(&cuda_surface, &res_desc) != cudaSuccess) THROW("Failed to create CUDA surface object");

        VkMemoryAllocateInfo debug_alloc_info = {
                .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                .pNext = &exp_mem_info,
                .allocationSize = debug_mem_reqs.size,
                .memoryTypeIndex = debug_mem_type_idx   
        };

        if (vkAllocateMemory(device, &debug_alloc_info, 0, &debug_overlay_memory) != VK_SUCCESS) THROW("Failed to allocate debug overlay memory");
        if (vkBindImageMemory(device, debug_overlay_image, debug_overlay_memory, 0) != VK_SUCCESS) THROW("Failed to bind debug overlay image memory");

        VkMemoryGetFdInfoKHR debug_get_fd_info = {
                .sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR,
                .pNext = 0,
                .memory = debug_overlay_memory,
                .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR
        };

        int debug_fd = -1;
        if (vkGetMemoryFdKHR(device, &debug_get_fd_info, &debug_fd) != VK_SUCCESS) THROW("Failed to get debug overlay memory FD");

        cudaExternalMemoryHandleDesc debug_cuda_ext_mem_desc = {};
        debug_cuda_ext_mem_desc.type = cudaExternalMemoryHandleTypeOpaqueFd;
        debug_cuda_ext_mem_desc.handle.fd = debug_fd;
        debug_cuda_ext_mem_desc.size = debug_mem_reqs.size;

        if (cudaImportExternalMemory(&debug_overlay_ext_mem, &debug_cuda_ext_mem_desc) != cudaSuccess) 
                THROW("Failed to import debug overlay external memory to CUDA");

        close(debug_fd);

        cudaExternalMemoryMipmappedArrayDesc debug_mipmap_desc = {};
        debug_mipmap_desc.extent = make_cudaExtent(current_extent.width, current_extent.height, 1);
        debug_mipmap_desc.formatDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
        debug_mipmap_desc.numLevels = 1;
        debug_mipmap_desc.flags = cudaArraySurfaceLoadStore;
        debug_mipmap_desc.formatDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
        debug_mipmap_desc.numLevels = 1;
        debug_mipmap_desc.flags = cudaArraySurfaceLoadStore;

        if (cudaExternalMemoryGetMappedMipmappedArray(&debug_overlay_mipmap, debug_overlay_ext_mem, &debug_mipmap_desc) != cudaSuccess) 
        THROW("Failed to get debug overlay CUDA mipmapped array");

        cudaArray_t debug_cuda_array = 0;
        if (cudaGetMipmappedArrayLevel(&debug_cuda_array, debug_overlay_mipmap, 0) != cudaSuccess) THROW("Failed to get debug overlay CUDA array level");

        cudaResourceDesc debug_res_desc = {};
        debug_res_desc.resType = cudaResourceTypeArray;
        debug_res_desc.res.array.array = debug_cuda_array;

        if (cudaCreateSurfaceObject(&debug_overlay_surface, &debug_res_desc) != cudaSuccess) THROW("Failed to create debug overlay CUDA surface object");
}


static inline void _init_window_and_device(void) {
        x_display = XOpenDisplay(NULL);
        if (!x_display) THROW("Failed to open X display\n");

        int32_t screen = DefaultScreen(x_display);
        Window root = RootWindow(x_display, screen);

        XSetWindowAttributes swa;
        swa.event_mask = ExposureMask | KeyPressMask | StructureNotifyMask;
        x_window = XCreateWindow(
                x_display, root, 0, 0, RENDER_WINDOW_WIDTH, RENDER_WINDOW_HEIGHT, 0,
                CopyFromParent, InputOutput, CopyFromParent, CWEventMask, &swa
        );
        XStoreName(x_display, x_window, RENDER_TITLE);
        XMapWindow(x_display, x_window);
        XSync(x_display, False);

        const char* inst_exts[] = { VK_KHR_SURFACE_EXTENSION_NAME, "VK_KHR_xlib_surface" };

        VkApplicationInfo appInfo = {
                .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                .pNext = 0,
                .pApplicationName = RENDER_TITLE,
                .applicationVersion = VK_MAKE_VERSION(1,0,0),
                .pEngineName = "none",
                .engineVersion = VK_MAKE_VERSION(1,0,0),
                .apiVersion = VK_API_VERSION_1_2
        };

        VkInstanceCreateInfo ic = {
                .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                .pNext = 0,
                .flags = 0,
                .pApplicationInfo = &appInfo,
                .enabledLayerCount = 0,
                .ppEnabledLayerNames = 0,
                .enabledExtensionCount = (uint32_t)(sizeof(inst_exts) / sizeof(inst_exts[0])),
                .ppEnabledExtensionNames = inst_exts
        };

        if (vkCreateInstance(&ic, 0, &instance) != VK_SUCCESS) THROW("Failed to create Vulkan instance");

        VkXlibSurfaceCreateInfoKHR x_sci = {
                .sType = VK_STRUCTURE_TYPE_XLIB_SURFACE_CREATE_INFO_KHR,
                .pNext = 0,
                .flags = 0,
                .dpy = x_display,
                .window = x_window
        };

        if (vkCreateXlibSurfaceKHR(instance, &x_sci, 0, &surface) != VK_SUCCESS) 
                THROW("Failed to create Vulkan Xlib surface");

        uint32_t dev_count = 0;
        vkEnumeratePhysicalDevices(instance, &dev_count, 0);
        if (dev_count == 0) THROW("Failed to find GPUs with Vulkan support");

        VkPhysicalDevice* devices = (VkPhysicalDevice*) malloc(sizeof(VkPhysicalDevice) * dev_count);
        if (vkEnumeratePhysicalDevices(instance, &dev_count, devices) != VK_SUCCESS) 
                THROW("Failed to enumerate physical devices");

        bool found = false;
        for (uint32_t i = 0; i < dev_count && !found; ++i) {
                VkPhysicalDevice pd = devices[i];
                uint32_t qcount = 0;
                vkGetPhysicalDeviceQueueFamilyProperties(pd, &qcount, 0);
                if (qcount == 0) continue;
                VkQueueFamilyProperties *qprops = 
                (VkQueueFamilyProperties*) malloc(sizeof(VkQueueFamilyProperties) * qcount);
                vkGetPhysicalDeviceQueueFamilyProperties(pd, &qcount, qprops);
                for (uint32_t q = 0; q < qcount; ++q) {
                        VkBool32 present = VK_FALSE;
                        vkGetPhysicalDeviceSurfaceSupportKHR(pd, q, surface, &present);
                        if ((qprops[q].queueFlags & VK_QUEUE_GRAPHICS_BIT) && present) {
                                phys_dev = pd;
                                graphics_queue_family = q;
                                found = true;
                                break;
                        }
                }
                free(qprops);
        }
        free(devices);
        if (phys_dev == VK_NULL_HANDLE) THROW("Failed to find a suitable GPU");
        if (!found) THROW("Failed to find a suitable queue family");

        float qprio = 1.0f;
        VkDeviceQueueCreateInfo qci = {
                .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                .pNext = 0,
                .flags = 0,
                .queueFamilyIndex = graphics_queue_family,
                .queueCount = 1,
                .pQueuePriorities = &qprio
        };

        const char *dev_exts[] = {
                VK_KHR_SWAPCHAIN_EXTENSION_NAME,
                VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
                VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
                VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME
        };

        VkDeviceCreateInfo dci = {
                .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                .pNext = 0,
                .flags = 0,
                .queueCreateInfoCount = 1,
                .pQueueCreateInfos = &qci,
                .enabledLayerCount = 0,
                .ppEnabledLayerNames = 0,
                .enabledExtensionCount = (uint32_t)(sizeof(dev_exts)/sizeof(dev_exts[0])),
                .ppEnabledExtensionNames = dev_exts,
                .pEnabledFeatures = 0
        };

        if (vkCreateDevice(phys_dev, &dci, 0, &device) != VK_SUCCESS) THROW("Failed to create logical device");
        vkGetDeviceQueue(device, graphics_queue_family, 0, &graphics_queue);
        _create_swapchain_and_shared_image();
}

static inline void _recreate_swapchain() {
        vkDeviceWaitIdle(device);

        if (cuda_surface) {
                cudaDestroySurfaceObject(cuda_surface);
                cuda_surface = 0;
        }
        if (debug_overlay_surface) {
                cudaDestroySurfaceObject(debug_overlay_surface);
                debug_overlay_surface = 0;
        }
        if (debug_overlay_mipmap) {
                cudaFreeMipmappedArray(debug_overlay_mipmap);
                debug_overlay_mipmap = 0;
        }
        if (debug_overlay_ext_mem) {
                cudaDestroyExternalMemory(debug_overlay_ext_mem);
                debug_overlay_ext_mem = 0;
        }
        if (debug_overlay_memory != VK_NULL_HANDLE) {
                vkFreeMemory(device, debug_overlay_memory, 0);
                debug_overlay_memory = VK_NULL_HANDLE;
        }
        if (debug_overlay_image != VK_NULL_HANDLE) {
                vkDestroyImage(device, debug_overlay_image, 0);
                debug_overlay_image = VK_NULL_HANDLE;
        }
        if (cuda_mipmap) {
                cudaFreeMipmappedArray(cuda_mipmap);
                cuda_mipmap = 0;
        }
        if (cuda_ext_mem) {
                cudaDestroyExternalMemory(cuda_ext_mem);
                cuda_ext_mem = 0;
        }
        if (shared_memory != VK_NULL_HANDLE) {
                vkFreeMemory(device, shared_memory, 0);
                shared_memory = VK_NULL_HANDLE;
        }
        if (shared_image != VK_NULL_HANDLE) {
                vkDestroyImage(device, shared_image, 0);
                shared_image = VK_NULL_HANDLE;
        }
        if (swapchain != VK_NULL_HANDLE) {
                vkDestroySwapchainKHR(device, swapchain, 0);
                swapchain = VK_NULL_HANDLE;
        }

        _create_swapchain_and_shared_image();
}

static inline void _destroy_window(void) {
        if (cuda_surface) {
                cudaDestroySurfaceObject(cuda_surface);
                cuda_surface = 0;
        }
        if (cuda_mipmap) {
                cudaFreeMipmappedArray(cuda_mipmap);
                cuda_mipmap = 0;
        }
        if (cuda_ext_mem) {
                cudaDestroyExternalMemory(cuda_ext_mem);
                cuda_ext_mem = 0;
        }
        if (shared_memory != VK_NULL_HANDLE) {
                vkFreeMemory(device, shared_memory, 0);
                shared_memory = VK_NULL_HANDLE;
        }
        if (shared_image != VK_NULL_HANDLE) {
                vkDestroyImage(device, shared_image, 0);
                shared_image = VK_NULL_HANDLE;
        }

        if (swapchain != VK_NULL_HANDLE) {
                vkDestroySwapchainKHR(device, swapchain, 0);
                swapchain = VK_NULL_HANDLE;
        }

        if (device != VK_NULL_HANDLE) {
                vkDestroyDevice(device, 0);
                device = VK_NULL_HANDLE;
        }

        if (surface != VK_NULL_HANDLE) {
                vkDestroySurfaceKHR(instance, surface, 0);
                surface = VK_NULL_HANDLE;
        }

        if (instance != VK_NULL_HANDLE) {
                vkDestroyInstance(instance, 0);
                instance = VK_NULL_HANDLE;
        }

        if (x_window) {
                XDestroyWindow(x_display, x_window);
                x_window = 0;
        }

        if (x_display) {
                XCloseDisplay(x_display);
                x_display = 0;
        }
}



void render_setup() {
        _init_window_and_device();
        tileinfo_devtables_init();
        spriteinfo_devtables_init();
}


void render_cleanup() {
        _destroy_window();
}


static inline void _pre_render(uint32_t* image_index, cudaSurfaceObject_t *out_surf, VkResult* res) {
        if (swapchain == VK_NULL_HANDLE) THROW("Swapchain not initialized in render_frame");

        VkResult r = vkAcquireNextImageKHR(device, swapchain, UINT64_MAX, VK_NULL_HANDLE, VK_NULL_HANDLE, image_index);
        if (r != VK_SUCCESS) THROW("vkAcquireNextImageKHR returned %d\n", r);
        if (*image_index >= swapchain_image_count) THROW("Invalid image index in pre_render");
        *out_surf = cuda_surface;
        *res = r;
}

static inline void _post_render(uint32_t image_index, cudaSurfaceObject_t surf, VkResult* res) {
        if (image_index >= swapchain_image_count) THROW("Invalid image index in post_render");
        if (cudaDeviceSynchronize() != cudaSuccess) THROW("Failed to synchronize CUDA device");
        
        VkCommandPool cmd_pool = VK_NULL_HANDLE;
        VkCommandPoolCreateInfo cpi = {
                .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                .pNext = 0,
                .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
                .queueFamilyIndex = graphics_queue_family
        };
        if (vkCreateCommandPool(device, &cpi, 0, &cmd_pool) != VK_SUCCESS) THROW("Failed to create command pool");
        
        VkCommandBufferAllocateInfo cbi = {
                .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                .pNext = 0,
                .commandPool = cmd_pool,
                .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                .commandBufferCount = 1
        };
        VkCommandBuffer cmd;
        if (vkAllocateCommandBuffers(device, &cbi, &cmd) != VK_SUCCESS) THROW("Failed to allocate command buffer");
        
        VkCommandBufferBeginInfo bi = {
                .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                .pNext = 0,
                .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
                .pInheritanceInfo = 0
        };
        if (vkBeginCommandBuffer(cmd, &bi) != VK_SUCCESS) THROW("Failed to begin command buffer");
        
        VkImageMemoryBarrier barrier1 = {
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                .pNext = 0,
                .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
                .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
                .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = shared_image,
                .subresourceRange = {
                        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                        .baseMipLevel = 0,
                        .levelCount = 1,
                        .baseArrayLayer = 0,
                        .layerCount = 1
                }
        };
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, 0, 0, 0, 1, &barrier1);
        
        VkImageMemoryBarrier barrier2 = {
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                .pNext = 0,
                .srcAccessMask = 0,
                .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = swapchain_images[image_index],
                .subresourceRange = {
                        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                        .baseMipLevel = 0,
                        .levelCount = 1,
                        .baseArrayLayer = 0,
                        .layerCount = 1
                }
        };
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, 0, 0, 0, 1, &barrier2);
        
        VkClearColorValue clear_color = {{0.0f, 0.0f, 0.0f, 1.0f}};
        VkImageSubresourceRange range = {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1
        };
        vkCmdClearColorImage(cmd, swapchain_images[image_index], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clear_color, 1, &range);

        const uint8_t scale = (uint8_t) min((float) render_extent.width / (float) RENDER_WIDTH, (float) render_extent.height / (float) RENDER_HEIGHT);
        VkImageBlit blit_region = {
                .srcSubresource = {
                        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                        .mipLevel = 0,
                        .baseArrayLayer = 0,
                        .layerCount = 1
                },
                .srcOffsets = {{0, 0, 0}, {RENDER_WIDTH, RENDER_HEIGHT, 1}},
                .dstSubresource = {
                        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                        .mipLevel = 0,
                        .baseArrayLayer = 0,
                        .layerCount = 1
                },
                .dstOffsets = {
                        {render_offset.x, render_offset.y, 0},
                        {render_offset.x + RENDER_WIDTH*scale, render_offset.y + RENDER_HEIGHT*scale, 1}
                }
        };
        vkCmdBlitImage(
                cmd, shared_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, swapchain_images[image_index], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 
                1, &blit_region, VK_FILTER_NEAREST
        );

        VkImageMemoryBarrier debug_barrier = {
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                .pNext = 0,
                .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
                .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
                .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = debug_overlay_image,
                .subresourceRange = {
                        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                        .baseMipLevel = 0,
                        .levelCount = 1,
                        .baseArrayLayer = 0,
                        .layerCount = 1
                }
        };
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, 0, 0, 0, 1, &debug_barrier);

        VkImageBlit debug_blit = {
                .srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
                .srcOffsets = {{0, 0, 0}, {(int32_t) current_extent.width, (int32_t) current_extent.height, 1}},
                .dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
                .dstOffsets = {{0, 0, 0}, {(int32_t) current_extent.width, (int32_t) current_extent.height, 1}}
        };
        vkCmdBlitImage(
                cmd, debug_overlay_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, swapchain_images[image_index], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                &debug_blit, VK_FILTER_LINEAR
        );
        
        VkImageMemoryBarrier barrier3 = {
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                .pNext = 0,
                .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
                .dstAccessMask = 0,
                .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                .newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = swapchain_images[image_index],
                .subresourceRange = {
                        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                        .baseMipLevel = 0,
                        .levelCount = 1,
                        .baseArrayLayer = 0,
                        .layerCount = 1
                }
        };
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, 0, 0, 0, 1, &barrier3);

        VkImageMemoryBarrier restore_barriers[2];
        restore_barriers[0] = (VkImageMemoryBarrier){
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                .pNext = 0,
                .srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
                .dstAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
                .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = shared_image,
                .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}
        };
        restore_barriers[1] = (VkImageMemoryBarrier){
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                .pNext = 0,
                .srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
                .dstAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
                .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = debug_overlay_image,
                .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}
        };
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, 0, 0, 0, 2, restore_barriers);

        
        if (vkEndCommandBuffer(cmd) != VK_SUCCESS) THROW("Failed to end command buffer");
        
        VkSubmitInfo si = {
                .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                .pNext = 0,
                .waitSemaphoreCount = 0,
                .pWaitSemaphores = 0,
                .pWaitDstStageMask = 0,
                .commandBufferCount = 1,
                .pCommandBuffers = &cmd,
                .signalSemaphoreCount = 0,
                .pSignalSemaphores = 0
        };
        VkResult submit_r = vkQueueSubmit(graphics_queue, 1, &si, VK_NULL_HANDLE);
        if (submit_r != VK_SUCCESS) THROW("Failed to submit command buffer: %d", submit_r);
        vkQueueWaitIdle(graphics_queue);
        
        vkFreeCommandBuffers(device, cmd_pool, 1, &cmd);
        vkDestroyCommandPool(device, cmd_pool, 0);


        VkPresentInfoKHR pinfo = {
                .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
        };

        pinfo.swapchainCount = 1;
        pinfo.pSwapchains = &swapchain;
        pinfo.pImageIndices = &image_index;
        pinfo.pResults = 0;

        *res = vkQueuePresentKHR(graphics_queue, &pinfo);
        if (*res == VK_ERROR_OUT_OF_DATE_KHR || *res == VK_SUBOPTIMAL_KHR) _recreate_swapchain();
}


void render(void) {
        VkResult res = VK_SUCCESS;
        uint32_t img_idx = 0;
        cudaSurfaceObject_t surf = 0;

        _pre_render(&img_idx, &surf, &res);

        kernel_draw(surf, RENDER_WIDTH, RENDER_HEIGHT);

        _post_render(img_idx, surf, &res);
}

Display* render_get_xdisplay(void) {
        return x_display;
}

Window render_get_xwindow(void) {
        return x_window;
}