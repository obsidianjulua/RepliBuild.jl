# Example: Multi-Module Project

Build a project using multiple external libraries (OpenCV + Qt5).

## Project Overview

Create an image viewer application combining:
- Qt5 for GUI
- OpenCV for image processing

## Step 1: Initialize Project

```julia
using RepliBuild

RepliBuild.init("image_viewer")
cd("image_viewer")
```

## Step 2: Configure Modules

Edit `replibuild.toml`:

```toml
[project]
name = "ImageViewer"
version = "1.0.0"
description = "Image viewer with OpenCV processing"

[dependencies]
# Use RepliBuild modules
modules = ["OpenCV", "Qt5"]

# Or use JLL packages
jll_packages = ["OpenCV_jll", "Qt5Base_jll"]

[compilation]
sources = ["src/image_processor.cpp"]
headers = ["include/image_processor.h"]
include_dirs = ["include"]

cxx_standard = "c++11"
optimization = "2"

[output]
library_name = "libimageviewer"
julia_module_name = "ImageViewer"

[bindings]
namespaces = ["ImageProc"]
export_classes = []
export_functions = []
generate_high_level = true
```

## Step 3: Create Image Processor

### Header

Create `include/image_processor.h`:

```cpp
#ifndef IMAGE_PROCESSOR_H
#define IMAGE_PROCESSOR_H

#include <opencv2/opencv.hpp>
#include <string>

// C API for Julia
extern "C" {
    void* image_load(const char* path);
    void image_free(void* img);
    int image_width(void* img);
    int image_height(void* img);
    void* image_grayscale(void* img);
    void* image_blur(void* img, int kernel_size);
    void* image_resize(void* img, int width, int height);
    void image_save(void* img, const char* path);
}

namespace ImageProc {

class ImageProcessor {
public:
    static cv::Mat load(const std::string& path);
    static void save(const cv::Mat& img, const std::string& path);
    static cv::Mat grayscale(const cv::Mat& img);
    static cv::Mat blur(const cv::Mat& img, int kernel_size);
    static cv::Mat resize(const cv::Mat& img, int width, int height);
    static cv::Mat rotate(const cv::Mat& img, double angle);
    static cv::Mat flip(const cv::Mat& img, int flip_code);
};

} // namespace ImageProc

#endif
```

### Implementation

Create `src/image_processor.cpp`:

```cpp
#include "image_processor.h"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace ImageProc {

cv::Mat ImageProcessor::load(const std::string& path) {
    return cv::imread(path, cv::IMREAD_COLOR);
}

void ImageProcessor::save(const cv::Mat& img, const std::string& path) {
    cv::imwrite(path, img);
}

cv::Mat ImageProcessor::grayscale(const cv::Mat& img) {
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    return gray;
}

cv::Mat ImageProcessor::blur(const cv::Mat& img, int kernel_size) {
    cv::Mat blurred;
    cv::GaussianBlur(img, blurred,
                     cv::Size(kernel_size, kernel_size),
                     0);
    return blurred;
}

cv::Mat ImageProcessor::resize(const cv::Mat& img, int width, int height) {
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(width, height));
    return resized;
}

cv::Mat ImageProcessor::rotate(const cv::Mat& img, double angle) {
    cv::Point2f center(img.cols / 2.0, img.rows / 2.0);
    cv::Mat rotation_matrix = cv::getRotationMatrix2D(center, angle, 1.0);
    cv::Mat rotated;
    cv::warpAffine(img, rotated, rotation_matrix, img.size());
    return rotated;
}

cv::Mat ImageProcessor::flip(const cv::Mat& img, int flip_code) {
    cv::Mat flipped;
    cv::flip(img, flipped, flip_code);
    return flipped;
}

} // namespace ImageProc

// C API implementation
void* image_load(const char* path) {
    cv::Mat* img = new cv::Mat(ImageProc::ImageProcessor::load(path));
    return img->empty() ? nullptr : img;
}

void image_free(void* img) {
    delete static_cast<cv::Mat*>(img);
}

int image_width(void* img) {
    return static_cast<cv::Mat*>(img)->cols;
}

int image_height(void* img) {
    return static_cast<cv::Mat*>(img)->rows;
}

void* image_grayscale(void* img) {
    cv::Mat result = ImageProc::ImageProcessor::grayscale(*static_cast<cv::Mat*>(img));
    return new cv::Mat(result);
}

void* image_blur(void* img, int kernel_size) {
    cv::Mat result = ImageProc::ImageProcessor::blur(*static_cast<cv::Mat*>(img), kernel_size);
    return new cv::Mat(result);
}

void* image_resize(void* img, int width, int height) {
    cv::Mat result = ImageProc::ImageProcessor::resize(*static_cast<cv::Mat*>(img), width, height);
    return new cv::Mat(result);
}

void image_save(void* img, const char* path) {
    ImageProc::ImageProcessor::save(*static_cast<cv::Mat*>(img), path);
}
```

## Step 4: Build

```julia
using RepliBuild

# RepliBuild automatically resolves OpenCV and Qt5 modules
RepliBuild.compile()
```

## Step 5: Create Julia API

Create `julia/ImageProcessorAPI.jl`:

```julia
module ImageProcessorAPI

const libviewer = joinpath(@__DIR__, "../build/libimageviewer.so")

# Image handle type
mutable struct Image
    ptr::Ptr{Cvoid}

    function Image(path::String)
        ptr = ccall((:image_load, libviewer), Ptr{Cvoid}, (Ptr{UInt8},), path)
        ptr == C_NULL && error("Failed to load image: $path")

        img = new(ptr)
        finalizer(img) do im
            ccall((:image_free, libviewer), Cvoid, (Ptr{Cvoid},), im.ptr)
        end
        return img
    end

    # Internal constructor
    Image(ptr::Ptr{Cvoid}, managed::Bool) = begin
        img = new(ptr)
        if managed
            finalizer(img) do im
                ccall((:image_free, libviewer), Cvoid, (Ptr{Cvoid},), im.ptr)
            end
        end
        return img
    end
end

# Properties
function Base.size(img::Image)
    width = ccall((:image_width, libviewer), Cint, (Ptr{Cvoid},), img.ptr)
    height = ccall((:image_height, libviewer), Cint, (Ptr{Cvoid},), img.ptr)
    return (height, width)
end

# Image operations
function grayscale(img::Image)
    ptr = ccall((:image_grayscale, libviewer), Ptr{Cvoid}, (Ptr{Cvoid},), img.ptr)
    return Image(ptr, true)
end

function blur(img::Image, kernel_size::Int=5)
    # Ensure odd kernel size
    kernel_size = kernel_size % 2 == 0 ? kernel_size + 1 : kernel_size

    ptr = ccall((:image_blur, libviewer), Ptr{Cvoid},
                (Ptr{Cvoid}, Cint), img.ptr, kernel_size)
    return Image(ptr, true)
end

function resize(img::Image, width::Int, height::Int)
    ptr = ccall((:image_resize, libviewer), Ptr{Cvoid},
                (Ptr{Cvoid}, Cint, Cint), img.ptr, width, height)
    return Image(ptr, true)
end

function save(img::Image, path::String)
    ccall((:image_save, libviewer), Cvoid,
          (Ptr{Cvoid}, Ptr{UInt8}), img.ptr, path)
end

# Pipeline operations
function process_pipeline(input_path::String, output_path::String;
                         grayscale_filter::Bool=false,
                         blur_kernel::Int=0,
                         resize_width::Int=0,
                         resize_height::Int=0)
    img = Image(input_path)
    println("Loaded image: $(size(img))")

    if grayscale_filter
        img = grayscale(img)
        println("Applied grayscale")
    end

    if blur_kernel > 0
        img = blur(img, blur_kernel)
        println("Applied blur (kernel=$blur_kernel)")
    end

    if resize_width > 0 && resize_height > 0
        img = resize(img, resize_width, resize_height)
        println("Resized to: ($resize_height, $resize_width)")
    end

    save(img, output_path)
    println("Saved to: $output_path")
end

export Image, size, grayscale, blur, resize, save, process_pipeline

end # module
```

## Step 6: Use the API

Create `test_viewer.jl`:

```julia
include("julia/ImageProcessorAPI.jl")
using .ImageProcessorAPI

# Load an image
img = Image("test_image.jpg")
println("Image size: $(size(img))")

# Apply grayscale
gray = grayscale(img)
save(gray, "output_gray.jpg")

# Apply blur
blurred = blur(img, 15)
save(blurred, "output_blur.jpg")

# Resize
small = resize(img, 320, 240)
save(small, "output_small.jpg")

# Process pipeline
process_pipeline("test_image.jpg", "output_processed.jpg",
                grayscale_filter=true,
                blur_kernel=5,
                resize_width=640,
                resize_height=480)
```

## Step 7: Create Batch Processor

```julia
function batch_process(input_dir::String, output_dir::String)
    mkpath(output_dir)

    for file in readdir(input_dir)
        if endswith(file, r"\.(jpg|jpeg|png)$"i)
            input_path = joinpath(input_dir, file)
            output_path = joinpath(output_dir, "processed_$file")

            println("Processing: $file")
            try
                process_pipeline(input_path, output_path,
                               grayscale_filter=false,
                               blur_kernel=3,
                               resize_width=800,
                               resize_height=600)
            catch e
                println("  Error: $e")
            end
        end
    end

    println("Batch processing complete!")
end

# Example: batch_process("input_images/", "output_images/")
```

## Complete Project Structure

```
image_viewer/
├── replibuild.toml
├── src/
│   └── image_processor.cpp
├── include/
│   └── image_processor.h
├── julia/
│   └── ImageProcessorAPI.jl
├── build/
│   └── libimageviewer.so
├── test_viewer.jl
└── test_image.jpg
```

## Module Resolution

RepliBuild automatically:

1. Resolves `OpenCV` module
   - Checks for OpenCV_jll
   - Falls back to pkg-config `opencv4`
   - Adds include dirs and link flags

2. Resolves `Qt5` module (if used)
   - Checks for Qt5Base_jll
   - Falls back to system Qt5
   - Adds Qt include dirs and libraries

## Benefits

- **Simple configuration** - Just list module dependencies
- **Cross-platform** - Modules handle platform differences
- **Flexible** - Use JLL or system libraries
- **Maintainable** - Module updates don't break projects

## Next Steps

- Add more image processing operations
- Create Qt GUI for the viewer
- Add threading for batch processing
- See **[Advanced Topics](../advanced/daemons.md)** for optimization
