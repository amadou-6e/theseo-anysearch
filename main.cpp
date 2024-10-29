#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <httplib.h>
#include <thread>
#include <vector>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <iostream>
#include "stb_image_write.h"

// Global variables for frame sharing
std::vector<unsigned char> latest_frame;
std::mutex frame_mutex;
std::condition_variable frame_cv;
std::atomic<bool> running{true};

// Debug logging with timestamps
void log(const std::string &message)
{
    time_t now = time(0);
    std::string timestamp = ctime(&now);
    timestamp = timestamp.substr(0, timestamp.length() - 1); // Remove newline
    std::cout << "[" << timestamp << "] " << message << std::endl;
}

// OpenGL error checking
bool checkGLError(const char *operation)
{
    GLenum error;
    bool hasError = false;
    while ((error = glGetError()) != GL_NO_ERROR)
    {
        std::cerr << "OpenGL error after " << operation << ": 0x"
                  << std::hex << error << std::dec << std::endl;
        hasError = true;
    }
    return !hasError;
}

// Function to capture GLFW framebuffer
std::vector<unsigned char> captureFramebuffer(GLFWwindow *window)
{
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);

    if (width == 0 || height == 0)
    {
        log("Invalid framebuffer size");
        return std::vector<unsigned char>();
    }

    std::vector<unsigned char> buffer(width * height * 4);

    // Read pixels from framebuffer
    glReadBuffer(GL_BACK);
    if (!checkGLError("glReadBuffer"))
    {
        log("Failed to set read buffer");
        return std::vector<unsigned char>();
    }

    glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, buffer.data());
    if (!checkGLError("glReadPixels"))
    {
        log("Failed to read pixels");
        return std::vector<unsigned char>();
    }

    // Flip image vertically
    std::vector<unsigned char> flipped(width * height * 4);
    for (int y = 0; y < height; y++)
    {
        memcpy(flipped.data() + y * width * 4,
               buffer.data() + (height - 1 - y) * width * 4,
               width * 4);
    }

    // Convert to PNG
    int png_size = 0;
    unsigned char *rgb_buffer = new unsigned char[width * height * 3];

    // Convert RGBA to RGB
    for (int i = 0; i < width * height; i++)
    {
        rgb_buffer[i * 3] = flipped[i * 4];
        rgb_buffer[i * 3 + 1] = flipped[i * 4 + 1];
        rgb_buffer[i * 3 + 2] = flipped[i * 4 + 2];
    }

    unsigned char *png_data = stbi_write_png_to_mem(rgb_buffer, width * 3,
                                                    width, height, 3, &png_size);
    delete[] rgb_buffer;

    std::vector<unsigned char> png_buffer;
    if (png_data && png_size > 0)
    {
        png_buffer.assign(png_data, png_data + png_size);
        STBIW_FREE(png_data);
        log("Generated PNG: " + std::to_string(png_size) + " bytes");
    }
    else
    {
        log("Failed to generate PNG");
    }

    return png_buffer;
}

// GLFW rendering thread
void renderThread()
{
    log("Starting render thread");

    // Initialize GLFW
    if (!glfwInit())
    {
        log("Failed to initialize GLFW");
        return;
    }

    // Use OpenGL 2.1 for simpler rendering
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);

    GLFWwindow *window = glfwCreateWindow(800, 600, "GLFW Stream", NULL, NULL);
    if (!window)
    {
        log("Failed to create GLFW window");
        glfwTerminate();
        return;
    }

    glfwMakeContextCurrent(window);
    log("GLFW window created");

    // Initialize GLEW
    GLenum err = glewInit();
    if (err != GLEW_OK)
    {
        log("Failed to initialize GLEW: " + std::string((char *)glewGetErrorString(err)));
        glfwDestroyWindow(window);
        glfwTerminate();
        return;
    }
    log("GLEW initialized");

    // Simple animation variables
    float angle = 0.0f;

    while (running && !glfwWindowShouldClose(window))
    {
        // Clear the screen
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // Use legacy OpenGL for simpler rendering
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(-1, 1, -1, 1, -1, 1);

        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();

        // Rotate triangle
        glRotatef(angle, 0, 0, 1);
        angle += 1.0f;

        // Draw a simple triangle
        glBegin(GL_TRIANGLES);
        glColor3f(1.0f, 0.0f, 0.0f);
        glVertex2f(-0.5f, -0.5f);
        glColor3f(0.0f, 1.0f, 0.0f);
        glVertex2f(0.5f, -0.5f);
        glColor3f(0.0f, 0.0f, 1.0f);
        glVertex2f(0.0f, 0.5f);
        glEnd();

        if (!checkGLError("rendering"))
        {
            log("Error during rendering");
        }

        glfwSwapBuffers(window);

        // Capture and store frame
        auto frame = captureFramebuffer(window);
        if (!frame.empty())
        {
            std::lock_guard<std::mutex> lock(frame_mutex);
            latest_frame = std::move(frame);
            frame_cv.notify_one();
        }

        glfwPollEvents();
        std::this_thread::sleep_for(std::chrono::milliseconds(16));
    }

    glfwDestroyWindow(window);
    glfwTerminate();
    log("Render thread terminated");
}

int main()
{
    log("Starting application");

    // Start GLFW rendering thread
    std::thread render_thread(renderThread);

    // HTTP server setup
    httplib::Server svr;

    // Add server logging
    svr.set_logger([](const auto &req, const auto & /*res*/)
                   { log("Incoming request: " + req.path); });

    // Add error handler
    svr.set_error_handler([](const auto & /*req*/, auto &res)
                          { log("Error occurred: " + std::to_string(res.status)); });

    // Serve HTML page
    svr.Get("/", [](const httplib::Request &, httplib::Response &res)
            {
        log("Serving main page");
        res.set_content(R"(
            <!DOCTYPE html>
            <html>
            <head>
                <title>GLFW Stream</title>
                <style>
                    body { margin: 0; display: flex; flex-direction: column; align-items: center; min-height: 100vh; background: #333; color: white; }
                    img { max-width: 800px; width: 100%; height: auto; }
                    #status { margin: 10px 0; }
                    #debug { margin: 10px; padding: 10px; background: #444; border-radius: 4px; }
                </style>
            </head>
            <body>
                <div id="status">Connecting...</div>
                <img src="/frame" id="stream" onerror="this.style.display='none'">
                <div id="debug"></div>
                <script>
                    const img = document.getElementById('stream');
                    const status = document.getElementById('status');
                    const debug = document.getElementById('debug');
                    let retryCount = 0;
                    
                    function updateImage() {
                        const timestamp = new Date().getTime();
                        fetch('/frame?' + timestamp, {
                            timeout: 2000 // 2 second timeout
                        })
                        .then(response => {
                            debug.textContent = `Response status: ${response.status}`;
                            if (!response.ok) throw new Error(`HTTP ${response.status}`);
                            return response.blob();
                        })
                        .then(blob => {
                            debug.textContent += `, Blob size: ${blob.size} bytes`;
                            img.style.display = 'block';
                            img.src = URL.createObjectURL(blob);
                            status.textContent = 'Connected';
                            status.style.color = '#4CAF50';
                            retryCount = 0;
                            setTimeout(updateImage, 16);
                        })
                        .catch(error => {
                            retryCount++;
                            status.textContent = `Connection error (Retry ${retryCount})`;
                            status.style.color = '#f44336';
                            debug.textContent = `Error: ${error.message} at ${new Date().toLocaleTimeString()}`;
                            setTimeout(updateImage, 1000);
                        });
                    }
                    updateImage();
                </script>
            </body>
            </html>
        )", "text/html"); });

    // Serve frames
    svr.Get("/frame", [](const httplib::Request &, httplib::Response &res)
            {
        log("Frame requested");
        std::unique_lock<std::mutex> lock(frame_mutex);
        if (frame_cv.wait_for(lock, std::chrono::seconds(1), [] { return !latest_frame.empty(); })) {
            log("Sending frame: " + std::to_string(latest_frame.size()) + " bytes");
            res.set_content(reinterpret_cast<char*>(latest_frame.data()), 
                          latest_frame.size(), 
                          "image/png");
        } else {
            log("Frame timeout");
            res.status = 408;  // Request Timeout
        } });

    log("Starting server on port 8080");
    if (!svr.listen("0.0.0.0", 8080))
    {
        log("Failed to start server");
        running = false;
        render_thread.join();
        return 1;
    }

    running = false;
    render_thread.join();
    log("Application terminated");

    return 0;
}