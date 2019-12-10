////////////////////////////////////////////////////////////////////////////////

// OpenGL Helpers to reduce the clutter
#include "helpers.h"
// GLFW is necessary to handle the OpenGL context
#include <GLFW/glfw3.h>
// Linear Algebra Library
#include <Eigen/Dense>
// Timer
#include <chrono>

#include <iostream>
#include <random>
#include <cmath>
#define _USE_MATH_DEFINES

#define DR_FLAC_IMPLEMENTATION
#include "extras/dr_flac.h"  /* Enables FLAC decoding. */
#define DR_MP3_IMPLEMENTATION
#include "extras/dr_mp3.h"   /* Enables MP3 decoding. */
#define DR_WAV_IMPLEMENTATION
#include "extras/dr_wav.h"   /* Enables WAV decoding. */
#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"

#include <unsupported/Eigen/FFT>

////////////////////////////////////////////////////////////////////////////////

// VertexBufferObject wrapper
VertexBufferObject VBO;
VertexBufferObject VBO_C;

// number of vertices
int total_vertices = 0;

// vertex positions
Eigen::MatrixXf V;

// per-vertex colors
Eigen::MatrixXf C;
Eigen::MatrixXf colors(3,9); 

// view transformation
Eigen::Matrix4f view(4,4);

// audio stuff
std::string filename = "";
ma_result result;
ma_decoder decoder;
ma_device_config deviceConfig;
ma_device device;

// fft stuff
Eigen::FFT<float> fft;

// random
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_int_distribution<> dis(0, 8);

// constants
int CHANNEL_COUNT, WIDTH, HEIGHT;

////////////////////////////////////////////////////////////////////////////////

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
	// Get viewport size (canvas in number of pixels)
	int width, height;
	glfwGetFramebufferSize(window, &width, &height);

	// Get the size of the window (may be different than the canvas size on retina displays)
	int width_window, height_window;
	glfwGetWindowSize(window, &width_window, &height_window);

	// Get the position of the mouse in the window
	double xpos, ypos;
	glfwGetCursorPos(window, &xpos, &ypos);

	// Deduce position of the mouse in the viewport
	double highdpi = (double) width / (double) width_window;
	xpos *= highdpi;
	ypos *= highdpi;

	// Convert screen position to world coordinates
	Eigen::Vector4f p_screen(xpos,height-1-ypos,0,1);
    Eigen::Vector4f p_canonical((p_screen[0]/width)*2-1,(p_screen[1]/height)*2-1,0,1); // NOTE: y axis is flipped in glfw
    Eigen::Vector4f p_world = view.inverse() * p_canonical;

	if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
		/*
		// sanity check
		V.conservativeResize(Eigen::NoChange, total_vertices+1);
		C.conservativeResize(Eigen::NoChange, total_vertices+1);

		V.col(total_vertices) << p_world.x(), p_world.y();
		C.col(total_vertices) << colors.col(dis(gen));

		total_vertices++;
		*/
	}

	// Upload the change to the GPU
	VBO.update(V);
	VBO_C.update(C);
}

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
	switch (key) {
		default:
			break;
	}

	// Upload the change to the GPU
	VBO.update(V);
	VBO_C.update(C);
}

void data_callback(ma_device* pDevice, void* pOutput, const void* pInput, ma_uint32 frameCount)
{
    ma_decoder* pDecoder = (ma_decoder*)pDevice->pUserData;
    if (pDecoder == NULL) {
        return;
    }
	
	/*
	// normal playback
	ma_decoder_read_pcm_frames(pDecoder, pOutput, frameCount);
	*/

	// adapted from https://github.com/dr-soft/miniaudio/blob/master/examples/simple_mixing.c
	int N = 4096;
	float* pOutputF32 = (float*)pOutput;
	float temp[N];
    ma_uint32 tempCapInFrames = ma_countof(temp) / CHANNEL_COUNT;
    ma_uint32 totalFramesRead = 0;

    while (totalFramesRead < frameCount) {
        ma_uint32 iSample;
        ma_uint32 framesReadThisIteration;
        ma_uint32 totalFramesRemaining = frameCount - totalFramesRead;
        ma_uint32 framesToReadThisIteration = tempCapInFrames;
        if (framesToReadThisIteration > totalFramesRemaining) {
            framesToReadThisIteration = totalFramesRemaining;
        }

        framesReadThisIteration = (ma_uint32)ma_decoder_read_pcm_frames(pDecoder, temp, framesToReadThisIteration);
        if (framesReadThisIteration == 0) {
            break;
        }

		std::vector<float> timevec;
		std::vector<std::complex<float>> freqvec;

		// collect samples
        for (iSample = 0; iSample < framesReadThisIteration*CHANNEL_COUNT; ++iSample) {
			timevec.push_back(temp[iSample]);
			pOutputF32[totalFramesRead*CHANNEL_COUNT + iSample] = temp[iSample]; // play audio
        }

		// perform fft
		fft.fwd(freqvec, timevec);

		total_vertices = freqvec.size()/2;
		V.conservativeResize(2, total_vertices);
		C.conservativeResize(3, total_vertices);

		float space = 2.0/total_vertices;
		float pos = -total_vertices*space/2.0;
		
		std::vector<float> freqvec_abs;
		for (int i = 0; i < freqvec.size()/2; i++) {
			freqvec_abs.push_back(std::abs(freqvec[i])); // get magnitudes of complex numbers
		}

		float max = *max_element(freqvec_abs.begin(), freqvec_abs.end()); 
		float min = *min_element(freqvec_abs.begin(), freqvec_abs.end()); 

		for (int i = 0; i < freqvec_abs.size(); i++) {
			float data = -1 + ((1-(-1))/(max-min))*(freqvec_abs[i]-min); // map to canonical range
			V.col(i) << pos, data;
			C.col(i) << colors.col(dis(gen));
			pos += space;
		}

        totalFramesRead += framesReadThisIteration;

        if (framesReadThisIteration < framesToReadThisIteration) {
            break;
        }
    }

    (void)pInput;
}

int main(int argc, char *argv[]) {
	// read in audio file
	if (argc < 2) {
        printf("Usage: ./cadence <filename>\n");
        return -1;
    }
	filename = argv[1];

	colors <<
		0.345, 0.486, 0.486,
		0, 0.247, 0.368,
		0, 0.486, 0.518,
		0.910, 0.839, 0.4,
		0.620, 0.651, 0.082,
		0.733, 0.878, 0.808,
		0.996, 0.863, 0.757,
		0.969, 0.631, 0.549,
		0.945, 0.341, 0.247;

	view <<
		1,	0,	0,	0,
		0,	1,	0,	0,
		0,	0,	1,	0,
		0,	0,	0,	1;

	// Initialize the GLFW library
	if (!glfwInit()) {
		return -1;
	}

	// Activate supersampling
	glfwWindowHint(GLFW_SAMPLES, 8);

	// Ensure that we get at least a 3.2 context
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);

	// On apple we have to load a core profile with forward compatibility
#ifdef __APPLE__
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

	// Create a windowed mode window and its OpenGL context
	GLFWwindow * window = glfwCreateWindow(640, 480, "cadence: a real-time music visualizer", NULL, NULL);
	if (!window) {
		glfwTerminate();
		return -1;
	}

	// Make the window's context current
	glfwMakeContextCurrent(window);

	// Load OpenGL and its extensions
	if (!gladLoadGL()) {
		printf("Failed to load OpenGL and its extensions");
		return(-1);
	}
	printf("OpenGL Version %d.%d loaded", GLVersion.major, GLVersion.minor);

	int major, minor, rev;
	major = glfwGetWindowAttrib(window, GLFW_CONTEXT_VERSION_MAJOR);
	minor = glfwGetWindowAttrib(window, GLFW_CONTEXT_VERSION_MINOR);
	rev = glfwGetWindowAttrib(window, GLFW_CONTEXT_REVISION);
	printf("OpenGL version recieved: %d.%d.%d\n", major, minor, rev);
	printf("Supported OpenGL is %s\n", (const char*)glGetString(GL_VERSION));
	printf("Supported GLSL is %s\n", (const char*)glGetString(GL_SHADING_LANGUAGE_VERSION));

	// Initialize the VAO
	// A Vertex Array Object (or VAO) is an object that describes how the vertex
	// attributes are stored in a Vertex Buffer Object (or VBO). This means that
	// the VAO is not the actual object storing the vertex data,
	// but the descriptor of the vertex data.
	VertexArrayObject VAO;
	VAO.init();
	VAO.bind();

	// Initialize the VBO with the vertices data
	// A VBO is a data container that lives in the GPU memory
	VBO.init();
	V.resize(2, 1);
	VBO.update(V);

	VBO_C.init();
	C.resize(3, 1);
	VBO_C.update(C);

	// Initialize the OpenGL Program
	// A program controls the OpenGL pipeline and it must contains
	// at least a vertex shader and a fragment shader to be valid
	Program program;
	const GLchar* vertex_shader = R"(
		#version 150 core

		in vec2 position;
		uniform mat4 view;
		in vec3 color;
        out vec3 f_color;

		void main() {
			gl_Position = view * vec4(position, 0.0, 1.0);
			f_color = color;
		}
	)";

	const GLchar* fragment_shader = R"(
		#version 150 core

		in vec3 f_color;
		out vec4 outColor;

		void main() {
		    outColor = vec4(f_color, 1.0);
		}
	)";

	// Compile the two shaders and upload the binary to the GPU
	// Note that we have to explicitly specify that the output "slot" called outColor
	// is the one that we want in the fragment buffer (and thus on screen)
	program.init(vertex_shader, fragment_shader, "outColor");
	program.bind();

	// The vertex shader wants the position of the vertices as an input.
	// The following line connects the VBO we defined above with the position "slot"
	// in the vertex shader
	program.bindVertexAttribArray("position", VBO);
	program.bindVertexAttribArray("color", VBO_C);

	// Register the keyboard callback
	glfwSetKeyCallback(window, key_callback);

	// Register the mouse callback
	glfwSetMouseButtonCallback(window, mouse_button_callback);

	// play sound (https://github.com/dr-soft/miniaudio/blob/master/examples/simple_playback.c)
	result = ma_decoder_init_file(("../sounds/" + filename).c_str(), NULL, &decoder);
	if (result != MA_SUCCESS) {
		printf("Audio file does not exist or cannot be opened.\n");
        return -1;
    }

	deviceConfig = ma_device_config_init(ma_device_type_playback);
    deviceConfig.playback.format = decoder.outputFormat;
    deviceConfig.playback.channels = decoder.outputChannels;
	CHANNEL_COUNT = decoder.outputChannels;
    deviceConfig.sampleRate = decoder.outputSampleRate;
    deviceConfig.dataCallback = data_callback;
    deviceConfig.pUserData = &decoder;

	if (ma_device_init(NULL, &deviceConfig, &device) != MA_SUCCESS) {
        printf("Failed to open playback device.\n");
        ma_decoder_uninit(&decoder);
        return -1;
    }

    if (ma_device_start(&device) != MA_SUCCESS) {
        printf("Failed to start playback device.\n");
        ma_device_uninit(&device);
        ma_decoder_uninit(&decoder);
        return -1;
    }

	// Loop until the user closes the window
	while (!glfwWindowShouldClose(window)) {
		// Set the size of the viewport (canvas) to the size of the application window (framebuffer)
		// int width, height;
		glfwGetFramebufferSize(window, &WIDTH, &HEIGHT);
		glViewport(0, 0, WIDTH, HEIGHT);

		// Bind your VAO (not necessary if you have only one)
		VAO.bind();

		// Bind your program
		program.bind();

		glUniformMatrix4fv(program.uniform("view"), 1, GL_FALSE, view.data());

		// Clear the framebuffer
		glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);

		// visualize data
		VBO.update(V);
		VBO_C.update(C);
		glDrawArrays(GL_LINE_STRIP, 0, total_vertices);

		// Swap front and back buffers
		glfwSwapBuffers(window);

		// Poll for and process events
		glfwPollEvents();
	}

	ma_device_uninit(&device);
    ma_decoder_uninit(&decoder);

	// Deallocate opengl memory
	program.free();
	VAO.free();
	VBO.free();
	VBO_C.free();

	// Deallocate glfw internals
	glfwTerminate();
	return 0;
}
