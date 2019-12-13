////////////////////////////////////////////////////////////////////////////////

// OpenGL Helpers to reduce the clutter
#include "helpers.h"
// GLFW is necessary to handle the OpenGL context
#include <GLFW/glfw3.h>
// Linear Algebra Library
#include <Eigen/Dense>

#include <iostream>
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

// constants
#define	FREQUENCY_BINS	32
#define	SMOOTHING_CONSTANT	0.00007
#define GAMMA	2.0
#define SCALE	0.05
#define BAR_SPACE	0.03
#define TOTAL_VERTICES	FREQUENCY_BINS*6+5 // each bar defined by 6 vertices
int WIDTH, HEIGHT, CHANNEL_COUNT, SAMPLE_RATE;

// VertexBufferObject wrapper
VertexBufferObject VBO;
VertexBufferObject VBO_C;

// vertex positions
Eigen::MatrixXf V(2, TOTAL_VERTICES);

// per-vertex colors
Eigen::MatrixXf C(3, TOTAL_VERTICES);

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
std::vector<float> freqvec_smooth(FREQUENCY_BINS, 0.0);
float max = 0;

////////////////////////////////////////////////////////////////////////////////

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

	// accessing samples adapted from https://github.com/dr-soft/miniaudio/blob/master/examples/simple_mixing.c
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
		int total_samples = framesReadThisIteration*CHANNEL_COUNT;
        for (iSample = 0; iSample < total_samples; ++iSample) {
			float multiplier = 0.5 * (1 - cos(2*M_PI*iSample/total_samples));
			timevec.push_back(multiplier * temp[iSample]); // apply hann window
			pOutputF32[totalFramesRead*CHANNEL_COUNT + iSample] = temp[iSample]; // play audio
        }

		// perform fft
		fft.fwd(freqvec, timevec);

		// frequency range and time smoothing adapted from https://dlbeer.co.nz/articles/fftvis.html
		float smoothing = pow(SMOOTHING_CONSTANT, ((float) freqvec.size()) / SAMPLE_RATE);
		float frequencies = freqvec.size()/2;
		int freq_start = 0;
		for (int i = 0; i < FREQUENCY_BINS; i++) {
			int freq_end = round(pow(((float) (i+1)) / FREQUENCY_BINS, GAMMA) * frequencies);
			if (freq_end > frequencies) freq_end = frequencies;
			float freq_width = freq_end - freq_start;
			if (freq_width <= 0) freq_width = 1;

			float bin_power = 0;
			for (int j = 0; j < freq_width; j++) {
				float power = pow(std::abs(freqvec[freq_start + j]), 2);
				if (power > bin_power) bin_power = power; // find max power of frequency bin
			}

			bin_power = log(bin_power);
			if (bin_power < 0) bin_power = 0;

			freqvec_smooth[i] = freqvec_smooth[i] * smoothing + (bin_power * (1 - smoothing) * SCALE); // apply smoothing to avoid jittery display

			freq_start = freq_end;
		}

		// draw bars
		float space = (2.0-(BAR_SPACE*(FREQUENCY_BINS-1)))/FREQUENCY_BINS;
		float pos = -FREQUENCY_BINS*space/(2.0-(BAR_SPACE*(FREQUENCY_BINS-1)));

		float curr_max = *max_element(freqvec_smooth.begin(), freqvec_smooth.end());
		if (curr_max > max) max = curr_max;
		
		for (int i = 0; i < FREQUENCY_BINS; i++) {
			float data = -1 + ((1-(-1))/max) * freqvec_smooth[i]; // map to canonical range, scaled by max frequency of audio slice

			// change color based on bar height
			Eigen::Vector3f color;
			if (data < -0.5) {
				color = Eigen::Vector3f(0.6, 0.725, 0.596);
			} else if (data < 0) {
				color = Eigen::Vector3f(0.992, 0.807, 0.666);
			} else if (data < 0.5) {
				color = Eigen::Vector3f(0.956, 0.513, 0.49);
			} else {
				color = Eigen::Vector3f(0.921, 0.286, 0.376);
			}

			V.col(i*6) << pos, -1; // lower left
			V.col(i*6+1) << pos, data; // upper left
			V.col(i*6+2) << pos+space, data; // upper right
			V.col(i*6+3) << pos+space, -1; // lower right

			// repeated vertices
			V.col(i*6+4) << V.col(i*6);
			V.col(i*6+5) << V.col(i*6+2);
			
			for (int j = 0; j <= 5; j++) {
				C.col(i*6+j) << color;
			}

			pos += space + BAR_SPACE;
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
	VBO.update(V);

	VBO_C.init();
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
	SAMPLE_RATE = decoder.outputSampleRate;
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
		glClearColor(0.152f, 0.211f, 0.231f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);

		// visualize data
		VBO.update(V);
		VBO_C.update(C);
		glDrawArrays(GL_TRIANGLES, 0, TOTAL_VERTICES);

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
