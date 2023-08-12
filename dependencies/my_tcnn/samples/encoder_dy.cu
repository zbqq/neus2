#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/common_device.h>

#include <tiny-cuda-nn/config.h>

#include <stbi/stbi_wrapper.h>

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace tcnn;
using precision_t = network_precision_t;
using PARAMS_T = network_precision_t;

template <typename T>
__global__ void debug_log(
	const uint32_t  n_elements,
	tcnn::MatrixView<T> output
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;
	printf("%d: %.10f\n",i, (float)output(i,0));
}


// template <typename T>
// __global__ void set_constant_value_view_vector(
// 	const uint32_t n_elements,
// 	const uint32_t n_pos_dim,
// 	const T value,
// 	tcnn::MatrixView<T> output
// ) {
// 	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
// 	if (i >= n_elements) return;

// 	const uint32_t elem_idx = i / n_pos_dim;
// 	const uint32_t dim_idx = i - elem_idx * n_pos_dim;

// 	output(dim_idx, elem_idx) = value;
// }


int main(int argc, char* argv[]) {
	json pos_encoding = {
			// {"otype", "HashGrid"},
			// {"n_levels", 12},
			// {"n_features_per_level", 2},
			// {"log2_hashmap_size", 15},
			// {"base_resolution", 16},
			// {"per_level_scale", 1.5},
			// {"interpolation", "Linear"}
			{"otype", "HashGrid"},
			{"n_levels", 16},
			{"n_features_per_level", 2},
			{"log2_hashmap_size", 19},
			{"base_resolution", 16},
			{"per_level_scale", 1.5},
			{"interpolation", "Linear"}
	};

	// std::shared_ptr<tcnn::Encoding<T>> m_pos_encoding = tcnn::create_encoding<T>(
	// std::shared_ptr<tcnn::Encoding<precision_t>> m_pos_encoding = tcnn::create_encoding<precision_t>(
	auto m_pos_encoding = tcnn::create_encoding<precision_t>(
			3, pos_encoding, 16u);

	std::unique_ptr<Context> pos_encoding_ctx;

	const uint32_t batch_size = 128;
	// const uint32_t batch_size = 1;

	cudaStream_t stream;
	CUDA_CHECK_THROW(cudaStreamCreate(&stream));

	// GPUMemory<char> m_params_buffer;
	// auto n_params = m_pos_encoding->n_params();
	// m_params_buffer.resize(sizeof(PARAMS_T) * n_params * 3 + sizeof(float) * n_params * 1);
	// // m_params_buffer.memset(0);

	// auto m_params_full_precision = (float*)(m_params_buffer.data());
	// auto m_params                = (PARAMS_T*)(m_params_buffer.data() + sizeof(float) * n_params);
	// auto m_params_backward       = (PARAMS_T*)(m_params_buffer.data() + sizeof(float) * n_params + sizeof(PARAMS_T) * n_params);
	// auto m_param_gradients       = (PARAMS_T*)(m_params_buffer.data() + sizeof(float) * n_params + sizeof(PARAMS_T) * n_params * 2);
	// auto m_params_inference = m_params;

	// m_params_buffer.resize(sizeof(PARAMS_T) * n_params * 3);
	// // m_params_buffer.memset(0);

	// auto m_params_full_precision = (float*)(m_params_buffer.data());
	// auto m_params                = (PARAMS_T*)(m_params_buffer.data() + sizeof(float) * n_params);
	// auto m_params_backward       = (PARAMS_T*)(m_params_buffer.data() + sizeof(float) * n_params + sizeof(PARAMS_T) * n_params);
	// auto m_param_gradients       = (PARAMS_T*)(m_params_buffer.data() + sizeof(float) * n_params + sizeof(PARAMS_T) * n_params * 2);
	// auto m_params_inference = m_params;

	// default_rng_t m_rng{1337};

	// m_pos_encoding->initialize_params(
	// 	m_rng,
	// 	m_params_full_precision,
	// 	m_params,
	// 	m_params_inference,
	// 	m_params_backward,
	// 	m_param_gradients
	// );

	GPUMemory<char> m_params_buffer;
	auto n_params = m_pos_encoding->n_params();
	printf("n_params:%d\n",n_params);
	// m_params_buffer.resize(sizeof(PARAMS_T) * n_params);
	m_params_buffer.resize(sizeof(float) * n_params + sizeof(PARAMS_T) * n_params);
	auto m_params_full_precision = (float*)(m_params_buffer.data());

	auto m_params                = (PARAMS_T*)(m_params_buffer.data() + sizeof(float) * n_params);
	auto m_params_backward       = (PARAMS_T*)(m_params_buffer.data() + sizeof(float) * n_params);
	auto m_param_gradients       = (PARAMS_T*)(m_params_buffer.data() + sizeof(float) * n_params);
	auto m_params_inference = m_params;
	pcg32 m_rng{1337};
	// default_rng_t m_rng{1337};
	// m_pos_encoding->initialize_params(m_rng, m_params_full_precision, nullptr, nullptr, nullptr, nullptr);
	m_pos_encoding->initialize_params(
		m_rng,
		m_params_full_precision,
		m_params,
		m_params_inference,
		m_params_backward,
		m_param_gradients
	);

	// generate_random_uniform<PARAMS_T>(m_rng, n_params, m_params, 1.0f, 1.0f);
	generate_random_uniform<PARAMS_T>(m_rng, n_params, m_params, 0.0f, 1.0f);
	// generate_random_uniform<PARAMS_T>(m_rng, n_params, m_params, -0.1f, 0.1f);
	// generate_random_logistic<PARAMS_T>(m_rng, n_params, m_params, (PARAMS_T)0.0f, (PARAMS_T)0.1f);

	tcnn::GPUMatrix<float, RM> input(m_pos_encoding->input_width(), batch_size, stream);

	tcnn::GPUMatrixDynamic<precision_t> output(m_pos_encoding->padded_output_width(), batch_size, stream, m_pos_encoding->preferred_output_layout());

	linear_kernel(set_constant_value_view_vector<float>, 
		0, stream, 
		m_pos_encoding->input_width() * batch_size , m_pos_encoding->input_width(), float(1.0f), input.view());

	// tcnn::linear_kernel(set_constant_value_view_vector<float>, 0, stream,
	// 	batch_size * m_pos_encoding->input_width(), m_pos_encoding->input_width(), 1.0f, input.view());


	printf("input_width,batch:%d,%d\n",m_pos_encoding->input_width(),batch_size);
	printf("output_width,batch:%d,%d\n",m_pos_encoding->padded_output_width(),batch_size);
	// linear_kernel(debug_log<float>, 
	// 	0, stream, 
	// 	m_pos_encoding->input_width(), input.view());
	// linear_kernel(debug_log<precision_t>, 
	// 	0, stream, 
	// 	m_pos_encoding->padded_output_width(), output.view());

	// m_pos_encoding->initialize_params(
	// 	rnd,
	// 	params_full_precision + offset,
	// 	params + offset,
	// 	inference_params + offset,
	// 	backward_params + offset,
	// 	gradients + offset,
	// 	scale
	// );
	// m_pos_encoding->forward(
	pos_encoding_ctx = m_pos_encoding->forward(
		stream,
		input,
		&output,
		// nullptr,
		false,
		// prepare_input_gradients
		// false
		true
	);

	printf("output\n");

	// linear_kernel(debug_log<precision_t>, 
	// 	0, stream, 
	// 	m_pos_encoding->padded_output_width(), output.view());

	// m_pos_encoding->backward(
	// 		stream,
	// 		pos_encoding_ctx,
	// 		input,
	// 		output,
	// 		dL_ddensity_network_input,
	// 		dL_dinput ? &dL_dpos_encoding_input : nullptr,
	// 		use_inference_params,
	// 		param_gradients_mode
	// 	);



		tcnn::GPUMatrix<precision_t> pos_encoding_dy{m_pos_encoding->padded_output_width(), batch_size};
		tcnn::GPUMatrix<float> dL_dsdf_dinput{ m_pos_encoding->input_width(), batch_size};
		dL_dsdf_dinput.memset_async(stream, 0);
		pos_encoding_dy.memset_async(stream, 0);
		linear_kernel(set_constant_value_view_vector<float>, 
			0, stream, 
			m_pos_encoding->input_width() * batch_size , m_pos_encoding->input_width(), float(1.0f), dL_dsdf_dinput.view());

		// m_pos_encoding->cal_dy(
		// 	stream,
		// 	*pos_encoding_ctx,
		// 	dL_dsdf_dinput,
		// 	&pos_encoding_dy,
		// 	false,
		// 	tcnn::EGradientMode::Ignore
		// );
		printf("dy\n");
		// linear_kernel(debug_log<precision_t>, 
		// 0, stream, 
		// m_pos_encoding->padded_output_width(), pos_encoding_dy.view());
	

	// free_all_gpu_memory_arenas();
}