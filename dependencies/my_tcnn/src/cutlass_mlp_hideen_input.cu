/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *//*
 */

/** @file   cutlass_mlp.cu
 *  @author Thomas Müller, NVIDIA
 *  @brief  CUTLASS implementation of an optimized multi-layer perceptron. Supports online training
 *          and simultaneous inference.
 */

#include <tiny-cuda-nn/networks/cutlass_mlp.h>

#include <tiny-cuda-nn/cutlass_matmul.h>

TCNN_NAMESPACE_BEGIN

template <typename T>
__global__ void set_constant_value_view_vector(
	const uint32_t n_elements,
	const uint32_t n_pos_dim,
	const T value,
	tcnn::MatrixView<T> output
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	const uint32_t elem_idx = i / n_pos_dim;
	const uint32_t dim_idx = i - elem_idx * n_pos_dim;

	output(dim_idx, elem_idx) = value;
}

template <typename T>
__global__ void matrix_multiple(
	const uint32_t n_elements,
	const uint32_t row,
	const uint32_t col,
	const uint32_t batch,
	tcnn::MatrixView<T> back,
	tcnn::MatrixView<T> front,
	tcnn::MatrixView<T> output
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	const uint32_t row_id = i / col;
	const uint32_t col_id = i - row_id * col;
	for (uint32_t j = 0; j < batch; j++){
	// for (uint32_t j = 0; j < 1; j++){
		output(row_id, col_id) += back(row_id,j) * front(col_id, j);
		printf("row:%d,col:%d,output,back,front:%f,%f,%f\n",row_id,col_id,(float)output(row_id, col_id),(float)back(row_id,j),(float)front(col_id, j));
	}

}

template <typename T>
__global__ void debug_log(
	const uint32_t  n_elements,
	tcnn::MatrixView<T> output
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;
	printf("%d: %.10f\n",i, (float)output(i,0));
}

template <typename T>
CutlassMLP<T>::CutlassMLP(
	uint32_t input_width,
	uint32_t network_width,
	uint32_t output_width,
	uint32_t n_hidden_layers,
	Activation activation,
	Activation output_activation
) :
m_input_width{input_width},
m_network_width{network_width},
m_output_width{output_width},
m_n_hidden_layers{n_hidden_layers},
m_activation{activation},
m_output_activation{output_activation},
m_can_fuse_activation{activation != Activation::Sine}
{
	m_padded_output_width = next_multiple(m_output_width, tensorcore_width);

	if (n_hidden_layers > 0) {
		m_n_hidden_matmuls = n_hidden_layers-1;
	} else {
		m_n_hidden_matmuls = 0;
	}

	// Create matrices related to weights
	if (n_hidden_layers == 0) {
		m_weight_matrices.emplace_back(nullptr, m_padded_output_width, m_input_width);
		m_weight_matrices_inference.emplace_back(nullptr, m_padded_output_width, m_input_width);
		m_weight_matrices_full_precision.emplace_back(nullptr, m_padded_output_width, m_input_width);
		m_gradient_matrices.emplace_back(nullptr, m_padded_output_width, m_input_width);
	} else {
		m_weight_matrices.emplace_back(nullptr, m_network_width, m_input_width);
		m_weight_matrices_inference.emplace_back(nullptr, m_network_width, m_input_width);
		m_weight_matrices_full_precision.emplace_back(nullptr, m_network_width, m_input_width);
		m_gradient_matrices.emplace_back(nullptr, m_network_width, m_input_width);

		for (uint32_t i = 0; i < m_n_hidden_matmuls; ++i) {
			m_weight_matrices.emplace_back(nullptr, m_network_width, m_network_width);
			m_weight_matrices_inference.emplace_back(nullptr, m_network_width, m_network_width);
			m_weight_matrices_full_precision.emplace_back(nullptr, m_network_width, m_network_width);
			m_gradient_matrices.emplace_back(nullptr, m_network_width, m_network_width);
		}

		m_weight_matrices.emplace_back(nullptr, m_padded_output_width, m_network_width);
		m_weight_matrices_inference.emplace_back(nullptr, m_padded_output_width, m_network_width);
		m_weight_matrices_full_precision.emplace_back(nullptr, m_padded_output_width, m_network_width);
		m_gradient_matrices.emplace_back(nullptr, m_padded_output_width, m_network_width);
	}

	// Determine total number of memory entries and set it
	m_total_n_params = 0;
	for (const auto& m : m_weight_matrices) {
		m_total_n_params += m.n_elements();
	}

	// 1 stream per matrix.
	m_training_splitk_streams.resize(m_n_hidden_layers + 1);
	m_training_splitk_events.resize(m_n_hidden_layers + 1);

	for (size_t i = 0; i < m_training_splitk_streams.size(); ++i) {
		CUDA_CHECK_THROW(cudaStreamCreate(&m_training_splitk_streams[i]));
		CUDA_CHECK_THROW(cudaEventCreate(&m_training_splitk_events[i]));
	}
}

template <typename T>
CutlassMLP<T>::~CutlassMLP() {
	for (size_t i = 0; i < m_training_splitk_streams.size(); ++i) {
		free_gpu_memory_arena(m_training_splitk_streams[i]);

		CUDA_CHECK_PRINT(cudaEventDestroy(m_training_splitk_events[i]));
		CUDA_CHECK_PRINT(cudaStreamDestroy(m_training_splitk_streams[i]));
	}
}

template <typename CutlassLayer, typename T>
bool compute_layer(
	cudaStream_t stream,
	bool is_inference,
	Activation activation,
	const GPUMatrix<T, RM>& weights,
	const GPUMatrixDynamic<T>& input,
	GPUMatrixDynamic<T>& output,
	GPUMatrixDynamic<T>& activation_output
) {
	bool can_fuse_activation = true;
	if (!is_inference) {
		// Never disallow fusing if the caller passes the same output and activation_output buffers... in that case,
		// invertibility of the activation function may be ignored.
		// can_fuse_activation &= activation != Activation::Sine || &output == &activation_output;
		can_fuse_activation = false;
	}

	if (can_fuse_activation) {
		fc_multiply<CutlassLayer>(stream, weights, input, output, activation);
	} else {
		fc_multiply<CutlassLayer>(stream, weights, input, output);
		activation_gpu(stream, activation, output, activation_output);
	}

	// return can_fuse_activation;
	return true;
}

template <typename CutlassLayer, typename T>
bool compute_inference_layer(
	cudaStream_t stream,
	Activation activation,
	const GPUMatrix<T, RM>& weights,
	const GPUMatrixDynamic<T>& input,
	GPUMatrixDynamic<T>& output
) {
	return compute_layer<CutlassLayer>(stream, true, activation, weights, input, output, output);
}

template <typename T>
void CutlassMLP<T>::inference_mixed_precision_impl(cudaStream_t stream, const GPUMatrixDynamic<T>& input, GPUMatrixDynamic<T>& output, bool use_inference_params) {
	// If there are no hidden layers, the network is just a simple matmul.
	if (m_n_hidden_layers == 0) {
		compute_inference_layer<LastLayer>(stream, m_output_activation, input_weight_matrix(use_inference_params), input, output);
		return;
	}

	uint32_t batch_size = input.n();
	GPUMatrix<T> inference_tmp[2] = {
		GPUMatrix<T>{m_network_width, batch_size, stream},
		GPUMatrix<T>{m_network_width, batch_size, stream},
	};

	m_inference_graph.capture_and_execute(stream, false, [&]() {
		// Run the actual network
		{
			uint32_t tmp_idx = 0;

			// Input layer
			compute_inference_layer<FullLayer>(stream, m_activation, input_weight_matrix(use_inference_params), input, inference_tmp[tmp_idx++ % 2]);

			// Hidden layers
			for (uint32_t i = 0; i < m_n_hidden_matmuls; ++i) {
				compute_inference_layer<FullLayer>(stream, m_activation, weight_matrix_at(use_inference_params, i), inference_tmp[(tmp_idx + 1) % 2], inference_tmp[tmp_idx % 2]);
				++tmp_idx;
			}

			// Output
			compute_inference_layer<LastLayer>(stream, m_output_activation, output_weight_matrix(use_inference_params), inference_tmp[(tmp_idx + 1) % 2], output);
		}
	});
}

template <typename T>
std::unique_ptr<Context> CutlassMLP<T>::forward_impl(cudaStream_t stream, const GPUMatrixDynamic<T>& input, GPUMatrixDynamic<T>* output, bool use_inference_params, bool prepare_input_gradients) {
	// If there are no hidden layers, the network is just a simple matmul. No tmp buffers required
	if (m_n_hidden_layers == 0) {
		if (output) {
			// compute_layer<LastLayer>(stream, false, m_output_activation, input_weight_matrix(use_inference_params), input, *output, *output);
			compute_layer<LastLayer>(stream, true, m_output_activation, input_weight_matrix(use_inference_params), input, *output, *output);
		}
		return std::make_unique<ForwardContext>(); // Nothing to save -- empty context
	}

	// Make sure our temporary buffers have the correct size for the given batch size
	uint32_t batch_size = input.n();
	auto forward = allocate_forward_buffers(stream, batch_size);

	// Run the actual network
	uint32_t tmp_idx = 0;

	bool fused = compute_layer<FullLayer>(
		stream,
		false,
		m_activation,
		input_weight_matrix(use_inference_params),
		input,
		forward->hidden_input.at(tmp_idx),
		forward->hidden.at(tmp_idx)
		// forward->hidden.at(tmp_idx),
		// m_can_fuse_activation ? forward->hidden.at(tmp_idx) : forward->hidden.at(tmp_idx+1)
	);
	tmp_idx += fused ? 1 : 2;

	// layers
	for (uint32_t i = 0; i < m_n_hidden_matmuls; ++i) {
		fused = compute_layer<FullLayer>(
			stream,
			false,
			m_activation,
			weight_matrix_at(use_inference_params, i),
			forward->hidden.at(tmp_idx-1),
			forward->hidden_input.at(tmp_idx),
			forward->hidden.at(tmp_idx)
			// forward->hidden.at(tmp_idx),
			// m_can_fuse_activation ? forward->hidden.at(tmp_idx) : forward->hidden.at(tmp_idx+1)
		);
		tmp_idx += fused ? 1 : 2;
	}

	if (output) {
		// compute_layer<LastLayer>(stream, false, m_output_activation, output_weight_matrix(use_inference_params), forward->hidden.at(tmp_idx-1), *output, *output);
		compute_layer<LastLayer>(stream, true, m_output_activation, output_weight_matrix(use_inference_params), forward->hidden.at(tmp_idx-1), *output, *output);
	}

	return forward;
}

template <typename T>
void CutlassMLP<T>::backward_impl(
	cudaStream_t stream,
	const Context& ctx,
	const GPUMatrixDynamic<T>& input,
	const GPUMatrixDynamic<T>& output,
	const GPUMatrixDynamic<T>& dL_doutput,
	GPUMatrixDynamic<T>* dL_dinput,
	bool use_inference_params,
	EGradientMode param_gradients_mode
) {
	// Make sure our temporary buffers have the correct size for the given batch size
	uint32_t batch_size = dL_doutput.n();

	std::vector<GPUMatrix<T>> backward_tmp(num_forward_activations());
	for (uint32_t i = 0; i < num_forward_activations(); ++i) {
		backward_tmp[i] = GPUMatrix<T>{m_network_width, batch_size, stream};
	}

	// Compute transfer of output activation in-place... it's treated specially for performance reasons
	GPUMatrixDynamic<T> backward_output_tmp;
	if (m_output_activation != Activation::None) {
		backward_output_tmp = {m_padded_output_width, batch_size, stream, dL_doutput.layout()};
		activation_backward_output_gpu(stream, dL_doutput.n_elements(), m_output_activation, output.data(), dL_doutput.data(), backward_output_tmp.data());
		// activation_backward_output_gpu: stream, input_elements, activation, activation_value, input_value, output_value)
	}

	// Backprop
	// - weight_gradient.T = activation * output_gradient.T
	// - input_gradient = weights.T * output_gradient
	// - RELU: pre_activation_gradinet = post_activation_gradient if val > 0 else 0

	const float param_gradient_beta = param_gradients_mode == EGradientMode::Accumulate ? 1.0f : 0.0f;

	{
		const auto& forward = dynamic_cast<const ForwardContext&>(ctx);

		int split_k_factor = batch_size / std::min((uint32_t)(1 << 12), batch_size);

		const GPUMatrixDynamic<T>& tmp_dL_doutput = m_output_activation == Activation::None ? dL_doutput : backward_output_tmp;

		// If there are no hidden layers, the network is just a simple matmul
		if (m_n_hidden_layers == 0) {
			if (param_gradients_mode != EGradientMode::Ignore) {
				cudaEventRecord(m_training_splitk_events.at(0), stream);
				cudaStreamWaitEvent(m_training_splitk_streams.at(0), m_training_splitk_events.at(0), 0);

				// Compute weight gradients
				fc_multiply_split_k<LastLayerK>(m_training_splitk_streams.at(0), tmp_dL_doutput, input.transposed(), input_gradient_matrix(), split_k_factor, param_gradient_beta);

				cudaEventRecord(m_training_splitk_events.at(0), m_training_splitk_streams.at(0));
			}

			if (dL_dinput) {
				fc_multiply<FullLayer>(stream, input_weight_matrix(use_inference_params).transposed(), tmp_dL_doutput, *dL_dinput);
			}

			if (param_gradients_mode != EGradientMode::Ignore) {
				cudaStreamWaitEvent(stream, m_training_splitk_events.at(0), 0);
			}
			return;
		}

		uint32_t tmp_idx = (m_can_fuse_activation ? (m_n_hidden_matmuls+1) : ((m_n_hidden_matmuls+1) * 2)) - 1;
		uint32_t backward_tmp_idx = 0;

		if (param_gradients_mode != EGradientMode::Ignore) {
			// Output layer
			cudaEventRecord(m_training_splitk_events.at(backward_tmp_idx), stream);
			cudaStreamWaitEvent(m_training_splitk_streams.at(backward_tmp_idx), m_training_splitk_events.at(backward_tmp_idx), 0);

			// Compute weight gradients
			fc_multiply_split_k<LastLayerK>(m_training_splitk_streams.at(backward_tmp_idx), tmp_dL_doutput, forward.hidden.at(tmp_idx).transposed(), output_gradient_matrix(), split_k_factor, param_gradient_beta);

			cudaEventRecord(m_training_splitk_events.at(backward_tmp_idx), m_training_splitk_streams.at(backward_tmp_idx));
		}

		if (!m_can_fuse_activation) {
			fc_multiply<FullLayer>(stream, output_weight_matrix(use_inference_params).transposed(), tmp_dL_doutput, backward_tmp.at(backward_tmp_idx));
			activation_backward_gpu(stream, m_activation, forward.hidden.at(tmp_idx-1), backward_tmp.at(backward_tmp_idx));
		} else {
			fc_multiply<FullLayer>(stream, output_weight_matrix(use_inference_params).transposed(), tmp_dL_doutput, forward.hidden.at(tmp_idx), backward_tmp.at(backward_tmp_idx), m_activation, true);
		}

		tmp_idx -= m_can_fuse_activation ? 1 : 2;
		++backward_tmp_idx;

		// layers
		for (uint32_t i = 0; i < m_n_hidden_matmuls; ++i) {
			uint32_t matrix_idx = m_n_hidden_matmuls - i - 1;

			if (param_gradients_mode != EGradientMode::Ignore) {
				cudaEventRecord(m_training_splitk_events.at(backward_tmp_idx), stream);
				cudaStreamWaitEvent(m_training_splitk_streams.at(backward_tmp_idx), m_training_splitk_events.at(backward_tmp_idx), 0);
				fc_multiply_split_k<FullLayerK>(m_training_splitk_streams.at(backward_tmp_idx), backward_tmp.at(backward_tmp_idx-1), forward.hidden.at(tmp_idx).transposed(), gradient_matrix_at(matrix_idx), split_k_factor, param_gradient_beta);
				cudaEventRecord(m_training_splitk_events.at(backward_tmp_idx), m_training_splitk_streams.at(backward_tmp_idx));
			}

			if (!m_can_fuse_activation) {
				fc_multiply<FullLayer>(stream, weight_matrix_at(use_inference_params, matrix_idx).transposed(), backward_tmp.at(backward_tmp_idx-1), backward_tmp.at(backward_tmp_idx));
				activation_backward_gpu(stream, m_activation, forward.hidden.at(tmp_idx-1), backward_tmp.at(backward_tmp_idx));
			} else {
				fc_multiply<FullLayer>(stream, weight_matrix_at(use_inference_params, matrix_idx).transposed(), backward_tmp.at(backward_tmp_idx-1), forward.hidden.at(tmp_idx), backward_tmp.at(backward_tmp_idx), m_activation, true);
			}

			tmp_idx -= m_can_fuse_activation ? 1 : 2;
			++backward_tmp_idx;
		}

		if (param_gradients_mode != EGradientMode::Ignore) {
			cudaEventRecord(m_training_splitk_events.at(backward_tmp_idx), stream);
			cudaStreamWaitEvent(m_training_splitk_streams.at(backward_tmp_idx), m_training_splitk_events.at(backward_tmp_idx), 0);
			fc_multiply_split_k<FullLayerK>(m_training_splitk_streams.at(backward_tmp_idx), backward_tmp.at(backward_tmp_idx-1), input.transposed(), input_gradient_matrix(), split_k_factor, param_gradient_beta);
			cudaEventRecord(m_training_splitk_events.at(backward_tmp_idx), m_training_splitk_streams.at(backward_tmp_idx));
		}

		// If requested, compute sensitivity of loss w.r.t. inputs
		if (dL_dinput) {
			// optimization opportunity to only compute sensitivity w.r.t selected SUBSET of inputs. Useful for NFs, where conditional dims stay the same.
			fc_multiply<FullLayer>(stream, input_weight_matrix(use_inference_params).transposed(), backward_tmp.at(backward_tmp_idx-1), *dL_dinput);
		}
	}

	if (param_gradients_mode != EGradientMode::Ignore) {
		// All the per-layer split-k matrix multiplications summing over
		// the batch are computed in parallel streams to the actual
		// backpropagation. Here, we need to wait for all of these to complete.
		for (auto& event : m_training_splitk_events) {
			cudaStreamWaitEvent(stream, event, 0);
		}
	}
}

// Assume that derivative is a square matrix. n_row == n_col.
template <typename T, bool SET_ROW = true>
__global__ void apply_relu_to_derivative_by_forward(
	const uint32_t n_elements,
	const uint32_t n_row,
	const uint32_t n_col,
	const tcnn::MatrixView<T> forward,
	tcnn::MatrixView<T> derivative
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	const uint32_t i_row = i / n_row;
	if (forward(i_row, 0) < T(0)) {
		const uint32_t i_col = i - n_row * i_row;
		if (SET_ROW) {
			derivative(i_row, i_col) = T(0);
		}
		else {
			derivative(i_col, i_row) = T(0);
		}
	}
}

template <typename T, bool SET_ROW = true>
__global__ void apply_relu_to_derivative_by_forward_batch(
	const uint32_t n_elements,
	const uint32_t n_pos_dim,
	const tcnn::MatrixView<T> forward,
	tcnn::MatrixView<T> derivative
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	const uint32_t elem_idx = i / n_pos_dim;
	const uint32_t dim_idx = i - elem_idx * n_pos_dim;
	// printf("apply relu:%f!\n",(float)forward(dim_idx, elem_idx));
	// printf("elem_idx:%d,dim_idx:%d,value: %f!\n",elem_idx,dim_idx,(float)forward(dim_idx, elem_idx));
	if (forward(dim_idx, elem_idx) < T(0)) {
	// if (forward(dim_idx, elem_idx) <= T(0)) {
		// printf("elem_idx:%d,dim_idx:%d\n",elem_idx,dim_idx);
		derivative(dim_idx, elem_idx) = T(0);
	}
}

template <typename T>
__global__ void kernel_compute_update_weight(
	const uint32_t n_elements,
	const uint32_t n_row,
	const uint32_t n_col,
	const tcnn::MatrixView<T> tmp_matrix_front,
	const tcnn::MatrixView<T> tmp_matrix_back,
	tcnn::MatrixView<T> update_weight,
	EGradientMode param_gradients_mode
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	const uint32_t i_row = i / n_row;
	const uint32_t i_col = i - n_row * i_row;

	T tmp_matrix_front_sum = 0.0f;
	T tmp_matrix_back_sum = 0.0f;
	for (uint32_t k = 0; k < n_row; ++k) {
		tmp_matrix_front_sum += tmp_matrix_front(k, i_col);
	}
	for (uint32_t l = 0; l < n_col; ++l) {
		tmp_matrix_back_sum += tmp_matrix_back(i_row, l);
	}
	// printf("tmp_matrix_back_sum %f\n",tmp_matrix_back_sum);
	// printf("tmp_matrix_front_sum %f\n",tmp_matrix_front_sum);
	if (param_gradients_mode == EGradientMode::Accumulate){
		update_weight(i_row, i_col) += tmp_matrix_front_sum * tmp_matrix_back_sum;
	}
	else if (param_gradients_mode == EGradientMode::Overwrite){
		update_weight(i_row, i_col) = tmp_matrix_front_sum * tmp_matrix_back_sum;
	}
	// printf("update_weight: %f", update_weight(i_row, i_col));
}

template <typename T>
void CutlassMLP<T>::backward_backward_input_impl(
	cudaStream_t stream,
	const Context& ctx,
	const GPUMatrixDynamic<T>& input,
	const GPUMatrixDynamic<T>& dL_ddLdinput,
	const GPUMatrixDynamic<T>& dL_doutput,
	GPUMatrixDynamic<T>* dL_ddLdoutput,
	GPUMatrixDynamic<T>* dL_dinput,
	bool use_inference_params,
	EGradientMode param_gradients_mode
) {

	// there exists m_hidden_layers + 1 layers in total.

	// dL_doutput:: dy for grid.h
	// Make sure our temporary buffers have the correct size for the given batch size
	uint32_t batch_size = dL_doutput.n();
	uint32_t target_batch = 0; // lxt: Should run for loop? Or pick a random batch? 
	// printf("******get into backward_backward_input********\n");
	// std::vector<GPUMatrix<T>> backward_tmp(num_forward_activations());
	// for (uint32_t i = 0; i < num_forward_activations(); ++i) {
	// 	backward_tmp[i] = GPUMatrix<T>{m_network_width, batch_size, stream};
	// }
	// printf("dl_dldinput shape:%d,%d\n",dL_ddLdinput.m(),dL_ddLdinput.n());

	// printf("num_forward_activations():%d\n",num_forward_activations());
	// printf("m_n_hidden_matmuls:%d\n",m_n_hidden_matmuls);
	// printf("m_hidden_layers:%d\n",m_n_hidden_layers);
	// uint32_t num_tmp_matrix = num_forward_activations() + 2;
	uint32_t num_tmp_matrix = m_n_hidden_layers + 3; // since that there exist one more default output layer
	std::vector<GPUMatrix<T>> tmp_front_multiply(num_tmp_matrix);

	
	// tmp_front_multiply[0].set_size_unsafe(input_weight_matrix(use_inference_params).m(), batch_size);	
	// initialize
	// tcnn::linear_kernel(set_constant_value_view_vector<T>, 0, stream,
	// 	batch_size*input_weight_matrix(use_inference_params).m() , input_weight_matrix(use_inference_params).m(), 1.0f, tmp_front_multiply[0].view());

	// tcnn::linear_kernel(debug_log<T>, 0, stream,
				// dL_ddLdinput.m(),dL_ddLdinput.view());
	for (uint32_t i = 1; i < num_tmp_matrix - 2; ++i) {
		uint32_t matrix_size = full_weight_matrix_at(use_inference_params, i - 1).m();
		tmp_front_multiply[i].set_size_unsafe(matrix_size, batch_size);
	}
	auto tmp_front_multiply_alloc = GPUMatrixBase::allocate_shared_memory(stream, tmp_front_multiply);
	tmp_front_multiply[0] = GPUMatrix<T>{dL_ddLdinput.data(),input_weight_matrix(use_inference_params).n(),batch_size};
	// tmp_front_multiply[0] = dL_ddLdinput;
	// printf("lay_out_dl_dldinput,layout_tmp_front:%d,%d\n",dL_ddLdinput.layout(),tmp_front_multiply[0].layout());
	std::vector<T> tmp_fron_multiply_cpu(tmp_front_multiply[0].m());
	CUDA_CHECK_THROW(cudaMemcpyAsync(tmp_fron_multiply_cpu.data(), tmp_front_multiply[0].data(), sizeof(T) * tmp_front_multiply[0].m(), cudaMemcpyDeviceToHost, stream));
	CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
		for (size_t j = 0; j < tmp_fron_multiply_cpu.size(); ++j) {
			// tcnn::tlog::info() << "tmp_back_multiply_CPU_debug" << j << " = " << (float)tmp_back_multiply_CPU_debug[j];
			printf("j %d:%f\n",j,(float)tmp_fron_multiply_cpu[j]);
		}
	// tmp_front_multiply[0] = dL_ddLdinput;
	// tcnn::linear_kernel(set_constant_value_view_vector<T>, 0, stream,
	// 	batch_size*input_weight_matrix(use_inference_params).n() , input_weight_matrix(use_inference_params).n(), (T)(1.0f), tmp_front_multiply[0].view()); // get average for all batches


	// for (uint32_t i = 1; i < num_tmp_matrix - 2; ++i) {
	// 	tmp_front_multiply[i].memset_async(stream, 0);
	// }

	// tcnn::linear_kernel(set_constant_value_view_vector<T>, 0, stream,
		// batch_size*input_weight_matrix(use_inference_params).n() , input_weight_matrix(use_inference_params).n(), (T)(1.0f), tmp_front_multiply[0].view()); // get average for all batches

	std::vector<GPUMatrix<T>> tmp_back_multiply(num_tmp_matrix);
	float before_matrix_size = output_weight_matrix(use_inference_params).m();
	// tmp_back_multiply[num_tmp_matrix-1].set_size_unsafe(before_matrix_size, batch_size);	
	tmp_back_multiply[num_tmp_matrix-1] = GPUMatrix<T>{before_matrix_size, batch_size};	
	// initialize
	for (uint32_t i = num_tmp_matrix - 2; i > 1; --i) {
		uint32_t matrix_size = full_weight_matrix_at(use_inference_params, i-1).n();
		tmp_back_multiply[i].set_size_unsafe(matrix_size, batch_size);
	}
	auto tmp_back_multiply_alloc = GPUMatrixBase::allocate_shared_memory(stream, tmp_back_multiply);

	// for (uint32_t i = num_tmp_matrix - 2; i > 1; --i) {
	// 	tmp_back_multiply[i].memset_async(stream, 0);
	// }
	// for (auto& tmp_back: tmp_back_multiply){
	// 	tmp_back.memset(0);
	// } // NAN occur in this part.
	printf("batch_size:%d\n",batch_size);
	tcnn::linear_kernel(set_constant_value_view_vector<T>, 0, stream,
		// batch_size*before_matrix_size , before_matrix_size, (T)(1.0f * batch_size), tmp_back_multiply[num_tmp_matrix-1].view()); // get average for all batches
		batch_size*before_matrix_size , before_matrix_size, (T)(1.0f * 128), tmp_back_multiply[num_tmp_matrix-1].view()); // get average for all batches
	// tcnn::linear_kernel(set_constant_value_view_vector<T>, 0, stream,
	// 	batch_size*before_matrix_size , before_matrix_size, (T)(1.0f/batch_size), tmp_back_multiply[num_tmp_matrix-1].view()); // get average for all batches
	// tcnn::linear_kernel(set_constant_value_view_vector<T>, 0, stream,
	// 	batch_size*before_matrix_size , before_matrix_size, 1.0f, tmp_back_multiply[num_tmp_matrix-1].view());
	// tmp_back_multiply[num_tmp_matrix-1].memset(0);
	// for (uint32_t i = num_tmp_matrix - 2; i > 1; --i) {
	// 	uint32_t matrix_size = full_weight_matrix_at(use_inference_params, i-1).n();
	// 	// tmp_back_multiply[i].memset(0);
	// 	tcnn::linear_kernel(set_constant_value_view_vector<T>, 0, stream,
	// 		batch_size*matrix_size , matrix_size, T(0.0f), tmp_back_multiply[i].view());
	// }

	// static_assert(m_output_activation != Activation::None, "Assume no output activation is used.");	

	const float param_gradient_beta = param_gradients_mode == EGradientMode::Accumulate ? 1.0f : 0.0f;

	{
		const auto& forward = dynamic_cast<const ForwardContext&>(ctx);

		int split_k_factor = batch_size / std::min((uint32_t)(1 << 12), batch_size);


		

		for (uint32_t i = 1; i < num_tmp_matrix - 2; ++i) {


			auto& cur_tmp_front_multiply = tmp_front_multiply.at(i);
			// printf("i:%d\n",i);
			// printf("before m,n:%d,%d\n",tmp_front_multiply.at(i-1).m(),tmp_front_multiply.at(i-1).n());
			// printf("weight m,n:%d,%d\n",full_weight_matrix_at(use_inference_params, i-1).m(),full_weight_matrix_at(use_inference_params, i-1).n());
			// printf("now m,n:%d,%d\n",cur_tmp_front_multiply.m(),cur_tmp_front_multiply.n());
			// printf("hidden m,n:%d,%d\n",forward.hidden.at(i-1).m(),forward.hidden.at(i-1).n());
			// auto relu_input_view = forward.hidden.at(i+1).view();
			// auto relu_input_view = forward.hidden.at(i-1).view();
			// auto relu_input_view = forward.hidden_input.at(i-1).view();
			// relu_input_view.advance_cols(target_batch);

			// printf("dim, batch:%d,%d\n",cur_tmp_front_multiply.m(),cur_tmp_front_multiply.n());
			// std::vector<T> relu_input_view_CPU_debug(cur_tmp_front_multiply.m());
			// CUDA_CHECK_THROW(cudaMemcpyAsync(relu_input_view_CPU_debug.data(), forward.hidden_input.at(i-1).data(), sizeof(T) * cur_tmp_front_multiply.m(), cudaMemcpyDeviceToHost, stream));
			// CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
			// for (size_t j = 0; j < relu_input_view_CPU_debug.size(); ++j) {
			// 	// tcnn::tlog::info() << "tmp_back_multiply_CPU_debug" << j << " = " << (float)tmp_back_multiply_CPU_debug[j];
			// 	printf("j %d:%f\n",j,(float)relu_input_view_CPU_debug[j]);
			// }

			// fc_multiply<FullLayer>(stream,
			// 	full_weight_matrix_at(use_inference_params, i-1),
			// 	tmp_front_multiply.at(i-1),
			// 	cur_tmp_front_multiply);

			// tcnn::linear_kernel(apply_relu_to_derivative_by_forward_batch<T, true>, 0, stream,
			// 	cur_tmp_front_multiply.m() * cur_tmp_front_multiply.n(),
			// 	cur_tmp_front_multiply.m(),
			// 	relu_input_view,
			// 	cur_tmp_front_multiply.view());

			// activation_backward_gpu(stream, m_activation, relu_input_view, cur_tmp_front_multiply);

			// consider relu
			fc_multiply<FullLayer>(stream, 
				full_weight_matrix_at(use_inference_params, i-1),
				tmp_front_multiply.at(i-1), 
				// forward.hidden.at(i-1), 
				forward.hidden_input.at(i-1), 
				cur_tmp_front_multiply, 
				m_activation, true);

		}

	// 	// exit(1);

		for (uint32_t i = num_tmp_matrix - 2; i > 1; --i) {
		// for (uint32_t i = num_tmp_matrix - 1; i > 1; --i) {
			// relu
			auto& cur_tmp_back_multiply = tmp_back_multiply.at(i);
			// printf("i:%d\n",i);
			// printf("before m,n:%d,%d\n",tmp_back_multiply.at(i+1).m(),tmp_back_multiply.at(i+1).n());
			// printf("weight m,n:%d,%d\n",full_weight_matrix_at(use_inference_params, i-1).m(),full_weight_matrix_at(use_inference_params, i-1).n());
			// printf("now m,n:%d,%d\n",cur_tmp_back_multiply.m(),cur_tmp_back_multiply.n());
			// printf("hidden m,n:%d,%d\n",forward.hidden.at(i-2).m(),forward.hidden.at(i-2).n());
			// printf("hidden m,n:%d,%d\n",forward.hidden.at(i-1).m(),forward.hidden.at(i-1).n());
			// auto relu_input_view = forward.hidden.at(i-2).view();
			// auto relu_input_view = forward.hidden_input.at(i-2).view();
			// relu_input_view.advance_cols(target_batch);
			
			// fc_multiply<FullLayer>(stream,
			// 	full_weight_matrix_at(use_inference_params, i-1).transposed(),
			// 	tmp_back_multiply.at(i+1),
			// 	cur_tmp_back_multiply);

			// tcnn::linear_kernel(apply_relu_to_derivative_by_forward_batch<T, true>, 0, stream,
			// 	cur_tmp_back_multiply.m() * cur_tmp_back_multiply.n(),
			// 	cur_tmp_back_multiply.m(),
			// 	relu_input_view,
			// 	cur_tmp_back_multiply.view());

			// activation_backward_gpu(stream, m_activation, relu_input_view, cur_tmp_back_multiply);

			fc_multiply<FullLayer>(stream, 
				full_weight_matrix_at(use_inference_params, i-1).transposed(),
				tmp_back_multiply.at(i+1),
				// forward.hidden.at(i-2),  // TO DO:: check -2 or -1
				forward.hidden_input.at(i-2),  // TO DO:: check -2 or -1
				cur_tmp_back_multiply, 
				m_activation, true);

		}

	

		// for (auto& tmp_front: tmp_front_multiply){
		// 	tmp_front.memset(0);
		// }
		// for (auto& tmp_back: tmp_back_multiply){
		// 	tmp_back.memset(0);
		// } // NAN occur in this part.
		for (uint32_t i = 1; i < num_tmp_matrix - 1; ++i) {
			auto& gradient_matrix = m_gradient_matrices.at(i-1);
			

			// printf("gradient_matrix_shape:%d,%d\n",gradient_matrix.m(),gradient_matrix.n());
			// printf("tmp_front_multiply:%d,%d\n",tmp_front_multiply.at(i-1).m(),tmp_front_multiply.at(i-1).n());
			// printf("tmp_back_multiply:%d,%d\n",tmp_back_multiply.at(i+1).m(),tmp_back_multiply.at(i+1).n());

			// printf("tmp_front,%d,%d\n",tmp_front_multiply.at(i-1).m(),tmp_front_multiply.at(i-1).n());
			// std::vector<T> tmp_front_multiply_CPU_debug(tmp_front_multiply.at(i-1).m());
			// CUDA_CHECK_THROW(cudaMemcpyAsync(tmp_front_multiply_CPU_debug.data(), tmp_front_multiply.at(i-1).data(), sizeof(T) * tmp_front_multiply.at(i-1).m(), cudaMemcpyDeviceToHost, stream));
			// CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
			// for (size_t j = 0; j < tmp_front_multiply_CPU_debug.size(); ++j) {
			// 	// tcnn::tlog::info() << "tmp_back_multiply_CPU_debug" << j << " = " << (float)tmp_back_multiply_CPU_debug[j];
			// 	printf("%d:%f\n",j,(float)tmp_front_multiply_CPU_debug[j]);
			// }

			// printf("tmp_back,%d,%d\n",tmp_back_multiply.at(i+1).m(),tmp_back_multiply.at(i+1).n());
			// std::vector<T> tmp_back_multiply_CPU_debug(tmp_back_multiply.at(i+1).m());
			// CUDA_CHECK_THROW(cudaMemcpyAsync(tmp_back_multiply_CPU_debug.data(), tmp_back_multiply.at(i+1).data(), sizeof(T) * tmp_back_multiply.at(i+1).m(), cudaMemcpyDeviceToHost, stream));
			// CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
			// for (size_t j = 0; j < tmp_back_multiply_CPU_debug.size(); ++j) {
			// 	// tcnn::tlog::info() << "tmp_back_multiply_CPU_debug" << j << " = " << (float)tmp_back_multiply_CPU_debug[j];
			// 	printf("%d:%f\n",j,(float)tmp_back_multiply_CPU_debug[j]);
			// }
			


			// tcnn::linear_kernel(matrix_multiple<T>, 0, stream,
			// 		tmp_back_multiply.at(i+1).m()*tmp_front_multiply.at(i-1).m(),
			// 		tmp_back_multiply.at(i+1).m(),
			// 		tmp_front_multiply.at(i-1).m(),
			// 		tmp_back_multiply.at(i+1).n(),
			// 		tmp_back_multiply.at(i+1).view(),
			// 		tmp_front_multiply.at(i-1).view(),
			// 		gradient_matrix.view()
			// 		);
			// fc_multiply<FullLayer>(stream,
				// tmp_back_multiply.at(i+1),
				// tmp_front_multiply.at(i-1).transposed(),
				// gradient_matrix);
			if (param_gradients_mode != EGradientMode::Ignore) {
				// printf("1\n");
				cudaEventRecord(m_training_splitk_events.at(i-1), stream);
				cudaStreamWaitEvent(m_training_splitk_streams.at(i-1), m_training_splitk_events.at(i-1), 0);
				fc_multiply_split_k<FullLayerK>(m_training_splitk_streams.at(i-1), tmp_back_multiply.at(i+1), tmp_front_multiply.at(i-1).transposed(), gradient_matrix, split_k_factor, param_gradient_beta);
				cudaEventRecord(m_training_splitk_events.at(i-1), m_training_splitk_streams.at(i-1));
			}

			// tcnn::linear_kernel(debug_log<T>, 0, stream,
				// gradient_matrix.n(),gradient_matrix.view());
				// printf("gradient_matrix\n");
				// std::vector<T> gradient_matrix_CPU_debug(gradient_matrix.m());
				// CUDA_CHECK_THROW(cudaMemcpyAsync(gradient_matrix_CPU_debug.data(), gradient_matrix.data(), sizeof(T) * gradient_matrix.m(), cudaMemcpyDeviceToHost, stream));
				// CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
				// for (size_t j = 0; j < gradient_matrix_CPU_debug.size(); ++j) {
				// 	// tcnn::tlog::info() << "tmp_back_multiply_CPU_debug" << j << " = " << (float)tmp_back_multiply_CPU_debug[j];
				// 	printf("j %d:%f\n",j,(float)gradient_matrix_CPU_debug[j]);
				// }
		}

	}

	if (param_gradients_mode != EGradientMode::Ignore) {

		for (auto& event : m_training_splitk_events) {
			cudaStreamWaitEvent(stream, event, 0);
		}
	}
	// auto& gradient_matrix = m_gradient_matrices.at(0);
	// printf("gradient_matrix\n");
	// std::vector<T> gradient_matrix_CPU_debug(gradient_matrix.m());
	// CUDA_CHECK_THROW(cudaMemcpyAsync(gradient_matrix_CPU_debug.data(), gradient_matrix.data(), sizeof(T) * gradient_matrix.m(), cudaMemcpyDeviceToHost, stream));
	// CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
	// for (size_t j = 0; j < gradient_matrix_CPU_debug.size(); ++j) {
	// 	// tcnn::tlog::info() << "tmp_back_multiply_CPU_debug" << j << " = " << (float)tmp_back_multiply_CPU_debug[j];
	// 	printf("j %d:%f\n",j,(float)gradient_matrix_CPU_debug[j]);
	// }
	// printf("finish backward_backward_input\n");
}

// template <typename T>
// void CutlassMLP<T>::backward_backward_input_impl(
// 	cudaStream_t stream,
// 	const Context& ctx,
// 	const GPUMatrixDynamic<T>& input,
// 	const GPUMatrixDynamic<T>& dL_ddLdinput,
// 	const GPUMatrixDynamic<T>& dL_doutput,
// 	GPUMatrixDynamic<T>* dL_ddLdoutput,
// 	GPUMatrixDynamic<T>* dL_dinput,
// 	bool use_inference_params,
// 	EGradientMode param_gradients_mode
// ) {

// 	// there exists m_hidden_layers + 1 layers in total.

// 	// Make sure our temporary buffers have the correct size for the given batch size
// 	uint32_t batch_size = dL_doutput.n();
// 	uint32_t target_batch = 0; // lxt: Should run for loop? Or pick a random batch? 
// 	printf("******get into backward_backward_input********\n");
// 	// std::vector<GPUMatrix<T>> backward_tmp(num_forward_activations());
// 	// for (uint32_t i = 0; i < num_forward_activations(); ++i) {
// 	// 	backward_tmp[i] = GPUMatrix<T>{m_network_width, batch_size, stream};
// 	// }
// 	printf("dl_dldinput shape:%d,%d\n",dL_ddLdinput.m(),dL_ddLdinput.n());

// 	printf("num_forward_activations():%d\n",num_forward_activations());
// 	printf("m_n_hidden_matmuls:%d\n",m_n_hidden_matmuls);
// 	printf("m_hidden_layers:%d\n",m_n_hidden_layers);
// 	// uint32_t num_tmp_matrix = num_forward_activations() + 2;
// 	uint32_t num_tmp_matrix = m_n_hidden_layers + 3;
// 	std::vector<GPUMatrix<T>> tmp_front_multiply(num_tmp_matrix);

// 	// uint32_t before_matrix_size = weight_matrix_at(use_inference_params, 0).transposed().n();
// 	uint32_t before_matrix_size = input_weight_matrix(use_inference_params).transposed().m();
// 	tmp_front_multiply[0].set_size_unsafe(before_matrix_size, before_matrix_size);

// 	// for (uint32_t i = 0; i < num_tmp_matrix; ++i) {
// 	// note that weight_matrix_at will add index by 1 automatically
// 	for (uint32_t i = 1; i < num_tmp_matrix - 2; ++i) {
// 		//tmp_front_multiply[i].set_size_unsafe(m_network_width, batch_size); // wym: num_forward_activations: num_hidden_layers
// 		// printf("before access memory!\n");
// 		uint32_t matrix_size = full_weight_matrix_at(use_inference_params, i - 1).transposed().n();
// 		// printf("%d, before_matrix_size:%d, after_matrix_size:%d\n",i, before_matrix_size,matrix_size);
// 		tmp_front_multiply[i].set_size_unsafe(before_matrix_size, matrix_size);
// 	}
// 	// for (uint32_t i = 1; i < num_tmp_matrix - 2; ++i) {
// 	// 	//tmp_front_multiply[i].set_size_unsafe(m_network_width, batch_size); // wym: num_forward_activations: num_hidden_layers
// 	// 	// printf("before access memory!\n");
// 	// 	uint32_t matrix_size = weight_matrix_at(use_inference_params, i - 1).transposed().n();
// 	// 	// printf("%d, before_matrix_size:%d, after_matrix_size:%d\n",i, before_matrix_size,matrix_size);
// 	// 	tmp_front_multiply[i].set_size_unsafe(before_matrix_size, matrix_size);
// 	// }
// 	// exit(1);
// 	auto tmp_front_multiply_alloc = GPUMatrixBase::allocate_shared_memory(stream, tmp_front_multiply);

// 	std::vector<GPUMatrix<T>> tmp_back_multiply(num_tmp_matrix);
// 	// for (uint32_t i = 0; i < num_tmp_matrix; ++i) {
// 		//tmp_back_multiply[i].set_size_unsafe(m_network_width, batch_size); // wym: num_forward_activations: num_hidden_layers
// 		// uint32_t matrix_size = weight_matrix_at(use_inference_params, m_n_hidden_matmuls - i).transposed().n();
// 		// tmp_back_multiply[i].set_size_unsafe(matrix_size, matrix_size);
// 	// }
// 	before_matrix_size = output_weight_matrix(use_inference_params).transposed().n();
// 	tmp_back_multiply[num_tmp_matrix-1].set_size_unsafe(before_matrix_size, before_matrix_size);	

// 	for (uint32_t i = num_tmp_matrix - 2; i >1; --i) {
// 		//tmp_back_multiply[i].set_size_unsafe(m_network_width, batch_size); // wym: num_forward_activations: num_hidden_layers
// 		// uint32_t matrix_size = full_weight_matrix_at(use_inference_params, i-1).transposed().n();
// 		uint32_t matrix_size = full_weight_matrix_at(use_inference_params, i-1).transposed().m();
// 		// uint32_t matrix_size = weight_matrix.at(use_inference_params, i-1).transposed().n();
// 		// tmp_back_multiply[i].set_size_unsafe(matrix_size, matrix_size);
// 		tmp_back_multiply[i].set_size_unsafe(before_matrix_size, matrix_size);
// 	}
	
// 	auto tmp_back_multiply_alloc = GPUMatrixBase::allocate_shared_memory(stream, tmp_back_multiply);



// 	// static_assert(m_output_activation != Activation::None, "Assume no output activation is used.");	

// 	const float param_gradient_beta = param_gradients_mode == EGradientMode::Accumulate ? 1.0f : 0.0f;

// 	{
// 		const auto& forward = dynamic_cast<const ForwardContext&>(ctx);

// 		int split_k_factor = batch_size / std::min((uint32_t)(1 << 12), batch_size);

// 		// const GPUMatrixDynamic<T>& tmp_dL_doutput = m_output_activation == Activation::None ? dL_doutput : backward_output_tmp;
// 		// const GPUMatrixDynamic<T>& tmp_dL_doutput = dL_doutput;

// 		// If there are no hidden layers, the network is just a simple matmul
// 		// static_assert(m_n_hidden_layers == 0, "no hidden layers not checked!");	

// 		// static_assert(m_can_fuse_activation == false, "sine activation not chekced!");	

// 		uint32_t num_hidden_layers = m_n_hidden_matmuls + 1;
		
// 		tmp_front_multiply[0].initialize_diagonal(1); //单位矩阵
// 		// tmp_front_multiply[0].memset(0);
		
// 		// for (uint32_t i = 1; i < num_hidden_layers ; ++i) {
// 		// for (uint32_t i = 1; i < num_hidden_layers + 1 ; ++i) {

// 		// 	// cudaEventRecord(m_training_splitk_events.at(backward_tmp_idx), stream);
// 		// 	// cudaStreamWaitEvent(m_training_splitk_streams.at(backward_tmp_idx), m_training_splitk_events.at(backward_tmp_idx), 0);
// 		// 	// cudaEventRecord(m_training_splitk_events.at(backward_tmp_idx), m_training_splitk_streams.at(backward_tmp_idx));

// 		// 	// relu
// 		// 	auto& cur_tmp_front_multiply = tmp_front_multiply.at(i);
// 		// 	// auto relu_input_view = forward.hidden.at(i+1).view();
// 		// 	auto relu_input_view = forward.hidden.at(i-1).view();
// 		// 	relu_input_view.advance_cols(target_batch);
// 		// 	printf("before m,n:%d,%d\n",tmp_front_multiply.at(i-1).m(),tmp_front_multiply.at(i-1).n());
// 		// 	printf("weight m,n:%d,%d\n",weight_matrix_at(use_inference_params, i-1).transposed().m(),weight_matrix_at(use_inference_params, i-1).transposed().n());
// 		// 	printf("now m,n:%d,%d\n",cur_tmp_front_multiply.m(),cur_tmp_front_multiply.n());
// 		// 	// fc_multiply<FullLayer>(stream,
// 		// 	// 	weight_matrix_at(use_inference_params, i-1).transposed(),
// 		// 	// 	tmp_front_multiply.at(i-1),
// 		// 	// 	cur_tmp_front_multiply);
// 		// 	fc_multiply<FullLayer>(stream,
// 		// 		tmp_front_multiply.at(i-1),
// 		// 		weight_matrix_at(use_inference_params, i-1).transposed(),
// 		// 		cur_tmp_front_multiply);
// 		// 	tcnn::linear_kernel(apply_relu_to_derivative_by_forward<T, true>, 0, stream,
// 		// 		cur_tmp_front_multiply.m() * cur_tmp_front_multiply.n(),
// 		// 		cur_tmp_front_multiply.m(),
// 		// 		cur_tmp_front_multiply.n(),
// 		// 		relu_input_view,
// 		// 		cur_tmp_front_multiply.view());
// 		// }

// 		// printf("****network config******\n");
// 		// for (uint32_t i = 0; i <= num_hidden_layers; i++ ){
// 		// 	printf("layer %d, weight m,n:%d,%d\n",i,full_weight_matrix_at(use_inference_params, i).m(),full_weight_matrix_at(use_inference_params, i).n());
// 		// 	// printf("layer %d, weight m,n:%d,%d\n",i,full_weight_matrix_at(use_inference_params, i).transposed().m(),full_weight_matrix_at(use_inference_params, i).transposed().n());
// 		// }
// 		// exit(0);
// 		for (uint32_t i = 1; i < num_hidden_layers + 1 ; ++i) {

// 			// cudaEventRecord(m_training_splitk_events.at(backward_tmp_idx), stream);
// 			// cudaStreamWaitEvent(m_training_splitk_streams.at(backward_tmp_idx), m_training_splitk_events.at(backward_tmp_idx), 0);
// 			// cudaEventRecord(m_training_splitk_events.at(backward_tmp_idx), m_training_splitk_streams.at(backward_tmp_idx));

// 			// relu
// 			auto& cur_tmp_front_multiply = tmp_front_multiply.at(i);
// 			// auto relu_input_view = forward.hidden.at(i+1).view();
// 			auto relu_input_view = forward.hidden.at(i-1).view();
// 			relu_input_view.advance_cols(target_batch);
// 			// printf("before m,n:%d,%d\n",tmp_front_multiply.at(i-1).m(),tmp_front_multiply.at(i-1).n());
// 			// printf("weight m,n:%d,%d\n",full_weight_matrix_at(use_inference_params, i-1).transposed().m(),full_weight_matrix_at(use_inference_params, i-1).transposed().n());
// 			// printf("now m,n:%d,%d\n",cur_tmp_front_multiply.m(),cur_tmp_front_multiply.n());
// 			// fc_multiply<FullLayer>(stream,
// 			// 	weight_matrix_at(use_inference_params, i-1).transposed(),
// 			// 	tmp_front_multiply.at(i-1),
// 			// 	cur_tmp_front_multiply);
// 			fc_multiply<FullLayer>(stream,
// 				tmp_front_multiply.at(i-1),
// 				// weight_matrix_at(use_inference_params, i-1).transposed(),
// 				full_weight_matrix_at(use_inference_params, i-1).transposed(),
// 				cur_tmp_front_multiply);
// 			// tcnn::linear_kernel(apply_relu_to_derivative_by_forward<T, true>, 0, stream,
// 			// 	cur_tmp_front_multiply.m() * cur_tmp_front_multiply.n(),
// 			// 	cur_tmp_front_multiply.m(),
// 			// 	cur_tmp_front_multiply.n(),
// 			// 	relu_input_view,
// 			// 	cur_tmp_front_multiply.view());
// 		}

// 		// exit(1);
// 		tmp_back_multiply[num_hidden_layers+2].initialize_diagonal(1); //单位矩阵
// 		// tmp_back_multiply[num_hidden_layers+1].memset(0); //单位矩阵 ## 内存访问出错

// 		// wym warning: activation should be operated on column. After that, just multiply by weight and a row ReLU.
// 		// printf("len:%d\n",forward.hidden.size());

// 		// for (uint32_t i = 0; i<forward.hidden.size();i++){
// 		// 	printf("forward %d, shape %d, %d\n",i, forward.hidden.at(i).m(), forward.hidden.at(i).n());
// 		// }
// 		for (uint32_t i = num_hidden_layers + 1; i > 1; --i) {
// 			// relu
// 			auto& cur_tmp_back_multiply = tmp_back_multiply.at(i);
// 			// auto relu_input_view = forward.hidden.at(i+1).view();
// 			auto relu_input_view = forward.hidden.at(i-2).view();
// 			relu_input_view.advance_cols(target_batch);
			
// 			// printf("before m,n:%d,%d\n",tmp_back_multiply.at(i+1).m(),tmp_back_multiply.at(i+1).n());
// 			// printf("weight m,n:%d,%d\n",full_weight_matrix_at(use_inference_params, i-1).transposed().m(),full_weight_matrix_at(use_inference_params, i-1).transposed().n());
// 			// printf("now m,n:%d,%d\n",cur_tmp_back_multiply.m(),cur_tmp_back_multiply.n());
// 			fc_multiply<FullLayer>(stream,
// 				// weight_matrix_at(use_inference_params, i+1).transposed(),
// 				tmp_back_multiply.at(i+1),
// 				full_weight_matrix_at(use_inference_params, i-1),
// 				cur_tmp_back_multiply);
// 			// tcnn::linear_kernel(apply_relu_to_derivative_by_forward<T, false>, 0, stream,
// 			// 	cur_tmp_back_multiply.m() * cur_tmp_back_multiply.n(),
// 			// 	cur_tmp_back_multiply.m(),
// 			// 	cur_tmp_back_multiply.n(),
// 			// 	relu_input_view,
// 			// 	cur_tmp_back_multiply.view());
			
// 			// fc_multiply<FullLayer>(stream, 
// 			// 	weight_matrix_at(use_inference_params, i+1).transposed(), 
// 			// 	tmp_back_multiply.at(i+1), 
// 			// 	forward.hidden.at(i+1), 
// 			// 	tmp_back_multiply.at(i), 
// 			// 	m_activation, true);
// 		}

	
// 		// printf("m_gradient_matrices size: %d\n",m_gradient_matrices.size());
// 		for (auto& tmp_front: tmp_front_multiply){
// 			tmp_front.memset(0);
// 		}
// 		for (auto& tmp_back: tmp_back_multiply){
// 			tmp_back.memset(0);
// 		}
// 		for (uint32_t i = 1; i < num_hidden_layers + 2; ++i) {
// 			// printf("before shape:%d,%d\n",tmp_front_multiply.at(i-1).m(),tmp_front_multiply.at(i-1).n());
// 			// printf("after shape:%d,%d\n",tmp_back_multiply.at(i+1).m(),tmp_back_multiply.at(i+1).n());
// 			auto& gradient_matrix = m_gradient_matrices.at(i-1);
// 			tcnn::linear_kernel(kernel_compute_update_weight<T>, 0, stream,
// 				gradient_matrix.m() * gradient_matrix.n(),
// 				gradient_matrix.m(),
// 				gradient_matrix.n(),
// 				tmp_front_multiply.at(i-1).view(),
// 				// tmp_back_multiply.at(i+1).view(),
// 				tmp_back_multiply.at(i+1).transposed().view(),
// 				gradient_matrix.view(),
// 				param_gradients_mode
// 				);
// 		}

// 	}

// 	if (param_gradients_mode != EGradientMode::Ignore) {
// 		// All the per-layer split-k matrix multiplications summing over
// 		// the batch are computed in parallel streams to the actual
// 		// backpropagation. Here, we need to wait for all of these to complete.
// 		for (auto& event : m_training_splitk_events) {
// 			cudaStreamWaitEvent(stream, event, 0);
// 		}
// 	}
// 	printf("finish backward_backward_input\n");
// }

template <typename T>
std::unique_ptr<typename CutlassMLP<T>::ForwardContext> CutlassMLP<T>::allocate_forward_buffers(cudaStream_t stream, uint32_t batch_size) {
	auto forward = std::make_unique<ForwardContext>();

	forward->hidden.resize(num_forward_activations());
	for (uint32_t i = 0; i < num_forward_activations(); ++i) {
		forward->hidden[i] = GPUMatrix<T>{m_network_width, batch_size, stream};
	}

	forward->hidden_input.resize(num_forward_activations());
	for (uint32_t i = 0; i < num_forward_activations(); ++i) {
		forward->hidden_input[i] = GPUMatrix<T>{m_network_width, batch_size, stream};
	}

	return forward;
}

template <typename T>
void CutlassMLP<T>::set_params(T* params, T* inference_params, T* backward_params, T* gradients) {
	size_t current_pos = 0;
	for (size_t i = 0; i < m_weight_matrices.size(); ++i) {
		m_weight_matrices[i].set_data_unsafe(params + current_pos);
		m_weight_matrices_inference[i].set_data_unsafe(inference_params + current_pos);
		m_gradient_matrices[i].set_data_unsafe(gradients + current_pos);
		current_pos += m_weight_matrices[i].n_elements();
	}
}

template <typename T>
void CutlassMLP<T>::initialize_params(pcg32& rnd, float* params_full_precision, T* params, T* inference_params, T* backward_params, T* gradients, float scale) {
	set_params(params, inference_params, backward_params, gradients);

	size_t current_pos = 0;
	for (size_t i = 0; i < m_weight_matrices_full_precision.size(); ++i) {
		m_weight_matrices_full_precision[i].set_data_unsafe(params_full_precision + current_pos);
		current_pos += m_weight_matrices_full_precision[i].n_elements();

		if (m_activation == Activation::Sine) {
			if (i == 0) {
				m_weight_matrices_full_precision[i].initialize_siren_uniform_first(rnd, scale);
			} else {
				m_weight_matrices_full_precision[i].initialize_siren_uniform(rnd, scale);
			}
		} else {
			m_weight_matrices_full_precision[i].initialize_xavier_uniform(rnd, scale);
		}
	}
}

// Explicitly instantiate CutlassMLP classes.
template class CutlassMLP<network_precision_t>;

TCNN_NAMESPACE_END
