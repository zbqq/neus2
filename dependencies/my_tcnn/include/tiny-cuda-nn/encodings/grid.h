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

/** @file   grid.h
 *  @author Thomas Müller, NVIDIA & Alex Evans, NVIDIA & Jianfei Guo, Shanghai AI Lab
 *  @brief  Trainable hierarchy of N-D grids of floating point values.
 *          The grids can be backed by dense memory, tiled memory, or by hash tables.
 */

#pragma once

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/common_device.h>
#include <tiny-cuda-nn/encoding.h>
#include <tiny-cuda-nn/encodings/oneblob.h>
#include <tiny-cuda-nn/gpu_memory.h>
#include <tiny-cuda-nn/multi_stream.h>
#include <tiny-cuda-nn/random.h>

#include <stdexcept>
#include <stdint.h>
#include <string>
#include <vector>


TCNN_NAMESPACE_BEGIN

#define RESAMPLE_KERNEL_DEBUG 0
#define RESMAPLE_CHECKED_BY_DENSITY_GRID 0
#define TV_LOSS_DEBUG 0
#define progressive_freq_debug 1

enum class GridType {
	Hash,
	Dense,
	Tiled,
};

template <typename T>
__global__ void debug_log_full(
	const uint32_t  n_elements,
	const uint32_t  col_num,
	tcnn::MatrixView<T> output
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;
	const uint32_t row = i / col_num;
	const uint32_t col = i - row * col_num;
	printf("%d,%d: %.06f\n",row,col,(float)output(row,col));
}

template <typename T, typename TIn=T>
__global__ void set_grid_value(
	const uint32_t n_elements,
	TIn* input,
	T* output
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	output[i] = (T)input[i];
}

// template <typename T>
// __global__ void debug_log(
// 	const uint32_t  n_elements,
// 	const uint32_t  row_num,
// 	tcnn::MatrixView<T> output
// ) {
// 	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
// 	if (i >= n_elements) return;
// 	printf("in gird.h %d: %.10f\n",i, (float)output(i,row_num));
// }

inline GridType string_to_grid_type(const std::string& grid_type) {
	if (equals_case_insensitive(grid_type, "Hash")) {
		return GridType::Hash;
	} else if (equals_case_insensitive(grid_type, "Dense")) {
		return GridType::Dense;
	} else if (equals_case_insensitive(grid_type, "Tiled") || equals_case_insensitive(grid_type, "Tile")) {
		return GridType::Tiled;
	}

	throw std::runtime_error{std::string{"Invalid grid type: "} + grid_type};
}

inline std::string to_string(GridType grid_type) {
	switch (grid_type) {
		case GridType::Hash: return "Hash";
		case GridType::Dense: return "Dense";
		case GridType::Tiled: return "Tiled";
		default: throw std::runtime_error{std::string{"Invalid grid type"}};
	}
}

template <uint32_t N_DIMS>
__device__ uint32_t fast_hash(const uint32_t pos_grid[N_DIMS]) {
	static_assert(N_DIMS <= 7, "fast_hash can only hash up to 7 dimensions.");

	// While 1 is technically not a good prime for hashing (or a prime at all), it helps memory coherence
	// and is sufficient for our use case of obtaining a uniformly colliding index from high-dimensional
	// coordinates.
	constexpr uint32_t primes[7] = {1u, 2654435761u, 805459861u, 3674653429u, 2097192037u, 1434869437u, 2165219737u};

	uint32_t result = 0;
	#pragma unroll
	for (uint32_t i = 0; i < N_DIMS; ++i) {
		result ^= pos_grid[i] * primes[i];
	}

	return result;
}

template <uint32_t N_DIMS, uint32_t N_FEATURES_PER_LEVEL>
__device__ uint32_t grid_index(const GridType grid_type, const uint32_t feature, const uint32_t hashmap_size, const uint32_t grid_resolution, const uint32_t pos_grid[N_DIMS]) {
	uint32_t stride = 1;
	uint32_t index = 0;

	// The second part of the loop condition is needed to avoid integer overflows in finer levels.
	#pragma unroll
	for (uint32_t dim = 0; dim < N_DIMS && stride <= hashmap_size; ++dim) {
		index += pos_grid[dim] * stride;
		stride *= grid_resolution;
	}

	if (grid_type == GridType::Hash && hashmap_size < stride) {
		index = fast_hash<N_DIMS>(pos_grid);
	}

	return (index % hashmap_size) * N_FEATURES_PER_LEVEL + feature;
}

__device__ inline float random_val(uint32_t seed, uint64_t idx) {
	pcg32 rng(((uint64_t)seed << 32) | (uint64_t)idx);
	return rng.next_float();
}

template <typename T>
__global__ void extract_position(
	const uint32_t num_elements,
	PitchedPtr<const float> data_in,
	T* __restrict__ output
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= num_elements) return;

	const uint32_t dim_idx = threadIdx.y;

	output[i + dim_idx * num_elements] = (T)data_in(i)[dim_idx];
}

template <typename T, uint32_t N_POS_DIMS, uint32_t N_FEATURES_PER_LEVEL>
__global__ void kernel_grid(
	const uint32_t num_elements,
	const uint32_t num_grid_features, // m_n_features: 16
	const uint32_t* hashmap_offset_table,
	const uint32_t* resolution_table,
	const float* scale_table,
	const uint32_t valid_level,
	// const uint32_t base_resolution,
	// const float log2_per_level_scale,
	const float quantize_threshold,
	float max_level,
	const float* __restrict__ max_level_gpu,
	const InterpolationType interpolation_type,
	const GridType grid_type,
	const T* __restrict__ grid,
	MatrixView<const float> positions_in,
	T* __restrict__ encoded_positions,
	float* __restrict__ dy_dx
) {
	const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= num_elements) return;

	const uint32_t level = blockIdx.y; // <- the level is the same for all threads
	if (level > valid_level) {
		if (encoded_positions) {
			#pragma unroll
			for (uint32_t f = 0; f < N_FEATURES_PER_LEVEL; ++f) {
				encoded_positions[i + (level * N_FEATURES_PER_LEVEL + f) * num_elements] = (T)0.0f;
			}
		}

		// Gradient is zero for zeroed-out dimensions.
		if (dy_dx) {
			#pragma unroll
			for (uint32_t f = 0; f < N_FEATURES_PER_LEVEL; ++f) {
				((vector_fullp_t<N_POS_DIMS>*)dy_dx)[i + (level * N_FEATURES_PER_LEVEL + f) * num_elements] = {0};
			}
		}

		return;
	}

	if (max_level_gpu) {
		max_level = (max_level_gpu[i] * num_grid_features) / N_FEATURES_PER_LEVEL;
	} else {
		max_level = (max_level * num_grid_features) / N_FEATURES_PER_LEVEL;
	}

	if (level >= max_level + 1e-3f) { // don't look up
		if (encoded_positions) {
			#pragma unroll
			for (uint32_t f = 0; f < N_FEATURES_PER_LEVEL; ++f) {
				encoded_positions[i + (level * N_FEATURES_PER_LEVEL + f) * num_elements] = (T)0.0f;
			}
		}

		// Gradient is zero for zeroed-out dimensions.
		if (dy_dx) {
			#pragma unroll
			for (uint32_t f = 0; f < N_FEATURES_PER_LEVEL; ++f) {
				((vector_fullp_t<N_POS_DIMS>*)dy_dx)[i + (level * N_FEATURES_PER_LEVEL + f) * num_elements] = {0};
			}
		}

		return;
	}

	grid += hashmap_offset_table[level] * N_FEATURES_PER_LEVEL;
	const uint32_t hashmap_size = hashmap_offset_table[level + 1] - hashmap_offset_table[level];

	// const float scale = exp2f(level * log2_per_level_scale) * base_resolution - 1.0f;
	// const uint32_t grid_resolution = ((uint32_t)ceil(scale) + 1);
	const float scale = scale_table[level];
	const uint32_t grid_resolution = resolution_table[level];

	float pos[N_POS_DIMS];
	float pos_derivative[N_POS_DIMS];
	uint32_t pos_grid[N_POS_DIMS];

	if (interpolation_type == InterpolationType::Nearest || interpolation_type == InterpolationType::Linear) {
		#pragma unroll
		for (uint32_t dim = 0; dim < N_POS_DIMS; ++dim) {
			pos_fract(positions_in(dim, i), &pos[dim], &pos_derivative[dim], &pos_grid[dim], scale, identity_fun, identity_derivative);
		}
	} else {
		#pragma unroll
		for (uint32_t dim = 0; dim < N_POS_DIMS; ++dim) {
			pos_fract(positions_in(dim, i), &pos[dim], &pos_derivative[dim], &pos_grid[dim], scale, smoothstep, smoothstep_derivative);
		}
	}

	auto grid_val = [&](const uint32_t local_pos[N_POS_DIMS]) {
		uint32_t index = grid_index<N_POS_DIMS, N_FEATURES_PER_LEVEL>(grid_type, 0, hashmap_size, grid_resolution, local_pos);
		return *(vector_t<T, N_FEATURES_PER_LEVEL>*)&grid[index];
	};

	if (interpolation_type == InterpolationType::Nearest) {
		auto result = grid_val(pos_grid);

		if (encoded_positions) {
			#pragma unroll
			for (uint32_t f = 0; f < N_FEATURES_PER_LEVEL; ++f) {
				encoded_positions[i + (level * N_FEATURES_PER_LEVEL + f) * num_elements] = result[f];
			}
		}

		// Gradient is zero when there's no interpolation.
		if (dy_dx) {
			#pragma unroll
			for (uint32_t f = 0; f < N_FEATURES_PER_LEVEL; ++f) {
				((vector_fullp_t<N_POS_DIMS>*)dy_dx)[i + (level * N_FEATURES_PER_LEVEL + f) * num_elements] = {0};
			}
		}

		return;
	}

	if (encoded_positions) {
		// N-linear interpolation
		vector_t<T, N_FEATURES_PER_LEVEL> result = {};

		#pragma unroll
		for (uint32_t idx = 0; idx < (1 << N_POS_DIMS); ++idx) {
			float weight = 1;
			uint32_t pos_grid_local[N_POS_DIMS];

			#pragma unroll
			for (uint32_t dim = 0; dim < N_POS_DIMS; ++dim) {
				if ((idx & (1<<dim)) == 0) {
					weight *= 1 - pos[dim];
					pos_grid_local[dim] = pos_grid[dim];
				} else {
					weight *= pos[dim];
					pos_grid_local[dim] = pos_grid[dim] + 1;
				}
			}

			auto val = grid_val(pos_grid_local);

			#pragma unroll
			for (uint32_t feature = 0; feature < N_FEATURES_PER_LEVEL; ++feature) {
				float data = (float)((T*)&val)[feature];
				if (fabsf(data) < quantize_threshold) data = 0.f;
				((T*)&result)[feature] += (T)(weight * data);
			}
		}

		#pragma unroll
		for (uint32_t f = 0; f < N_FEATURES_PER_LEVEL; ++f) {
			encoded_positions[i + (level * N_FEATURES_PER_LEVEL + f) * num_elements] = result[f];
		}
	}

	// Gradient
	if (dy_dx) {
		vector_fullp_t<N_POS_DIMS> grads[N_FEATURES_PER_LEVEL] = {};

		#pragma unroll
		for (uint32_t grad_dim = 0; grad_dim < N_POS_DIMS; ++grad_dim) {
			#pragma unroll
			for (uint32_t idx = 0; idx < (1 << (N_POS_DIMS-1)); ++idx) {
				float weight = scale;
				uint32_t pos_grid_local[N_POS_DIMS];

				#pragma unroll
				for (uint32_t non_grad_dim = 0; non_grad_dim < N_POS_DIMS-1; ++non_grad_dim) {
					const uint32_t dim = non_grad_dim >= grad_dim ? (non_grad_dim+1) : non_grad_dim;

					if ((idx & (1<<non_grad_dim)) == 0) {
						weight *= 1 - pos[dim];
						pos_grid_local[dim] = pos_grid[dim];
					} else {
						weight *= pos[dim];
						pos_grid_local[dim] = pos_grid[dim] + 1;
					}
				}

				pos_grid_local[grad_dim] = pos_grid[grad_dim];
				auto val_left = grid_val(pos_grid_local);
				pos_grid_local[grad_dim] = pos_grid[grad_dim] + 1;
				auto val_right = grid_val(pos_grid_local);

				#pragma unroll
				for (uint32_t feature = 0; feature < N_FEATURES_PER_LEVEL; ++feature) {
					grads[feature][grad_dim] += weight * ((float)val_right[feature] - (float)val_left[feature]) * pos_derivative[grad_dim];
				}
			}
		}

		#pragma unroll
		for (uint32_t f = 0; f < N_FEATURES_PER_LEVEL; ++f) {
			((vector_fullp_t<N_POS_DIMS>*)dy_dx)[i + (level * N_FEATURES_PER_LEVEL + f) * num_elements] = grads[f];
		}
	}
}

template <typename T, typename GRAD_T, uint32_t N_POS_DIMS, uint32_t N_FEATURES_PER_LEVEL, uint32_t N_FEATURES_PER_THREAD>
__global__ void kernel_grid_backward(
	const uint32_t num_elements,
	const uint32_t num_grid_features,
	const uint32_t* hashmap_offset_table,
	const uint32_t* resolution_table,
	const float* scale_table,
	const uint32_t valid_level,
	// const uint32_t base_resolution,
	// const float log2_per_level_scale,
	float max_level,
	const float* __restrict__ max_level_gpu,
	const bool stochastic_interpolation,
	const InterpolationType interpolation_type,
	const GridType grid_type,
	GRAD_T* __restrict__ grid_gradient,
	MatrixView<const float> positions_in,
	const T* __restrict__ dL_dy
) {
	const uint32_t i = ((blockIdx.x * blockDim.x + threadIdx.x) * N_FEATURES_PER_THREAD) / N_FEATURES_PER_LEVEL;
	if (i >= num_elements) return;

	const uint32_t level = blockIdx.y ; // <- the level is the same for all threads.
	if (level > valid_level) return;
	const uint32_t feature = (blockIdx.x * blockDim.x + threadIdx.x) * N_FEATURES_PER_THREAD - i * N_FEATURES_PER_LEVEL;

	if (max_level_gpu) {
		max_level = (max_level_gpu[i] * num_grid_features) / N_FEATURES_PER_LEVEL;
	} else {
		max_level = (max_level * num_grid_features) / N_FEATURES_PER_LEVEL;
	}

	if (level > max_level + 1e-3f) {
		return;
	}

	grid_gradient += hashmap_offset_table[level] * N_FEATURES_PER_LEVEL;
	const uint32_t hashmap_size = hashmap_offset_table[level + 1] - hashmap_offset_table[level];

	// const float scale = exp2f(level * log2_per_level_scale) * base_resolution - 1.0f;
	// const uint32_t grid_resolution = ((uint32_t)ceil(scale) + 1);
	const float scale = scale_table[level];
	const uint32_t grid_resolution = resolution_table[level];

	auto add_grid_gradient = [&](const uint32_t local_pos[N_POS_DIMS], const vector_t<T, N_FEATURES_PER_THREAD>& grad, const float weight) {
		uint32_t index = grid_index<N_POS_DIMS, N_FEATURES_PER_LEVEL>(grid_type, feature, hashmap_size, grid_resolution, local_pos);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600 // atomicAdd(__half2) is only supported with compute capability 60 and above
		if (N_FEATURES_PER_THREAD > 1 && std::is_same<GRAD_T, __half>::value) {
			for (uint32_t f = 0; f < N_FEATURES_PER_THREAD; f += 2) {
				__half2 v = {(__half)((float)grad[f] * weight), (__half)((float)grad[f+1] * weight)};
				atomicAdd((__half2*)&grid_gradient[index + f], v);
			}
		} else
#endif
		{
			if (std::is_same<GRAD_T, __half>::value) {
				// Should never happen
				//printf("Attempted to use atomicAdd(__half)\n")
			} else {
				for (uint32_t f = 0; f < N_FEATURES_PER_THREAD; ++f) {
					atomicAdd((float*)&grid_gradient[index + f], (float)grad[f] * weight);
				}
			}
		}
	};

	float pos[N_POS_DIMS];
	uint32_t pos_grid[N_POS_DIMS];

	if (interpolation_type == InterpolationType::Nearest || interpolation_type == InterpolationType::Linear) {
		#pragma unroll
		for (uint32_t dim = 0; dim < N_POS_DIMS; ++dim) {
			pos_fract(positions_in(dim, i), &pos[dim], &pos_grid[dim], scale, identity_fun);
		}
	} else {
		#pragma unroll
		for (uint32_t dim = 0; dim < N_POS_DIMS; ++dim) {
			pos_fract(positions_in(dim, i), &pos[dim], &pos_grid[dim], scale, smoothstep);
		}
	}

	vector_t<T, N_FEATURES_PER_THREAD> grad;

	#pragma unroll
	for (uint32_t f = 0; f < N_FEATURES_PER_THREAD; ++f) {
		grad[f] = dL_dy[i + (level * N_FEATURES_PER_LEVEL + feature + f) * num_elements];
	}

	if (interpolation_type == InterpolationType::Nearest) {
		add_grid_gradient(pos_grid, grad, 1.0f);
		return;
	}

	if (stochastic_interpolation) {
		float sample = random_val(1337, i + level * num_elements);
		uint32_t pos_grid_local[N_POS_DIMS];

		#pragma unroll
		for (uint32_t dim = 0; dim < N_POS_DIMS; ++dim) {
			if (sample >= pos[dim]) {
				pos_grid_local[dim] = pos_grid[dim];
			} else {
				pos_grid_local[dim] = pos_grid[dim] + 1;
			}
		}

		add_grid_gradient(pos_grid_local, grad, 1.0f);
		return;
	}

	// N-linear interpolation
	#pragma unroll
	for (uint32_t idx = 0; idx < (1 << N_POS_DIMS); ++idx) {
		float weight = 1;
		uint32_t pos_grid_local[N_POS_DIMS];

		#pragma unroll
		for (uint32_t dim = 0; dim < N_POS_DIMS; ++dim) {
			if ((idx & (1<<dim)) == 0) {
				weight *= 1 - pos[dim];
				pos_grid_local[dim] = pos_grid[dim];
			} else {
				weight *= pos[dim];
				pos_grid_local[dim] = pos_grid[dim] + 1;
			}
		}

		add_grid_gradient(pos_grid_local, grad, weight);
	}
}

inline __device__ uint32_t tv_cascaded_grid_idx_at(float* pos, uint32_t mip, uint32_t NERF_GRIDSIZE) {
	int i[3];
	float mip_scale = scalbnf(1.0f, -mip);
	for (uint32_t ii = 0; ii < 3; ii++){
		pos[ii] -= 0.5f;
		pos[ii] *= mip_scale;
		pos[ii] += 0.5f;
		i[ii] = (int)(pos[ii] * NERF_GRIDSIZE);
	}



	if (i[0] < -1 || i[0] > NERF_GRIDSIZE || i[1] < -1 || i[1] > NERF_GRIDSIZE || i[2] < -1 || i[2] > NERF_GRIDSIZE) {
		printf("WTF %d %d %d\n", i[0], i[1], i[2]);
	}

	uint32_t idx = morton3D(
		clamp(i[0], 0, (int)NERF_GRIDSIZE-1),
		clamp(i[1], 0, (int)NERF_GRIDSIZE-1),
		clamp(i[2], 0, (int)NERF_GRIDSIZE-1)
	);

	return idx;
}

inline __host__ __device__ uint32_t tv_grid_mip_offset(uint32_t mip, uint32_t NERF_GRIDSIZE) {
	return (NERF_GRIDSIZE * NERF_GRIDSIZE * NERF_GRIDSIZE) * mip;
}

inline __device__ bool tv_density_grid_occupied_at(float* pos, const uint8_t* density_grid_bitfield, uint32_t mip, uint32_t NERF_GRIDSIZE) {
	uint32_t idx = tv_cascaded_grid_idx_at(pos, mip, NERF_GRIDSIZE);
	return density_grid_bitfield[idx/8+tv_grid_mip_offset(mip, NERF_GRIDSIZE)/8] & (1<<(idx%8));
}

#if TV_LOSS_DEBUG
template <typename T, typename GRAD_T, uint32_t N_POS_DIMS, uint32_t N_FEATURES_PER_LEVEL>
__global__ void kernel_tv_debug(
	const uint64_t num_elements,
	const uint32_t level,
	const float return_ratio,
	const uint32_t hashmap_offset,
	const uint32_t hashmap_size,
	const float scale,
	const uint32_t grid_resolution,
	const float loss_scale,
	uint8_t *density_grid,
	uint32_t max_cascade,
	uint32_t nerf_gridsize,
	const GridType grid_type,
	const T* __restrict__ grid,
	GRAD_T* __restrict__ grid_gradient,
	uint32_t* count
) {
	const uint64_t i = (uint64_t)blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
	if (i >= num_elements) return;
	float sample = random_val(1337, i);
	if (sample < return_ratio) return;

	float pos[N_POS_DIMS];
	uint32_t pos_grid[N_POS_DIMS];

	// vector_fullp_t<N_POS_DIMS> cur_pos;
	float cur_pos[N_POS_DIMS];
	uint32_t cur_pos_grid[N_POS_DIMS];
	uint64_t cur_pos_idx = i;

	#pragma unroll
	for (uint32_t dim = 0; dim < N_POS_DIMS; ++dim) {
		cur_pos_grid[dim] = cur_pos_idx % grid_resolution;
		cur_pos_idx /= grid_resolution;
		// cur_pos[dim] = (float)cur_pos_grid[dim] / (float)(grid_resolution - 1);
		cur_pos[dim] = (float)cur_pos_grid[dim] / (float)(grid_resolution - 1);
	}

	if(!tv_density_grid_occupied_at(cur_pos, density_grid, max_cascade, nerf_gridsize)){
	// if(!tv_density_grid_occupied_at(cur_pos, density_grid, 0, nerf_gridsize)){
		return;
	}

	atomicAdd(count, 1u);
}
#endif


template <typename T, typename GRAD_T, uint32_t N_POS_DIMS, uint32_t N_FEATURES_PER_LEVEL>
__global__ void kernel_tv_backward(
	const uint64_t num_elements,
	const uint32_t level,
	// const float return_ratio,
	const bool random_sample,
	const uint32_t random_seed,
	// default_rng_t rng,
	// const uint32_t num_grid_features,
	// const uint32_t* hashmap_offset_table,
	const uint32_t hashmap_offset,
	const uint32_t hashmap_size,
	// const uint32_t base_resolution,
	// const float log2_per_level_scale,
	const float scale,
	const uint32_t grid_resolution,
	const float loss_scale,
	uint8_t *density_grid,
	uint32_t max_cascade,
	uint32_t nerf_gridsize,
	const GridType grid_type,
	const T* __restrict__ grid,
	GRAD_T* __restrict__ grid_gradient
) {
	const uint64_t i = (uint64_t)blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
	if (i >= num_elements) return;
	// float sample = random_val(1337, i);
	// if (sample < return_ratio) return;


	// grid += hashmap_offset_table[level] * N_FEATURES_PER_LEVEL;
	// grid_gradient += hashmap_offset_table[level] * N_FEATURES_PER_LEVEL;

	// const uint32_t hashmap_size = hashmap_offset_table[level + 1] - hashmap_offset_table[level];

	// const float scale = exp2f(level * log2_per_level_scale) * base_resolution - 1.0f;
	// const uint32_t grid_resolution = ((uint32_t)ceil(scale) + 1);

	float pos[N_POS_DIMS];
	uint32_t pos_grid[N_POS_DIMS];

	// vector_fullp_t<N_POS_DIMS> cur_pos;
	float cur_pos[N_POS_DIMS];
	uint32_t cur_pos_grid[N_POS_DIMS];

	if (random_sample) {
		// rng.advance(i*N_POS_DIMS);
		#pragma unroll
		for (uint32_t dim = 0; dim < N_POS_DIMS; ++dim) {
			cur_pos_grid[dim] = (uint32_t)floor((float)grid_resolution * random_val(random_seed, i*N_POS_DIMS+dim));
			// cur_pos_grid[dim] = (uint32_t)floor((float)grid_resolution * rng.next_float());
			cur_pos[dim] = (float)cur_pos_grid[dim] / (float)(grid_resolution - 1);
		}
	}
	else {
		uint64_t cur_pos_idx = i;
		#pragma unroll
		for (uint32_t dim = 0; dim < N_POS_DIMS; ++dim) {
			cur_pos_grid[dim] = cur_pos_idx % grid_resolution;
			cur_pos_idx /= grid_resolution;
			// cur_pos[dim] = (float)cur_pos_grid[dim] / (float)(grid_resolution - 1);
			cur_pos[dim] = (float)cur_pos_grid[dim] / (float)(grid_resolution - 1);
		}
	}
	

	if(!tv_density_grid_occupied_at(cur_pos, density_grid, max_cascade, nerf_gridsize)){
	// if(!tv_density_grid_occupied_at(cur_pos, density_grid, 0, nerf_gridsize)){
		return;
	}


	grid += hashmap_offset * N_FEATURES_PER_LEVEL;
	grid_gradient += hashmap_offset * N_FEATURES_PER_LEVEL;

	auto grid_val = [&](const uint32_t local_pos[N_POS_DIMS]) {
		uint32_t index = grid_index<N_POS_DIMS, N_FEATURES_PER_LEVEL>(grid_type, 0, hashmap_size, grid_resolution, local_pos);
		return *(vector_t<T, N_FEATURES_PER_LEVEL>*)&grid[index];
	};

	auto add_grid_gradient = [&](const uint32_t local_pos[N_POS_DIMS], const vector_t<T, N_FEATURES_PER_LEVEL>& grad, const float weight) {
		uint32_t index = grid_index<N_POS_DIMS, N_FEATURES_PER_LEVEL>(grid_type, 0, hashmap_size, grid_resolution, local_pos);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600 // atomicAdd(__half2) is only supported with compute capability 60 and above
		if (N_FEATURES_PER_LEVEL > 1 && std::is_same<GRAD_T, __half>::value) {
			for (uint32_t f = 0; f < N_FEATURES_PER_LEVEL; f += 2) {
				__half2 v = {(__half)((float)grad[f] * weight), (__half)((float)grad[f+1] * weight)};
				atomicAdd((__half2*)&grid_gradient[index + f], v);
			}
		} else
#endif
		{
			if (std::is_same<GRAD_T, __half>::value) {
				// Should never happen
				//printf("Attempted to use atomicAdd(__half)\n")
			} else {
				for (uint32_t f = 0; f < N_FEATURES_PER_LEVEL; ++f) {
					atomicAdd((float*)&grid_gradient[index + f], (float)grad[f] * weight);
				}
			}
		}
	};

	auto v000_grid_pos = cur_pos_grid;
	// auto v100_grid_pos = ;
	// auto v010_grid_pos = ;
	// auto v001_grid_pos = ;
	// if (N_POS_DIMS != 3){
	// 	printf("Teach me coding!!\n");
	// 	exit(1);
	// }

	uint32_t v100_grid_pos[N_POS_DIMS];
	for(uint32_t i = 0; i < N_POS_DIMS; i++){
		if (i == 0){
			v100_grid_pos[i] = v000_grid_pos[i] + ((v000_grid_pos[i] + 1) == grid_resolution) ? 0 : 1;
		}
		else{
			v100_grid_pos[i] = v000_grid_pos[i];
		}
	}

	uint32_t v010_grid_pos[N_POS_DIMS];
	for(uint32_t i = 0; i < N_POS_DIMS; i++){
		if (i == 1){
			v010_grid_pos[i] = v000_grid_pos[i] + ((v000_grid_pos[i] + 1) == grid_resolution) ? 0 : 1;
		}
		else{
			v010_grid_pos[i] = v000_grid_pos[i];
		}
	}

	uint32_t v001_grid_pos[N_POS_DIMS];
	for(uint32_t i = 0; i < N_POS_DIMS; i++){
		if (i == 2){
			v001_grid_pos[i] = v000_grid_pos[i] + ((v000_grid_pos[i] + 1) == grid_resolution) ? 0 : 1;
		}
		else{
			v001_grid_pos[i] = v000_grid_pos[i];
		}
	}


	auto v000 = grid_val(v000_grid_pos);
	auto v100 = grid_val(v100_grid_pos);
	auto v010 = grid_val(v010_grid_pos);
	auto v001 = grid_val(v001_grid_pos);

	vector_t<T, N_FEATURES_PER_LEVEL> gptr000;
	vector_t<T, N_FEATURES_PER_LEVEL> gptr100;
	vector_t<T, N_FEATURES_PER_LEVEL> gptr010;
	vector_t<T, N_FEATURES_PER_LEVEL> gptr001;

	// CUDA_CHECK_THROW(cudaMemsetAsync(gptr000.data(), 0, N_FEATURES_PER_LEVEL * sizeof(T), stream));
	// CUDA_CHECK_THROW(cudaMemsetAsync(gptr100.data(), 0, N_FEATURES_PER_LEVEL * sizeof(T), stream));
	// CUDA_CHECK_THROW(cudaMemsetAsync(gptr010.data(), 0, N_FEATURES_PER_LEVEL * sizeof(T), stream));
	// CUDA_CHECK_THROW(cudaMemsetAsync(gptr001.data(), 0, N_FEATURES_PER_LEVEL * sizeof(T), stream));
	memset(gptr000.data, 0, N_FEATURES_PER_LEVEL * sizeof(T));
	memset(gptr100.data, 0, N_FEATURES_PER_LEVEL * sizeof(T));
	memset(gptr010.data, 0, N_FEATURES_PER_LEVEL * sizeof(T));
	memset(gptr001.data, 0, N_FEATURES_PER_LEVEL * sizeof(T));

	#pragma unroll
	for (uint32_t feature = 0; feature < N_FEATURES_PER_LEVEL; ++feature) {
		float v000_data = (float)((T*)&v000)[feature];
		float v100_data = (float)((T*)&v100)[feature];
		float v010_data = (float)((T*)&v010)[feature];
		float v001_data = (float)((T*)&v001)[feature];

		float dx = (v100_data - v000_data);
		float dy = (v010_data - v000_data);
		float dz = (v001_data - v000_data);
		const float idelta = loss_scale * rsqrtf(1e-9f + dx * dx + dy * dy + dz * dz);

		// dx scaling? //

		if (dx != 0.f) atomicAdd(&gptr100[feature], (T)(dx * idelta)); // remove atomicAdd ?
		if (dy != 0.f) atomicAdd(&gptr010[feature], (T)(dy * idelta));
		if (dz != 0.f) atomicAdd(&gptr001[feature], (T)(dz * idelta));
		atomicAdd(&gptr000[feature], -(dx + dy + dz) * idelta);
	}

	add_grid_gradient(v000_grid_pos, gptr000, 1.0f);
	add_grid_gradient(v100_grid_pos, gptr100, 1.0f);
	add_grid_gradient(v010_grid_pos, gptr010, 1.0f);
	add_grid_gradient(v001_grid_pos, gptr001, 1.0f);
}

template <typename T>
__global__ void transpose_encoded_position(
	const uint32_t n_elements,
	const T* __restrict__ encoded_positions,
	PitchedPtr<T> output
) {
	const uint32_t i = threadIdx.y + blockIdx.x * blockDim.y;
	if (i >= n_elements) return;

	const uint32_t elem_idx = i;
	const uint32_t dim_idx = threadIdx.x;

	output(elem_idx)[dim_idx] = encoded_positions[elem_idx + n_elements * dim_idx];
}

template <typename T>
__global__ void transpose_gradients(
	const uint32_t n_elements,
	T* __restrict__ transposed_dL_dy,
	PitchedPtr<const T> dL_dy
) {
	const uint32_t i = threadIdx.y + blockIdx.x * blockDim.y;
	if (i >= n_elements) return;

	const uint32_t elem_idx = i;
	const uint32_t dim_idx = threadIdx.x;

	transposed_dL_dy[elem_idx + n_elements * dim_idx] = dL_dy(elem_idx)[dim_idx];
}

template <typename T, uint32_t N_POS_DIMS>
__global__ void kernel_grid_backward_input(
	const uint32_t num_elements,
	const uint32_t num_grid_features,
	const T* dL_dy_rm,
	const float* __restrict__ dy_dx,
	MatrixView<float> dL_dx
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= num_elements) return;

	vector_fullp_t<N_POS_DIMS> result = {0};

	for (int k = 0; k < num_grid_features; ++k) {
		float dL_dy_local = (float)dL_dy_rm[i + k * num_elements];
		auto dy_dx_local = ((vector_fullp_t<N_POS_DIMS>*)dy_dx)[i + k * num_elements];

		#pragma unroll
		for (uint32_t dim = 0; dim < N_POS_DIMS; ++dim) {
			result[dim] += dL_dy_local * dy_dx_local[dim];
		}
	}

	#pragma unroll
	for (uint32_t dim = 0; dim < N_POS_DIMS; ++dim) {
		dL_dx(dim, i) = result[dim];
	}
}

template <typename T, uint32_t N_POS_DIMS>
__global__ void kernel_cal_dy(
	const uint32_t num_elements,
	const uint32_t batch_size,
	const uint32_t num_grid_features,
	const float* dL_dx_rm,
	const float* __restrict__ dy_dx,
	// MatrixView<float> dy_dx,
	MatrixView<T> dy
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= num_elements) return;


	for (int dim = 0; dim < N_POS_DIMS; ++dim) {
		float dL_dx_local = (float)dL_dx_rm[i + dim * batch_size];
		// auto dy_dx_local = ((vector_fullp_t<num_grid_features>*)dy_dx)[i + k * num_elements];

		// #pragma unroll
		for (uint32_t k = 0; k < num_grid_features; ++k) {
			// result[dim] += dL_dx_local * dy_dx_local[dim];
			// auto dy_dx_local = dy_dx(k * N_POS_DIMS + dim, i);
			auto dy_dx_local = dy_dx[k * N_POS_DIMS * batch_size + i * N_POS_DIMS + dim];
			// auto dy_dx_local = ((vector_fullp_t<N_POS_DIMS>*)dy_dx)[i + k * num_elements];
			// result[k] += dL_dx_local * dy_dx_local;
			dy(k, i) += dL_dx_local * dy_dx_local;
			// result[k] += dL_dx_local * dy_dx_local[dim];
			// printf("dim:%d,feature_dim:%d, dl_dx_local:%f, dy_dx:%f\n",dim, k,  dL_dx_local,(float)dy_dx_local);
			// result[dim] += dL_dx_local * dy_dx[(i + dim*num_elements) * N_POS_DIMS + k];
		}
	}

	// #pragma unroll
	// for (uint32_t k = 0; k < num_grid_features; ++k) {
		// dy(k, i) = result[k];
	// }

	// naive no accelerate

	// for (uint32_t dim = 0; dim < num_grid_features; ++dim) {
	// 	for (int k = 0; k < N_POS_DIMS; ++k) {
	// 		float dL_dx_local = (float)dL_dx_rm[i + k * num_elements];
	// 		// printf("dl_dx_local:%f, dy_dx:%f\n",dL_dx_local,(float)dy_dx[(i + dim*num_elements) * N_POS_DIMS + k]);
	// 		dy(dim, i) += (T)(dL_dx_local * dy_dx[(i + dim*num_elements) * N_POS_DIMS + k]);
	// 	}
	// }	
}

template <typename T, typename GRAD_T, uint32_t N_POS_DIMS, uint32_t N_FEATURES_PER_LEVEL, uint32_t N_FEATURES_PER_THREAD>
__global__ void kernel_grid_backward_input_backward_grid(
	const uint32_t num_elements,
	const uint32_t num_grid_features,
	const uint32_t* hashmap_offset_table,
	const uint32_t* resolution_table,
	const float* scale_table,
	const uint32_t valid_level,
	// const uint32_t base_resolution,
	// const float log2_per_level_scale,
	float max_level,
	const float* __restrict__ max_level_gpu,
	// const bool stochastic_interpolation, // TODO: is this needed?
	const InterpolationType interpolation_type,
	const GridType grid_type,
	// inputs
	MatrixView<const float> dL_ddLdx,
	MatrixView<const float> positions_in,
	const T* __restrict__ dL_dy,
	// outputsk
	GRAD_T* __restrict__ grid_gradient
) {
	const uint32_t i = ((blockIdx.x * blockDim.x + threadIdx.x) * N_FEATURES_PER_THREAD) / N_FEATURES_PER_LEVEL;
	if (i >= num_elements) return;

	const uint32_t level = blockIdx.y ; // <- the level is the same for all threads.
	if (level > valid_level) return;
	const uint32_t feature = (blockIdx.x * blockDim.x + threadIdx.x) * N_FEATURES_PER_THREAD - i * N_FEATURES_PER_LEVEL;

	if (max_level_gpu) {
		max_level = (max_level_gpu[i] * num_grid_features) / N_FEATURES_PER_LEVEL;
	} else {
		max_level = (max_level * num_grid_features) / N_FEATURES_PER_LEVEL;
	}

	if (level > max_level + 1e-3f) {
		return;
	}

	grid_gradient += hashmap_offset_table[level] * N_FEATURES_PER_LEVEL;
	const uint32_t hashmap_size = hashmap_offset_table[level + 1] - hashmap_offset_table[level];

	// const float scale = exp2f(level * log2_per_level_scale) * base_resolution - 1.0f;
	// const uint32_t grid_resolution = ((uint32_t)ceil(scale) + 1);
	const float scale = scale_table[level];
	const uint32_t grid_resolution = resolution_table[level];

	auto add_grid_gradient = [&](const uint32_t local_pos[N_POS_DIMS], const vector_t<T, N_FEATURES_PER_THREAD>& grad, const float weight) {
		uint32_t index = grid_index<N_POS_DIMS, N_FEATURES_PER_LEVEL>(grid_type, feature, hashmap_size, grid_resolution, local_pos);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600 // atomicAdd(__half2) is only supported with compute capability 60 and above
		if (N_FEATURES_PER_THREAD > 1 && std::is_same<GRAD_T, __half>::value) {
			for (uint32_t f = 0; f < N_FEATURES_PER_THREAD; f += 2) {
				__half2 v = {(__half)((float)grad[f] * weight), (__half)((float)grad[f+1] * weight)};
				atomicAdd((__half2*)&grid_gradient[index + f], v);
			}
		} else
#endif
		{
			if (std::is_same<GRAD_T, __half>::value) {
				// Should never happen
				//printf("Attempted to use atomicAdd(__half)\n")
			} else {
				for (uint32_t f = 0; f < N_FEATURES_PER_THREAD; ++f) {
					atomicAdd((float*)&grid_gradient[index + f], (float)grad[f] * weight);
				}
			}
		}
	};

	float pos[N_POS_DIMS];
	float pos_derivative[N_POS_DIMS];
	uint32_t pos_grid[N_POS_DIMS];

	if (interpolation_type == InterpolationType::Nearest || interpolation_type == InterpolationType::Linear) {
		#pragma unroll
		for (uint32_t dim = 0; dim < N_POS_DIMS; ++dim) {
			pos_fract(positions_in(dim, i), &pos[dim], &pos_derivative[dim], &pos_grid[dim], scale, identity_fun, identity_derivative);
		}
	} else {
		#pragma unroll
		for (uint32_t dim = 0; dim < N_POS_DIMS; ++dim) {
			pos_fract(positions_in(dim, i), &pos[dim], &pos_derivative[dim], &pos_grid[dim], scale, smoothstep, smoothstep_derivative);
		}
	}

	vector_t<T, N_FEATURES_PER_THREAD> grad;

	#pragma unroll
	for (uint32_t f = 0; f < N_FEATURES_PER_THREAD; ++f) {
		grad[f] = dL_dy[i + (level * N_FEATURES_PER_LEVEL + feature + f) * num_elements];
	}

	if (interpolation_type == InterpolationType::Nearest) {
		// d(dydx)_dgrid is zero when there's no interpolation.
		return;
	}

	// for N-linear interpolation
	#pragma unroll
	for (uint32_t grad_dim = 0; grad_dim < N_POS_DIMS; ++grad_dim) {
		float grad_in = scale * dL_ddLdx(grad_dim, i) * pos_derivative[grad_dim];
		#pragma unroll
		for (uint32_t idx = 0; idx < (1 << (N_POS_DIMS-1)); ++idx) {
			float weight = grad_in;
			uint32_t pos_grid_local[N_POS_DIMS];

			#pragma unroll
			for (uint32_t non_grad_dim = 0; non_grad_dim < N_POS_DIMS-1; ++non_grad_dim) {
				const uint32_t dim = non_grad_dim >= grad_dim ? (non_grad_dim+1) : non_grad_dim;

				if ((idx & 1<<non_grad_dim) == 0) {
					weight *= 1 - pos[dim];
					pos_grid_local[dim] = pos_grid[dim];
				} else {
					weight *= pos[dim];
					pos_grid_local[dim] = pos_grid[dim] + 1;
				}
			}

			// left
			pos_grid_local[grad_dim] = pos_grid[grad_dim];
			add_grid_gradient(pos_grid_local, grad, -weight);
			// right
			pos_grid_local[grad_dim] = pos_grid[grad_dim] + 1;
			add_grid_gradient(pos_grid_local, grad, weight);
		}
	}
}

template <typename T, uint32_t N_POS_DIMS, uint32_t N_FEATURES_PER_LEVEL, uint32_t N_FEATURES_PER_THREAD>
__global__ void kernel_grid_backward_input_backward_input(
	const uint32_t num_elements,
	const uint32_t num_grid_features,
	const uint32_t* hashmap_offset_table,
	const uint32_t* resolution_table,
	const float* scale_table,
	const uint32_t valid_level,
	// const uint32_t base_resolution,
	// const float log2_per_level_scale,
	const float quantize_threshold,
	float max_level,
	const float* __restrict__ max_level_gpu,
	const InterpolationType interpolation_type,
	const GridType grid_type,
	// inputs
	MatrixView<const float> dL_ddLdx,
	MatrixView<const float> positions_in,
	const T* __restrict__ dL_dy,
	const T* __restrict__ grid,
	// outputs
	MatrixView<float> dL_dx
) {
	const uint32_t i = ((blockIdx.x * blockDim.x + threadIdx.x) * N_FEATURES_PER_THREAD) / N_FEATURES_PER_LEVEL;
	if (i >= num_elements) return;

	const uint32_t level = blockIdx.y ; // <- the level is the same for all threads.
	if (level > valid_level) return;
	const uint32_t feature = (blockIdx.x * blockDim.x + threadIdx.x) * N_FEATURES_PER_THREAD - i * N_FEATURES_PER_LEVEL;

	if (max_level_gpu) {
		max_level = (max_level_gpu[i] * num_grid_features) / N_FEATURES_PER_LEVEL;
	} else {
		max_level = (max_level * num_grid_features) / N_FEATURES_PER_LEVEL;
	}

	if (level > max_level + 1e-3f) {
		return;
	}

	grid += hashmap_offset_table[level] * N_FEATURES_PER_LEVEL;
	const uint32_t hashmap_size = hashmap_offset_table[level + 1] - hashmap_offset_table[level];

	// const float scale = exp2f(level * log2_per_level_scale) * base_resolution - 1.0f;
	// const uint32_t grid_resolution = ((uint32_t)ceil(scale) + 1);
	const float scale = scale_table[level];
	const uint32_t grid_resolution = resolution_table[level];

	float pos[N_POS_DIMS];
	float pos_derivative[N_POS_DIMS];
	float pos_2nd_derivative[N_POS_DIMS];
	uint32_t pos_grid[N_POS_DIMS];

	if (interpolation_type == InterpolationType::Nearest || interpolation_type == InterpolationType::Linear) {
		#pragma unroll
		for (uint32_t dim = 0; dim < N_POS_DIMS; ++dim) {
			pos_fract(positions_in(dim, i), &pos[dim], &pos_derivative[dim], &pos_2nd_derivative[dim], &pos_grid[dim], scale, identity_fun, identity_derivative, identity_2nd_derivative);
		}
	} else {
		#pragma unroll
		for (uint32_t dim = 0; dim < N_POS_DIMS; ++dim) {
			pos_fract(positions_in(dim, i), &pos[dim], &pos_derivative[dim], &pos_2nd_derivative[dim], &pos_grid[dim], scale, smoothstep, smoothstep_derivative, smoothstep_2nd_derivative);
		}
	}

	vector_t<T, N_FEATURES_PER_THREAD> grad;

	#pragma unroll
	for (uint32_t f = 0; f < N_FEATURES_PER_THREAD; ++f) {
		grad[f] = dL_dy[i + (level * N_FEATURES_PER_LEVEL + feature + f) * num_elements];
	}

	if (interpolation_type == InterpolationType::Nearest) {
		// d(dydx)_dx is zero when there's no interpolation
		return;
	}

	// for N-linear interpolation

	auto calc_dLdx = [&](const uint32_t local_pos[N_POS_DIMS], const float weight) {
		uint32_t index = grid_index<N_POS_DIMS, N_FEATURES_PER_LEVEL>(grid_type, feature, hashmap_size, grid_resolution, local_pos);
		float dL_dx_dim = 0;
		#pragma unroll
		for (uint32_t f=0; f < N_FEATURES_PER_THREAD; ++f) {
			dL_dx_dim += (float)grid[index + f] * (float)grad[f] * weight;
		}
		return dL_dx_dim;
	};

	vector_t<float, N_POS_DIMS> grad_in_diag;
	vector_t<float, N_POS_DIMS> grad_in_other;
	#pragma unroll
	for (uint32_t grad_dim = 0; grad_dim < N_POS_DIMS; ++grad_dim) {
		// from diagonal part of Hessian
		grad_in_diag[grad_dim] = scale * scale * dL_ddLdx(grad_dim, i) * pos_2nd_derivative[grad_dim];
		// from other part of Hessian
		grad_in_other[grad_dim] = scale * scale * dL_ddLdx(grad_dim, i) * pos_derivative[grad_dim]; // will do " * pos_derivative[real_other_grad_dim] " later
	}

	static constexpr bool dimension_greater_than_1 = (N_POS_DIMS > 1);
	#pragma unroll
	for (uint32_t grad_dim = 0; grad_dim < N_POS_DIMS; ++grad_dim) {
		float grad_out = 0;
		#pragma unroll
		for (uint32_t idx = 0; idx < (1 << (N_POS_DIMS-1)); ++idx) {
			// from diagonal part of Hessian; d(doutput_d[grad_dim])_d[grad_dim]
			// NOTE: LinearInterpolations' diagonal part is 0.
			if (interpolation_type == InterpolationType::Smoothstep) {
				float weight_2nd_diag = grad_in_diag[grad_dim];
				uint32_t pos_grid_local[N_POS_DIMS];

				#pragma unroll
				for (uint32_t non_grad_dim = 0; non_grad_dim < N_POS_DIMS-1; ++non_grad_dim) {
					const uint32_t dim = non_grad_dim >= grad_dim ? (non_grad_dim+1) : non_grad_dim;
					// real non_grad_dim
					if ((idx & 1<<non_grad_dim) == 0) {
						weight_2nd_diag *= 1 - pos[dim];
						pos_grid_local[dim] = pos_grid[dim];
					} else {
						weight_2nd_diag *= pos[dim];
						pos_grid_local[dim] = pos_grid[dim] + 1;
					}
				}

				// left
				pos_grid_local[grad_dim] = pos_grid[grad_dim];
				grad_out += calc_dLdx(pos_grid_local, -weight_2nd_diag);
				// right
				pos_grid_local[grad_dim] = pos_grid[grad_dim] + 1;
				grad_out += calc_dLdx(pos_grid_local, weight_2nd_diag);
			}

			// from other part of Hessian; d(doutput_d[real_other_grad_dim])_d[grad_dim]
			if (dimension_greater_than_1) {
				#pragma unroll
				for (uint32_t other_grad_dim = 0; other_grad_dim < N_POS_DIMS-1; ++other_grad_dim) {
					const uint32_t real_other_grad_dim = other_grad_dim >= grad_dim ? (other_grad_dim+1) : other_grad_dim;
					float weight_2nd_other = grad_in_other[real_other_grad_dim] * pos_derivative[grad_dim];
					uint32_t pos_grid_local[N_POS_DIMS];

					#pragma unroll
					for (uint32_t non_grad_dim = 0; non_grad_dim < N_POS_DIMS-1; ++non_grad_dim) {
						// real non_grad_dim
						const uint32_t dim = non_grad_dim >= real_other_grad_dim ? (non_grad_dim+1) : non_grad_dim;
						if ((idx & 1<<non_grad_dim) == 0) {
							if (dim != grad_dim) {
								weight_2nd_other *= 1 - pos[dim];
							} else {
								weight_2nd_other *= -1;
							}
							pos_grid_local[dim] = pos_grid[dim];
						} else {
							if (dim != grad_dim) {
								weight_2nd_other *= pos[dim];
							}
							pos_grid_local[dim] = pos_grid[dim] + 1;
						}
					}

					// left
					pos_grid_local[real_other_grad_dim] = pos_grid[real_other_grad_dim];
					grad_out += calc_dLdx(pos_grid_local, -weight_2nd_other);
					// right
					pos_grid_local[real_other_grad_dim] = pos_grid[real_other_grad_dim] + 1;
					grad_out += calc_dLdx(pos_grid_local, weight_2nd_other);
				}
			}
		}

		atomicAdd((float*)&dL_dx(grad_dim, i), grad_out);
	}
}

template <typename T, uint32_t N_POS_DIMS>
__global__ void kernel_grid_backward_input_backward_dLdoutput(
	const uint32_t num_elements,
	const uint32_t num_grid_features,
	// inputs
	MatrixView<const float> dL_ddLdx,
	const float* __restrict__ dy_dx,
	const T* dL_dy_rm,
	// ouputs
	tcnn::MatrixView<T> dL_ddLdy
) {
	const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= num_elements) return;

	for (uint32_t k=0; k < num_grid_features; ++k) {
		auto dy_dx_local = ((tcnn::vector_fullp_t<N_POS_DIMS>*)dy_dx)[i + k * num_elements];

		float result = 0;
		#pragma unroll
		for (uint32_t grad_dim = 0; grad_dim < N_POS_DIMS; ++grad_dim) {
			result += dy_dx_local[grad_dim] * dL_ddLdx(grad_dim, i);
		}

		dL_ddLdy(k, i) = (T)result;
	}
}

template <typename T, uint32_t N_POS_DIMS, uint32_t N_FEATURES_PER_LEVEL>
__global__ void kernel_resample(
	const uint64_t num_elements,
	const uint32_t level,
	const uint32_t* hashmap_offset_table,
	const uint32_t* hashmap_offset_table_new,
	const float scale,
	const uint32_t grid_resolution,
	const float scale_new,
	const uint32_t grid_resolution_new,
	const float quantize_threshold,
	const InterpolationType interpolation_type,
	const uint8_t* density_grid,
	const uint32_t max_cascade,
	const uint32_t nerf_gridsize,
	const GridType grid_type,
	const T* __restrict__ grid,
	float* __restrict__ grid_new
) {
	const uint64_t i = (uint64_t)blockIdx.x * (uint64_t)blockDim.x + (uint64_t)threadIdx.x;
	if (i >= num_elements) return;

	// const uint32_t level = blockIdx.y; // <- the level is the same for all threads
	// const uint32_t map_size_new = map_offset_table_new[level+1] - map_offset_table_new[level];
	// if (i >= map_size_new) return;
	// const float scale_new = exp2f(level * log2_per_level_scale) * base_resolution_new - 1.0f;
	// const uint32_t grid_resolution_new = ((uint32_t)ceil(scale_new) + 1);
	// if (i >= pow(grid_resolution_new, N_POS_DIMS)) return;

	grid += hashmap_offset_table[level] * N_FEATURES_PER_LEVEL;
	grid_new += hashmap_offset_table_new[level] * N_FEATURES_PER_LEVEL;
	const uint32_t hashmap_size = hashmap_offset_table[level + 1] - hashmap_offset_table[level];
	const uint32_t hashmap_size_new = hashmap_offset_table_new[level + 1] - hashmap_offset_table_new[level];

	// const float scale = exp2f(level * log2_per_level_scale) * base_resolution - 1.0f;
	// const uint32_t grid_resolution = ((uint32_t)ceil(scale) + 1);

	float pos[N_POS_DIMS];
	uint32_t pos_grid[N_POS_DIMS];

	float cur_pos[N_POS_DIMS];
	// vector_fullp_t<N_POS_DIMS> cur_pos;
	uint32_t cur_pos_grid[N_POS_DIMS];
	uint64_t cur_pos_idx = i;
	if (interpolation_type == InterpolationType::Nearest || interpolation_type == InterpolationType::Linear) {
		#pragma unroll
		for (uint32_t dim = 0; dim < N_POS_DIMS; ++dim) {
			cur_pos_grid[dim] = cur_pos_idx % grid_resolution_new;
			cur_pos_idx /= grid_resolution_new;
			cur_pos[dim] = (float)cur_pos_grid[dim] / (float)(grid_resolution_new - 1);
			pos_fract_resample(cur_pos[dim], &pos[dim], &pos_grid[dim], scale, identity_fun);
		}
	} else {
		#pragma unroll
		for (uint32_t dim = 0; dim < N_POS_DIMS; ++dim) {
			cur_pos_grid[dim] = cur_pos_idx % grid_resolution_new;
			cur_pos_idx /= grid_resolution_new;
			cur_pos[dim] = (float)cur_pos_grid[dim] / (float)(grid_resolution_new - 1);
			pos_fract_resample(cur_pos[dim], &pos[dim], &pos_grid[dim], scale, smoothstep);
		}
	}

#if RESMAPLE_CHECKED_BY_DENSITY_GRID
	if(!tv_density_grid_occupied_at(cur_pos, density_grid, max_cascade, nerf_gridsize)){
	// if(!tv_density_grid_occupied_at(cur_pos, density_grid, 0, nerf_gridsize)){
		return;
	}
#endif

#if RESAMPLE_KERNEL_DEBUG
	// debug
	if (i == 1997) {
		printf("i: %zu, level: %d, scale_new: %f, grid_resolution_new: %d\n", i, level, scale_new, grid_resolution_new);
		printf("level: %d, scale_new: %f, grid_resolution_new: %d\n", level, scale_new, grid_resolution_new);
		printf("scale: %f, grid_resolution: %d, hashmap_size: %d, hashmap_size_new: %d\n", scale, grid_resolution, hashmap_size, hashmap_size_new);
		printf("pos: (%f, %f, %f)\n", pos[0], pos[1], pos[2]);
		printf("pos_grid: (%d, %d, %d)\n", pos_grid[0], pos_grid[1], pos_grid[2]);
		printf("cur_pos: (%f, %f, %f)\n", cur_pos[0], cur_pos[1], cur_pos[2]);
		printf("cur_pos_grid: (%d, %d, %d)\n", cur_pos_grid[0], cur_pos_grid[1], cur_pos_grid[2]);
	}
#endif

	auto grid_val = [&](const uint32_t local_pos[N_POS_DIMS]) {
		uint32_t index = grid_index<N_POS_DIMS, N_FEATURES_PER_LEVEL>(grid_type, 0, hashmap_size, grid_resolution, local_pos);
		return *(vector_t<T, N_FEATURES_PER_LEVEL>*)&grid[index];
	};

	// N-linear interpolation
	vector_t<T, N_FEATURES_PER_LEVEL> result = {};

	#pragma unroll
	for (uint32_t idx = 0; idx < (1 << N_POS_DIMS); ++idx) {
		float weight = 1;
		uint32_t pos_grid_local[N_POS_DIMS];

		#pragma unroll
		for (uint32_t dim = 0; dim < N_POS_DIMS; ++dim) {
			if ((idx & (1<<dim)) == 0) {
				weight *= 1 - pos[dim];
				pos_grid_local[dim] = pos_grid[dim];
			} else {
				weight *= pos[dim];
				pos_grid_local[dim] = pos_grid[dim] + 1;
			}
		}

		auto val = grid_val(pos_grid_local);

#if RESAMPLE_KERNEL_DEBUG
		// debug
		if (i == 1997) {
			printf("pos_grid_local: (%d, %d, %d), weight: %f\n", pos_grid_local[0], pos_grid_local[1], pos_grid_local[2], weight);
		}
#endif

		#pragma unroll
		for (uint32_t feature = 0; feature < N_FEATURES_PER_LEVEL; ++feature) {
			float data = (float)((T*)&val)[feature];
			if (fabsf(data) < quantize_threshold) data = 0.f;
			((T*)&result)[feature] += (T)(weight * data);
		}
	}


	uint32_t index = grid_index<N_POS_DIMS, N_FEATURES_PER_LEVEL>(grid_type, 0, hashmap_size_new, grid_resolution_new, cur_pos_grid);
// #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600 // atomicAdd(__half2) is only supported with compute capability 60 and above
// 	if (N_FEATURES_PER_LEVEL > 1 && std::is_same<T, __half>::value) {
// 		for (uint32_t f = 0; f < N_FEATURES_PER_LEVEL; f += 2) {
// 			__half2 v = {(__half)((float)result[f]), (__half)((float)result[f+1])};
// 			float data = (float)v;
// 			if (fabsf(data) >= quantize_threshold) {
// 				atomicExch((__half2*)&grid_new[index + f], v);
// 			}
// 		}
// 	} else
// #endif
// 	{
// 		if (std::is_same<T, __half>::value) {
// 			// Should never happen
// 			//printf("Attempted to use atomicAdd(__half)\n")
// 		} else {
// 			for (uint32_t f = 0; f < N_FEATURES_PER_LEVEL; ++f) {
// 				float data = (float)result[f];
// 				if (fabsf(data) >= quantize_threshold) {
// 					atomicExch((float*)&grid_new[index + f], data);
// 				}
// 			}
// 		}
// 	}

	// auto grid_val_new = [&](const uint32_t local_pos[N_POS_DIMS]) {
	// 	uint32_t index = grid_index<N_POS_DIMS, N_FEATURES_PER_LEVEL>(grid_type, 0, hashmap_size_new, grid_resolution_new, local_pos);
	// 	// return *(vector_t<T, N_FEATURES_PER_LEVEL>*)&grid_new[index];
	// 	return &grid_new[index];
	// };
	
	#pragma unroll
	for (uint32_t f = 0; f < N_FEATURES_PER_LEVEL; ++f) {
		float data = (float)result[f];
		if (fabsf(data) >= quantize_threshold) {
			// atomicAdd((float*)&(grid_new[index + f]), data); // memory access fail
			// if ((float)result[f] > (float)grid_new[index+f] || (float)result[f] < (float)grid_new[index+f])
			if (fabsf(data) > fabsf(grid_new[index+f]))
				atomicExch((float*)&(grid_new[index + f]), data); // memory access fail
				// grid_new[index + f] = result[f];
			// ((T*)grid_val_new(cur_pos_grid))[f] = result[f];image.png
		}
	}

}

template <typename T>
class GridEncoding : public Encoding<T> {
public:
	virtual uint32_t n_pos_dims() const = 0;
	virtual uint32_t n_features_per_level() const = 0;

	virtual size_t level_n_params(uint32_t level) const = 0;
	virtual size_t level_params_offset(uint32_t level) const = 0;

	float max_level() const {
		return m_max_level;
	}

	void set_max_level(float value) {
		m_max_level = value;
	}

	float* max_level_gpu() const {
		return m_max_level_gpu;
	}

	void set_max_level_gpu(float* value) {
		m_max_level_gpu = value;
	}

	float quantize_threshold() const {
		return m_quantize_threshold;
	}

	void set_quantize_threshold(float value) {
		m_quantize_threshold = value;
	}

protected:
	// Disables lookups of finer levels than this.
	// The default value of 1000 effectively disables the feature
	float m_max_level = 1000.f;

	// If this pointer is non-null, it is expected to point to per-element m_max_level
	float* m_max_level_gpu = nullptr;

	// Features with values less then this threshold are quantized to zero
	float m_quantize_threshold = 0.f;
};

template <typename T, uint32_t N_POS_DIMS=3, uint32_t N_FEATURES_PER_LEVEL=2>
class GridEncodingTemplated : public GridEncoding<T> {
public:
#if TCNN_MIN_GPU_ARCH >= 62 || TCNN_MIN_GPU_ARCH == 60
	// The GPUs that we tested this on do not have an efficient 1D fp16
	// atomicAdd feature. Thus, we accumulate gradients at fp32 if we're
	// forced to use 1D atomicAdds. As soon as 2D or higher is possible,
	// we can make use the efficient atomicAdd(half2) function.
	using grad_t = std::conditional_t<N_FEATURES_PER_LEVEL == 1, float, T>;
#else
	// atomicAdd(__half2) is only supported with compute capability 60 and above.
	// Since atomicAdd(__half) is relatively slow / doesn't exist for low compute
	// capabilities, accumulate in fp32 instead.
	using grad_t = float;
#endif

	GridEncodingTemplated(
		uint32_t n_features,
		uint32_t log2_hashmap_size,
		uint32_t base_resolution,
		float per_level_scale,
		bool stochastic_interpolation,
		InterpolationType interpolation_type,
		GridType grid_type,
		float valid_level_scale,
		float base_valid_level_scale,
		uint32_t base_training_step
	) :
	m_n_features{n_features},
	m_log2_hashmap_size{log2_hashmap_size},
	m_base_resolution{base_resolution},
	m_per_level_scale{per_level_scale},
	m_stochastic_interpolation{stochastic_interpolation},
	m_interpolation_type{interpolation_type},
	m_grid_type{grid_type},
	m_valid_level_scale{valid_level_scale},
	m_base_valid_level_scale{base_valid_level_scale},
	m_base_training_step{base_training_step}
	{
		m_n_levels = div_round_up(m_n_features, N_FEATURES_PER_LEVEL);
		m_valid_level = m_n_levels;
		uint32_t offset = 0;

		m_hashmap_offsets_table_cpu.resize(m_n_levels + 1);
		m_resolution_table_cpu.resize(m_n_levels);
		m_scale_table_cpu.resize(m_n_levels);

		for (uint32_t i = 0; i < m_n_levels; ++i) {
			// Compute dense params required for the given level
			const float scale = exp2f(i * std::log2(per_level_scale)) * base_resolution - 1.0f;
			const uint32_t resolution = (uint32_t)(ceilf(scale)) + 1;
			// m_scale_table_cpu[i] = scale;
			m_scale_table_cpu[i] = resolution - 1;
			m_resolution_table_cpu[i] = resolution;


			uint32_t max_params = std::numeric_limits<uint32_t>::max()/2;
			uint32_t params_in_level = std::pow((float)resolution, N_POS_DIMS) > (float)max_params ? max_params : powi(resolution, N_POS_DIMS);

			// Make sure memory accesses will be aligned
			params_in_level = next_multiple(params_in_level, 8u);

			if (grid_type == GridType::Dense) {
				// No-op
			} else if (grid_type == GridType::Tiled) {
				// If tiled grid needs fewer params than dense, then use fewer and tile.
				params_in_level = std::min(params_in_level, powi(base_resolution, N_POS_DIMS));
			} else if (grid_type == GridType::Hash) {
				// If hash table needs fewer params than dense, then use fewer and rely on the hash.
				params_in_level = std::min(params_in_level, (1u << log2_hashmap_size));
			} else {
				throw std::runtime_error{std::string{"GridEncoding: invalid grid type "} + to_string(grid_type)};
			}

			m_hashmap_offsets_table_cpu[i] = offset;
			offset += params_in_level;

#ifdef TCNN_VERBOSE_MEMORY_ALLOCS
			std::cout << "GridEncoding at level " << i << ": resolution=" << resolution << " params_in_level=" << params_in_level << std::endl;
#endif
		}

		m_hashmap_offsets_table_cpu[m_n_levels] = offset;
		m_n_params = m_hashmap_offsets_table_cpu[m_n_levels] * N_FEATURES_PER_LEVEL;
		m_hashmap_offsets_table.resize(m_n_levels + 1);
		CUDA_CHECK_THROW(cudaMemcpy(m_hashmap_offsets_table.data(), m_hashmap_offsets_table_cpu.data(), (m_n_levels+1) * sizeof(uint32_t), cudaMemcpyHostToDevice));

		m_resolution_table.resize(m_n_levels);
		CUDA_CHECK_THROW(cudaMemcpy(m_resolution_table.data(), m_resolution_table_cpu.data(), (m_n_levels) * sizeof(uint32_t), cudaMemcpyHostToDevice));
		m_scale_table.resize(m_n_levels);
		CUDA_CHECK_THROW(cudaMemcpy(m_scale_table.data(), m_scale_table_cpu.data(), (m_n_levels) * sizeof(float), cudaMemcpyHostToDevice));

		m_n_padded_output_dims = m_n_output_dims = m_n_features;

		if (n_features % N_FEATURES_PER_LEVEL != 0) {
			throw std::runtime_error{"GridEncoding: number of grid features must be a multiple of n_features_per_level"};
		}
	}

	std::unique_ptr<Context> forward_impl(
		cudaStream_t stream,
		const GPUMatrixDynamic<float>& input,
		GPUMatrixDynamic<T>* output = nullptr,
		bool use_inference_params = false,
		bool prepare_input_gradients = false
	) override {
		auto forward = std::make_unique<ForwardContext>();
		const uint32_t num_elements = input.n();
		if ((!output && !prepare_input_gradients) || m_n_padded_output_dims == 0 || num_elements == 0) {
			return forward;
		}

		SyncedMultiStream synced_streams{stream, m_n_to_pad > 0 ? 2u : 1u};

		// Take care of padding on the auxiliary stream
		if (output && m_n_to_pad > 0) {
			if (output->layout() == AoS) {
				parallel_for_gpu_aos(synced_streams.get(1), num_elements, m_n_to_pad, [n_output_dims=m_n_output_dims, out=output->pitched_ptr()] __device__ (size_t elem, size_t dim) {
					out(elem)[n_output_dims + dim] = 0;
				});
			} else {
				parallel_for_gpu_aos(synced_streams.get(1), num_elements, m_n_to_pad, [num_elements, n_output_dims=m_n_output_dims, out_soa=output->data()] __device__ (size_t elem, size_t dim) {
					out_soa[elem + (n_output_dims + dim) * num_elements] = 0;
				});
			}
		}
		// Idea: each block only takes care of _one_ hash level (but may iterate over multiple input elements).
		// This way, only one level of the hashmap needs to fit into caches at a time (and it reused for consecutive
		// elements) until it is time to process the next level.

		static constexpr uint32_t N_THREADS_HASHGRID = 512;
		const dim3 blocks_hashgrid = { div_round_up(num_elements, N_THREADS_HASHGRID), m_n_levels, 1 };

		T* encoded_positions_soa = output ? output->data() : nullptr;
		GPUMemoryArena::Allocation workspace;
		if (output && output->layout() == AoS) {
			workspace = allocate_workspace(stream, num_elements * m_n_features * sizeof(T));
			encoded_positions_soa = (T*)workspace.data();
		}

		if (prepare_input_gradients) {
			forward->dy_dx = GPUMatrix<float, RM>{N_POS_DIMS * m_n_features, input.n(), stream};
		}
		kernel_grid<T, N_POS_DIMS, N_FEATURES_PER_LEVEL><<<blocks_hashgrid, N_THREADS_HASHGRID, 0, synced_streams.get(0)>>>(
			num_elements,
			m_n_features,
			m_hashmap_offsets_table.data(),
			m_resolution_table.data(),
			m_scale_table.data(),
			m_valid_level,
			// m_base_resolution,
			// std::log2(m_per_level_scale),
			this->m_quantize_threshold,
			this->m_max_level,
			this->m_max_level_gpu,
			m_interpolation_type,
			m_grid_type,
			use_inference_params ? m_grid_inference : m_grid,
			forward->positions.data() ? forward->positions.view() : input.view(),
			encoded_positions_soa,
			forward->dy_dx.data()
		);

		if (output && output->layout() == AoS) {
			// Transpose result (was stored row major due to coalescing)
			const dim3 threads_transpose = { m_n_levels * N_FEATURES_PER_LEVEL, 8, 1 };
			const uint32_t blocks_transpose = div_round_up(num_elements, threads_transpose.y);
			transpose_encoded_position<T><<<blocks_transpose, threads_transpose, 0, synced_streams.get(0)>>>(
				num_elements,
				encoded_positions_soa,
				output->pitched_ptr()
			);
		}
		return forward;
	}

	void backward_impl(
		cudaStream_t stream,
		const Context& ctx,
		const GPUMatrixDynamic<float>& input,
		const GPUMatrixDynamic<T>& output,
		const GPUMatrixDynamic<T>& dL_doutput,
		GPUMatrixDynamic<float>* dL_dinput = nullptr,
		bool use_inference_params = false,
		EGradientMode param_gradients_mode = EGradientMode::Overwrite
	) override {
		const uint32_t num_elements = input.n();
		if ((!dL_dinput && param_gradients_mode == EGradientMode::Ignore) || m_n_padded_output_dims == 0 || num_elements == 0) {
			return;
		}

		const auto& forward = dynamic_cast<const ForwardContext&>(ctx);

		const T* dL_dy_rm = dL_doutput.data();

		GPUMemoryArena::Allocation workspace;
		if (dL_doutput.layout() == CM) {
			workspace = allocate_workspace(stream, num_elements * m_n_features * sizeof(T));

			// Transpose dL_dy. Use the buffer previously occupied by the encoded positions
			const dim3 threads_transpose = { m_n_levels * N_FEATURES_PER_LEVEL, 8, 1 };
			const uint32_t blocks_transpose = div_round_up(num_elements, threads_transpose.y);
			transpose_gradients<T><<<blocks_transpose, threads_transpose, 0, stream>>>(
				num_elements,
				(T*)workspace.data(),
				dL_doutput.pitched_ptr()
			);

			dL_dy_rm = (const T*)workspace.data();
		}

		if (param_gradients_mode != EGradientMode::Ignore) {
			// We accumulate gradients with grad_t precision, which, for performance reasons, is not always T.
			// If not, accumulate in a temporary buffer and cast later.
			grad_t* grid_gradient;
			GPUMemoryArena::Allocation grid_gradient_tmp;

			if (!std::is_same<grad_t, T>::value) {
				grid_gradient_tmp = allocate_workspace(stream, m_n_params * sizeof(grad_t));
				grid_gradient = (grad_t*)grid_gradient_tmp.data();
			} else {
				grid_gradient = (grad_t*)m_grid_gradient;
			}

			if (param_gradients_mode == EGradientMode::Overwrite) {
				CUDA_CHECK_THROW(cudaMemsetAsync(grid_gradient, 0, n_params() * sizeof(grad_t), stream));
			}

			static constexpr uint32_t N_THREADS_HASHGRID = 256;
			static constexpr uint32_t N_FEATURES_PER_THREAD = std::min(2u, N_FEATURES_PER_LEVEL);

			const dim3 blocks_hashgrid = { div_round_up(num_elements * N_FEATURES_PER_LEVEL / N_FEATURES_PER_THREAD, N_THREADS_HASHGRID), m_n_levels, 1 };

			kernel_grid_backward<T, grad_t, N_POS_DIMS, N_FEATURES_PER_LEVEL, N_FEATURES_PER_THREAD><<<blocks_hashgrid, N_THREADS_HASHGRID, 0, stream>>>(
				num_elements,
				m_n_features,
				m_hashmap_offsets_table.data(),
				m_resolution_table.data(),
				m_scale_table.data(),
				m_valid_level,
				// m_base_resolution,
				// std::log2(m_per_level_scale),
				this->m_max_level,
				this->m_max_level_gpu,
				m_stochastic_interpolation,
				m_interpolation_type,
				m_grid_type,
				grid_gradient,
				forward.positions.data() ? forward.positions.view() : input.view(), // positions SoA
				dL_dy_rm // gradients SoA
			);

			if (!std::is_same<grad_t, T>::value) {
				parallel_for_gpu(stream, n_params(), [grad=m_grid_gradient, grad_tmp=grid_gradient] __device__ (size_t i) {
					grad[i] = (T)grad_tmp[i];
				});
			}
		}

		if (!dL_dinput) {
			return;
		}

		linear_kernel(kernel_grid_backward_input<T, N_POS_DIMS>, 0, stream,
			num_elements,
			m_n_features,
			dL_dy_rm,
			forward.dy_dx.data(),
			dL_dinput->view()
		);
	}

	void backward_backward_input_impl(
		cudaStream_t stream,
		const Context& ctx,
		const GPUMatrixDynamic<float>& input,
		const GPUMatrixDynamic<float>& dL_ddLdinput,
		const GPUMatrixDynamic<T>& dL_doutput,
		GPUMatrixDynamic<T>* dL_ddLdoutput = nullptr,
		GPUMatrixDynamic<float>* dL_dinput = nullptr,
		bool use_inference_params = false,
		EGradientMode param_gradients_mode = EGradientMode::Overwrite
	) override {
		const uint32_t num_elements = input.n();
		if ((!dL_ddLdoutput && param_gradients_mode == EGradientMode::Ignore) || m_n_padded_output_dims == 0 || num_elements == 0) {
			return;
		}

		const auto& forward = dynamic_cast<const ForwardContext&>(ctx);

		const T* dL_dy_rm = dL_doutput.data();

		GPUMemoryArena::Allocation workspace;
		if (dL_doutput.layout() == CM) {
			workspace = allocate_workspace(stream, num_elements * m_n_features * sizeof(T));

			// Transpose dL_dy. Use the buffer previously occupied by the encoded positions
			const dim3 threads_transpose = { m_n_levels * N_FEATURES_PER_LEVEL, 8, 1 };
			const uint32_t blocks_transpose = div_round_up(num_elements, threads_transpose.y);
			transpose_gradients<T><<<blocks_transpose, threads_transpose, 0, stream>>>(
				num_elements,
				(T*)workspace.data(),
				dL_doutput.pitched_ptr()
			);

			dL_dy_rm = (const T*)workspace.data();
		}

		if (param_gradients_mode != EGradientMode::Ignore) {
			// We accumulate gradients with grad_t precision, which, for performance reasons, is not always T.
			// If not, accumulate in a temporary buffer and cast later.
			grad_t* grid_gradient;
			GPUMemoryArena::Allocation grid_gradient_tmp;

			if (!std::is_same<grad_t, T>::value) {
				grid_gradient_tmp = allocate_workspace(stream, m_n_params * sizeof(grad_t));
				grid_gradient = (grad_t*)grid_gradient_tmp.data();
			} else {
				grid_gradient = (grad_t*)m_grid_gradient;
			}

			if (param_gradients_mode == EGradientMode::Overwrite) {
				CUDA_CHECK_THROW(cudaMemsetAsync(grid_gradient, 0, n_params() * sizeof(grad_t), stream));
			}

			static constexpr uint32_t N_THREADS_HASHGRID = 256;
			static constexpr uint32_t N_FEATURES_PER_THREAD = std::min(2u, N_FEATURES_PER_LEVEL);

			const dim3 blocks_hashgrid = { div_round_up(num_elements * N_FEATURES_PER_LEVEL / N_FEATURES_PER_THREAD, N_THREADS_HASHGRID), m_n_levels, 1 };

			// from dL_d(dL_dx) to dL_dgrid
			kernel_grid_backward_input_backward_grid<T, grad_t, N_POS_DIMS, N_FEATURES_PER_LEVEL, N_FEATURES_PER_THREAD><<<blocks_hashgrid, N_THREADS_HASHGRID, 0, stream>>>(
				num_elements,
				m_n_features,
				m_hashmap_offsets_table.data(),
				m_resolution_table.data(),
				m_scale_table.data(),
				m_valid_level,
				// m_base_resolution,
				// std::log2(m_per_level_scale),
				this->m_max_level,
				this->m_max_level_gpu,
				m_interpolation_type,
				m_grid_type,
				// inputs
				dL_ddLdinput.view(),
				forward.positions.data() ? forward.positions.view() : input.view(), // positions SoA
				dL_dy_rm, // gradients SoA
				// outputs
				grid_gradient
			);

			if (!std::is_same<grad_t, T>::value) {
				parallel_for_gpu(stream, n_params(), [grad=m_grid_gradient, grad_tmp=grid_gradient] __device__ (size_t i) {
					grad[i] = (T)grad_tmp[i];
				});
			}
		}

		if (dL_ddLdoutput) {
			// from dL_d(dL_dx) to dL_doutput
			linear_kernel(kernel_grid_backward_input_backward_dLdoutput<T, N_POS_DIMS>, 0, stream,
				num_elements,
				m_n_features,
				// inputs
				dL_ddLdinput.view(),
				forward.dy_dx.data(),
				dL_dy_rm,
				// outputs
				dL_ddLdoutput->view()
			);
		}

		if (dL_dinput) {
			static constexpr uint32_t N_THREADS_HASHGRID = 256;
			static constexpr uint32_t N_FEATURES_PER_THREAD = std::min(2u, N_FEATURES_PER_LEVEL);

			const dim3 blocks_hashgrid = { div_round_up(num_elements * N_FEATURES_PER_LEVEL / N_FEATURES_PER_THREAD, N_THREADS_HASHGRID), m_n_levels, 1 };

			// from dL_d(dL_dx) to dL_dx
			kernel_grid_backward_input_backward_input<T, N_POS_DIMS, N_FEATURES_PER_LEVEL, N_FEATURES_PER_THREAD><<<blocks_hashgrid, N_THREADS_HASHGRID, 0, stream>>>(
				num_elements,
				m_n_features,
				m_hashmap_offsets_table.data(),
				m_resolution_table.data(),
				m_scale_table.data(),
				m_valid_level,
				// m_base_resolution,
				// std::log2(m_per_level_scale),
				this->m_quantize_threshold,
				this->m_max_level,
				this->m_max_level_gpu,
				m_interpolation_type,
				m_grid_type,
				// inputs
				dL_ddLdinput.view(),
				forward.positions.data() ? forward.positions.view() : input.view(),
				dL_dy_rm,
				use_inference_params ? m_grid_inference : m_grid,
				// outputs
				dL_dinput->view()
			);
		}
	}

	void tv_backward(
		cudaStream_t stream,
		const float loss_scale, 
		uint8_t * density_grid,
		uint32_t max_cascade,
		uint32_t nerf_cascade,
		// const uint32_t random_seed,
		default_rng_t& rng,
		const uint32_t max_num_elements = 1e7,
		bool use_inference_params = false) 
	override {

		// float return_ratio[16] = {0, 0, 0, 0, 0, 0, 0, 0,
		// 							0.2, 0.5, 0.75, 0.8, 0.85, 0.9, 0.95, 0.98};

		static constexpr uint32_t N_THREADS_HASHGRID = 256;
		static constexpr uint32_t N_FEATURES_PER_THREAD = std::min(2u, N_FEATURES_PER_LEVEL);

		// for (uint32_t i = m_n_levels/2 ; i < m_n_levels/2 + 1; ++i) {
		// for (uint32_t i = m_n_levels/2 ; i < m_n_levels; ++i) {
		// for (uint32_t i = 4 ; i < m_n_levels; ++i) {
		for (uint32_t i = 0 ; i < m_n_levels && i < m_valid_level; ++i) {
			// const float scale = exp2f(i * std::log2(m_per_level_scale)) * m_base_resolution - 1.0f;
			// const uint32_t resolution = (uint32_t)(ceilf(scale)) + 1;
			const float scale = m_scale_table_cpu[i];
			const uint32_t resolution = m_resolution_table_cpu[i];
			
			uint64_t num_elements = powi64(resolution, N_POS_DIMS);
			bool random_sample = false;
			uint32_t random_seed = rng.next_uint();
			if (num_elements > max_num_elements) {
				num_elements = max_num_elements;
				random_sample = true;
			}
			const dim3 blocks_hashgrid = { (uint32_t)div_round_up(num_elements, (uint64_t)N_THREADS_HASHGRID), 1, 1 };
			
			const uint32_t hashmap_offset = m_hashmap_offsets_table_cpu[i];
			const uint32_t hashmap_size = m_hashmap_offsets_table_cpu[i+1] - m_hashmap_offsets_table_cpu[i];
			kernel_tv_backward<T, grad_t, N_POS_DIMS, N_FEATURES_PER_LEVEL><<<blocks_hashgrid, N_THREADS_HASHGRID, 0, stream>>>(
				num_elements,
				i, // level
				// return_ratio[i],
				random_sample,
				random_seed,
				// rng,
				// m_n_features,
				hashmap_offset,
				hashmap_size,
				// m_base_resolution,
				// std::log2(m_per_level_scale),
				scale,
				resolution,
				// (float)loss_scale / (float)(resolution * resolution * resolution),
				(float)loss_scale  / (float) num_elements, // divide by number
				density_grid,
				max_cascade,
				nerf_cascade,
				m_grid_type,
				m_grid,
				(grad_t*)m_grid_gradient
			);
			rng.advance();
#if TV_LOSS_DEBUG
			GPUMemory<uint32_t> count(1);
			CUDA_CHECK_THROW(cudaMemset(count.data(), 0, sizeof(uint32_t)));
			kernel_tv_debug<T, grad_t, N_POS_DIMS, N_FEATURES_PER_LEVEL><<<blocks_hashgrid, N_THREADS_HASHGRID, 0, stream>>>(
				num_elements,
				i, // level
				return_ratio[i],
				hashmap_offset,
				hashmap_size,
				scale,
				resolution,
				(float)loss_scale,
				density_grid,
				max_cascade,
				nerf_cascade,
				m_grid_type,
				m_grid,
				(grad_t*)m_grid_gradient,
				count.data()
			);
			uint32_t count_cpu = 0;
			CUDA_CHECK_THROW(cudaMemcpy(&count_cpu, count.data(), sizeof(uint32_t) * 1, cudaMemcpyDeviceToHost));
			printf("level: %d, tv_loss grid count: %d, num_elements: %zu, ratio: %f\n", i, count_cpu, num_elements, (double)count_cpu/(double)num_elements);
			count.free_memory();
#endif
		}
		return;
	}

	uint32_t input_width() const override {
		return N_POS_DIMS;
	}

	uint32_t padded_output_width() const override {
		return m_n_padded_output_dims;
	}

	uint32_t output_width() const override {
		return m_n_padded_output_dims;
	}

	uint32_t required_input_alignment() const override {
		return 1;
	}


// #if RESAMPLE_BY_SCALE
// 	// change per_level_scale
// 	std::pair<uint32_t, uint32_t> downsample(
// 		cudaStream_t stream,
// 		uint32_t downsample_scale=2
// 	) override {
// 		float top_scale = exp2f((m_n_levels - 1) * std::log2(m_per_level_scale)) * m_base_resolution - 1.0f;
// 		uint32_t top_resolution = (uint32_t)(ceilf(top_scale)) + 1;
		
// 		// float per_level_scale_new = exp2f(std::log2((top_resolution / downsample_scale) / m_base_resolution) / (m_n_levels - 1));
// 		float per_level_scale_new = exp2f(std::log2((top_resolution / downsample_scale - 1) / m_base_resolution) / (m_n_levels - 1));
// #if RESAMPLE_KERNEL_DEBUG
// 		printf("per_level_scale_new: %f\n", per_level_scale_new);
// #endif

// 		top_scale = exp2f((m_n_levels - 1) * std::log2(per_level_scale_new)) * m_base_resolution - 1.0f;
// 		top_resolution = (uint32_t)(ceilf(top_scale)) + 1;
// 		// per_level_scale = np.exp2(np.log2(N_max * bound / N_min) / (n_levels - 1))

// 		// uint32_t base_resolution_new = m_base_resolution / downsample_scale;
// 		// if (m_base_resolution % downsample_scale != 0) {
// 		// 	throw std::runtime_error{"GridEncoding Downsample: base_resolution must be a multiple of downsample_scale"};
// 		// }
// 		m_resample_scale /= (float)downsample_scale;
// 		resample(stream, m_base_resolution, per_level_scale_new);
// 		return std::pair<uint32_t, uint32_t>(m_base_resolution, top_resolution);
// 	}

// 	std::pair<uint32_t, uint32_t> upsample(
// 		cudaStream_t stream,
// 		uint32_t upsample_scale=2
// 	) override {
// 		float top_scale = exp2f((m_n_levels - 1) * std::log2(m_per_level_scale)) * m_base_resolution - 1.0f;
// 		uint32_t top_resolution = (uint32_t)(ceilf(top_scale)) + 1;
		
// 		float per_level_scale_new = exp2f(std::log2((top_resolution * upsample_scale) / m_base_resolution) / (m_n_levels - 1));
// 		// float per_level_scale_new = exp2f(std::log2((top_resolution * upsample_scale - 1) / m_base_resolution) / (m_n_levels - 1));
// #if RESAMPLE_KERNEL_DEBUG
// 		printf("per_level_scale_new: %f\n", per_level_scale_new);
// #endif

// 		top_scale = exp2f((m_n_levels - 1) * std::log2(per_level_scale_new)) * m_base_resolution - 1.0f;
// 		top_resolution = (uint32_t)(ceilf(top_scale)) + 1;
// 		// per_level_scale = np.exp2(np.log2(N_max * bound / N_min) / (n_levels - 1))

// 		m_resample_scale *= (float)upsample_scale;
// 		if (m_resample_scale > 1) {
// 			throw std::runtime_error{"GridEncoding Upsample: base_resolution exceeds"};
// 		}
// 		resample(stream, m_base_resolution, per_level_scale_new);
// 		return std::pair<uint32_t, uint32_t>(m_base_resolution, top_resolution);
// 	}
// #else
// 	/* dont recommend to use float version, please only use it for verify */
// 	std::pair<uint32_t, uint32_t> downsample(
// 		cudaStream_t stream,
// 		float downsample_scale
// 	) override {
// 		uint32_t base_resolution_new = std::ceil((float)m_base_resolution / downsample_scale);
// 		// if (m_base_resolution % downsample_scale != 0) {
// 		// 	throw std::runtime_error{"GridEncoding Downsample: base_resolution must be a multiple of downsample_scale"};
// 		// }
// 		m_resample_scale /= ((float)m_base_resolution / (float)base_resolution_new);
// 		resample(stream, base_resolution_new, m_per_level_scale);
// 		const float top_scale = exp2f((m_n_levels - 1) * std::log2(m_per_level_scale)) * m_base_resolution - 1.0f;
// 		const uint32_t top_resolution = (uint32_t)(ceilf(top_scale)) + 1;
// 		return std::pair<uint32_t, uint32_t>(m_base_resolution, top_resolution);
// 	}

// 	/* dont recommend to use float version, please only use it for verify */
// 	std::pair<uint32_t, uint32_t> upsample(
// 		cudaStream_t stream,
// 		float upsample_scale
// 	) override {
// 		uint32_t base_resolution_new = std::ceil((float)m_base_resolution * upsample_scale);
// 		m_resample_scale /= ((float)m_base_resolution / (float)base_resolution_new);
// 		if (m_resample_scale > 1) {
// 			throw std::runtime_error{"GridEncoding Upsample: base_resolution exceeds"};
// 		}
// 		resample(stream, base_resolution_new, m_per_level_scale);
// 		const float top_scale = exp2f((m_n_levels - 1) * std::log2(m_per_level_scale)) * m_base_resolution - 1.0f;
// 		const uint32_t top_resolution = (uint32_t)(ceilf(top_scale)) + 1;
// 		return std::pair<uint32_t, uint32_t>(m_base_resolution, top_resolution);
// 	}

// 	std::pair<uint32_t, uint32_t> downsample(
// 		cudaStream_t stream,
// 		uint32_t downsample_scale=2
// 	) override {
// 		uint32_t base_resolution_new = m_base_resolution / downsample_scale;
// 		if (m_base_resolution % downsample_scale != 0) {
// 			throw std::runtime_error{"GridEncoding Downsample: base_resolution must be a multiple of downsample_scale"};
// 		}
// 		m_resample_scale /= (float)downsample_scale;
// 		resample(stream, base_resolution_new, m_per_level_scale);
// 		const float top_scale = exp2f((m_n_levels - 1) * std::log2(m_per_level_scale)) * m_base_resolution - 1.0f;
// 		const uint32_t top_resolution = (uint32_t)(ceilf(top_scale)) + 1;
// 		return std::pair<uint32_t, uint32_t>(m_base_resolution, top_resolution);
// 	}

// 	std::pair<uint32_t, uint32_t> upsample(
// 		cudaStream_t stream,
// 		uint32_t upsample_scale=2
// 	) override {
// 		uint32_t base_resolution_new = m_base_resolution * upsample_scale;
// 		m_resample_scale *= (float)upsample_scale;
// 		if (m_resample_scale > 1) {
// 			throw std::runtime_error{"GridEncoding Upsample: base_resolution exceeds"};
// 		}
// 		resample(stream, base_resolution_new, m_per_level_scale);
// 		const float top_scale = exp2f((m_n_levels - 1) * std::log2(m_per_level_scale)) * m_base_resolution - 1.0f;
// 		const uint32_t top_resolution = (uint32_t)(ceilf(top_scale)) + 1;
// 		return std::pair<uint32_t, uint32_t>(m_base_resolution, top_resolution);
// 	}
// #endif
// resample by m_resolution_table
std::pair<uint32_t, uint32_t> upsample(
		cudaStream_t stream,
		const uint8_t* density_grid,
		const uint32_t max_cascade,
		const uint32_t nerf_gridsize,
		uint32_t upsample_scale=2,
		uint32_t upsample_start_level=8
	) override {
		// float top_scale = exp2f((m_n_levels - 1) * std::log2(m_per_level_scale)) * m_base_resolution - 1.0f;
		// uint32_t top_resolution = (uint32_t)(ceilf(top_scale)) + 1;
		std::vector<uint32_t> resolution_table_cpu_new(m_resolution_table_cpu); // deep copy
		std::vector<float> scale_table_cpu_new(m_scale_table_cpu);
		
		// float per_level_scale_new = exp2f(std::log2((top_resolution / downsample_scale) / m_base_resolution) / (m_n_levels - 1));
		float per_level_scale_new = exp2f(std::log2(((float)m_resolution_table_cpu[m_n_levels-1] * (float)upsample_scale - 0.5) / (float)m_resolution_table_cpu[upsample_start_level]) / (float)(m_n_levels - upsample_start_level - 1));
#if RESAMPLE_KERNEL_DEBUG
		printf("[upsample] per_level_scale_new: %f\n", per_level_scale_new);
#endif
		for (uint32_t i = upsample_start_level + 1; i < m_n_levels; ++i) {
			// Compute dense params required for the given level
			const float scale = exp2f((i - upsample_start_level) * std::log2(per_level_scale_new)) * m_resolution_table_cpu[upsample_start_level] - 1.0f;
			// const float scale = exp2f(i * std::log2(m_per_level_scale)) * base_resolution_new - 1.0f;
			const uint32_t resolution = (uint32_t)(ceilf(scale)) + 1;
			scale_table_cpu_new[i] = resolution - 1;
			resolution_table_cpu_new[i] = resolution;
		}

		m_resample_scale *= (float)upsample_scale; // useless
		if (m_resample_scale > 1) {
			throw std::runtime_error{"GridEncoding Upsample: base_resolution exceeds"};
		}
		resample(stream, m_base_resolution, m_per_level_scale, resolution_table_cpu_new, scale_table_cpu_new, upsample_start_level, density_grid, max_cascade, nerf_gridsize);
		
		return std::pair<uint32_t, uint32_t>(m_base_resolution, m_resolution_table_cpu[m_n_levels-1]);
	}

std::pair<uint32_t, uint32_t> reset_gridlevel(
		cudaStream_t stream,
		const uint8_t* density_grid,
		const uint32_t max_cascade,
		const uint32_t nerf_gridsize
	) override {
		std::vector<uint32_t> resolution_table_cpu_new(m_n_levels); // deep copy
		std::vector<float> scale_table_cpu_new(m_n_levels);
		
		for (uint32_t i = 0; i < m_n_levels; ++i) {
			// Compute dense params required for the given level
			const float scale = exp2f(i * std::log2(m_per_level_scale)) * m_base_resolution - 1.0f;
			const uint32_t resolution = (uint32_t)(ceilf(scale)) + 1;
			scale_table_cpu_new[i] = resolution - 1;
			resolution_table_cpu_new[i] = resolution;
		}

		m_resample_scale = 1.0f;
		resample(stream, m_base_resolution, m_per_level_scale, resolution_table_cpu_new, scale_table_cpu_new, 0, density_grid, max_cascade, nerf_gridsize);
		
		return std::pair<uint32_t, uint32_t>(m_base_resolution, m_resolution_table_cpu[m_n_levels-1]);
	}

// resample by m_resolution_table
std::pair<uint32_t, uint32_t> downsample(
		cudaStream_t stream,
		const uint8_t* density_grid,
		const uint32_t max_cascade,
		const uint32_t nerf_gridsize,
		uint32_t downsample_scale=2,
		uint32_t downsample_start_level=8
	) override {
		// float top_scale = exp2f((m_n_levels - 1) * std::log2(m_per_level_scale)) * m_base_resolution - 1.0f;
		// uint32_t top_resolution = (uint32_t)(ceilf(top_scale)) + 1;
		std::vector<uint32_t> resolution_table_cpu_new(m_resolution_table_cpu); // deep copy
		std::vector<float> scale_table_cpu_new(m_scale_table_cpu);
		
		// float per_level_scale_new = exp2f(std::log2((top_resolution / downsample_scale) / m_base_resolution) / (m_n_levels - 1));
		float per_level_scale_new = exp2f(std::log2(((float)m_resolution_table_cpu[m_n_levels-1] / (float)downsample_scale - 1) / (float)m_resolution_table_cpu[downsample_start_level]) / (float)(m_n_levels - downsample_start_level - 1));
#if RESAMPLE_KERNEL_DEBUG
		printf("[downsample] per_level_scale_new: %f\n", per_level_scale_new);
#endif
		for (uint32_t i = downsample_start_level + 1; i < m_n_levels; ++i) {
			// Compute dense params required for the given level
			const float scale = exp2f((i - downsample_start_level) * std::log2(per_level_scale_new)) * m_resolution_table_cpu[downsample_start_level] - 1.0f;
			// const float scale = exp2f(i * std::log2(m_per_level_scale)) * base_resolution_new - 1.0f;
			const uint32_t resolution = (uint32_t)(ceilf(scale)) + 1;
			// scale_table_cpu_new[i] = scale;
			scale_table_cpu_new[i] = resolution - 1;
			resolution_table_cpu_new[i] = resolution;
		}

		m_resample_scale /= (float)downsample_scale; // useless
		resample(stream, m_base_resolution, m_per_level_scale, resolution_table_cpu_new, scale_table_cpu_new, downsample_start_level, density_grid, max_cascade, nerf_gridsize);
		
		return std::pair<uint32_t, uint32_t>(m_base_resolution, m_resolution_table_cpu[m_n_levels-1]);
	}

	/* author: qinhan */
	void resample(
		cudaStream_t stream,
		uint32_t base_resolution_new,
		float per_level_scale_new,
		const std::vector<uint32_t>& resolution_table_cpu_new,
		const std::vector<float>& scale_table_cpu_new,
		const uint32_t resample_start_level,
		const uint8_t* density_grid,
		const uint32_t max_cascade,
		const uint32_t nerf_gridsize
	) {
		// create new hashmap config
		std::vector<uint32_t> hashmap_offsets_table_cpu_new(m_n_levels + 1);

		uint32_t offset = 0;

		for (uint32_t i = 0; i < m_n_levels; ++i) {
			// Compute dense params required for the given level
			const float scale = scale_table_cpu_new[i];
			const uint32_t resolution = resolution_table_cpu_new[i];

			uint32_t max_params = std::numeric_limits<uint32_t>::max()/2;
			uint32_t params_in_level = std::pow((float)resolution, N_POS_DIMS) > (float)max_params ? max_params : powi(resolution, N_POS_DIMS);

			// Make sure memory accesses will be aligned
			params_in_level = next_multiple(params_in_level, 8u);

			if (m_grid_type == GridType::Dense) {
				// No-op
			} else if (m_grid_type == GridType::Tiled) {
				// If tiled grid needs fewer params than dense, then use fewer and tile.
				params_in_level = std::min(params_in_level, powi(base_resolution_new, N_POS_DIMS));
			} else if (m_grid_type == GridType::Hash) {
				// If hash table needs fewer params than dense, then use fewer and rely on the hash.
				params_in_level = std::min(params_in_level, (1u << m_log2_hashmap_size));
			} else {
				throw std::runtime_error{std::string{"GridEncoding: invalid grid type "} + to_string(m_grid_type)};
			}

			hashmap_offsets_table_cpu_new[i] = offset;
			offset += params_in_level;

#ifdef TCNN_VERBOSE_MEMORY_ALLOCS
			std::cout << "GridEncoding at level " << i << ": resolution=" << resolution << " params_in_level=" << params_in_level << std::endl;
#endif
		}

		hashmap_offsets_table_cpu_new[m_n_levels] = offset;
		uint32_t n_params_new = hashmap_offsets_table_cpu_new[m_n_levels] * N_FEATURES_PER_LEVEL;
		GPUMemory<uint32_t> hashmap_offsets_table_new(m_n_levels + 1);
		CUDA_CHECK_THROW(cudaMemcpy(hashmap_offsets_table_new.data(), hashmap_offsets_table_cpu_new.data(), (m_n_levels+1) * sizeof(uint32_t), cudaMemcpyHostToDevice));

		GPUMemory<char> grid_tmp_buffer;
		// grid_tmp_buffer.resize(sizeof(T) * n_params_new);
		grid_tmp_buffer.resize(sizeof(float) * n_params_new);
		grid_tmp_buffer.memset(0);

		static constexpr uint32_t N_THREADS_HASHGRID = 256;
		static constexpr uint32_t N_FEATURES_PER_THREAD = std::min(2u, N_FEATURES_PER_LEVEL);

		// for (uint32_t i = resample_start_level; i < m_n_levels; ++i) {
		for (uint32_t i = 0; i < m_n_levels; ++i) {
			const float scale = scale_table_cpu_new[i];
			const uint32_t resolution = resolution_table_cpu_new[i];
			
			const float scale_old = m_scale_table_cpu[i];
			const uint32_t resolution_old = m_resolution_table_cpu[i];
			
			const uint64_t num_elements = powi64(resolution, N_POS_DIMS);
			const dim3 blocks_hashgrid = { (uint32_t)div_round_up(num_elements, (uint64_t)N_THREADS_HASHGRID), 1, 1 };
			kernel_resample<T, N_POS_DIMS, N_FEATURES_PER_LEVEL><<<blocks_hashgrid, N_THREADS_HASHGRID, 0, stream>>>(
				num_elements,
				i, // level
				m_hashmap_offsets_table.data(),
				hashmap_offsets_table_new.data(),
				scale_old,
				resolution_old,
				scale,
				resolution,
				this->m_quantize_threshold,
				m_interpolation_type,
				density_grid,
				max_cascade,
				nerf_gridsize,
				m_grid_type,
				m_grid,
				(float*)grid_tmp_buffer.data()
			);
		}

		m_per_level_scale = per_level_scale_new;
		m_base_resolution = base_resolution_new;
		m_hashmap_offsets_table_cpu.swap(hashmap_offsets_table_cpu_new);
		std::vector<uint32_t>().swap(hashmap_offsets_table_cpu_new); // clear

		m_resolution_table_cpu = resolution_table_cpu_new; // deep copy
		m_scale_table_cpu = scale_table_cpu_new;
		CUDA_CHECK_THROW(cudaMemcpy(m_resolution_table.data(), m_resolution_table_cpu.data(), (m_n_levels) * sizeof(uint32_t), cudaMemcpyHostToDevice));
		CUDA_CHECK_THROW(cudaMemcpy(m_scale_table.data(), m_scale_table_cpu.data(), (m_n_levels) * sizeof(float), cudaMemcpyHostToDevice));

		CUDA_CHECK_THROW(cudaMemcpy(m_hashmap_offsets_table.data(), hashmap_offsets_table_new.data(), (m_n_levels+1) * sizeof(uint32_t), cudaMemcpyDeviceToDevice));

		CUDA_CHECK_THROW(cudaMemset(m_grid, 0, n_params() * sizeof(T)));
		CUDA_CHECK_THROW(cudaMemcpy(m_grid, grid_tmp_buffer.data(), sizeof(T) * n_params_new, cudaMemcpyDeviceToDevice));

		tcnn::linear_kernel(set_grid_value<T, float>, 0, stream, n_params_new,
					(float*)grid_tmp_buffer.data(), m_grid);
		if (m_grid_inference != m_grid) {
			CUDA_CHECK_THROW(cudaMemcpy(m_grid_inference, m_grid, sizeof(T) * n_params_new, cudaMemcpyDeviceToDevice));
		}
		grid_tmp_buffer.free_memory();
		
		// memset grid gradient
		CUDA_CHECK_THROW(cudaMemsetAsync(m_grid_gradient, 0, n_params() * sizeof(grad_t), stream));

	}

	// void cal_dy(
	// 	cudaStream_t stream,
	// 	const Context& ctx,
	// 	const GPUMatrixDynamic<float>& loss,
	// 	GPUMatrixDynamic<T>* dy,
	// 	bool use_inference_params = false,
	// 	EGradientMode param_gradients_mode = EGradientMode::Overwrite
	// ) override {
	// 	printf("cal_dy gird override succeed!\n");
	// 	const uint32_t num_elements = loss.n();

	// 	const auto& forward = dynamic_cast<const ForwardContext&>(ctx);
		
	// 	const float* dx = loss.data();

	// 	GPUMemoryArena::Allocation workspace;
	// 	if (loss.layout() == CM) {
	// 		workspace = allocate_workspace(stream, num_elements * m_n_features * sizeof(float));

	// 		// Transpose dL_dy. Use the buffer previously occupied by the encoded positions
	// 		const dim3 threads_transpose = { m_n_levels * N_FEATURES_PER_LEVEL, 8, 1 };
	// 		const uint32_t blocks_transpose = div_round_up(num_elements, threads_transpose.y);
	// 		transpose_gradients<float><<<blocks_transpose, threads_transpose, 0, stream>>>(
	// 			num_elements,
	// 			(float*)workspace.data(),
	// 			loss.pitched_ptr()
	// 		);

	// 		dx = (const float*)workspace.data();
	// 	}

		

	// 	// linear_kernel(debug_log_full<float>, 0, stream,
	// 	// 	forward.dy_dx.m() * forward.dy_dx.n(),
	// 	// 	forward.dy_dx.n(),
	// 	// 	forward.dy_dx.view());
	// 	// linear_kernel(debug_log<float>, 0, stream,
	// 	// 	forward.dy_dx.m(),
	// 	// 	0,
	// 	// 	forward.dy_dx.view());

	// 	// linear_kernel(debug_log<float>, 0, stream,
	// 	// 	forward.dy_dx.m(),
	// 	// 	1,
	// 	// 	forward.dy_dx.view());
			
	// 	// linear_kernel(debug_log<float>, 0, stream,
	// 	// 	forward.dy_dx.m(),
	// 	// 	2,
	// 	// 	forward.dy_dx.view());

	// 	// linear_kernel(debug_log<float>, 0, stream,
	// 	// 	forward.dy_dx.m(),
	// 	// 	3,
	// 	// 	forward.dy_dx.view());


			
	// 	// printf("shape:%d,%d\n",forward.dy_dx.m(),forward.dy_dx.n());
	// 	// printf("m_n_features,N_FEATURES:%d,%d\n",m_n_features,N_FEATURES);
	// 	// printf("num_elements:%d\n",num_elements);
	// 	linear_kernel(kernel_cal_dy<T, N_POS_DIMS>, 
	// 		0, 
	// 		stream,
	// 		num_elements,
	// 		forward.dy_dx.n(),
	// 		m_n_features,
	// 		dx,
	// 		forward.dy_dx.data(),
	// 		// forward.dy_dx.view(),
	// 		// (GPUMatrix<float, CM>(forward.dy_dx.data(), forward.dy_dx.m(), forward.dy_dx.n())).view(),
	// 		dy->view()
	// 	);
		
	// 	// linear_kernel(debug_log<float>, 0, stream, forward.dy_dx.m(), 0, forward.dy_dx.view());
	// 	// printf("dy\n");
	// 	// linear_kernel(debug_log<T>, 0, stream, dy->m(), 0, dy->view());
	// 	return;
	// }

	void set_alignment(uint32_t alignment) override {
		alignment = lcm(alignment, min_alignment());
		m_n_padded_output_dims = next_multiple(m_n_output_dims, alignment);
		m_n_to_pad = m_n_padded_output_dims - m_n_output_dims;
	}

	uint32_t min_alignment() const override {
		return N_FEATURES_PER_LEVEL;
	}

	MatrixLayout preferred_output_layout() const override {
		return SoA;
	}

	T* params() const override{ return m_grid; }

	T* params_inference() const override{ return m_grid_inference; }

	T* params_gradients() const override{ return m_grid_gradient; }

	T* params_backward() const override{ return nullptr; }

	void set_params(T* params, T* inference_params, T* backward_params, T* gradients) override {
		m_grid = params;
		m_grid_inference = inference_params;
		m_grid_gradient = gradients;
	}

	void initialize_params(pcg32& rnd, float* params_full_precision, T* params, T* inference_params, T* backward_params, T* gradients, float scale = 1) override {
		set_params(params, inference_params, backward_params, gradients);

		// Initialize the hashgrid from the GPU, because the number of parameters can be quite large.
		generate_random_uniform<float>(rnd, n_params(), params_full_precision, -1e-4f, 1e-4f);
	}

	size_t n_params() const override {
		return m_n_params;
	}

	size_t level_n_params(uint32_t level) const override {
		return level_params_offset(level + 1) - level_params_offset(level);
	}

	size_t level_params_offset(uint32_t level) const override {
		return m_hashmap_offsets_table_cpu.at(level);
	}

	std::vector<std::pair<uint32_t, uint32_t>> layer_sizes() const override {
		// Even though we have parameters, they can't really be considered a "layer".
		// So we return an empty array here.
		return {};
	}

	uint32_t n_pos_dims() const override {
		return N_POS_DIMS;
	}

	uint32_t n_features_per_level() const override {
		return N_FEATURES_PER_LEVEL;
	}

	json hyperparams() const override {
		json result = {
			{"otype", "Grid"},
			{"type", to_string(m_grid_type)},
			{"n_levels", m_n_levels},
			{"n_features_per_level", N_FEATURES_PER_LEVEL},
			{"base_resolution", m_base_resolution},
			{"per_level_scale", m_per_level_scale},
			{"interpolation", to_string(m_interpolation_type)},
			{"resample_scale", m_resample_scale},
		};

		if (m_grid_type == GridType::Hash) {
			result["log2_hashmap_size"] = m_log2_hashmap_size;
		}

		return result;
	}

	void set_training_step(int training_step) override { 
		m_training_step = training_step;
		if (m_training_step <= 0) { // when predicting global movement 
			m_valid_level = m_n_levels;
			return;
		}
		m_valid_level = min(m_n_levels, (uint32_t)ceil(m_base_valid_level_scale * m_n_levels + m_valid_level_scale * max(0, (int)(m_training_step - m_base_training_step))));
		#if progressive_freq_debug
			if (m_training_step % 50 == 0) {
				// printf("m_valid_level_float:%f\n",m_base_valid_level_scale * m_n_levels + m_valid_level_scale * max(0, (int)(m_training_step - m_base_training_step)));
				printf("m_training_step:%d, m_valid_level:%d\n", m_training_step, m_valid_level);
			}
		#endif
	}

	const uint32_t* hashmap_offsets_table() const override {
		return m_hashmap_offsets_table.data();
	}

	const T* grid(bool use_inference_params) const override {
		if (use_inference_params) return m_grid_inference;
		return m_grid;
	}

private:
	struct ForwardContext : public Context {
		GPUMatrix<float, RM> positions;
		GPUMatrix<float, RM> dy_dx;
	};

	uint32_t m_n_features;
	uint32_t m_n_levels;
	uint32_t m_n_params;
	std::vector<uint32_t> m_hashmap_offsets_table_cpu;
	GPUMemory<uint32_t> m_hashmap_offsets_table;
	std::vector<uint32_t> m_resolution_table_cpu;
	GPUMemory<uint32_t> m_resolution_table;
	std::vector<float> m_scale_table_cpu;
	GPUMemory<float> m_scale_table;
	uint32_t m_log2_hashmap_size;
	uint32_t m_base_resolution;

	uint32_t m_n_dims_to_pass_through;

	// derived sizes
	uint32_t m_n_output_dims;
	uint32_t m_n_padded_output_dims;
	uint32_t m_n_to_pad = 0;

	float m_per_level_scale;
	
	// progressively increase valid level
	// m_valid_level = m_base_valid_level_scale * m_n_levels + m_valid_level_scale * max(m_training_step - m_base_training_step, 0)
	uint32_t m_valid_level;
	int m_training_step;
	uint32_t m_base_training_step;
	float m_base_valid_level_scale;
	float m_valid_level_scale;

	bool m_stochastic_interpolation;
	InterpolationType m_interpolation_type;
	GridType m_grid_type;

	// Storage of params
	T* m_grid;
	T* m_grid_inference;
	T* m_grid_gradient;

	// resample factor
	float m_resample_scale = 1.0f;
};

template <typename T, uint32_t N_FEATURES_PER_LEVEL>
GridEncoding<T>* create_grid_encoding_templated(uint32_t n_dims_to_encode, const json& encoding) {
	const uint32_t log2_hashmap_size = encoding.value("log2_hashmap_size", 19u);
	const std::string encoding_type = encoding.value("otype", "Grid");
	const std::string default_type = equals_case_insensitive(encoding_type, "TiledGrid") ? "Tiled" : (equals_case_insensitive(encoding_type, "DenseGrid") ? "Dense" : "Hash");

	uint32_t n_features;
	if (encoding.contains("n_features") || encoding.contains("n_grid_features")) {
		n_features = encoding.contains("n_features") ? encoding["n_features"] : encoding["n_grid_features"];
		if (encoding.contains("n_levels")) {
			throw std::runtime_error{"GridEncoding: may not specify n_features and n_levels simultaneously (one determines the other)"};
		}
	} else {
		n_features = N_FEATURES_PER_LEVEL * encoding.value("n_levels", 16u);
	}

#define TCNN_GRID_PARAMS \
	n_features, \
	log2_hashmap_size, \
	encoding.value("base_resolution", 16u), \
	encoding.value("per_level_scale", 2.0f), \
	encoding.value("stochastic_interpolation", false), \
	string_to_interpolation_type(encoding.value("interpolation", "Linear")), \
	string_to_grid_type(encoding.value("type", default_type)), \
	encoding.value("valid_level_scale", 0.01f), \
	encoding.value("base_valid_level_scale", 0.5f), \
	encoding.value("base_training_step", 200u), \

	// If higher-dimensional hash encodings are desired, corresponding switch cases can be added
	switch (n_dims_to_encode) {
		// case 1: return new GridEncodingTemplated<T, 1, N_FEATURES_PER_LEVEL>{ TCNN_GRID_PARAMS };
		// case 2: return new GridEncodingTemplated<T, 2, N_FEATURES_PER_LEVEL,n_features>{ TCNN_GRID_PARAMS };
		// case 3: return new GridEncodingTemplated<T, 3, N_FEATURES_PER_LEVEL,n_features>{ TCNN_GRID_PARAMS };
		// case 4: return new GridEncodingTemplated<T, 4, N_FEATURES_PER_LEVEL,n_features>{ TCNN_GRID_PARAMS };
		case 2: return new GridEncodingTemplated<T, 2, N_FEATURES_PER_LEVEL>{ TCNN_GRID_PARAMS };
		case 3: return new GridEncodingTemplated<T, 3, N_FEATURES_PER_LEVEL>{ TCNN_GRID_PARAMS };
		case 4: return new GridEncodingTemplated<T, 4, N_FEATURES_PER_LEVEL>{ TCNN_GRID_PARAMS };
		// case 5: return new GridEncodingTemplated<T, 5, N_FEATURES_PER_LEVEL>{ TCNN_GRID_PARAMS };
		// case 6: return new GridEncodingTemplated<T, 6, N_FEATURES_PER_LEVEL>{ TCNN_GRID_PARAMS };
		// case 7: return new GridEncodingTemplated<T, 7, N_FEATURES_PER_LEVEL>{ TCNN_GRID_PARAMS };
		default: throw std::runtime_error{"GridEncoding: number of input dims must be 2 or 3."};
	}
#undef TCNN_GRID_PARAMS
}

template <typename T>
GridEncoding<T>* create_grid_encoding(uint32_t n_dims_to_encode, const json& encoding) {
	const uint32_t n_features_per_level = encoding.value("n_features_per_level", 2u);
	switch (n_features_per_level) {
		case 1: return create_grid_encoding_templated<T, 1>(n_dims_to_encode, encoding);
		case 2: return create_grid_encoding_templated<T, 2>(n_dims_to_encode, encoding);
		case 4: return create_grid_encoding_templated<T, 4>(n_dims_to_encode, encoding);
		case 8: return create_grid_encoding_templated<T, 8>(n_dims_to_encode, encoding);
		default: throw std::runtime_error{"GridEncoding: n_features_per_level must be 1, 2, 4, or 8."};
	}
}

TCNN_NAMESPACE_END
