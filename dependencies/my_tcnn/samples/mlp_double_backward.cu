#include <tiny-cuda-nn/common.h>


using namespace tcnn;
using precision_t = network_precision_t;

nlohmann::json pos_encoding = {
		"otype": "HashGrid",
		"n_levels": 16,
		"n_features_per_level": 2,
		"log2_hashmap_size": 19,
		"base_resolution": 16
}
// std::shared_ptr<tcnn::Encoding<T>> m_pos_encoding = tcnn::create_encoding<T>(n_pos_dims, pos_encoding, density_network.contains("otype") && (tcnn::equals_case_insensitive(density_network["otype"], "FullyFusedMLP") || tcnn::equals_case_insensitive(density_network["otype"], "MegakernelMLP")) ? 16u : 8u);
std::shared_ptr<tcnn::Encoding<T>> m_pos_encoding = tcnn::create_encoding<T>(
		3, pos_encoding, 16u);
		// 3, pos_encoding, density_network.contains("otype") && (tcnn::equals_case_insensitive(density_network["otype"], "FullyFusedMLP") || tcnn::equals_case_insensitive(density_network["otype"], "MegakernelMLP")) ? 16u : 8u);

nlohmann::json local_density_network_config = {
		"otype": "FullyFusedMLP",
		"activation": "ReLU",
		"output_activation": "None",
		"n_neurons": 64,
		"n_hidden_layers": 1
}

local_density_network_config["n_input_dims"] = m_pos_encoding->padded_output_width();
if (!density_network.contains("n_output_dims")) {
	local_density_network_config["n_output_dims"] = 16;
}
std::unique_ptr<tcnn::Network<T>> m_density_network = tcnn::create_network<T>(local_density_network_config);

auto forward = std::make_unique<ForwardContext>();
forward->density_network_input = tcnn::GPUMatrixDynamic<T>{m_pos_encoding->padded_output_width(), batch_size, stream, m_pos_encoding->preferred_output_layout()};
