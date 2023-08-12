# NeuS2: Fast Learning of Neural Implicit Surfaces for Multi-view Reconstruction

**This is an unofficial implementation.**

### [[Project]](https://vcai.mpi-inf.mpg.de/projects/NeuS2/)[ [Paper]](https://arxiv.org/abs/2212.05231)

[NeuS2](https://vcai.mpi-inf.mpg.de/projects/NeuS2/) is a method for fast neural surface reconstruction, which achieves two orders of magnitude improvement in terms of acceleration without compromising reconstruction quality, compared to [NeuS](https://lingjie0206.github.io/papers/NeuS/). To accelerate the training process, we integrate multi-resolution hash encodings into a neural surface representation and implement our whole algorithm in CUDA. In addition, we extend our method for reconstructing dynamic scenes with an incremental training strategy.

This project is an extension of [Instant-NGP](https://github.com/NVlabs/instant-ngp) enabling it to model neural surface representation and dynmaic scenes. We extended:
- dependencies/[tiny-cuda-nn](https://github.com/19reborn/my_tcnn)
  - add second-order derivative backpropagation computation for MLP;
  - add progressive training for Grid Encoding.
- neural-graphics-primitives
  - extend NeRF mode for **NeuS**;
  - add support for dynamic scenes.

Please see [Instant-NGP](https://github.com/NVlabs/instant-ngp) for original requirements and compilation instructions.


After building the requirements, use CMake to build the project:

```
cmake . -B build
cmake --build build --config RelWithDebInfo -j 
```

## Training

### Static Scene

You can specify a static scene by setting `--scene` to a `.json` file containing data descriptions.

The [DTU](https://roboimagedata.compute.dtu.dk/?page_id=36) Scan24 scene can be downloaded from [Google Drive](https://drive.google.com/file/d/1KkNkljeYNwg5dH_y080AlzslVl1RTnKy/view?usp=sharing):

```sh
./build/testbed --scene ${data_path}/transform.json
```
Or, you can run the experiment in an automated fashion through python bindings:

```sh
python scripts/run.py --mode nerf --scene ${data_path}/transform.json --name ${your_experiment_name} --network ${config_path}
```

The outputs and logs of the experiment can be found at `output/${your_experiment_name}/`.

### Dynamic Scene

To specify a dynamic scene, you should set `--scene` to a directory containing `.json` files that describe training frames.

```sh
./build/testbed --scene ${data_dirname}
```

Or, run `scripts/run_dynamic.py` using python:

```sh
python scripts/run_dynamic.py --mode nerf --scene ${data_dirname} --name ${your_experiment_name} --network ${config_path}
```

There are some hyperparameters of the network configuration, such as `configs/nerf/base.json`, to control the dynamic training process:
- `first_frame_max_training_step`: determine the number of training iterations for the first frame, default `2000`.
- `next_frame_max_training_step`: determine the number of training iterations for subsequent frames, default `1300`, including global transformation prediction.
- `predict_global_movement`: set `true` if use global transformation prediction.
- `predict_global_movement_training_step`: determine the number of training iterations for global transformation prediction, default `300`. Only valid when `predict_global_movement` is `true`.

Also, we provide scripts to reconstruct dynamic scenes by reconstructing static scene frame by frame.

```sh
python scripts/run_per_frame.py --base_dir ${data_dirname} --output_dir ${output_path} --config ${config_name}
```

Dynamic scene examples can be downloaded from [Google Drive](https://drive.google.com/file/d/1hvqaupbufxuadVMP_2reTAqnaEZ4xvhj/view?usp=sharing).

## Data Convention

Our NeuS2 implementation expects initial camera parameters to be provided in a `transforms.json` file, organized as follows:
```
{
	"from_na": true, # specify NeuS2 data format
	"w": 512, # image_width
	"h": 512, # image_height
	"aabb_scale": 1.0,
	"scale": 0.5,
	"offset": [
		0.5,
		0.5,
		0.5
	],
	"frames": [ # list of reference images & corresponding camera parameters
		{
			"file_path": "images/000000.png", # specify the image path (should be relative path)
			"transform_matrix": [ # specify extrinsic parameters of camera, a camera to world transform (shape: [4, 4])
				[
					0.9702627062797546,
					-0.01474287360906601,
					-0.2416049838066101,
					0.9490470290184021
				],
				[
					0.0074799139983952045,
					0.9994929432868958,
					-0.0309509988874197,
					0.052045613527297974
				],
				[
					0.2419387847185135,
					0.028223415836691856,
					0.9698809385299683,
					-2.6711924076080322
				],
				[
					0.0,
					0.0,
					0.0,
					1.0
				]
			],
			"intrinsic_matrix": [ # specify intrinsic parameters of camera (shape: [4, 4])
				[
					2892.330810546875,
					-0.00025863019982352853,
					823.2052612304688,
					0.0
				],
				[
					0.0,
					2883.175537109375,
					619.0709228515625,
					0.0
				],
				[
					0.0,
					0.0,
					1.0,
					0.0
				],
				[
					0.0,
					0.0,
					0.0,
					1.0
				]
			]
		},
		...
	]
}
```
Each `transforms.json` file contains data about a single frame, including camera parameters and image paths. You can specify specific transform files, such as `transforms_test.json` and `transforms_train.json`, to use for training and testing with data splitting.

For example, you can organize your dynamic scene data as:
```
<case_name>
|-- images
   |-- 000280 # target frame of the scene
      |-- image_c_000_f_000280.png
      |-- image_c_001_f_000280.png
      ...
   |-- 000281
      |-- image_c_000_f_000281.png
      |-- image_c_001_f_000281.png
      ...
   ...
|-- train
   |-- transform_000280.json
   |-- transform_000281.json
   ...
|-- test
   |-- transform_000280.json
   |-- transform_000281.json
   ...
```

Images are four-dimensional, with three channels for RGB and one channel for the mask.

A data conversion tool is provided at `tools/data_format_from_neus.py`, which can transform [NeuS](https://lingjie0206.github.io/papers/NeuS/) data to the data format used in this project.

## Citation

Respect to Wang et al. for their work, which will certainly boost neural graphics research.
```bibtex
@misc{https://doi.org/10.48550/arxiv.2212.05231,
  doi = {10.48550/ARXIV.2212.05231},
  url = {https://arxiv.org/abs/2212.05231},
  author = {Wang, Yiming and Han, Qin and Habermann, Marc and Daniilidis, Kostas and Theobalt, Christian and Liu, Lingjie},
  title = {NeuS2: Fast Learning of Neural Implicit Surfaces for Multi-view Reconstruction},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}

```

Also, if you find this repository useful, please consider citing:
```bibtex
@misc{zhu2023neus2cuda,
  title={NeuS2-Custom-Implementation},
  author={Chengxuan Zhu},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished={\url{https://github.com/FreeButUselessSoul/neus2}},
  year={2023}
}
```