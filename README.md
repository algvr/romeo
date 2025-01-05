# ROMEO: Revisiting Optimization Methods for Reconstructing 3D Human-Object Interaction Models From Images

### 🏆 Best Paper Award at [T-CAP Workshop](https://sites.google.com/view/t-cap-2024/home), ECCV 2024

## [[Paper]](https://algvr.github.io/romeo/static/pdfs/romeo.pdf) [[Website]](https://algvr.github.io/romeo/)

<img src="https://algvr.github.io/romeo/static/images/method_figure.png" alt="Method Figure" style="width: 90%; margin-bottom: 8px;"/>

We present ROMEO, a method for reconstructing 3D human-object interaction models from images. Depth-size ambiguities caused by unknown object and human sizes make the joint reconstruction of humans and objects into a plausible configuration matching the observed image a difficult task. Data-driven methods struggle with reconstructing 3D human-object interaction models when it comes to unseen object categories or object shapes, due to the difficulty of obtaining sufficient and diverse 3D training data, and often even of acquiring object meshes for training. To address these challenges, we propose a novel method that does not require any manual human-object contact annotations or 3D data supervision. ROMEO integrates the flexibility of optimization-based methods and the effectiveness of foundation models with large modeling capacity in a plug-and-play fashion.


## About this repository

This repository is designed to facilitate replication of and follow-up efforts on our work ROMEO, published in [T-CAP @ ECCV 2024](https://sites.google.com/view/t-cap-2024/home). It is meant to be used with the BEHAVE and InterCap datasets, but can easily be extended to work with others.

The main files are:
- `preprocess.py`: for preprocessing the images for which to reconstruct human-object interaction scenes 
- `reconstruct.py`: for reconstructing human-object interaction scenes from the preprocessed images
- `evaluate.py`: for evaluating the reconstructions
- `config/defaults.py`: for viewing and modifying default configuration parameters

## Preparation

### Preparing environment

This repository is designed to work with Python 3.11. Please run `setup.sh` to install the required packages into the current environment.

### Downloading data

Please download this [TAR archive](https://drive.google.com/file/d/1R1ce_jHzOa72rS26nSZVtNh_daxI9X76/view?usp=sharing) and extract it into the repository's root directory to prepare .

This repository currently supports the the [BEHAVE](https://virtualhumans.mpi-inf.mpg.de/behave/) and [InterCap](https://intercap.is.tue.mpg.de/) datasets. You can download them at their respective project pages. For ease of use with our framework, extract the datasets into `data/original/behave/` and `data/original/intercap/`, respectively. Note that to follow 
our evaluation protocol on BEHAVE, only captures from Date 3 are needed. Similarly, for evaluation on InterCap, only a subset of sequences from Subjects 09 and 10 are needed.

### Preprocessing data

To automatically reconstruct the PARE parameters of the person in the image, as well as predict object instance masks and depth maps for a dataset, please run:

```
python3 preprocess.py --dataset=<dataset_name> --input_dir=<dataset_dir>
```

Provide `--input_dir=intercap_keyframes` to select the frames we used to evaluate on InterCap.

## Reconstruction

After preparing the data as described above, please run:

```
python3 reconstruct.py --dataset=<dataset_name> --input_dir=<directory_with_preprocessed_images>
```

Provide `--input_dir=data/preprocessed/<dataset_name>/cropped_rgb/` to use the images produced by the previous preprocessing step. Alternatively, provide `--input_dir=intercap_keyframes` to select the frames we used to evaluate on InterCap.
See `reconstruct.py` for documentation on customization options.

## Evaluation

To evaluate the reconstructions obtained with our framework, please run:
```
python3 evaluate.py --input_dir=<directory_with_reconstructions>
```

# Acknowledgements

This project was made possible by the following prior works, to which the authors wish to express their gratitude:

- [lang-segment-anything](https://github.com/luca-medeiros/lang-segment-anything)
- [PARE](https://github.com/mkocabas/PARE/)
- [PHOSA](https://jasonyzhang.com/phosa/)
- [ZoeDepth](https://github.com/isl-org/ZoeDepth)

# Citation

If you found our work useful, please consider citing it:

```
@inproceedings{
  gavryushin2024romeo,
  title={ROMEO: Revisiting Optimization Methods for Reconstructing 3D Human-Object Interaction Models From Images},
  author={Gavryushin, Alexey and Liu, Yifei and Huang, Daoji and Kuo, Yen-Ling and Valentin, Julien and van Gool, Luc and Hilliges, Otmar and Wang, Xi},
  booktitle={T-CAP Workshop at ECCV 2024},
  year={2024},
  url={https://romeo.github.io/}
}
```

