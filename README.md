# Sentence Deletion in Document Simplification
This repo contains codes for the following paper:
```
@inproceedings{zhong2020discourse,
  title={Discourse level factors for sentence deletion in text simplification},
  author={Zhong, Yang and Jiang, Chao and Xu, Wei and Li, Junyi Jessy},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={34},
  number={05},
  pages={9709--9716},
  year={2020}
}
```
If you would like to refer to it, please cite the paper mentioned above.

# Getting Started
These instructions will introduce the dataset and get you running the codes.

## Dataset
Please contact the authors with the permits for accessing Newsela's data. Then unzip the data folder under the root directory.

## Requirements
- Python 3.6 or Higher
- Pytorch = 1.5.0
- Pandas, Numpy, Pickle, Scipy
- GLoVe Embedding (unzip glove.42B.300d.txt into current directory)

You can run
```
pip install -r requirments.txt
```
to install all dependencies.

## Run the code

Current support **FNN embedding** and **FNN with position features** settings.   
 "G7" and "G4" in ``class_to_train`` parameters refers to Middle School Sentence Deletion and Elementary School Sentence Deletion respectively.


For **FNN embedding**, run
```
python main.py -model FullNN -conc 0 -class_to_train G7 -epochs 100 -batch_size 16 -binning 10 -add_features False
```
For **FNN with position features**, run
```
python main.py -model FullNN -conc 3 -class_to_train G7 -epochs 100 -batch_size 16 -binning 10 -add_features True
```
