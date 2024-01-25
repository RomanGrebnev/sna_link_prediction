# Group project for Social Network Analysis course. Topic 11: Link prediction.

## Group members

- Grégoire de Lambertye - 12202211
- Zsombor Iszak - 11709501
- Roman Grebnev - 12202120
- Emile Johnston - 12229987
- Maximilian Maul - 11818418

## Project description

The goal of this project is to explore the approaches for link prediction task on "Der Standard" data. 

## Structure of the repository

Please note! We don't provide data files in git repository owing to it's large size. Please put the data files in the data folder as shown below before running the code.

```
.
├── data                                            # data folder
│   ├── Postings_01052019_15052019.csv              # postings data
│   ├── Postings_16052019_31052019.csv              # postings data
│   ├── Votes_01052019_15052019.csv                 # votes data
│   └── Votes_16052019_31052019.csv                 # votes data
├── notebooks                                       # folder with notebooks and scripts for data preprocessing and utility code
│   ├── notebooks_exploratory_data_analysis             # folder with notebooks for exploratory data analysis
|   |   └── ...
│   ├── data_loaders                                # folder with data loaders for deep learning approaches
|   |   └── ...
│   ├── graphsage                                   # folder with graph sage implementation
|   |   └── ...
│   ├── light_gcn                                   # folder with light gcn implementation
|   |   └── ...
│   ├── ndl_similarity                              # folder with non-deep learning similarity based approaches implementations
|   |   └── ...
│   ├── dl1_v1_graphsage_custom_sampling.ipynb      # graph sage training notebook using custom sampling
│   ├── dl1_v2_graphsage_heterodata.ipynb           # graph sage training notebook using pyG sampling
│   ├── dl2_lightgcn.ipynb                          # light gcn training notebook
│   ├── graphs_definitions.ipynb                    # experiments with graph definitions
│   ├── ndl_dataprep_dataloader.ipynb               # data preparation for non-deep learning approaches
│   ├── ndl_modeling.ipynb                          # non-deep learning approaches modeling
│   ├── ndl_result_visualization.ipynb              # non-deep learning approaches result visualization
|   └── utils.py                                    # utility functions
conda_environment.yml                               # conda environment file
.gitignore                                          # gitignore file
```

## How to run the code

For convenience we provide a conda environment file with all the dependencies. To install the environment run the following command:

```conda env create -f environment.yml```

Please note! The environment was tested on Linux machine with Ubuntu 22.04.3 LTS. In case there are some issues with installation of the environment, please contact us. 

After the environment is installed, activate it:

```conda activate sna_link_prediction```

Installed environment can be used further to run the code in the notebooks.
