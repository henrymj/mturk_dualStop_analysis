# mturk_dualStop_analysis

Code and Data to replicate analyses investigating interference of SSRT across dual tasks for the paper "Towards a taxonomy of inhibition-related processes: A dual-task approach", by Patrick G. Bissett, Henry M. Jones, McKenzie P. Hagen, Tung Bui, Jamie K. Li, James M. Shine, & Russell A Poldrack (2021).
  
To rerun, download the repo and change the path to the raw data in raw_data_path.txt.
  
Notebooks are organized into the type of analysis applied (discovery or validation), and the dataset the analysis is applied to (discovery, validation, or all).
  
`analysis-discovery_dataset-discovery.ipynb` runs the qa and analysis performed on the discovery dataset during the exploration phase of this project.
  
`analysis-validation_dataset-validation.ipynb` runs the preregistered analysis on the validation dataset.
  
`analysis-validation_dataset-all.ipynb` runs the post-hoc analysis excluding short SSD subjects on the combined dataset.
  
`display_tables.ipynb` reads in the tables saved out by the above notebooks, cleans them, and displays them for use in the paper.