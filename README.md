# mturk_dualStop_analysis

Code and Data to replicate analyses investigating interference of SSRT across dual tasks for Bissett et al. 2021.
  
To rerun, download the repo and either A) change the `project_dir` in `dual_data_utils.py` (line 534), or B) when rerunning the notebooks, include the `project_dir` argument when calling `make_clean_concat_data` towards the top of each notebook.
  
Notebooks are organized into the type of analysis applied (discovery or validation), and the dataset the analysis is applied to (discovery, validation, or all).
  
`discovery_analysis_dataset-discovery.ipynb` runs the qa and analysis performed on the discovery dataset during the exploration phase of this project.
  
`validation_analysis_dataset-validation.ipynb` runs the preregistered analysis on the validation dataset.
  
`validation_analysis_dataset-all.ipynb` runs the post-hoc analysis excluding short SSD subjects on the combined dataset.
  
`display_tables.ipynb` reads in the tables saved out by the above notebooks, cleans them, and displays them for use in the paper.