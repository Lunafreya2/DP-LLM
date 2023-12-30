# DP-LLM

Saad's crash course to LLM training with Differential Privacy enabled gradient optimizers.

## Getting started

1. Download the .tgz sample data file from: [ADD LINK HERE]
1. Extract the .tgz file using the command  
   `tar -xvzf /path/to/file`
1. Run the `print_data_samples.py` file to visualize the data samples. The file takes **1** argument: [path/to/data/file.csv].

## Plan so far

- [ ] Get ChatGPT to generate fake data for prelim. testing.
  - [ ] Define the labels.  
  - [ ] Define the prompt for limiting the scope of chatGPT responses.
  - [ ] Get the final csv file
- [ ] Get the base distillGPT2 trained (reproduce the online examples).
  - [ ] Find an appropriate example online.  
  - [ ] Reproduce the example locally.  
  - [ ] Experiment as needed.  
- [ ] Attach privacy engine.
  - [ ] Use the newly released private transformers library to enable fine-tuning with DP.  
