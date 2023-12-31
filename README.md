# DP-LLM

Saad's crash course to LLM training with Differential Privacy enabled gradient optimizers.

## Getting started

1. Run the `distilgpt2_generate.py` python script to generate a sample senetence using the pretrained `distilgpt2` model. The script takes a single argument enclosed within quotation marks.  

Example command:

```shell

python python-source/distilgpt2_generate.py "I would like to"
```

<!-- OUTDATED -->
<!-- 1. Download the .tgz sample data file from: [ADD LINK HERE]
1. Extract the .tgz file using the command  
   `tar -xvzf /path/to/file`
2. Run the `print_data_samples.py` python script to visualize the data samples. The file takes **1** argument: [path/to/data/file.csv].
3. Run the `preprocess_data.py` python script to preprocess data into the correct format for `distilgpt2` model. The python script takes **1** argument: [path/to/data/file.csv].     -->

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
