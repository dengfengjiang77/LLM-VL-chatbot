New Learning in LLM


INTRODUCTION:
useful tips for using LLM model




# VL Models Research:
## - Yi-6B
### Company: 01-ai
### Download: 16g/Completed
### Happy Path: Failed maybe cuz by the GPU storage
## xgen-mm-phi3-mini-base-r-v1
### Company: Salesforce
### Download: 20g/Completed
### Happy Path: not start
## cogVLM2
### Company: THUDM
### Download: 40g/not start
### Happy Path: not start
## OpenGVLab/Mini-InternVL-Chat-2B-V1-5
### Company: OpenGVLab，Shanghai
### Download: 4g/ongoing
### Happy Path: not start



# # Yi-6b

**Run a Demo for the Model**
    open anaconda environment
    $activate myenv2 
    $cd C:\Users\LH641\Documents\Dengfeng\LLM\Yi\VL   
    #Run a single_inference to test the happy path#
    $python single_inference.py --model-path "C:\Users\LH641\Documents\Dengfeng\LLM\Yi\model" --image-file "C:\Users\LH641\Documents\Dengfeng\LLM\Yi\VL\images\cats.jpg" --question "Describe the image in detail"

05/22 errors in model, need check the version of pytorch, 2.30, for CUDA?
05/30 errors, You can't move a model that has some modules offloaded to cpu or disk. I think reason is the GPU limitation on my laptop

xgen-mm-phi3-mini-base-r-v1

introduction:
-  pretrained foundation model: xgen-mm-phi3-mini-base-r-v1 (5b parameters)
-  instruct fine-tuned model: xgen-mm-phi3-mini-instruct-r-v1 (supports flexible high-resolution image encoding with efficient visual token sampling.)
- 4.41.0 transformers library




Useful command:
    $activate xxx       # activate in anaconda environment
    $pip list           # list all the installed software in your environment    
    $nvidia-smi         # check the setting and status of your local GPU
    $pip install xxx    # install in python environment
    $conda install xxx  # install in conda environment
    $git clone xxx      # example : git clone https://github.com/01-ai/Yi.git