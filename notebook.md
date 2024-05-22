New Learning in LLM

VL Models Research:
- Yi-6B
- xgen-mm-phi3-mini-base-r-v1
- cogVLM2

Yi-6b

C:\Users\LH641\Documents\Dengfeng\Yi-main\VL
#Run a single_inference to test the happy path#
python single_inference.py --model-path "C:\Users\LH641\Documents\Dengfeng\Yi\model" --image-file "C:\Users\LH641\Documents\Dengfeng\Yi\VL\images" --question "Describe the image in detail"

05/22 errors in model, can not solve


xgen-mm-phi3-mini-base-r-v1

introduction:
-  pretrained foundation model: xgen-mm-phi3-mini-base-r-v1 (5b parameters)
-  instruct fine-tuned model: xgen-mm-phi3-mini-instruct-r-v1 (supports flexible high-resolution image encoding with efficient visual token sampling.)
- 4.41.0 transformers library
