The repository is based on [SparseGPT](https://github.com/IST-DASLab/sparsegpt) code 

## Dependencies

* `torch`: tested on v2.2.1
* `transformers`: tested on v4.35.2
* `datasets`: tested on v2.16.1

## Usage

The simplest way to try this out is to run the following command:

```
python inference_demo.py --execution_mode 1 --compressed_model_path elvircrn/llama2-7b-double-sparse-sparsity0.7-wikitext2-final --pretrained_model_path <Llama-2-7b-hf_path>
```
