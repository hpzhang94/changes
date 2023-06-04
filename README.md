# CHANGES
Source code for ACL 2023 CODI workshop paper: Contrastive Hierarchical Discourse Graph for Scientific Document Summarization



## Installation

Input the command for environment setup:

```
pip install -r requirement.txt
```



## Data

Please download the dataset at:



## Train

```
python train_e2e.py --model_save_path [model save path] 
		--data_path [data save path]
		--data_name [pubmed or arxiv] 
		--hidden [hidden size] 
		--lr [lr] 
		--dropout [dropout rate] 
		--sepochs [epoch num] 
		--train_c [the flag to indcate if the model will update contrast model's parameter]
		--cweight_path [contrast model parameter path] 
                --sweight_path [summarization model parameter path]
```



## Test

```
python test.py --data_path  [data save path]
	 --data_name [pubmed or arxiv] 
	 --hidden [hidden size] 
	 --cweight_path [contrast model parameter path] 
	 --sweight_path [summarization model parameter path]
```

## References

```

@misc{zhang2023contrastive,
      title={Contrastive Hierarchical Discourse Graph for Scientific Document Summarization}, 
      author={Haopeng Zhang and Xiao Liu and Jiawei Zhang},
      year={2023},
      eprint={2306.00177},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
