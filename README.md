# R-GQA


Code for ``Retrieval-Augmented Generative Question Answering for Event Argument Extraction'' [[link](https://arxiv.org/abs/2004.13625)]


If you use our code, please cite:

	@inproceedings{R-GQA,
	    title = {Retrieval-Augmented Generative Question Answering for Event Argument Extraction},
	    author={Du, Xinya and Ji, Heng},
	    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing",
	    year = "2022",
	    publisher = "Association for Computational Linguistics",
	}
	

## Dependencies 
- sentence_transformers=2.1.0
- pytorch=1.6 
- transformers=3.1.0
- pytorch-lightning=1.0.6
- spacy=3.0 # conflicts with transformers
- pytorch-struct=0.4 


## Datasets

- ACE05 (Access from [LDC](https://catalog.ldc.upenn.edu/LDC2006T06) and preprocessing following [OneIE](http://blender.cs.illinois.edu/software/oneie/). In this repo, we provide toy example under ```./data_toy```.
- WikiEvents: The original dataset release is [here](https://github.com/raspberryice/gen-arg). 
<!--- Also available on AWS [here](s3://gen-arg-data/wikievents/))-->
<!-- - RAMS (Download at [https://nlp.jhu.edu/rams/]) -->

<!-- You can download the data through the AWS cli or AWS console. 
Alternatively, you can download individual files by 
- `wget https://gen-arg-data.s3.us-east-2.amazonaws.com/wikievents/data/<split>.jsonl` for split={train, dev,test}.
- `wget https://gen-arg-data.s3.us-east-2.amazonaws.com/wikievents/data/coref/<split>.jsonlines` for split={train, dev, test}. -->


## Train/Test on the toy dataset

- Train ```./scripts/toy_train_ace_ir_yn.sh```

- Test ```./scripts/toy_test_ace_ir_yn.sh```

- Evaluation (on toy test file)

```bash
DATA_DIR=data_toy/ace/json
CKPT_NAME=gen_ir_yn

python src/genie/scorer.py --gen-file=checkpoints/${CKPT_NAME}-pred/predictions.jsonl --dataset=ACE \
	--test-file=${DATA_DIR}/toy.test.oneie.json \
	--output-file=${DATA_DIR}/predict.toy.test.oneie.json \
```

## Trigger Extraction

In this repo, we tackle the task where the event trigger is provided.
You are welcome to use our prior implementations for EEQA [[link](https://github.com/xinyadu/eeqa)] to get trigger extraction results first.

## Model Checkpoints 

We provide ```epoch=5.ckpt``` for directly running inference on the toy data [here](https://drive.google.com/file/d/1FdjPJ1cWy4y9F0nsquYvsjG6H-iSg905/view?usp=share_link).

  

