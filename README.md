# factoid_QA_with_distant_spervision

Codes for [Factoid Question Answering With Distant Supervision](http://www.mdpi.com/1099-4300/20/6/439/pdf). 

I am cleaning the codes for uploading, and some description should be added. 



### Requirements
- GPU and CUDA 8 are required
- python >=3.5 
- pytorch 0.3.0
- numpy
- pandas
- msgpack
- spacy 1.x
- cupy
- pynvrtc

### Download Data
Please download data files from [google drive](https://drive.google.com/drive/folders/1EI47PfmeZRfpAUdNq2EI7um_sxlV8prv?usp=sharing), and put the files under the "dat" file. 
Specifically, download these four files, 
```
questions_dis_data_150htmls_using_abstext.txt
triple_weight_by_search.txt
new_mined_paraphrase0124.txt
WebQA.v1.0.tar.gz   # is it proper to upload this dataset? 
```
Then unzip the WebQA data with ```tar -zxvf WebQA.v1.0.tar.gz```. 

### Model training
Train the model via runing 

```
cd DSRC
python train_model.py
```

Please refer to ```parameters.py``` for configuration details, where ```train_idx``` is consponding to different experimental configurations in the paper. 

### Automatic training data generation via distant supervision 
Besides the generated training data, we also released the data used to generate the training data, training sample selection and ming the distant paraphrases. 

#### Training data generation via distant supervision
Coming soon. 

#### Training sample selection 

#### Distant paraphrase Minging


### Credits
Autor of sru: [Tao Lei](https://github.com/taolei87/sru).

Author of the Document Reader model: [Danqi Chen](https://github.com/danqi).

Author of the original Pytorch implementation: [Runqi Yang](https://hitvoice.github.io/about/). 

Most of the pytorch model code is borrowed from [Facebook/ParlAI](https://github.com/facebookresearch/ParlAI/) under a BSD-3 license.

