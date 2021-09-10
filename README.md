# Identifying and Benchmarking Natural Out-of-Context Prediction Problems

This repository contains data and code for the paper **"Identifying and Benchmarking Natural
Out-of-Context Prediction Problems"** by David Madras and Rich Zemel, including instructions for running the NOOCh benchmark.

> ## Abstract
>  Deep learning systems frequently fail at out-of-context (OOC) prediction, a problem of paramount importance in practical applications. However, most benchmarks for OOC prediction feature restricted or synthetically-created shifts, thus only measuring performance on a narrow range of realistic OOC problems. In this work, we introduce novel methods for leveraging annotations to automatically identify candidate OOC examples in existing computer vision datasets. Utilizing two contrasting notions of context, we present NOOCh: a suite of naturally-occurring "challenge sets'', intended for evaluating a model's OOC performance. Experimentally, we evaluate and compare several robust learning methods on these challenge sets, identifying opportunities for further research in OOC prediction.

## Why NOOCh?
NOOCh (Naturally-Occuring Out-of-context Challenge sets) is a collection of images automatically identified within the [COCO-Stuff](https://github.com/nightrome/cocostuff) dataset. In each image, some object of interest's presence (or absence) is "out-of-context" (OOC). This yields a set of _hard positives_ and _hard negatives_ for each of 12 objects of interest.

There are two versions of this benchmark, each corresponding to a different notion of context. In the first version, an image is deemed OOC due to the presence or absence of useful context cues - this version of the benchmark is called NOOCh-CE (Co-occurence/Extractibility - see the paper for details). In the second version, an image is deemed OOC due to an unusual overall gist of the scene as determined by a word embedding - this version of the benchmark is called NOOCh-Gist. Here, we show examples of hard positives (top row) and negatives (bottom) row from the "kite", "sports\_ball" and "surfboard" tasks in NOOCh-CE.

![example nooch images, showing hard positive and negative examples for the kite, sports ball, and surfboard tasks](images/nooch_example_images.png "Examples of hard positive and negatives from three NOOCh tasks")

The metric that we found most interesting for evaluating model robustness on this dataset was AUC-Hard - a model's AUC on all hard examples (positive and negative included). However, we encourage others to look at a range of metrics; evaluation or robustness is complicated and one size does not necessarily fit all.

"Nooch" is a common slang term for [nutritional yeast](https://en.wikipedia.org/wiki/Nutritional_yeast), a popular food product which is an excellent source of many B vitamins! We hope that, just like adding some nooch to your food, adding some NOOCh to your model evaluations can help to make them stronger and more healthy :)

 
## Using NOOCh
Running your model on this benchmark doesn't require all the code in this repo! Just follow these steps.
1. Download COCO-Stuff (see below).
2. Create a folder `data` in your current working directory, and copy `cocostuff_split_image_ids.npz`, `category_id_list.npy` and `category_map.npy` into it.
3. Copy the folder `nooch` with all of its contents into your current working directory.
4. Copy `src/utils/cocostuff_loading_utils.py` and `src/utils/nooch.py` into your current working directory (or somewhere you can import from it).
5. Get data loaders from `src/load_cocostuff` .
6. Train your model and evaluate.

If `COCODIR` is the name of the directory you stored COCO-Stuff in (step 1), and you are interested in the `car` task from NOOCh, then your code could look something like this:
```
from cocostuff_loading_utils import load_cocostuff
import nooch

train_loader, valid_loader, test_loader = load_cocostuff(COCODIR, "car")

for images, labels, areas, other_info in train_loader: # or valid_loader
	# train

all_labels, all_logits, all_image_ids = [], [], []
for images, labels, areas, other_info in test_loader:
	all_labels.append(labels)
	all_logits.append(model(images))
	all_image_ids.append(other_info[:, -1]) # example ids
	
	
all_labels, all_logits, all_image_ids = (torch.cat(all_labels).detach().cpu().numpy(),
					torch.cat(all_logits).detach().cpu().numpy()
					 torch.cat(all_image_ids).detach().cpu().numpy(), 
					 )


print('AUC (Hard examples): {:.3f}'.format(
	nooch.evaluate('car', all_labels, all_logits, all_image_ids, 
			metric='auc', subgroup='hard', criterion='gist'))
```
`areas` contains the percentage of each image taken up by each object category (indexed by `category_id_list.npy`), and `other_info` contains example ids for each image (as well as environments, if they have been defined).

## Prequisites for using NOOCh
- Python 3.6.8
- Numpy 1.19.4
- Torch 1.7.1
- Torchvision 0.8.2
- Pillow 8.0.1
- Scikit-learn 0.0
Note: if installing torch fails, try something like: `pip install torch==1.7.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html`.

## Downloading COCO-Stuff
The COCO-Stuff dataset can be downloaded from [here](https://github.com/nightrome/cocostuff#downloads). The images and some of the annotations are originally from the COCO dataset, which does not make its test set available, so you'll need to download what's listed as the "train" and "validation" data - these will be merged in our code to create our own train/validation/test splits.

```
# Download everything
wget --directory-prefix=downloads http://images.cocodataset.org/zips/train2017.zip
wget --directory-prefix=downloads http://images.cocodataset.org/zips/val2017.zip
wget --directory-prefix=downloads http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuff_trainval2017.zip
wget --directory-prefix=downloads http://images.cocodataset.org/annotations/annotations_trainval2017.zip

# Unpack everything
mkdir -p dataset/images
unzip downloads/train2017.zip -d dataset/images/
unzip downloads/val2017.zip -d dataset/images/
unzip downloads/annotations_trainval2017.zip -d dataset/
unzip downloads/stuff_trainval2017.zip -d dataset/annotations/

# Rename directory
mv dataset/annotations dataset/annotations-json
```

Or follow these steps:
1. Create a directory where the dataset will be stored (we'll call it $COCODIR here).
2. Download images ([train2017.zip](http://images.cocodataset.org/zips/train2017.zip) and [test2017.zip](http://images.cocodataset.org/zips/val2017.zip)). These are quite large (19GB total).
3. Unzip both image files and place the resulting folders under $COCODIR/images.
4. Download JSON-style annotations for "things" ([annotations_trainval2017.zip](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)) - these are from COCO as well.
5. Download annotations for "stuff" ([stuff_trainval2017.zip](http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuff_trainval2017.zip)).
6. Unzip all annotations and place the resulting JSON files under $COCODIR/annotations-json.

The final folder structure should look like:
 - $COCODIR
     * images  
         * train2017
         * val2017
     * annotations-json
         * instances_train2017.json   
         * instances_val2017.json  
         * stuff_train2017.json   
         * stuff_val2017.json  
         * etc.

## Running the Code

To run a model on the `car` task, where `$COCODIR` is the dataset you stored COCO-Stuff in, and `$RESULTS_DIR` is where you want to store all the experimental results:

#### Approaches Not Requiring Side Information
ERM:
```python src/run_model.py --method=erm --labels=car --datadir=$COCODIR --results_dir=$RESULTS_DIR```

Label reweighting with reweighting parameter = 1:
```python src/run_model.py --method=erm --reweighted_erm --reweight_exponent=1 --reweight_type=class --labels=car --datadir=$COCODIR --results_dir=$RESULTS_DIR```

Label undersampling with undersampling parameter = 1:
```python src/run_model.py --method=erm  --weighted_sampler=1.0 --reweight_type=class  --labels=car --datadir=$COCODIR --results_dir=$RESULTS_DIR```

Focal loss with eta = 1:
```python src/run_model.py --method=erm  --focal_loss --focal_eta=1.0 --labels=car --datadir=$COCODIR --results_dir=$RESULTS_DIR```


#### Approaches Requiring Side Information (i.e. environment-based)

Assuming 4 environments defined as (car, road), (no car, road), (car, no road), (no car, no road) - this type of environment assignment is the default and the only one available through load_cocostuff(...). 

GDRO with adjustment parameter = 1:
```python src/run_model.py --method=gdro --make_environments --environments=car,road --gdro_adjustment=1 --labels=car --datadir=$COCODIR --results_dir=$RESULTS_DIR```

IRM with penalty parameter = 1:
```python src/run_model.py --method=irm --make_environments --environments=car,road --irm_penalty_coefficient=1 --labels=car --datadir=$COCODIR --results_dir=$RESULTS_DIR```

Environment reweighting with reweighting parameter = 1:
```python src/run_model.py --method=erm --make_environments --environments=car,road --reweighted_erm --reweight_exponent=1 --reweight_type=env --labels=car --datadir=$COCODIR --results_dir=$RESULTS_DIR```

Environment undersampling with undersampling parameter = 1:
```python src/run_model.py --method=erm  --make_environments --environments=car,road  --weighted_sampler=1.0 --reweight_type=envs  --labels=car --datadir=$COCODIR --results_dir=$RESULTS_DIR```
