import json
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch
from torchvision import transforms
from collections import Counter


def load_cocostuff(datadir, label, 
        split_npz=os.path.join('data', 'cocostuff_split_inds.npz'), 
        env_labels='', batch_size=32, no_cuda=False, 
        weighted_sampler=0, reweight_type=''):
    # Quick and dirty way to get COCO-Stuff loaders.
    # For more options, see cclu.prepare_cocostuff_minibatching.
    # datadir: where the COCO-Stuff data is stored: should have images and
    #   annotations-json as subdirectories.
    # label: the name of an object (e.g. "car", "surfboard").
    # split_npz: where the COCO-Stuff splits are stored.
    # env_labels: a comma-separated list of the object names which will define
    #   four environments (e.g. "car,road", "surfboard,sea").
    # batch_size: size of minibatches.
    # no_cuda: True to not use CUDA.
    # weighted_sampler: set to something larger than 0 to do undersampling -
    #   higher is stronger undersampling.
    # reweight_type: either class or envs - dictates what values guide
    #   undersampling.
    cat_map = np.load(os.path.join('data', 'category_map.npy'), allow_pickle=True).item()
    cat_id_list = list(np.load(os.path.join('data', 'category_id_list.npy')))
    label_ind = cat_id_list.index(cat_map[label])
    label_function = lambda x: x[:, label_ind:label_ind + 1]

    if env_labels != '':
        env_labels = env_labels.split(',')
        env_inds = [cat_id_list.index(cat_map[lab]) for lab in env_labels]
        def env_function(x):
            env_data = x[env_inds] > 0
            envs = (env_data[:, 0] * 2 + env_data[:, 1]).reshape([-1, 1])
            return envs
    else:
        env_function = None

    train_loader, valid_loader, test_loader = prepare_cocostuff_minibatching(
            datadir, batch_size, label_function, no_cuda=no_cuda, 
            split_npz=split_npz, use_10k=False, environment_function=env_function, 
            weighted_sampler=weighted_sampler, reweight_type=reweight_type)

    return train_loader, valid_loader, test_loader 

class CocostuffDataset(Dataset):

    def __init__(self, dat, images, tforms):
        self.dat = dat
        self.images = images
        self.to_tensor_transform = transforms.ToTensor()
        self.transforms = tforms

    def __len__(self):
        return len(self.dat['image_ids'])

    def __getitem__(self, idx):
        image_id = self.dat['image_ids'][idx]
        image_fname = self.images[image_id]['file_name']
        image = Image.open(image_fname).convert('RGB')
        image = self.to_tensor_transform(image)
        image = self.transforms(image)
        return (image,
                self.dat['labels'][idx], 
                self.dat['areas'][idx],
                self.dat['image_ids'][idx]
                )

class CocostuffLoader(object):

    def __init__(self, loader, name, no_cuda=False, environment_function=None,
            environment_counter=None, class_counts=None):
        self.loader = loader
        self.iter_loader = iter(self.loader)
        self.name = name
        self.no_cuda = no_cuda
        self.environment_function = environment_function
        # counts of examples in each environment across the training set
        self.environment_counter = environment_counter
        # counts of examples in each class across the training set
        self.class_probs = device(torch.Tensor(
                np.array([(class_counts[k] / 
                    sum([class_counts[j] for j in class_counts.keys()]))
                for k in sorted(class_counts.keys())]) 
                ), self.no_cuda)
        if not self.environment_counter is None:
            self.env_probs = device(torch.Tensor(
                    np.array([(self.environment_counter[k] / 
                        sum([self.environment_counter[j] 
                            for j in self.environment_counter.keys()]))
                    for k in sorted(self.environment_counter.keys())]) 
                    ), self.no_cuda)

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.loader)

    def __next__(self):
        try:
            data, labels, areas, image_ids = next(self.iter_loader)
        except StopIteration:
            self.iter_loader = iter(self.loader)
            raise StopIteration

        # other_info will be ordered [areas, environments, image_ids]
        # depending on which are specified
        if not self.environment_function is None:
            envs = torch.Tensor(self.environment_function(areas))
            other_info = device(
                    torch.cat([envs, image_ids.reshape([-1, 1])], dim=1), self.no_cuda)
        else:
            other_info = device(image_ids, self.no_cuda).reshape([-1, 1])

        data = device(data, self.no_cuda)
        labels = device(labels, self.no_cuda)
        areas = device(areas, self.no_cuda)
        return data, labels, areas, other_info


def merge_cocostuff_jsons(annotations_file_train_things,
            annotations_file_val_things,
            annotations_file_train_stuff,
            annotations_file_val_stuff):

    # The COCO-Stuff dataset annotations come in several json files. These
    # reference the original training and validation sets. Since COCO was/is
    # a competition, the original test set is not available. So in this 
    # function, we merge the original training and validation jsons, in order
    # to later make our own training/validation/test split. We also merge
    # the thing and stuff annotations, since we do not draw a distinction here.

    with open(annotations_file_train_things, 'r') as f:
        annotations_train_things = json.load(f)
    with open(annotations_file_val_things, 'r') as f:
        annotations_val_things = json.load(f)
    with open(annotations_file_train_stuff, 'r') as f:
        annotations_train_stuff = json.load(f)
    with open(annotations_file_val_stuff, 'r') as f:
        annotations_val_stuff = json.load(f)

    # adjust file paths
    for i in range(len(annotations_train_things['images'])):
        annotations_train_things['images'][i]['file_name'] = os.path.join('train2017', 
                annotations_train_things['images'][i]['file_name'])
    for i in range(len(annotations_val_things['images'])):
        annotations_val_things['images'][i]['file_name'] = os.path.join('val2017',
                annotations_val_things['images'][i]['file_name'])

    # merge the original train and valid thing annotations
    annotations_things = {}
    annotations_things['images'] = (annotations_train_things['images'] + 
            annotations_val_things['images'])
    annotations_things['annotations'] = (annotations_train_things['annotations'] + 
            annotations_val_things['annotations'])
    annotations_things['categories'] = annotations_train_things['categories']

    # merge the original train and valid stuff annotations
    annotations_stuff = {}
    annotations_stuff['annotations'] = (list(filter(lambda anno: anno['category_id'] != 183, 
                annotations_train_stuff['annotations'])) +
            list(filter(lambda anno: anno['category_id'] != 183, 
                annotations_val_stuff['annotations'])))
    annotations_stuff['categories'] = list(filter(lambda cat: cat['name'] != 'other',
                annotations_train_stuff['categories']))

    # merge the thing and stuff annotations
    annotations = {}
    annotations['images'] = annotations_things['images']
    annotations['annotations'] = (annotations_things['annotations'] + 
            annotations_stuff['annotations'])
    annotations['categories'] = (annotations_things['categories'] + 
            annotations_stuff['categories'])
    return annotations

            
def prepare_cocostuff_minibatching(
        datadir, 
        batch_size, 
        label_function,
        no_cuda=False,
        split_npz=None,
        use_10k=False,
        environment_function=None,
        weighted_sampler=0,
        reweight_type=''
        ):
   
    if not use_10k:
        # if we are using the full dataset, need to merge separate jsons
        annotations_file_train_things = os.path.join(datadir, 'annotations-json/instances_train2017.json')
        annotations_file_val_things = os.path.join(datadir, 'annotations-json/instances_val2017.json')
        annotations_file_train_stuff = os.path.join(datadir, 'annotations-json/stuff_train2017.json')
        annotations_file_val_stuff = os.path.join(datadir, 'annotations-json/stuff_val2017.json')
        annotations = merge_cocostuff_jsons(annotations_file_train_things,
                annotations_file_val_things,
                annotations_file_train_stuff,
                annotations_file_val_stuff)
    else:
        # if we are using the small version, just need one json
        annotations_file = os.path.join(datadir, 'annotations-json/cocostuff-10k-v1.1.json')
        with open(annotations_file, 'r') as f:
            annotations = json.load(f)

    # create mapping from image id to image data
    images = {
        image['id']: image for image in annotations['images']
    }

    # fill in mapping with annotations and file paths
    imgdir = os.path.join(datadir, 'images')
    for image_id in images:
        images[image_id]['annotations'] = []
        images[image_id]['file_name'] = os.path.join(imgdir, 
            images[image_id]['file_name'])

    for anno in annotations['annotations']:
        anno_image_id = anno['image_id']
        images[anno_image_id]['annotations'].append(anno)

    # get sorted list of image ids
    image_id_list = np.array(sorted(list(images.keys())))

    # create label arrays for categories present + area taken up
    category_id_list = sorted(list(set(
        [c['id'] for c in annotations['categories']])))

    # aggregate annotations (labels and areas)
    labels = np.zeros((len(image_id_list), len(category_id_list)))
    areas = np.zeros((len(image_id_list), len(category_id_list)))
    for i, image_id in enumerate(image_id_list):
        image_size = images[image_id]['height'] * images[image_id]['width']
        category_in_image_list = [anno['category_id'] for anno in images[image_id]['annotations']]
        # we normalize areas by image size - area is in [0, 1]
        category_areas_in_image_list = [anno['area'] / image_size for anno in images[image_id]['annotations']]
        
        category_indices = list(set([category_id_list.index(c) for c in category_in_image_list]))
        labels[i, category_indices] = 1
        
        for j, area_amount in enumerate(category_areas_in_image_list):
            category_j_index = category_id_list.index(category_in_image_list[j])
            areas[i, category_j_index] += area_amount

    # get the labels from the categories we specified
    labels = label_function(labels)

    # split up image ids into training, validation, test
    dat = {}
    ntrain = len(image_id_list)

    if split_npz is None:
        raise Exception('No file containing splits was loaded!')
    else:
        # split_npz contains 3 tensors of ints, representing the image ids
        # in the train/valid/test set
        split_info = np.load(split_npz)
        test_ids, valid_ids, train_ids = (split_info['test_ids'], 
                split_info['valid_ids'], split_info['train_ids'])

    for split_name, split_ids in [
            ('train', train_ids), ('valid', valid_ids), ('test', test_ids)]:
        split_inds = np.array([np.where(image_id_list == img_id)[0][0]
            for img_id in split_ids])
        dat[split_name] = {
            'image_ids': split_ids,
            'labels': labels[split_inds],
            'areas': areas[split_inds]
        }

    # get counts of examples in each environment
    if not environment_function is None:
        environments = environment_function(dat['train']['areas'])
        env_counter = Counter(environments.flatten())
    else:
        env_counter = None

    # create loaders using image file list + and label/side info vectors
    loaders = {}

    tforms_test = torch.nn.Sequential(
        transforms.Resize((321, 321))
        )
    tforms_test = torch.jit.script(tforms_test)
    tforms = {'train': tforms_test,
            'valid': tforms_test,
            'test': tforms_test}

    class_counts = Counter(dat['train']['labels'].flatten())
    for split in dat:
        dataset = CocostuffDataset(dat[split], images, tforms[split])
        shuffle = (split == 'train')

        if split == 'train' and weighted_sampler > 0 and dat[split]['labels'].shape[1] == 1:
            if reweight_type == 'class':
                class_weights = np.array([(1 / class_counts[k]) ** weighted_sampler
                    for k in sorted(class_counts.keys())]) 
                sample_weights = class_weights[dat[split]['labels'].astype(int)].flatten()
            elif reweight_type == 'envs':
                env_weights = np.array([(1 / env_counter[k]) ** weighted_sampler
                    for k in sorted(env_counter.keys())]) 
                sample_weights = env_weights[environments.astype(int)].flatten()
            num_samples = len(dataset)
            sampler = WeightedRandomSampler(sample_weights, num_samples, replacement=True)
            shuffle=False
        else:
            sampler = None

        base_loader = torch.utils.data.DataLoader(
                dataset=dataset,
                batch_size=min(batch_size, len(dataset)),
                drop_last=False,
                shuffle=shuffle,
                sampler=sampler
                )
        loader = CocostuffLoader(
                base_loader, 
                split,
                no_cuda=no_cuda,
                environment_function=environment_function,
                environment_counter=env_counter,
                class_counts=class_counts,
                )
        loaders[split] = loader
    return loaders['train'], loaders['valid'], loaders['test']

def device(x, no_cuda):
    # helper function: apply .cuda() to tensor or not
    if no_cuda:
        return x
    else:
        return x.cuda()

