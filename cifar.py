%matplotlib inline

import pickle
import random

import cv2
import pandas as pd
import numpy as np

from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from tqdm import tqdm_notebook as tqdm

import torch
import torch.nn as nn
import torchvision

from torchvision.transforms import functional as F

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio

from IPython.display import Image


def read_cifar(basedir='cifar-100-python'):
    with open(f'{basedir}/train', 'rb') as train_fp, \
         open(f'{basedir}/test', 'rb') as test_fp, \
         open(f'{basedir}/meta', 'rb') as meta_fp:
            train = pickle.load(train_fp, encoding='bytes')
            test = pickle.load(test_fp, encoding='bytes')
            meta = pickle.load(meta_fp, encoding='bytes')
    return train, test, meta


def tsne_plot(tsne_features, df, meta, class_selection, sub_selection, title):
    fig = go.Figure()
    for i in sub_selection:
        filenames = df[df['label'] == i]['filename']
        fig.add_trace(go.Scatter(
            x=tsne_features[df['label'] == i][:,0],
            y=tsne_features[df['label'] == i][:,1],
            text=[str(i) for i in filenames.values],
            name=str(meta[b'fine_label_names'][i]),
            mode='markers',
        ))

    fig.update_layout(
        title=go.layout.Title(
            text=title,
    ))

    return fig


def make_dataset(cifar, dataset_def):
    train, test, _ = cifar
    train_class_selection, n_train_per_class, knowledge_class_selection, n_knowledge_per_class = \
        dataset_def['train_class_selection'], dataset_def['n_train_per_class'], \
        dataset_def['knowledge_class_selection'], dataset_def['n_knowledge_per_class']


    all_train_df = pd.DataFrame(list(zip(train[b'data'],
                                         train[b'filenames'],
                                         train[b'fine_labels'])),
                                columns=['data', 'filename', 'label'])

    all_test_df = pd.DataFrame(list(zip(test[b'data'],
                                        test[b'filenames'],
                                        test[b'fine_labels'])),
                               columns=['data', 'filename', 'label'])

    train_df = all_train_df.groupby('label', as_index=False).apply(lambda x: x.loc[np.random.choice(x.index, n_train_per_class, False),:])
    train_images = train_df[train_df['label'].isin(train_class_selection)]['data'].values
    train_labels = train_df[train_df['label'].isin(train_class_selection)]['label'].values
    train_filenames = train_df[train_df['label'].isin(train_class_selection)]['filename'].values

    knowledge_df = all_test_df.groupby('label', as_index=False).apply(lambda x: x.loc[np.random.choice(x.index, n_knowledge_per_class, False),:])
    knowledge_images = knowledge_df[knowledge_df['label'].isin(knowledge_class_selection)]['data'].values
    knowledge_labels = knowledge_df[knowledge_df['label'].isin(knowledge_class_selection)]['label'].values
    knowledge_filenames = knowledge_df[knowledge_df['label'].isin(knowledge_class_selection)]['filename'].values

    test_images = all_test_df[~all_test_df['data'].isin(knowledge_images)]['data'].values
    test_labels = all_test_df[~all_test_df['data'].isin(knowledge_images)]['label'].values
    test_filenames = all_test_df[~all_test_df['data'].isin(knowledge_images)]['filename'].values

    return (train_images, train_labels, train_filenames), \
           (knowledge_images, knowledge_labels, knowledge_filenames), \
           (test_images, test_labels, test_filenames)


def get_unseen_classes(n_total_classes, seen_classes, n=5):
    unseen_classes = []
    while len(unseen_classes) < n:
        sample = random.randint(0, n_total_classes-1)
        if sample in seen_classes or sample in unseen_classes:
            continue
        unseen_classes.append(sample)
    return unseen_classes


def to_torch(x, normalize=None):
    resized = cv2.resize(x.reshape((3, 32, 32)).transpose().swapaxes(0, 1),
                         (224, 224),
                         interpolation=cv2.INTER_CUBIC)
    tensor = torch.from_numpy(np.moveaxis(resized / 255., -1, 0).astype(np.float32))
    if normalize:
        return F.normalize(tensor, **normalize)
    return tensor


def extract_features(loader, model_name='densenet121', layer_name='features'):
    get_model = getattr(torchvision.models, model_name)

    model = get_model(pretrained=True)
    model = model.cuda()
    model.eval()

    # register hook to access to features in forward pass
    features = []
    def hook(module, input, output):
        N,C,H,W = output.shape
        output = output.reshape(N,C,-1)
        features.append(output.mean(dim=2).cpu().detach().numpy())
    handle = model._modules.get(layer_name).register_forward_hook(hook)

    for i_batch, inputs in tqdm(enumerate(loader), total=len(loader)):
        _ = model(inputs.cuda())

    features = np.concatenate(features)

    handle.remove()
    del model

    return features


n_total_classes = 100
n_train_classes = 50
n_knowledge_classes = 5

train_class_selection = random.sample(range(n_total_classes), n_train_classes)
knowlege_class_selection = get_unseen_classes(n_total_classes, train_class_selection, n_knowledge_classes)

dataset_def = {
    'train_class_selection': train_class_selection,
    'n_train_per_class': 30,
    'knowledge_class_selection': knowlege_class_selection,
    'n_knowledge_per_class': 5
}


train, knowledge, test = make_dataset(read_cifar('../data/cifar-100-python'), dataset_def)
len(train[0]), len(train[1]), len(knowledge[0]), len(knowledge[1]), len(test[0]), len(test[1])


data = np.vstack(train[0])
normalize = {
    'mean': [data[:,  :1024].mean(), data[:, 1024:2048].mean(), data[:, 2048:].mean()],
    'std': [data[:,  :1024].std(), data[:, 1024:2048].std(), data[:, 2048:].std()]
}
images = [to_torch(i, normalize) for i in train[0]]
loader = torch.utils.data.DataLoader(images, batch_size=2, shuffle=False, num_workers=1)

features = extract_features(loader)
features.shape


tsne = TSNE(n_components=2, learning_rate=400., init='pca', random_state=42)
tsne_features = tsne.fit_transform(features)

train_df = pd.DataFrame(list(zip(*train)), columns=['data', 'label', 'filename'])
_, _, meta = read_cifar('../data/cifar-100-python')
fig = tsne_plot(tsne_features, train_df, meta, train_class_selection,
    random.sample(train_class_selection, 5), 'TSNE lr=400., normalized images, densenet121')
fig.show()

train_df['label']
