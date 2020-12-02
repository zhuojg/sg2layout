import os
import numpy as np
import tensorflow as tf
import json
import random
import math
from models.sg2im import Sg2ImModel

# ---------------- define parameters ----------------
data_dir = './data/magazine'
layout_width = 64.
layout_height = 64.

batch_size = 32
num_iterations = 1e+6
learning_rate = 1e-4
eval_mode_after = 1e+5

num_train_samples = None
num_val_samples = 1024

embedding_dim = 128
gconv_dim = 128
gconv_hidden_dim = 512
gconv_num_layers = 5
mlp_normalization = 'none'
normalization = 'batch'
activation = 'leakyrelu-0.2'

print_every = 10
timing = False
checkpoint_every = 1e4
output_dir = './output'
checkpoint_dir = './ckpt'
checkpoint_max_to_keep = 20
restore_from_checkpooint = False

if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

if not os.path.isdir(checkpoint_dir):
    os.mkdir(checkpoint_dir)

# ---------- end -----------

# define dataset
dataset = tf.data.Dataset.list_files(os.path.join(data_dir, '*.json'))
dataset = dataset.batch(batch_size=batch_size)

# ---------------- define parameters ----------------
vocab = {
    'object_name_to_idx': {},
    'pred_name_to_idx': {},
}
vocab['object_name_to_idx']['__image__'] = 0
vocab['object_name_to_idx']['text over image'] = 1
vocab['object_name_to_idx']['image'] = 2
vocab['object_name_to_idx']['background'] = 3
vocab['object_name_to_idx']['header'] = 4
vocab['object_name_to_idx']['header over image'] = 5
vocab['object_name_to_idx']['text'] = 6

# build object_idx_to_name
name_to_idx = vocab['object_name_to_idx']
assert len(name_to_idx) == len(set(name_to_idx.values()))
max_object_idx = max(name_to_idx.values())
idx_to_name = ['NONE'] * (1 + max_object_idx)
for name, idx in vocab['object_name_to_idx'].items():
    idx_to_name[idx] = name
vocab['object_idx_to_name'] = idx_to_name

vocab['pred_idx_to_name'] = [
    '__in_image__',
    'left of',
    'right of',
    'above',
    'below',
    'inside',
    'surrounding',
]

# build pred_name_to_idx
vocab['pred_name_to_idx'] = {}
for idx, name in enumerate(vocab['pred_idx_to_name']):
    vocab['pred_name_to_idx'][name] = idx

# ---------- end -----------

# build models
kwargs = {
    'vocab': vocab,
    'embedding_dim': embedding_dim,
    'gconv_dim': gconv_dim,
    'gconv_hidden_dim': gconv_hidden_dim,
    'gconv_num_layers': gconv_num_layers,
    'mlp_normalization': mlp_normalization,
    'normalization': normalization,
    'activation': activation,
}
model = Sg2ImModel(**kwargs)

# define optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# define loss


def box_loss(bbox_pred, bbox):
    loss_box = tf.keras.losses.MSE(bbox_pred, bbox)
    return loss_box


# define checkpoint
ckpt = tf.train.Checkpoint(
    optimizer=optimizer,
    model=model
)
ckpt_manager = tf.train.CheckpointManager(
    ckpt, checkpoint_dir, max_to_keep=checkpoint_max_to_keep)


# define sample function
def sample(epoch):
    test_json = './test/test.json'
    scene_graphs = json.load(open(test_json))

    print(len(scene_graphs))
    
    objs, triples = model.encode_scene_graphs(scene_graphs)

    objs = tf.convert_to_tensor(objs)
    triples = tf.convert_to_tensor(triples)

    print(objs)

    bbox_pred = model(objs, triples)

    # TODO: draw bbox

# define training step
def train_step(layouts_json, is_training=True):
    """[summary]

    Args:
        layouts_json (tf.Tensor(dtype=string), batch_size * 1): A batch of layouts' file path
    """

    # -------------------------------------------------------------
    #
    # 1. we combine the obj in a batch into a big graph
    #
    # -------------------------------------------------------------
    obj_offset = 0
    # list of category index for all the objects
    all_obj = []

    # list of coordinates of bounding box for all the objects
    all_boxes = []

    # list of random relations between objects in every layout
    # every item is [s, p ,o]
    # s: index in all_obj
    # p: index of relationship
    # o: index in all_obj
    all_triples = []

    for item in layouts_json:
        layout = json.load(open(item.numpy()))
        cur_obj = []
        cur_boxes = []
        for category in layout.keys():
            for obj in layout[category]:
                all_obj.append(vocab['object_name_to_idx'][category])
                cur_obj.append(all_obj[-1])

                x0, y0, x1, y1 = obj
                all_boxes.append(tf.convert_to_tensor(
                    [x0 / layout_width, y0 / layout_height, x1 / layout_width, y1 / layout_height], dtype=tf.float32))
                cur_boxes.append(all_boxes[-1])

        # at the end of one layout add __image__ obj
        all_obj.append(vocab['object_name_to_idx']['__image__'])
        cur_obj.append(all_obj[-1])
        all_boxes.append(tf.convert_to_tensor([0, 0, 1, 1], dtype=tf.float32))
        cur_boxes.append(all_boxes[-1])

        # compute centers of obj in current layout
        obj_centers = []
        for box in cur_boxes:
            x0, y0, x1, y1 = box
            obj_centers.append([(x0 + x1) / 2, (y0 + y1) / 2])

        # calculate triples
        # every triple is [s, p, o]
        # where s and o is the index of objs in all_obj, not the category index
        # and p is the index of relationship
        whole_image_idx = vocab['object_name_to_idx']['__image__']

        for obj_index, obj in enumerate(cur_obj):
            if obj == whole_image_idx:
                continue

            other_obj = [obj_idx for obj_idx, obj in enumerate(cur_obj) if (
                obj_idx != obj_index and obj != whole_image_idx)]

            if len(other_obj) == 0:
                break

            other = random.choice(other_obj)
            if random.random() > 0.5:
                s, o = obj_index, other
            else:
                s, o = other, obj_index

            # check for inside / surrounding
            sx0, sy0, sx1, sy1 = cur_boxes[s]
            ox0, oy0, ox1, oy1 = cur_boxes[o]
            d0 = obj_centers[s][0] - obj_centers[o][0]
            d1 = obj_centers[s][1] - obj_centers[o][1]
            theta = math.atan2(d1, d0)

            if sx0 < ox0 and sx1 > ox1 and sy0 < oy0 and sy1 > oy1:
                p = 'surrounding'
            elif sx0 > ox0 and sx1 < ox1 and sy0 > oy0 and sy1 < oy1:
                p = 'inside'
            elif theta >= 3 * math.pi / 4 or theta <= -3 * math.pi / 4:
                p = 'left of'
            elif -3 * math.pi / 4 <= theta < -math.pi / 4:
                p = 'above'
            elif -math.pi / 4 <= theta < math.pi / 4:
                p = 'right of'
            elif math.pi / 4 <= theta < 3 * math.pi / 4:
                p = 'below'
            p = vocab['pred_name_to_idx'][p]

            all_triples.append([s + obj_offset, p, o + obj_offset])

        # Add __in_image__ triples
        O = len(cur_obj)
        in_image = vocab['pred_name_to_idx']['__in_image__']
        for i in range(O - 1):
            all_triples.append([i + obj_offset, in_image, O - 1 + obj_offset])

        obj_offset += O

    all_obj = tf.convert_to_tensor(all_obj)
    all_boxes = tf.convert_to_tensor(all_boxes)
    all_triples = tf.convert_to_tensor(all_triples)

    O = all_obj.shape[0]
    T = all_obj.shape[0]

    # split triples, s, p and o all have size (T, 1)
    s, p, o = tf.split(all_triples, num_or_size_splits=3, axis=1)
    s, p, o = [tf.squeeze(x, axis=1) for x in [s, p, o]]

    edges = tf.stack([s, o], axis=1)

    # -------------------------------------------------------------
    #
    # 2. run model, calculate loss and optimize model
    #
    # -------------------------------------------------------------
    with tf.GradientTape(persistent=True) as tape:
        bbox_pred = model(all_obj, all_triples)
        loss = box_loss(bbox_pred, all_boxes)

    if is_training:
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(
            zip(gradients, model.trainable_variables)
        )

    return loss


# define train function
def train():
    # TODO: add code of restoring
    epoch = 0
    iter_cnt = 0
    iter_every_epoch = len(dataset)

    sample(0)

    exit(0)

    for file_batch in dataset:
        loss = train_step(file_batch)

        iter_cnt += 1

        # if iter_cnt == eval_mode_after:
        #     print('switching to eval mode')
        #     # TODO: switch to eval mode

        if iter_cnt % print_every == 0:
            print(np.mean(loss))
            print('Epoch: {:.2f}, Iteration: {:6d}. Loss: {:.4f}'.format(
                iter_cnt / iter_every_epoch, iter_cnt, np.mean(loss)))

    epoch += 1

    if epoch % checkpoint_every:
        ckpt_manager.save()
        sample(epoch)


if __name__ == '__main__':
    train()
