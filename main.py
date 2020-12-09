import os
import numpy as np
import tensorflow as tf
import json
import random
import math
from models.sg2im import Sg2ImModel
from PIL import Image, ImageDraw
import argparse


# ---------------- sample function ----------------
def sample(prefix):
    layout_height = 64.
    layout_width = 64.

    colormap = ['#000000','#0000ff', '#00ff00', '#00ffff', '#ff0000', '#ff00ff', '#ffff00']

    test_json = args.sample_json
    scene_graphs = json.load(open(test_json))
    
    objs, triples = model.encode_scene_graphs(scene_graphs)

    objs = tf.convert_to_tensor(objs)
    triples = tf.convert_to_tensor(triples)

    bbox_pred = model(objs, triples).numpy()

    idx = 0
    group_objs = []
    group_bbox = []
    temp_objs = []
    temp_bbox = []

    while idx < len(objs):
        obj = objs[idx]
        bbox = bbox_pred[idx]
        temp_objs.append(obj)
        temp_bbox.append(bbox)

        if obj == 0:
            group_objs.append(temp_objs)
            group_bbox.append(temp_bbox)
            temp_objs = []
            temp_bbox = []

        idx += 1
    
    for i, group in enumerate(group_objs):
        canva = Image.fromarray(np.zeros((int(layout_height), int(layout_width))))
        canva = canva.convert('RGB')
        draw = ImageDraw.Draw(canva)
        for j, category_idx in enumerate(group):
            category_idx = category_idx.numpy()
            if category_idx == 0:
                continue

            box = group_bbox[i][j]

            x0, y0, x1, y1 = box

            x0 = int(x0 * layout_width)
            x1 = int(x1 * layout_width)

            y0 = int(y0 * layout_height)
            y1 = int(y1 * layout_height)
            
            draw.rectangle([x0, y0, x1, y1], outline=colormap[category_idx])
        
        canva = canva.convert('RGB')
        canva.save(os.path.join(args.output_dir, '%s_%d.png' % (prefix, i)))
        print('[Sample] image %s_%d.png saved.' % (prefix, i))


# ---------------- training one step ----------------
def train_step(layouts_json, is_training=True):
    """[summary]

    Args:
        layouts_json (tf.Tensor(dtype=string), batch_size * 1): A batch of layouts' file path
    """

    layout_height = 64.
    layout_width = 64.

    # -------------------------------------------------------------
    #
    # 1. combine the objects in one batch into a big graph
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

            # check for relationship
            sx0, sy0, sx1, sy1 = cur_boxes[s]
            ox0, oy0, ox1, oy1 = cur_boxes[o]
            d0 = obj_centers[s][0] - obj_centers[o][0]
            d1 = obj_centers[s][1] - obj_centers[o][1]
            theta = math.atan2(d1, d0)
            
            # TODO: should not hard code relationships
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
    T = all_triples.shape[0]

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


# ---------------- training pipeline ----------------
def train():
    if not os.path.isdir(args.output_dir):
            os.makedirs(args.output_dir)

    if not os.path.isdir(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    # ---------------- dataset ----------------
    dataset = tf.data.Dataset.list_files(os.path.join(args.data_dir, '*.json'))
    dataset = dataset.batch(batch_size=args.batch_size)
    
    # TODO: add code of restoring
    if args.checkpoint_path is not None:
        # model.restore(args.checkpoint_path)
        pass

    epoch = 0
    iter_cnt = 0
    iter_every_epoch = len(dataset)

    while iter_cnt < args.num_iter:
        for file_batch in dataset:
            loss = train_step(file_batch)

            iter_cnt += 1

            if iter_cnt % args.print_every == 0:
                print(np.mean(loss))
                print('Epoch: {:.2f}, Iteration: {:6d}. Loss: {:.4f}'.format(
                    iter_cnt / iter_every_epoch, iter_cnt, np.mean(loss)))

        epoch += 1

        if epoch % args.checkpoint_every:
            ckpt_manager.save()
            sample('train_%s' % epoch)


# ---------------- test pipeline ----------------
def test():
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
    
    ckpt.restore(args.checkpoint_path)
    sample('test')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # basic configuration
    parser.add_argument('--data_dir', default=None)
    parser.add_argument('--mode', default='test', choices=['test', 'train'])

    # model configuration
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--num_iter', default=1e+6, type=float)
    parser.add_argument('--embedding_dim', default=128, type=int)
    parser.add_argument('--gconv_dim', default=128, type=int)
    parser.add_argument('--gconv_hidden_dim', default=512, type=int)
    parser.add_argument('--gconv_num_layers', default=5, type=int)
    parser.add_argument('--mlp_normalization', default='none')
    parser.add_argument('--normalization', default='batch')
    parser.add_argument('--activation', default='leakyrelu-0.2')

    # print and ckpt configuration
    parser.add_argument('--print_every', default=10, type=int)
    parser.add_argument('--checkpoint_every', default=1e4, type=int)
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--checkpoint_dir', default=None)
    parser.add_argument('--checkpoint_max_to_keep', default=20, type=int)
    parser.add_argument('--checkpoint_path', default=None)
    parser.add_argument('--sample_json', default='./test/test.json')

    args = parser.parse_args()

    if args.mode not in ['train', 'test']:
        raise ValueError('unknown mode' % args.mode)

    # ---------------- vocabulary ----------------
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


    # ---------------- models ----------------
    model_kwargs = {
        'vocab': vocab,
        'embedding_dim': args.embedding_dim,
        'gconv_dim': args.gconv_dim,
        'gconv_hidden_dim': args.gconv_hidden_dim,
        'gconv_num_layers': args.gconv_num_layers,
        'mlp_normalization': args.mlp_normalization,
        'normalization': args.normalization,
        'activation': args.activation,
    }
    model = Sg2ImModel(**model_kwargs)


    # ---------------- optimizer ----------------
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)


    # ---------------- loss function ----------------
    def box_loss(bbox_pred, bbox):
        loss_box = tf.keras.losses.MSE(bbox_pred, bbox)
        return loss_box


    # ---------------- checkpoint manager ----------------
    ckpt = tf.train.Checkpoint(
        optimizer=optimizer,
        model=model
    )

    ckpt_manager = tf.train.CheckpointManager(
        ckpt, args.checkpoint_dir, max_to_keep=args.checkpoint_max_to_keep)

    pipeline_kwargs = {
        'model': model,
        'optimizer': optimizer,
        'ckpt': ckpt,
        'ckpt_manager': ckpt_manager
    }

    # -------------- start ---------------
    if args.mode == 'test':
        print('Start testing...')
        test()
    elif args.mode == 'train':
        print('Start training...')
        train()
