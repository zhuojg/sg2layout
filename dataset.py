import os
from PIL import Image
from scipy import ndimage
import numpy as np
import tensorflow as tf
import json
import random
import math


class MagazineDataLoader:
    def _build_vocab(self):
        self.vocab = {
            'object_name_to_idx': {},
            'pred_name_to_idx': {},
        }

        # build object_name_to_idx
        # magazine dataset labels start from 1
        self.vocab['object_name_to_idx']['__image__'] = 0
        self.vocab['object_name_to_idx']['text over image'] = 1
        self.vocab['object_name_to_idx']['image'] = 2
        self.vocab['object_name_to_idx']['background'] = 3
        self.vocab['object_name_to_idx']['header'] = 4
        self.vocab['object_name_to_idx']['header over image'] = 5
        self.vocab['object_name_to_idx']['text'] = 6

        # build object_idx_to_name
        name_to_idx = self.vocab['object_name_to_idx']
        assert len(name_to_idx) == len(set(name_to_idx.values()))
        max_object_idx = max(name_to_idx.values())
        idx_to_name = ['NONE'] * (1 + max_object_idx)
        for name, idx in self.vocab['object_name_to_idx'].items():
            idx_to_name[idx] = name
        self.vocab['object_idx_to_name'] = idx_to_name

        # build pred_idx_to_name
        self.vocab['pred_idx_to_name'] = [
            '__in_image__',
            'left of',
            'right of',
            'above',
            'below',
            'inside',
            'surrounding',
        ]

        # build pred_name_to_idx
        self.vocab['pred_name_to_idx'] = {}
        for idx, name in enumerate(self.vocab['pred_idx_to_name']):
            self.vocab['pred_name_to_idx'][name] = idx

    def __init__(self, json_files_path='./data/magazine'):
        dataset = tf.data.Dataset.list_files(os.path.join(json_files_path, '*.json'))

        self._build_vocab()

        def _process_layout_item(layout_path):
            f = open(layout_path)
            layout = json.load(f)

            layout_data = []

            for category in layout.keys():
                for item in layout[category]:
                    temp_data = {
                        'category_id': self.vocab['object_name_to_idx'][category],
                        'bbox': item
                    }
                    layout_data.append(temp_data)

            objs, boxes = [], []

            WW = HH = 64.

            for object_data in layout_data:
                objs.append(object_data['category_id'])

                x0, y0, x1, y1 = object_data['bbox']
                boxes.append(tf.convert_to_tensor(
                    [x0 / WW, y0 / HH, x1 / WW, y1 / HH], dtype=tf.float32))

            objs.append(self.vocab['object_name_to_idx']['__image__'])
            boxes.append(tf.convert_to_tensor([0, 0, 1, 1], dtype=tf.float32))

            # compute centers of all objects
            obj_centers = []
            for i, obj_idx in enumerate(objs):
                x0, y0, x1, y1 = boxes[i]
                obj_centers.append([(x0 + x1) / 2, (y0 + y1) / 2])
            obj_centers = tf.convert_to_tensor(obj_centers, dtype=tf.float32)

            triples = []
            __image__ = self.vocab['object_name_to_idx']['__image__']

            for item_idx, item in enumerate(objs):
                choices = [obj for obj in range(len(objs)) if (
                    obj != item_idx and obj != __image__)]
                if len(choices) == 0:
                    break

                other = random.choice(choices)
                if random.random() > 0.5:
                    s, o = item_idx, other
                else:
                    s, o = other, item_idx
                
                # check for inside / surrounding
                sx0, sy0, sx1, sy1 = boxes[s]
                ox0, oy0, ox1, oy1 = boxes[o]
                d = obj_centers[s] - obj_centers[o]
                theta = math.atan2(d[1], d[0])

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
                p = self.vocab['pred_name_to_idx'][p]
                triples.append([s, p, o])

            # Add __in_image__ triples
            O = len(objs)
            in_image = self.vocab['pred_name_to_idx']['__in_image__']
            for i in range(O - 1):
                triples.append([i, in_image, O - 1])

            objs = tf.convert_to_tensor(objs, dtype=tf.float32)
            boxes = tf.convert_to_tensor(boxes, dtype=tf.float32)
            triples = tf.convert_to_tensor(triples, dtype=tf.float32)

            return {'objs': objs, 'boxes': boxes, 'triples': triples}

        dataset = dataset.map(_process_layout_item)

        for item in dataset:
            print(item)
            break

        # convert layout_list to tf dataset
        # self.dataset = tf.data.Dataset.from_tensor_slices(layout_list)

    def __len__(self):
        return self.length


if __name__ == '__main__':
    dataset = MagazineDataLoader()
