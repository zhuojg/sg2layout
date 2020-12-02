import math
import tensorflow as tf
from models.graph import GraphTripleConv, GraphTripleConvNet
from models.layers import build_mlp


class Sg2ImModel(tf.keras.Model):
    def __init__(self, vocab, image_size=(64, 64), embedding_dim=64, gconv_dim=128, gconv_hidden_dim=512, gconv_pooling='avg', gconv_num_layers=5, normalization='batch', activation='leakyrelu-0.2', mlp_normalization='none', **kwargs):
        super(Sg2ImModel, self).__init__()

        if len(kwargs) > 0:
            print('WARNING: unexpected kwargs: ', kwargs)

        self.vocab = vocab
        self.image_size = image_size

        num_objs = len(vocab['object_idx_to_name'])
        num_preds = len(vocab['pred_idx_to_name'])

        self.obj_embeddings = tf.one_hot(num_objs + 1, depth=embedding_dim)
        self.pred_embeddings = tf.one_hot(num_preds, depth=embedding_dim)

        if gconv_num_layers == 0:
            # input: embedding_dim
            # output: gconv_dim
            self.gconv = tf.keras.layers.Dense(gconv_dim)
        elif gconv_num_layers > 0:
            gconv_kwargs = {
                'input_dim': embedding_dim,
                'output_dim': gconv_dim,
                'hidden_dim': gconv_hidden_dim,
                'pooling': gconv_pooling,
                'mlp_normalization': mlp_normalization,
            }
            self.gconv = GraphTripleConv(**gconv_kwargs)

        self.gconv_net = None
        if gconv_num_layers > 1:
            gconv_kwargs = {
                'input_dim': gconv_dim,
                'hidden_dim': gconv_hidden_dim,
                'pooling': gconv_pooling,
                'num_layers': gconv_num_layers - 1,
                'mlp_normalization': mlp_normalization,
            }
            self.gconv_net = GraphTripleConvNet(**gconv_kwargs)

        box_net_dim = 4
        box_net_layers = [gconv_dim, gconv_hidden_dim, box_net_dim]
        self.box_net = build_mlp(box_net_layers, batch_norm=mlp_normalization)

    def call(self, objs, triples):
        O = objs.shape[0]
        T = objs.shape[0]

        # split triples, s, p and o all have size (T, 1)
        s, p, o = tf.split(triples, num_or_size_splits=3, axis=1)
        # squeeze, so the result size is (T,)
        s, p, p = [tf.squeeze(x, axis=1) for x in [s, p ,o]]
        
        # `edges` has shape (T, 2)
        edges = tf.stack([s, o], axis=1)

        obj_vecs = self.obj_embeddings(objs)
        obj_vecs_orig = obj_vecs
        pred_vecs = self.pred_embeddings(p)

        if isinstance(self.gconv, tf.keras.layers.Dense):
            obj_vecs = self.gconv(obj_vecs)
        else:
            obj_vecs, pred_vecs = self.gconv(obj_vecs, pred_vecs, edges)
        
        if self.gconv_net is not None:
            obj_vecs, pred_vecs = self.gconv_net(obj_vecs, pred_vecs, edges)
        
        boxes_pred = self.box_net(obj_vecs)

        return boxes_pred

    def encode_scene_graphs(self, scene_graphs_json):
        objs = None
        triples = None

        return objs, triples

    def call_json(self, scene_graphs_json):
        objs, triples = self.encode_scene_graphs(scene_graphs_json)

        return self.call(objs, triples)
