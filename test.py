from models.sg2im import build_mlp

model = build_mlp([4, 4, 5, 6, 7, 2])
model.summary()
