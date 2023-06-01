from tinygrad.tensor import Tensor
from tinygrad.helpers import getenv

def train_resnet():
  # TODO: Resnet50-v1.5
  pass

def train_retinanet():
  from pycocotools.coco import COCO
  from datasets.openimages import openimages, iterate
  from models.retinanet import RetinaNet
  from models.resnet import ResNeXt50_32X4D
  import numpy as np

  coco = COCO(openimages())
  model = RetinaNet(ResNeXt50_32X4D())
  from tinygrad.jit import TinyJit


  # TODO: replace this with proper normalization
  input_mean = Tensor([0.485, 0.456, 0.406]).reshape(1, -1, 1, 1)
  input_std = Tensor([0.229, 0.224, 0.225]).reshape(1, -1, 1, 1)
  def input_fixup(x):
    x = x.permute([0,3,1,2]) / 255.0
    x -= input_mean
    x /= input_std
    return x

  mdlrun = TinyJit(lambda x: model(input_fixup(x)).realize())

  # TODO: make this proper batch size
  bs = 8
  epochs = 1

  # TODO: finish training loop for RetinaNet
  for epoch in range(epochs):
    for x, targets in iterate(coco, bs):
      print(x.shape)
      dat = Tensor(x.astype(np.float32))

      if dat.shape[0] == bs:
        out = mdlrun(dat).numpy()
      else:
        mdlrun.jit_cache = None
        out =  model(input_fixup(dat)).numpy()

      predictions = model.postprocess_detections(out, input_size=dat.shape[1:3], orig_image_sizes=[t["image_size"] for t in targets])
      
      # TODO: loss function, optimizer, etc.
      print(predictions.shape)

  pass

def train_unet3d():
  # TODO: Unet3d
  pass

def train_rnnt():
  # TODO: RNN-T
  pass

def train_bert():
  # TODO: BERT
  pass

def train_maskrcnn():
  # TODO: Mask RCNN
  pass

if __name__ == "__main__":
  Tensor.training = True

  for m in getenv("MODEL", "resnet,retinanet,unet3d,rnnt,bert,maskrcnn").split(","):
    nm = f"train_{m}"
    if nm in globals():
      print(f"training {m}")
      globals()[nm]()


