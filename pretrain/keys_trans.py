import torch, timm
state = torch.load('/Users/fanyaohou/Desktop/pretrain_model/resnet50_1kpretrained_timm_style.pth', 'cpu')
rec_state = torch.load('/Users/fanyaohou/Desktop/pretrain_model/recnet50_backbone.pth', 'cpu')

print(len(rec_state))