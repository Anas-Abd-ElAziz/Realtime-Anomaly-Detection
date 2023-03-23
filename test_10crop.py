import matplotlib.pyplot as plt
import torch
from sklearn.metrics import auc, roc_curve, precision_recall_curve
import numpy as np

def test(dataloader, model, args,device, root):
    with torch.no_grad():
        model.eval()
        pred = torch.zeros(0, device=device)

        input = np.load(root + "features/sample.npy")
        input = torch.from_numpy(np.expand_dims(input,axis=0)).to(device).permute(0,2,1,3)
        score_abnormal, score_normal, feat_select_abn, feat_select_normal, feat_abn_bottom, feat_select_normal_bottom, logits, \
            scores_nor_bottom, scores_nor_abn_bag, feat_magnitudes = model(inputs=input)
            
        return logits

