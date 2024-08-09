import torch.nn as nn
import argparse
# from utils import Logger
import torch
import torch.nn.functional as F


# def count_parameters(logger: Logger, model: nn.Module):
#     """Print no. of parameters in model, no. of traininable parameters,
#      no. of parameters in each dtype."""
#     total = sum(p.numel() for p in model.parameters())
#     trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     logger.write('\nModel details:')
#     logger.write('# parameters: '+str(total))
#     logger.write('# trainable parameters: '+str(trainable)+', '\
#                  +str(100*trainable/total)+'%')
#
#     dtypes = {}
#     for _, p in model.named_parameters():
#         dtype = p.dtype
#         if dtype not in dtypes:
#             dtypes[dtype] = 0
#         dtypes[dtype] += p.numel()
#     logger.write('#params by dtype:')
#     for k, v in dtypes.items():
#         logger.write(str(k)+': '+str(v)+', '+str(100*v/total)+'%')


class TimeSeriesModel(nn.Module):

    def binary_cls_final(self, logits, labels):
        if labels is not None:
            return F.binary_cross_entropy_with_logits(logits, labels, 
                                    pos_weight=self.pos_class_weight)
        else:
            return F.sigmoid(logits)
        
    def forecast_final(self, ts_emb, forecast_values, forecast_mask):
        pred = self.forecast_head(ts_emb) # bsz, features_num
        return (forecast_mask*(pred-forecast_values)**2).sum()/forecast_mask.sum()
