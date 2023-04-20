
from models import XVLMBase, build_mlp, load_pretrained

from models.xbert import BertConfig
from models.xroberta import RobertaConfig

import torch
from torch import nn
import torch.nn.functional as F
class MultiTask(XVLMBase):

    def __init__(self, config):
        config_text = RobertaConfig.from_json_file(config['text_config']) \
            if config['use_roberta'] else BertConfig.from_json_file(config['text_config'])
        self.num_text_layers = config_text.fusion_layer
        self.num_cross_layers = config_text.num_hidden_layers - config_text.fusion_layer
        config_text.num_hidden_layers = self.num_text_layers + 2 * self.num_cross_layers
        self.num_attention_heads = config_text.num_attention_heads

        super().__init__(config, load_vision_params=False, load_text_params=False,
                         use_contrastive_loss=True, use_matching_loss=True, use_mlm_loss=False, use_bbox_loss=False,
                         config_text=config_text)

        self.share_cross_attention(self.text_encoder.encoder)

        self.cls_head = build_mlp(input_dim=self.text_width, output_dim=2)
        self.init_params = ['cls_head.' + n for n, _ in self.cls_head.named_parameters()]

    def load_pretrained(self, ckpt_rpath, config, is_eval=False):
        state_dict = load_pretrained(ckpt_rpath, config, is_eval=is_eval, load_text=True)
        msg = self.load_state_dict(state_dict, strict=False)
        print('load checkpoint from %s' % ckpt_rpath)
        print("missing_keys: ", [p for p in msg.missing_keys if 'vision_encoder' not in p])
        print("unexpected_keys: ", msg.unexpected_keys)
        
    def forward(self,  image, text_ids, text_atts, targets, task_id, train=True, idx=None):
        image_embeds, image_atts = self.get_vision_embeds(image)
        if task_id == 0:
            image0_embeds, image1_embeds = torch.split(image_embeds, targets.size(0))

            output_cls = self.get_cross_embeds([image0_embeds, image1_embeds],
                                               [image_atts[:image0_embeds.size(0)], image_atts[image0_embeds.size(0):]],
                                               text_ids=text_ids, text_atts=text_atts)[:, 0, :]

            prediction = self.cls_head(output_cls)

            return F.cross_entropy(prediction, targets) if train else prediction
        elif task_id == 1:
            text_embeds = self.get_text_embeds(text_ids, text_atts)

            image_feat, text_feat = self.get_features(image_embeds, text_embeds)
            loss_itc = self.get_contrastive_loss(image_feat, text_feat, idx=idx)
            loss_itm = self.get_matching_loss(image_embeds, image_atts, image_feat, text_embeds, text_atts, text_feat,
                                              idx=idx)

            return loss_itc, loss_itm

    def share_cross_attention(self, model):
        for i in range(self.num_cross_layers):
            layer_num = self.num_text_layers + i * 2
            modules_0 = model.layer[layer_num].crossattention.self._modules
            modules_1 = model.layer[layer_num + 1].crossattention.self._modules

            for name in modules_0.keys():
                if 'key' in name or 'value' in name:
                    module_0 = modules_0[name]
                    module_1 = modules_1[name]
                    if hasattr(module_0, "weight"):
                        module_0.weight = module_1.weight
                        if hasattr(module_0, "bias"):
                            module_0.bias = module_1.bias
