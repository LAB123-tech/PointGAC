from timm.models.layers import trunc_normal_

from models.model_layers import *
from utils.builder import MODELS


@MODELS.register_module()
class PointTransformerCls(nn.Module):
    def __init__(self, config):
        super().__init__()
        super(PointTransformerCls, self).__init__()
        # --------------------------------------------------------------------------------------------------------------
        # Extract the required configurations for each module
        # --------------------------------------------------------------------------------------------------------------
        self.configs = self.extract_configs(config)
        # --------------------------------------------------------------------------------------------------------------
        # Shared Tokenizer – Positional Encoding – Student Encoder – Classification Head
        # --------------------------------------------------------------------------------------------------------------
        self.tokenizer = PointCloudTokenizer(**self.configs["tokenizer_config"])
        self.pos_enc = Mlp(**self.configs["pos_enc_config"])
        self.model = TransformerModule(**self.configs["model_config"])
        # --------------------------------------------------------------------------------------------------------------
        # CLS token and position
        # --------------------------------------------------------------------------------------------------------------
        self.cls_token = trunc_normal_(nn.Parameter(torch.zeros(1, 1, self.model.embed_dim)), std=0.02)
        self.cls_pos = trunc_normal_(nn.Parameter(torch.randn(1, 1, self.model.embed_dim)), std=0.02)
        self.cls_head = PointTransformerClsHead(**self.configs["cls_config"])

    @staticmethod
    def extract_configs(config):
        """
        Extract the configuration dictionaries for each module
        @param config:
        @return:
        """
        tokenizer_config = {
            "num_group": config.tokenizer.num_group,
            "group_size": config.tokenizer.group_size,
            "in_dim": config.tokenizer.in_dim,
            "out_dim": config.tokenizer.out_dim,
        }
        pos_enc_config = {
            "in_dim": config.pos_emd.in_dim,
            "hidden_dim": config.pos_emd.hidden_dim,
            "out_dim": config.pos_emd.out_dim,
        }
        model_config = {
            "embed_dim": config.model.trans_dim,
            "depth": config.model.encoder_depth,
            "num_heads": config.model.num_heads,
            "drop_path_rate": [x.item() for x in
                               torch.linspace(0, config.model.drop_path_rate, config.model.encoder_depth)]
        }
        cls_config = {
            "trans_dim": config.cls_head.in_dim,
            "hidden_dim": config.cls_head.hidden_dim,
            "cls_dim": config.cls_head.cls_dim
        }
        return {
            "tokenizer_config": tokenizer_config,
            "pos_enc_config": pos_enc_config,
            "model_config": model_config,
            "cls_config": cls_config
        }

    def preprocess(self, data):
        """
        Tokenizer and positional encoding.
        @param data: (B, N, C)
        @return:
        """
        center_data, center_tokens = self.tokenizer(data)
        pos_embed = self.pos_enc(center_data[:, :, :3])
        return center_data, center_tokens, pos_embed

    def forward(self, pts):
        center_data, center_tokens, pos_embed = self.preprocess(pts)
        # --------------------------------------------------------------------------------------------------------------
        # CLS embedding: (2, 1, 384)
        # CLS Position Encoding: (2, 1, 384)
        # --------------------------------------------------------------------------------------------------------------
        cls_features = self.cls_token.expand(pts.shape[0], -1, -1)
        cls_pos = self.cls_pos.expand(pts.shape[0], -1, -1)
        # --------------------------------------------------------------------------------------------------------------
        # Transformer: (2, 65, 384) -> (2, 65, 384)
        # --------------------------------------------------------------------------------------------------------------
        in_feature = torch.cat((cls_features, center_tokens), dim=1)
        in_pos = torch.cat((cls_pos, pos_embed), dim=1)
        out_feature = self.model(in_feature, in_pos)
        # --------------------------------------------------------------------------------------------------------------
        # Concatenate the CLS token and the max pooling result as the classification feature.
        # --------------------------------------------------------------------------------------------------------------
        concat_feature = torch.cat([out_feature[:, 0], out_feature[:, 1:].max(1)[0]], dim=-1)
        result = self.cls_head(concat_feature)
        return result
