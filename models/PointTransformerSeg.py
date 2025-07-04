from models.model_layers import *
from utils.builder import MODELS


@MODELS.register_module()
class PointTransformerSeg(nn.Module):
    def __init__(self, config):
        super().__init__()
        # --------------------------------------------------------------------------------------------------------------
        # Extract the required configurations for each module
        # --------------------------------------------------------------------------------------------------------------
        self.configs = self.extract_configs(config)
        # --------------------------------------------------------------------------------------------------------------
        # Shared Tokenizer – Positional Encoding – Student Encoder – Class Information – Upsampling – Segmentation Head
        # --------------------------------------------------------------------------------------------------------------
        self.tokenizer = PointCloudTokenizer(**self.configs["tokenizer_config"])
        self.pos_enc = Mlp(**self.configs["pos_enc_config"])
        self.model = TransformerModule(**self.configs["model_config"])
        self.label_conv = Conv(**self.configs["label_config"])
        self.propagation = FeaturePropagation(**self.configs["feature_propagation"])
        self.seg_head = PointTransformerSegHead(**self.configs["seg_head"])

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
        label_config = {
            "category_dim": config.label.cls_dim,
        }
        feature_propagation = {
            "in_channel": config.feature_propagation.in_dim,
            "hidden_dim": config.feature_propagation.hidden_dim
        }
        seg_head = {
            "in_channels": config.seg_head.in_dim,
            "hidden_dims": config.seg_head.hidden_dims,
            "part_dim": config.seg_head.part_dim
        }
        return {
            "tokenizer_config": tokenizer_config,
            "pos_enc_config": pos_enc_config,
            "model_config": model_config,
            "label_config": label_config,
            "feature_propagation": feature_propagation,
            "seg_head": seg_head
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

    def forward(self, pts, cls_label):
        B, N, C = pts.size()
        center_data, center_tokens, pos_embed = self.preprocess(pts)
        # --------------------------------------------------------------------------------------------------------------
        # 12-layer Transformer: (2, 128, 384) -> (2, 128, 384)
        # --------------------------------------------------------------------------------------------------------------
        out_feature = self.model(center_tokens, pos_embed)
        # --------------------------------------------------------------------------------------------------------------
        # Global features
        # Global max pooling and global average pooling features, given to each upsampled point
        # (2, 128, 384) -> (2, 1, 384) -> (2, 384, 2048)
        # --------------------------------------------------------------------------------------------------------------
        feature_max = torch.max(out_feature, 1)[0]
        feature_avg = torch.mean(out_feature, 1)
        feature_max = feature_max.unsqueeze(-1).repeat(1, 1, N)
        feature_avg = feature_avg.unsqueeze(-1).repeat(1, 1, N)
        # --------------------------------------------------------------------------------------------------------------
        # Global features
        # Classification feature of each target, given to each upsampled point
        # (2, 16, 1) -> (2, 64, 1) -> (2, 64, 2048)
        # --------------------------------------------------------------------------------------------------------------
        cls_label_one_hot = cls_label.view(B, self.configs["label_config"]["category_dim"], 1)
        cls_label_feature = self.label_conv(cls_label_one_hot).repeat(1, 1, N)
        # --------------------------------------------------------------------------------------------------------------
        # Concatenate global features and classification features as global feature for each point:
        # (2, 384 + 384 + 64, 2048) -> (2, 832, 2048)
        # --------------------------------------------------------------------------------------------------------------
        global_feature = torch.cat((feature_max, feature_avg, cls_label_feature), 1)
        # --------------------------------------------------------------------------------------------------------------
        # Upsample 128 center points to 2048 original input points through interpolation,
        # then concatenate each point's global features
        # (2, 384, 128) -> (2, 512, 2048) + (2, 832, 2048) -> (2, 1344, 2048)
        # --------------------------------------------------------------------------------------------------------------
        interpolate_feature = self.propagation(center_data, out_feature, pts, pts)
        seg_feature = torch.cat((interpolate_feature, global_feature), 1)
        # --------------------------------------------------------------------------------------------------------------
        # Segmentation head
        # --------------------------------------------------------------------------------------------------------------
        result = self.seg_head(seg_feature).permute(0, 2, 1)
        return result

