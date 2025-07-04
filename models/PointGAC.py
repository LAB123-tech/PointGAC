from models.model_layers import *
from utils.builder import MODELS

NORMALIZE_EPS = 1e-5


@MODELS.register_module()
class PointGAC(nn.Module):
    def __init__(self, config):
        super(PointGAC, self).__init__()
        # --------------------------------------------------------------------------------------------------------------
        # Extract the required configurations for each module
        # --------------------------------------------------------------------------------------------------------------
        self.configs = self.extract_configs(config)
        # --------------------------------------------------------------------------------------------------------------
        # Shared Tokenizer - Positional Encoding - Student Encoder - Student Decoder
        # --------------------------------------------------------------------------------------------------------------
        self.tokenizer = PointCloudTokenizer(**self.configs["tokenizer_config"])
        self.pos_enc = Mlp(**self.configs["pos_enc_config"])
        self.student_model = StudentModel(**self.configs["student_model_config"])
        self.teacher_encoder = None
        self.regressor = Mlp(**self.configs["regressor_config"])
        self.codebook_gen = CodebookGenerator(**self.configs["assignment_generator_config"])

    @staticmethod
    def extract_configs(config):
        """
        Extract the configuration dictionary for each module
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
        student_model_config = {
            "trans_dim": config.student_model.trans_dim,
            "encoder_depth": config.student_model.encoder_depth,
            "decoder_depth": config.student_model.decoder_depth,
            "num_heads": config.student_model.num_heads,
            "drop_path_rate": config.student_model.drop_path_rate,
            "mask_type": config.student_model.mask_type,
            "mask_ratio": config.student_model.mask_ratio,
        }
        regressor_config = {
            "in_dim": config.regressor.in_dim,
            "hidden_dim": config.regressor.hidden_dim,
            "out_dim": config.regressor.out_dim,
        }
        assignment_generator_config = {
            "num_words": config.dict.num_words,
            "num_channels": config.dict.dic_dim,
            "dic_momentum": config.dict.dic_momentum,
            "max_buffer_size": config.dict.max_buffer_size,
        }
        return {
            "tokenizer_config": tokenizer_config,
            "pos_enc_config": pos_enc_config,
            "student_model_config": student_model_config,
            "regressor_config": regressor_config,
            "assignment_generator_config": assignment_generator_config
        }

    def set_teacher(self):
        self.teacher_encoder = EMA(self.student_model.encoder, update_after_step=0)

    def preprocess(self, data):
        """
        Tokenizer and positional encoding.
        @param data: (B, N, C)
        @return:
        """
        center_data, center_tokens = self.tokenizer(data)
        pos_embed = self.pos_enc(center_data[:, :, :3])
        return center_data, center_tokens, pos_embed

    def forward_test(self, data):
        """
        Test forward inference
        @param data:
        @return:
        """
        center_data, center_tokens, pos_embed = self.preprocess(data)
        encoder_output = self.student_model.encoder(center_tokens, pos_embed)
        return encoder_output, center_data

    def forward(self, data):
        """
        Training forward inference
        @param data:
        @return:
        """
        center_data, center_tokens, pos_embed = self.preprocess(data)
        # --------------------------------------------------------------------------------------------------------------
        # Student model forward inference
        # --------------------------------------------------------------------------------------------------------------
        student_output, mask_mae = self.student_model(center_tokens, pos_embed, center_data)
        student_output = self.regressor(student_output)
        # --------------------------------------------------------------------------------------------------------------
        # Teacher model forward inference
        # --------------------------------------------------------------------------------------------------------------
        self.teacher_encoder.ema_model.eval()
        with torch.no_grad():
            teacher_output = self.teacher_encoder(center_tokens, pos_embed)
            codebook = self.codebook_gen(teacher_output)
        return student_output[mask_mae.bool()], teacher_output[mask_mae.bool()], codebook
