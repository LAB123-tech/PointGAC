from tqdm import tqdm

from datasets.dataset_utils.DatasetTransforms import Data_Augmentation
from utils import builder
from utils.main_util import *
from utils.visualize import visualize_pca_and_save, visualize_tsne


def test_net(args, config, device, logger):
    # ------------------------------------------------------------------------------------------------------------------
    # Build dataset
    # ------------------------------------------------------------------------------------------------------------------
    test_data_aug = Data_Augmentation(config.test_dataset.aug_list)
    test_dataloader = builder.dataset_builder(config.test_dataset)
    # ------------------------------------------------------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------------------------------------------------------
    model = builder.model_builder(config.model).to(device)
    # ------------------------------------------------------------------------------------------------------------------
    # Load model weights
    # ------------------------------------------------------------------------------------------------------------------
    load_model(args, logger, is_train=False, model=model)
    # ------------------------------------------------------------------------------------------------------------------
    # Start testing
    # ------------------------------------------------------------------------------------------------------------------
    test(model, test_dataloader, test_data_aug, device)


def test(model, test_dataloader, test_data_aug, device):
    test_features = []
    test_labels = []
    model.eval()
    with torch.no_grad():
        for idx, data in enumerate(tqdm(test_dataloader, desc="Testing Progress")):
            points = data[0].to(device)
            label = data[1].to(device)
            points = test_data_aug(points)
            feature, center = model.forward_test(points[:, :, :3])
            feature_global = torch.cat([feature.max(dim=1).values, feature.mean(dim=1)], dim=-1)
            test_features.append(feature_global)
            test_labels.append(label.detach())
            visualize_pca_and_save(points[0][:, :3], feature[0], center[0], idx)
        # --------------------------------------------------------------------------------------------------------------
        # Map point features to RGB space (optional visualization)
        # --------------------------------------------------------------------------------------------------------------
        # for i in range(2):
        #     visualize_pca_and_save(points[i][:, :3], feature[i], center[i], i)
        # --------------------------------------------------------------------------------------------------------------
        # Visualize TSNE distribution for the entire dataset
        # --------------------------------------------------------------------------------------------------------------
        test_features = torch.cat(test_features, dim=0)
        test_labels = torch.cat(test_labels, dim=0)
        visualize_tsne(test_features, test_labels, "test")
