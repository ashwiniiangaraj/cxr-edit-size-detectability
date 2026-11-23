"""Model registry"""

def get_detector_class(name: str):
    if name == "spectral_lr":
        from models.feature_lr import SpectralLR
        return SpectralLR
    if name == "handcrafted_lr":
        from models.feature_lr import HandcraftedLR
        return HandcraftedLR
    if name == "lbp_lr":                                   
        from models.lbp_lr import LBPLR
        return LBPLR
    if name == "cnn_densenet":
        from models.cnn_densenet import CNNDensenet
        return CNNDensenet
    if name == "supcon":
        from models.supcon import SupConDetector
        return SupConDetector
    if name == "npr":                                      
        from models.npr_cnn import NPRDetector
        return NPRDetector
    import config
    if name in config.CLIP_VARIANTS:
        from models.foundation_lr import make_frozen_detector
        return make_frozen_detector(name)
    raise KeyError(f"Unknown model '{name}'")


MODEL_LABELS = {
    "spectral_lr": "Spectral LR",
    "handcrafted_lr": "Handcrafted LR",
    "lbp_lr": "LBP LR",                                    
    "cnn_densenet": "CNN (DenseNet121)",
    "supcon": "SupCon",
    "npr": "NPR CNN",                                      
    "clip_frozen": "Frozen CLIP + LR",
    "biomedclip_frozen": "Frozen BiomedCLIP + LR",
    "rad_dino": "Frozen RAD-DINO + LR",                    
    "dinov2_frozen": "Frozen DINOv2 + LR",                 
}