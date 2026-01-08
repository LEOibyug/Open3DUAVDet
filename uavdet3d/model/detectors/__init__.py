from .detector_template import DetectorTemplate

from .key_point_pose import KeyPoint2Pose
from .center_det import CenterDet
from .center_dorn_det import CenterDORNDet
from .center_dorn_cs_det import CenterDORNCSDet

__all__ = {
    'DetectorTemplate': DetectorTemplate,
    'KeyPoint2Pose': KeyPoint2Pose,
    'CenterDet': CenterDet,
    'CenterDORNDet': CenterDORNDet,
    'CenterDORNCSDet': CenterDORNCSDet
}

def build_detector(model_cfg,  dataset):
    model = __all__[model_cfg.NAME](
        model_cfg=model_cfg,  dataset=dataset
    )

    return model


