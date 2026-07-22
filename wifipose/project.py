"""Project the SMPL teacher to 2D COCO-17 keypoints. CoMotion is amodal (it
infers truncated/occluded body parts), unlike per-frame keypoint detectors that
hallucinate off-frame joints. Root-relative torso-normalized coordinates are
intrinsics-invariant, so unit intrinsics are used."""
import numpy as np

# COCO-17 index -> SMPL-24 joint; eyes/ears have no SMPL joint (score 0)
COCO17_FROM_SMPL = [15, -1, -1, -1, -1, 16, 17, 18, 19, 20, 21, 1, 2, 4, 5, 7, 8]


def smpl_keypoints_2d(J_canon, R_can, pelvis, h):
    """[N,24,3] canonical joints (+ inverse transform) -> kp[N,17,3] (u, v, score)
    on the unit image plane."""
    J_cam = np.einsum("nji,nkj->nki", R_can, J_canon * h) + pelvis[:, None]
    z = np.clip(J_cam[..., 2:3], 1e-3, None)
    uv = J_cam[..., :2] / z
    kp = np.zeros((len(J_canon), 17, 3), np.float32)
    for c, s in enumerate(COCO17_FROM_SMPL):
        if s >= 0:
            kp[:, c, :2] = uv[:, s]
            kp[:, c, 2] = 1.0
    return kp
