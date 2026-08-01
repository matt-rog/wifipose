# Reports

These are the measurement files that the results section of the main README
quotes from. Each one was written by the script that produced it, measured on
the separate 36 second holdout recording, using models trained only on the 14
minute training session. They were copied here from the training machine, where
the recordings and a working copy of this repository live.

`wave_report.json` comes from `train/train_wave.py` and covers the one
dimensional arm elevation model, which is the clearest result in the project.

`skeleton_report.json` comes from `train/train_skeleton.py` and covers the 24
joint body model. `skeleton_raw_report.json` is the same model trained with the
bone length and velocity terms removed from the loss, and because it scores
essentially the same, those two terms are not contributing anything measurable.

`openpose_report.json` comes from `train/train_openpose.py` and covers the 17
keypoint model. Its `pck20` looks like a good result while its `wrist_r` is
close to zero, which means the model has learned a better average posture rather
than learning to follow the arms.

`densepose_report.json` comes from `train/train_densepose.py` and covers the
body-part model. The difference between `frame_fg_iou` and `shuffled_fg_iou` is
the part of the score that comes from tracking the person rather than from
placing a well positioned average body shape, and it is only 0.0016.

`ruview_skeleton_report.json` and `ruview_densepose_report.json` come from the
scripts in `ruview/`, and they are the RuView project's own models run on our
data using the same metrics.

The three files named `cmu_e1_report.json`, `cmu_e2_report.json` and
`cmu_e3_report.json` record three attempts to borrow architecture ideas from the
DensePose From WiFi paper. The first is a shared multi-task trunk, the second is
a deeper encoder-decoder, and the third converts the features into a
pseudo-image and feeds it to a frozen ImageNet-pretrained ResNet stem. The first
of these has the highest foreground IoU of anything in the project at 0.317, but
its tracking gap is -0.015, so what it has produced is a better static body
shape rather than a better tracker. The scripts that generated these three files
are in `experiments/`, where `cmu_mtn.py` covers the first two and `cmu_e3.py`
covers the third.
