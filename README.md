# wifipose

This project estimates human pose from the WiFi channel measurements of a
single antenna. A Raspberry Pi listens to an ordinary WiFi router and records
how the radio channel looks on each packet it hears. A camera films the person
at the same time, and vision models run on that video to produce labels. Small
networks are then trained to map the WiFi measurement to those labels. The
camera is only used to create the labels, and is not used when the trained
models run.

What comes out of this is limited. Arm motion is recovered at a level that is
clearly above chance, and the rest of the body is not.

## Setup

The access point is a GL-iNet travel router with the SSID "m&m", running on 5
GHz, primary channel 48, with an 80 MHz wide channel. It sends ordinary traffic
and is not modified in any way.

The receiver is a Raspberry Pi 3B+ with a BCM43455c0 chip. It runs
[nexmon_csi](https://github.com/seemoo-lab/nexmon_csi), a firmware patch that
makes the chip report channel state information, installed and controlled with
[picsi](https://github.com/nexmonster/picsi). The Pi has a single antenna, does
not transmit, and does not store anything locally. It forwards each measurement
to a laptop over UDP on a wired connection. We checked that the Pi really does
have only one receive chain rather than assuming it, by testing whether the
burst packets carry separate chains, and they do not.

The camera is a NexiGo USB webcam recording 1280x720 MJPG video. Exposure,
gain, autofocus and white balance are all locked before recording so that the
frame rate does not change as the light changes, and at the settings used it
records at about 15 frames per second. The laptop records the camera and the
WiFi stream in a single process, and stamps both with the same clock, so the
two can be aligned afterwards with a timestamp lookup and one constant offset.

The recording room is a domestic garage with the door shut and even lighting,
and with no window or other bright source behind the subject. There is one
person in the room throughout, standing and walking inside the camera frame
between the router and the Pi. The exact positions and separations of the
router, the Pi and the camera were never written down, so the rig cannot be
rebuilt to the same dimensions from what was recorded.

Labels and training run on a separate Linux machine with three RTX 3090 cards,
using [PyTorch](https://pytorch.org/),
[apple/ml-comotion](https://github.com/apple/ml-comotion) for the body pose
labels, and [detectron2](https://github.com/facebookresearch/detectron2) with
its [DensePose
project](https://github.com/facebookresearch/detectron2/tree/main/projects/DensePose)
for the body-part labels. The recording and rendering code on the laptop uses
NumPy and OpenCV.

## What was recorded

The dataset is made up of four takes. The first is a 30 second calibration take
in which the subject performs three sharp jumping jacks, and it exists so that
the constant time offset between the camera and the WiFi stream can be
measured. The second is a 14 minute training take of standing, walking and arm
motion, which is the only data that any model is ever trained on. The third is
a 2.5 minute take of the same room with nobody in it, which teaches the
body-part model what an empty room looks like and is also used afterwards to
check whether the model invents a person who is not there. The fourth is a
separate 36 second take recorded at a different time, and it is the only
recording that any number in this document is measured on.

## The approach

The channel measurement is a complex number for each of the subcarriers that
make up the WiFi channel. It describes the sum of every path the signal took
from the router to the Pi, which includes the direct path, the reflections off
the walls and the floor, and the reflections off the person.

Training a model directly on that measurement does not work across recordings.
Most of the energy in it comes from the static reflections that describe the
room, not the person, so the model ends up fitting the room. An earlier version
of this project that used the raw measurement reached a wrist correlation of
0.88 when tested inside the same recording, and 0.07 when tested on a different
one.

The alternative, which goes back to the CARM paper in 2015 and is standard in
this literature, is to use how fast the channel is changing rather than what
the channel is. When a body part moves, the length of the path that reflects
off it changes, and a path whose length is changing shifts the received signal
slightly in frequency. If you take a Fourier transform over the recent history
of each subcarrier, you get a spectrum in which the horizontal axis is speed
and the value is roughly how much reflected energy is moving at that speed.
Subtracting the average of the window first removes the static reflections,
which sit at zero and carry most of what is specific to the room. Replacing the
raw measurement with this spectrum is what moved the cross-recording numbers
from nothing to something.

The window is 2 seconds long and is resampled to 128 points, which gives 65
frequency bins spaced half a hertz apart, up to 32 hertz. At 5 GHz, one hertz
corresponds to a reflection path whose length is changing at about 5.7
centimetres per second. Waving an arm produces energy in the low tens of hertz,
walking is lower and more spread out, and an empty room is close to flat.

A single antenna sees one geometry between the transmitter and the receiver, so
it measures only one projection of velocity. It cannot tell which limb moved,
or in which direction, only that something moved at a particular speed. Telling
those apart requires more than one antenna.

## Inputs and outputs

For each session the recorder writes the raw measurements as `[M, 256, 2]`
16-bit integers, which is one real and imaginary pair for each of 256
subcarriers, for each of M packets, along with a timestamp per packet and the
transmitter address. The 14 minute training session produced about 133,000
packets over 843 seconds, which is roughly 158 packets per second after keeping
only the packets from the router.

Loading drops the 14 subcarriers that carry no data and resamples the remaining
242 down to 114, and takes the magnitude of each one, normalised per packet so
that the receiver's automatic gain control does not appear as signal. The phase
is thrown away, because on a single antenna it is corrupted by clock offsets
between the two radios, and measurements showed that including it made no
difference. The result is an array of shape `[M, 114]`.

The feature builder turns that into one vector per labelled video frame. It
takes the previous 2 seconds of packets, resamples them onto an even 128 point
grid because the packets do not arrive evenly, subtracts the mean, applies a
window function, and takes the magnitude of the Fourier transform along time.
That gives 65 bins for each of the 114 subcarriers, which is then averaged over
all subcarriers and over three groups of subcarriers separately, producing a
vector of 260 numbers per frame.

The labels come from the video. CoMotion produces a 24 joint body model per
frame, which is made relative to the hips, rotated to face a fixed direction
and divided by torso length, giving `[N, 24, 3]`. DensePose produces a 24 part
body surface map per frame, which is collapsed here to 5 coarse classes and
downsampled to a 36 by 64 grid. Training used about 8,400 labelled frames and
the holdout has about 1,070.

Four models are trained on the same 260 number input. One predicts a single
number, the average height of the two wrists. One predicts the 24 joint body,
which is 72 numbers. One predicts 17 two-dimensional keypoints, which is 34
numbers, relative to the hips and scaled by torso size because absolute
position in the image is not something the WiFi measurement can tell you. The
last predicts a class for each cell of the 36 by 64 grid. All four are two
hidden layers of 128 units, except the last, which projects up and then uses
two transposed convolutions to make the grid. They are small because 8,400
frames from a single session will not support anything larger, and every larger
or recurrent version we tried scored worse.

## Relation to prior work

The signal processing is conventional. CARM established that this kind of
Doppler feature carries human motion, SHARP does the same on commodity 802.11ac
hardware, and Widar3.0 is the standard reference for making WiFi sensing work
across environments. WiPose, Person-in-WiFi and DensePose From WiFi are the
line of work that predicts pose specifically, and all of them train a WiFi
model against a vision model running on synchronised video, which is the same
structure used here. The metric definitions follow MM-Fi, and the data
augmentation follows Strohmayer and Kampel.

There are four differences. The most important is that this rig has a single
antenna and a single link, where WiPose and DensePose From WiFi both use a
three by three array and Widar3.0 uses several links. That is a hardware
difference rather than a method difference, and it is the reason the results
are as coarse as they are. The second is that the pose labels come from
CoMotion rather than a frame-by-frame keypoint detector, because CoMotion
infers where a joint is when it is hidden or outside the frame instead of
dropping it, which stops the WiFi model from learning to imitate detector
failures. The third is that the empty room recording is used as training data,
which we have not seen done in this literature; without it the body-part model
draws a person on every frame of an empty room. The fourth is that the
evaluation is stricter than what these papers report.

## Results

These numbers are measured on the separate 36 second holdout recording, using
models trained only on the 14 minute training session. The report files they
are read from are in `reports/`.

A model that always predicts the average standing pose already scores well,
because most joints in most frames are near their average, so every result is
compared against that constant baseline. A model can also lower its average
error by predicting something stiller and closer to the mean while tracking
less, so every result is also paired with a measure of whether the prediction
moves with the person.

The arm elevation model, which predicts one number, reaches a correlation of
0.39 with the truth. A constant prediction scores zero by definition, so the
WiFi measurement is carrying arm position across recordings.

The 24 joint skeleton reaches 53.9 mm mean joint error against a constant pose
baseline of 79.6 mm. The correlation for wrist height is 0.22. The correlation
averaged over all three axes is 0.12, so the 0.22 figure is the best single
axis rather than a typical one. Replacing every non-arm joint with the training
average changes the error from 53.9 mm to 54.3 mm, which means the torso and
legs in the rendered skeleton come from the anatomical prior and not from the
WiFi signal. The bone length and velocity terms in the loss also turn out to
make no measurable difference, since removing them gives 53.8 mm and a wrist
correlation of 0.21.

The 17 keypoint model reaches PCK@0.2 of 0.73 against a constant baseline of
0.56. Its wrist correlation is 0.088, which is close to zero. It has learned a
better average posture than the constant baseline, and does not track the arms
frame by frame.

The body-part model reaches a foreground IoU of 0.26 against 0.09 for a static
blob, and marks only 2.6% of pixels as person on the held-out portion of the
empty room recording, so it is not hallucinating people. But scoring each
predicted map against its own frame gives 0.2786, and scoring it against a
randomly chosen other frame gives 0.2770. The difference of 0.0016 is the part
that is actually tracking, and it is negligible. This model produces a well
placed but essentially static body shape.

As an external comparison, `ruview/` runs ruvnet/RuView, the most visible open
source project in this area, on the same data with the same metrics. Its
skeleton model scores 85.0 mm against the 79.5 mm constant baseline with a
wrist correlation of -0.06, so it is worse than predicting the average pose.
Its segmentation model scores 0.12 foreground IoU. Its frame-to-frame tracking
gap is 0.043, which is larger than ours, so its predictions move with the
person more than ours do. They simply move to the wrong places.

The one dimensional arm elevation signal and the vertical component of the
skeleton are the only outputs that demonstrably track. The 2D keypoint and
body-part outputs look reasonable in the rendered videos and score well against
their baselines, but they are close to elaborate averages.

## Limits and things wrong with this

The holdout has been used to make choices. The window length, the transform
size, the ensembling, the loss weighting and the decision to augment some
models and not others were all settled by comparing holdout numbers over many
runs. That makes it a validation set rather than a genuine test set, and the
numbers above are therefore somewhat optimistic by an amount we have not
measured.

All of it rests on 36 seconds of one recording, with no error bars. The
effective number of independent samples is much smaller than the 1,070 frames
suggests, because consecutive frames share almost all of a 2 second input
window. Small differences between the numbers above should not be treated as
real.

There is one subject, one room and one rig, so the only kind of change being
tested is between two recordings of the same person in the same place. Nothing
here says anything about a different person, a different room or a different
router position.

The labels are themselves uncertain. They come from a monocular vision model
running on a 15 frame per second webcam, and depth is the weakest part of any
monocular pose estimate, and wrist height is the main tracking number reported
here. Some unknown share of the remaining error belongs to the labels rather
than to the WiFi model, and we never measured how accurate the labels are.

The millimetre figures are not really distances. The targets are divided by
torso length, and the conversion back to millimetres multiplies by a single
fixed constant for one person, so they should not be compared to figures from
papers that predict real scale.

There are also several smaller problems that should be tightened up. Any 2
second window containing at least 8 packets is treated as valid, even though
the normal rate is about 158 packets per second, which means that a window
holding only 8 packets gets stretched out to 128 points and then carries the
same weight in training as a window that was properly filled. Packet loss
between the Pi and the laptop is never checked, despite the fact that the
sequence numbers needed to check it are being recorded. Both streams are
timestamped at the moment they arrive at the laptop rather than at the moment
they were sent, and clock drift over a 14 minute recording is never rechecked
afterwards. Every result quoted above is an average over several random seeds,
which improves the correlation figures on its own, and we do not report the
single model numbers alongside them so that the two effects could be told
apart. There are no automated tests anywhere in the project, which means that a
mistake in something like subcarrier indexing would never announce itself and
would show up only as slightly worse results.

## Things that were tried and did not work

Using the phase as well as the magnitude changed nothing, because on one
antenna the phase is dominated by clock offsets between the two radios. This is
an argument for a second antenna, where taking the ratio between two antennas
cancels those offsets, rather than an argument against phase.

Recurrent models over a longer history improved the average error by 4% while
the prediction variance fell by nearly half and the wrist correlation dropped
from 0.26 to 0.16, which is the model learning to sit still rather than
learning to track.

Domain-adversarial training and CORAL across sub-regions of the training
session made things steadily worse as their weight increased. Different parts
of one recording are not different enough to be useful as separate domains.

Normalising the holdout features by their own statistics at test time made
things worse, so the difference between recordings is not a simple offset.

Three attempts to borrow architecture ideas from DensePose From WiFi were run
on this data, with their scripts in `experiments/` and their measurements in
`reports/`. The first was a shared multi-task trunk predicting segmentation and
pose together, the second was a deeper encoder-decoder in the style of their
modality translation network, and the third converted the features into a
pseudo-image and fed it to a frozen ImageNet-pretrained ResNet stem. They
scored 0.32, 0.21 and 0.28 foreground IoU, so the first is nominally the best
result we have, but all three have a frame-to-frame tracking gap of essentially
zero or slightly negative. They make the static body shape more accurate
without making it track.

## What would improve this

A second receive antenna would help more than anything else. Taking the ratio
between two antennas cancels the clock offsets that make the phase unusable
here, and it gives some directional information. This is the main difference
between the published systems that work across environments and this one.

The second most useful thing would be to record more than one session. Teaching
a model to ignore a particular kind of change requires showing it at least two
versions of that change, and a single session cannot provide that, which is why
the domain adaptation experiments described above went nowhere.

Beyond those two, subtracting the empty room background from the measurements
and pretraining on unlabelled WiFi data are both cheap enough to be worth
trying, although neither of them does anything about the fact that there is
only one antenna.

## Repository layout

The code that runs on the laptop and the Pi lives in `record/`, which contains
the Pi setup script, the synchronised recorder and the clock offset
calibration. Everything to do with generating labels from the video is in
`teacher/`. The shared code for loading measurements, building features and
computing metrics is in `wifipose/`, and the four trainers that use it are in
`train/`. The renderer that draws predictions next to the camera and the labels
is in `infer/`. The comparison against the RuView project is in `ruview/`,
where `ruview/RuView` is a git submodule pointing at that project. The rendered
comparisons are in `videos/`, and `reports/` holds the measurement files that
the results section quotes from.

The recordings themselves are too large to keep in the repository, so they live
on the training machine alongside a working copy of this code.

## Reproduce

```bash
./record/pi_bringup.sh <pi_ip>
python record/record.py --prefix sync    --secs 30
python record/record.py --prefix train   --secs 840
python record/record.py --prefix empty   --secs 150
python record/record.py --prefix holdout --secs 36
python record/sync_offset.py --prefix sync --mac <router_mac>

./teacher/run_comotion.sh train_video.avi <n_frames> train.pt
python teacher/comotion_targets.py --pt train.pt --frame-ts train_frame_ts.npy \
    --sync-offset <offset> --out train_Y.npz         # repeat for holdout
python teacher/densepose_gt.py --video train_video.avi --out train_dp.npz

python train/train_wave.py      --mac <router_mac>
python train/train_skeleton.py  --mac <router_mac>
python train/train_openpose.py  --mac <router_mac>
python train/train_densepose.py --mac <router_mac>

python infer/render_holdout.py  --mac <router_mac>
```

Training the four models requires numpy, opencv-python and torch, and will run
on a single GPU. Generating the labels in the middle step is heavier, and
requires both apple/ml-comotion and detectron2 with its DensePose project
installed.

## References

- [nexmon_csi](https://github.com/seemoo-lab/nexmon_csi) and [picsi](https://github.com/nexmonster/picsi)
- [CARM (MobiCom 2015)](https://www.sigmobile.org/mobicom/2015/papers/p65-wangA.pdf)
- [SHARP (IEEE TMC 2023)](https://arxiv.org/abs/2103.09924)
- [Widar3.0 (MobiSys 2019)](https://tns.thss.tsinghua.edu.cn/widar3.0/)
- [WiPose (MobiCom 2020)](https://cse.buffalo.edu/~lusu/papers/MobiCom2020.pdf)
- [DensePose From WiFi](https://arxiv.org/abs/2301.00250)
- [MM-Fi (NeurIPS 2023)](https://arxiv.org/abs/2305.10345)
- [SenseFi](https://github.com/xyanchen/WiFi-CSI-Sensing-Benchmark)
- [Strohmayer and Kampel, augmentation](https://arxiv.org/abs/2401.00964)
- [CoMotion](https://github.com/apple/ml-comotion)
- [DensePose / detectron2](https://github.com/facebookresearch/detectron2/tree/main/projects/DensePose)
- [RuView](https://github.com/ruvnet/RuView)
