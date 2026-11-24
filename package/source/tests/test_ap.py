import nibabel as nib
import numpy as np

from samson_mri.ap import box_agreement, check_range, length_ratio, linked_range, suggest_from_masks, template_positions
from samson_mri.viz import ap_figure, candidates_figure, orientation_figure, qc_figure


def test_positions_and_lengths():
    assert np.allclose(template_positions(6, 12, 22), [12, 14, 16, 18, 20, 22])
    assert np.isclose(length_ratio(6, 1.0, 12, 22), 1.25)
    assert linked_range(12, 6, 1.0, ratio=1.25) == (12, 22)
    assert linked_range(35, 16, 0.8, ratio=1.0)[1] == 39     # clipped to the template


def test_check_range():
    assert check_range(16, 4, 33, (0.1667, 0.1667, 0.8)) == []
    messages = check_range(6, 25, 30, (0.09, 0.09, 1.0))   # outside the centre and 5 mm vs 2 mm
    assert len(messages) == 2 and all({"short", "detail"} <= set(m) for m in messages)
    # x10 headers: no length check unless the scale is given
    assert check_range(16, 4, 20, (1.667, 1.667, 8.0)) == []
    assert len(check_range(16, 4, 20, (1.667, 1.667, 8.0), header_scale=0.1)) == 1


def test_suggestion_from_example_mask(example_mask):
    mask = nib.load(example_mask).get_fdata() > 0
    table = suggest_from_masks([mask], 0, 15, top=5, slice_mm=0.8)
    best = table.iloc[0]
    # The article used 4-33 for this dataset; the suggestion from one mask must be close to it
    assert abs(best.template_first - 4) <= 4 and abs(best.template_last - 33) <= 4
    assert box_agreement([mask], 0, 15, 4, 33) > 0.8
    assert table.box_iou.is_monotonic_decreasing


def test_figures(example_image, example_mask, d1_spec, fake_predictor):
    from samson_mri import SAMson
    from samson_mri.io import load_subject
    from samson_mri.pipeline import compute_boxes
    subject = load_subject(example_image, d1_spec)
    prompts = compute_boxes(subject, d1_spec)
    fig = ap_figure(subject.volume, subject.voxel_size, 4, 33, prompts=prompts, header_scale=0.1, title="x")
    assert len(fig.axes) == 2 + 2 * 6
    assert len(ap_figure(subject.volume, subject.voxel_size, 4, 33, 5, 7).axes) == 2 + 2 * 3
    orientation_figure(subject.volume, subject.voxel_size)
    result = SAMson(predictor=fake_predictor).segment(example_image, d1_spec, mode="semi-auto")
    qc_figure(result, reference=nib.load(example_mask).get_fdata())
    assert len(candidates_figure(result, 3).axes) == 4


def test_brain_zoom(example_image, d1_spec, fake_predictor):
    from samson_mri import SAMson
    from samson_mri.viz import brain_window
    result = SAMson(predictor=fake_predictor).segment(example_image, d1_spec, mode="semi-auto")
    height, width = result.full_auto.shape[-2:]
    x0, x1, y0, y1 = brain_window(result)
    assert -0.5 <= x0 < x1 <= width - 0.5 and -0.5 <= y0 < y1 <= height - 0.5
    assert np.isclose(x1 - x0, y1 - y0) or np.isclose(x1 - x0, width) or np.isclose(y1 - y0, height)   # square unless clipped
    boxes = np.asarray(result.prompts.boxes)
    assert x0 <= boxes[:, 0].min() and boxes[:, 2].max() <= x1 + 0.5 and y0 <= boxes[:, 1].min() and boxes[:, 3].max() <= y1 + 0.5
    for fig in (qc_figure(result, zoom=True), candidates_figure(result, 3, zoom=True)):
        ax = fig.axes[0]
        assert np.isclose(ax.get_xlim()[0], x0) and np.isclose(ax.get_ylim()[1], y0)   # y axis inverted (image)
    whole = qc_figure(result).axes[0]
    assert np.isclose(whole.get_xlim()[1] - whole.get_xlim()[0], width)
