"""
Regression test with SAM's real ViT-H model: the example subject must give exactly the masks published with the article.
Skipped when the checkpoint is not found (download it with `samson download-model`, or set SAMSON_CHECKPOINT).
"""
import pytest

from samson_mri.example import run_example
from samson_mri.model import CheckpointNotFound, find_checkpoint

try:
    CHECKPOINT = find_checkpoint("vit_h")
except CheckpointNotFound:
    CHECKPOINT = None


@pytest.mark.model
@pytest.mark.skipif(CHECKPOINT is None, reason="SAM's ViT-H checkpoint not found")
def test_example_reproduces_published_masks(tmp_path, capsys):
    assert run_example(str(tmp_path), checkpoint=CHECKPOINT) == 0
    output = capsys.readouterr().out
    assert output.count("identical to the article's published mask: yes") == 2
