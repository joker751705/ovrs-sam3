from __future__ import annotations

from importlib import import_module
from pathlib import Path
import sys
import types
import unittest

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "_ovrs_sam3_testpkg"

if PACKAGE_NAME not in sys.modules:
    package = types.ModuleType(PACKAGE_NAME)
    package.__path__ = [str(REPO_ROOT)]
    sys.modules[PACKAGE_NAME] = package

SemanticCriterionConfig = import_module(
    f"{PACKAGE_NAME}.config_dataclasses"
).SemanticCriterionConfig
criterion_module = import_module(
    f"{PACKAGE_NAME}.losses.semantic_criterion"
)
SemanticCriterion = criterion_module.SemanticCriterion
OUTPUT_KEYS = import_module(
    f"{PACKAGE_NAME}.models.task_modes"
).OUTPUT_KEYS


class SemanticCriterionDynamicMixTest(unittest.TestCase):
    def setUp(self):
        self.criterion = SemanticCriterion(
            SemanticCriterionConfig(
                mixed_bce_weight=1.0,
                final_dice_weight=0.0,
                mixed_bce_boundary_width=1,
            )
        )
        self.label_map = torch.tensor(
            [[
                [255, 255, 255, 255],
                [255, 0, 1, 255],
                [255, 255, 255, 255],
                [255, 255, 255, 255],
            ]],
            dtype=torch.long,
        )
        self.prompt_to_class_id = torch.tensor([0, 1, 2])

    def _context(self, progress: float):
        return self.criterion.prepare_streaming_context(
            targets={"label_map": self.label_map},
            num_prompt_channels=3,
            prompt_to_class_id=self.prompt_to_class_id,
            num_label_classes=3,
            target_hw=(4, 4),
            train_progress=progress,
        )

    def test_cosine_schedule_endpoints_and_midpoint(self):
        self.assertEqual(
            self.criterion._compute_mix_ratios(0.0),
            (0.0, 1.0),
        )
        gt_ratio, teacher_ratio = self.criterion._compute_mix_ratios(0.5)
        self.assertAlmostEqual(gt_ratio, 0.5)
        self.assertAlmostEqual(teacher_ratio, 0.5)
        gt_ratio, teacher_ratio = self.criterion._compute_mix_ratios(1.0)
        self.assertAlmostEqual(gt_ratio, 1.0)
        self.assertAlmostEqual(teacher_ratio, 0.0)

    def test_absent_prompt_is_excluded_from_mask_denom_and_gradient(self):
        context = self._context(0.5)
        self.assertEqual(
            context.presence_target.tolist(),
            [[True, True, False]],
        )

        prompt_boundaries = context.outer_boundary_by_class.index_select(
            1, self.prompt_to_class_id
        )
        prompt_masks = context.valid_mask[:, None] | prompt_boundaries
        expected_denom = int(prompt_masks[:, :2].sum().item())
        self.assertEqual(context.mixed_pixel_denom, expected_denom)

        final_logits = torch.linspace(-1.5, 1.5, 48).reshape(1, 3, 4, 4)
        final_logits.requires_grad_(True)
        teacher_logits = torch.linspace(1.0, -1.0, 48).reshape(1, 3, 4, 4)

        losses = self.criterion.forward_chunk(
            outputs={
                OUTPUT_KEYS.final_logits: final_logits,
                OUTPUT_KEYS.sam3_teacher_logits: teacher_logits,
            },
            context=context,
            class_start=0,
            class_end=3,
        )

        gt_target = (
            self.label_map[:, None]
            == self.prompt_to_class_id[None, :, None, None]
        ).to(dtype=final_logits.dtype)
        mixed_target = 0.5 * gt_target + 0.5 * teacher_logits.sigmoid()
        pair_mask = (
            context.presence_target[:, :, None, None]
            & prompt_masks
        )
        expected = (
            F.binary_cross_entropy_with_logits(
                final_logits,
                mixed_target,
                reduction="none",
            )
            * pair_mask
        ).sum() / expected_denom

        torch.testing.assert_close(losses["loss_mixed_bce"], expected)

        losses["total_loss"].backward()
        torch.testing.assert_close(
            final_logits.grad[:, 2],
            torch.zeros_like(final_logits.grad[:, 2]),
        )

    def test_mixed_target_matches_weighted_bces_on_common_mask(self):
        context = self._context(0.25)
        final_logits = torch.linspace(-1.0, 1.0, 48).reshape(1, 3, 4, 4)
        teacher_logits = torch.linspace(0.8, -0.8, 48).reshape(1, 3, 4, 4)

        losses = self.criterion.forward_chunk(
            outputs={
                OUTPUT_KEYS.final_logits: final_logits,
                OUTPUT_KEYS.sam3_teacher_logits: teacher_logits,
            },
            context=context,
            class_start=0,
            class_end=3,
        )

        gt_target = (
            self.label_map[:, None]
            == self.prompt_to_class_id[None, :, None, None]
        ).to(dtype=final_logits.dtype)
        prompt_boundaries = context.outer_boundary_by_class.index_select(
            1, self.prompt_to_class_id
        )
        pair_mask = (
            context.presence_target[:, :, None, None]
            & (context.valid_mask[:, None] | prompt_boundaries)
        ).to(dtype=final_logits.dtype)

        gt_bce = (
            F.binary_cross_entropy_with_logits(
                final_logits,
                gt_target,
                reduction="none",
            )
            * pair_mask
        ).sum() / context.mixed_pixel_denom
        teacher_bce = (
            F.binary_cross_entropy_with_logits(
                final_logits,
                teacher_logits.sigmoid(),
                reduction="none",
            )
            * pair_mask
        ).sum() / context.mixed_pixel_denom
        expected = (
            context.gt_mix_ratio * gt_bce
            + context.teacher_mix_ratio * teacher_bce
        )

        torch.testing.assert_close(losses["loss_mixed_bce"], expected)

    def test_final_step_uses_gt_without_teacher_logits(self):
        context = self._context(1.0)
        final_logits = torch.zeros((1, 3, 4, 4), requires_grad=True)

        losses = self.criterion.forward_chunk(
            outputs={OUTPUT_KEYS.final_logits: final_logits},
            context=context,
            class_start=0,
            class_end=3,
        )

        self.assertTrue(torch.isfinite(losses["total_loss"]))
        losses["total_loss"].backward()
        self.assertIsNotNone(final_logits.grad)

    def test_absent_only_chunk_does_not_require_teacher_logits(self):
        context = self._context(0.0)
        final_logits = torch.zeros((1, 1, 4, 4), requires_grad=True)

        losses = self.criterion.forward_chunk(
            outputs={OUTPUT_KEYS.final_logits: final_logits},
            context=context,
            class_start=2,
            class_end=3,
        )

        self.assertEqual(float(losses["total_loss"].item()), 0.0)
        losses["total_loss"].backward()
        torch.testing.assert_close(
            final_logits.grad,
            torch.zeros_like(final_logits.grad),
        )


if __name__ == "__main__":
    unittest.main()
