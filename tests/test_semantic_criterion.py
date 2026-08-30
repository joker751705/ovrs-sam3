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


class SemanticCriterionDynamicDistillTest(unittest.TestCase):
    FINAL_WEIGHT = 1.25
    DISTILL_WEIGHT = 0.4

    def setUp(self):
        self.criterion = SemanticCriterion(
            SemanticCriterionConfig(
                final_balanced_bce_weight=self.FINAL_WEIGHT,
                final_dice_weight=0.0,
                sam3_mask_distill_weight=self.DISTILL_WEIGHT,
                sam3_mask_distill_boundary_width=1,
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

    def _targets_and_distill_mask(self, context, dtype):
        target = (
            self.label_map[:, None]
            == self.prompt_to_class_id[None, :, None, None]
        ).to(dtype=dtype)
        prompt_boundaries = (
            context.distill_outer_boundary_by_class.index_select(
                1,
                self.prompt_to_class_id,
            )
        )
        distill_mask = (
            context.presence_target[:, :, None, None]
            & (context.valid_mask[:, None] | prompt_boundaries)
        )
        return target, distill_mask

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

    def test_context_uses_all_prompts_for_final_and_present_for_distill(self):
        context = self._context(0.5)
        self.assertEqual(
            context.presence_target.tolist(),
            [[True, True, False]],
        )
        self.assertEqual(
            context.total_valid_pixels,
            int(context.valid_mask.sum().item()) * 3,
        )

        _, distill_mask = self._targets_and_distill_mask(
            context,
            torch.float32,
        )
        self.assertEqual(
            context.distill_pixel_denom,
            int(distill_mask.sum().item()),
        )

    def test_final_bce_keeps_master_scope_and_absent_gradient(self):
        context = self._context(1.0)
        final_logits = torch.linspace(-1.5, 1.5, 48).reshape(1, 3, 4, 4)
        final_logits.requires_grad_(True)

        losses = self.criterion.forward_chunk(
            outputs={OUTPUT_KEYS.final_logits: final_logits},
            context=context,
            class_start=0,
            class_end=3,
        )

        target, _ = self._targets_and_distill_mask(
            context,
            final_logits.dtype,
        )
        expected_final = (
            F.binary_cross_entropy_with_logits(
                final_logits,
                target,
                reduction="none",
            )
            * context.valid_mask[:, None]
        ).sum() / context.total_valid_pixels
        torch.testing.assert_close(
            losses["loss_final_bce"],
            expected_final,
        )

        losses["total_loss"].backward()
        self.assertGreater(
            float(final_logits.grad[:, 2].abs().sum().item()),
            0.0,
        )

    def test_dynamic_distill_matches_mixed_target_and_two_bces(self):
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

        target, distill_mask = self._targets_and_distill_mask(
            context,
            final_logits.dtype,
        )
        teacher_prob = teacher_logits.sigmoid()
        mixed_target = (
            context.gt_mix_ratio * target
            + context.teacher_mix_ratio * teacher_prob
        )
        expected_mixed = (
            F.binary_cross_entropy_with_logits(
                final_logits,
                mixed_target,
                reduction="none",
            )
            * distill_mask
        ).sum() / context.distill_pixel_denom

        gt_bce = (
            F.binary_cross_entropy_with_logits(
                final_logits,
                target,
                reduction="none",
            )
            * distill_mask
        ).sum() / context.distill_pixel_denom
        teacher_bce = (
            F.binary_cross_entropy_with_logits(
                final_logits,
                teacher_prob,
                reduction="none",
            )
            * distill_mask
        ).sum() / context.distill_pixel_denom
        expected_two_bces = (
            context.gt_mix_ratio * gt_bce
            + context.teacher_mix_ratio * teacher_bce
        )

        torch.testing.assert_close(
            losses["loss_sam3_mask_distill_bce"],
            expected_mixed,
        )
        torch.testing.assert_close(expected_mixed, expected_two_bces)
        torch.testing.assert_close(
            losses["total_loss"],
            self.FINAL_WEIGHT * losses["loss_final_bce"]
            + self.DISTILL_WEIGHT * expected_mixed,
        )

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

    def test_absent_only_chunk_has_final_bce_but_no_distill(self):
        context = self._context(0.0)
        final_logits = torch.zeros((1, 1, 4, 4), requires_grad=True)

        losses = self.criterion.forward_chunk(
            outputs={OUTPUT_KEYS.final_logits: final_logits},
            context=context,
            class_start=2,
            class_end=3,
        )

        self.assertGreater(float(losses["loss_final_bce"].item()), 0.0)
        self.assertEqual(
            float(losses["loss_sam3_mask_distill_bce"].item()),
            0.0,
        )
        torch.testing.assert_close(
            losses["total_loss"],
            self.FINAL_WEIGHT * losses["loss_final_bce"],
        )
        losses["total_loss"].backward()
        self.assertGreater(float(final_logits.grad.abs().sum().item()), 0.0)

    def test_present_chunk_requires_teacher_before_final_step(self):
        context = self._context(0.0)
        final_logits = torch.zeros((1, 2, 4, 4), requires_grad=True)

        with self.assertRaisesRegex(ValueError, "teacher_mix_ratio"):
            self.criterion.forward_chunk(
                outputs={OUTPUT_KEYS.final_logits: final_logits},
                context=context,
                class_start=0,
                class_end=2,
            )


if __name__ == "__main__":
    unittest.main()
