"""Tests for pose-stage failure diagnosis.

The thing under test here is not a computation, it is a *claim about cause*.
COLMAP leaves an empty sparse directory in two situations that need opposite
remedies -- it was killed, or it ran and gave up -- and the pipeline used to
assert the second one unconditionally. These tests pin the distinction so the
message cannot quietly revert to guessing.
"""

from __future__ import annotations

from pathlib import Path

from dronemap.stage3_pose import _diagnose_empty_model


LOG_PATH = Path("data/runs/r/logs/colmap_mapper.log")


def _progress_log(n: int) -> str:
    """A mapper log that got as far as registering ``n`` images, then stopped.

    COLMAP emits ``num_reg_frames=N`` *before* the Nth registration completes,
    which is why the diagnosis reports ``max(N) + 1``.
    """
    return "\n".join(
        f"I20260905 10:14:11.956517  6596 incremental_pipeline.cc:537] "
        f"Registering image #{i} (num_reg_frames={i - 1})"
        for i in range(1, n + 1)
    )


def test_nonzero_exit_after_progress_is_a_process_failure():
    """The real regression: 87/130 registered, killed by a concurrent OpenMVS job.

    The old message called this "too little overlap ... or too few frames" and
    advised adding keyframes -- which would have worsened the memory pressure
    that actually killed it.
    """
    msg = _diagnose_empty_model(
        log_text=_progress_log(88),
        mapper_rc=-1073741819,  # 0xC0000005 on Windows
        n_images=130,
        run_id="fixture_closeup",
        log_path=LOG_PATH,
    )
    assert "PROCESS failure" in msg
    assert "CAPTURE failure" not in msg
    # It must report how far it actually got, as evidence against "too few frames".
    assert "88 of 130" in msg
    # And it must not hand out the counterproductive advice.
    assert "Do NOT lower pose.init_min_tri_angle" in msg
    assert "dronemap run --run-id fixture_closeup" in msg


def test_clean_exit_without_model_is_a_capture_failure():
    """Exit code 0 and no model means the mapper genuinely could not solve it."""
    msg = _diagnose_empty_model(
        log_text="",
        mapper_rc=0,
        n_images=40,
        run_id="weak_capture",
        log_path=LOG_PATH,
    )
    assert "CAPTURE failure" in msg
    assert "PROCESS failure" not in msg
    assert "0 of 40" in msg
    # The overlap advice must carry the coupling, because raising target_overlap
    # alone is bounded by sharpness_window and provably does nothing.
    assert "sharpness_window" in msg


def test_process_failure_with_no_progress_omits_the_concurrency_hint():
    """A mapper that died immediately is not evidence about concurrent load.

    With fewer than 3 registrations there is no "progressing normally" to point
    at, so the message must not speculate about memory exhaustion.
    """
    msg = _diagnose_empty_model(
        log_text=_progress_log(1),
        mapper_rc=-1073741819,
        n_images=130,
        run_id="r",
        log_path=LOG_PATH,
    )
    assert "PROCESS failure" in msg
    assert "memory exhaustion" not in msg
    assert "Do NOT lower" not in msg


def test_exit_code_is_rendered_in_both_signed_and_hex_form():
    """Windows NTSTATUS codes are searchable in hex, POSIX signals in decimal."""
    msg = _diagnose_empty_model(
        log_text=_progress_log(50),
        mapper_rc=-1073740791,  # 0xC0000409 STATUS_STACK_BUFFER_OVERRUN
        n_images=97,
        run_id="r",
        log_path=LOG_PATH,
    )
    assert "-1073740791" in msg
    assert "0xC0000409" in msg


def test_diagnosis_always_cites_the_log():
    """Every failure has to point at the evidence, in both branches."""
    for rc in (0, -1073741819):
        msg = _diagnose_empty_model(
            log_text=_progress_log(10),
            mapper_rc=rc,
            n_images=20,
            run_id="r",
            log_path=LOG_PATH,
        )
        assert "colmap_mapper.log" in msg
