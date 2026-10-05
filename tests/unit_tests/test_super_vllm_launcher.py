# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import signal
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


SCRIPT = Path(__file__).resolve().parents[2] / "benchmarks/nemotron_3.5_super/sbatch_external_vllm.sh"
BASH = shutil.which("bash")


@unittest.skipUnless(BASH, "Launcher tests require Bash")
class TestSuperVllmLauncher(unittest.TestCase):
    def setUp(self) -> None:
        workdir = TemporaryDirectory(prefix="gym-launcher-")
        self.addCleanup(workdir.cleanup)
        self.workdir = workdir.name
        # Never inherit cluster credentials, tuning overrides, or real sbatch commands.
        self.env = {
            "PATH": os.defpath,
            # Bypass the sleep() stubs even in background functions on macOS Bash.
            "TEST_SLEEP": shutil.which("sleep", path=os.defpath),
            "USER": "launcher-test",
            "MODEL": "/test/model",
            "CONTAINER": "/test/image.sqsh",
            "MOUNTS": "/test:/test",
            "VLLM_CONFIG": "/dev/null",
            "EXPERIMENT_NAME": "launcher-test",
            "NUM_PREFILL_NODES": "4",
            "NUM_DECODE_NODES": "4",
            "SLURM_PROCID": "0",
            "SLURM_JOB_ID": "12345",
            "SLURM_JOB_USER": "launcher-test",
            "ROUTER_NODE": "node0",
            "ALL_NODES": "node0 node1 node2 node3 node4 node5 node6 node7",
        }

    def run_shell(self, command: str, *args: str, env: dict[str, str] | None = None) -> tuple[int, str, str]:
        proc = subprocess.Popen(
            [BASH, "--noprofile", "--norc", "-c", command, "launcher-test", *args],
            env=self.env | (env or {}),
            cwd=self.workdir,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        try:
            stdout, stderr = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            # The failure-path tests must terminate instead of leaking a polling worker.
            os.killpg(proc.pid, signal.SIGKILL)
            stdout, stderr = proc.communicate()
            self.fail(f"Launcher did not terminate. stdout={stdout!r}, stderr={stderr!r}")
        return proc.returncode, stdout, stderr

    def require_batch_bash(self) -> None:
        # The batch script uses [[ -v ]] and wait -n -p; macOS ships Bash 3.2.
        status, _, _ = self.run_shell("(( BASH_VERSINFO[0] > 5 || (BASH_VERSINFO[0] == 5 && BASH_VERSINFO[1] >= 1) ))")
        if status:
            self.skipTest("Generated batch scripts require Bash 5.1 or newer")

    def capture_submission(self, *eval_args, env=None):
        # Capture generated commands and both sbatch calls without submitting jobs.
        stub = r"""
sbatch() {
    if [[ -n "${vllm_command:-}" ]]; then
        printf '%s\0' "$eval_command" "$vllm_command" "$batch_command" >&2
    fi
    printf '%s\0' "$@" >&2
    printf '\0' >&2
    printf '12345\n'
}
launcher_script=$1
shift
source "$launcher_script" "$@"
"""
        status, _, captured = self.run_shell(stub, str(SCRIPT), *eval_args, env=env)
        self.assertEqual(status, 0, captured)
        eval_command, pd_command, batch_command, submissions = captured.split("\0", 3)
        self.assertTrue(submissions.endswith("\0\0"))
        calls = [call.split("\0") for call in submissions.removesuffix("\0\0").split("\0\0")]
        return eval_command, pd_command, batch_command, calls

    def generate_commands(self, *overrides, env=None):
        eval_command, pd_command, _, _ = self.capture_submission("--config", "benchmark.yaml", *overrides, env=env)
        return eval_command, pd_command

    def eval_arguments(self, *overrides: str, env: dict[str, str] | None = None) -> list[str]:
        command, _ = self.generate_commands(*overrides, env=env)
        command = command.replace("source /opt/Gym_venv/bin/activate", ":").replace("cd /opt/Gym\n", ":\n")
        stubs = r"""
# Supply model parameters for the default /dev/null fixture; real configs replace these.
GYM_MODEL_PARAMS=(++policy_model.responses_api_models.vllm_model.sampling_overrides.temperature=1.0)
gym() {
    if [[ "$2" == run ]]; then printf '%s\0' "$@"; fi
}
date() { printf '%s\n' "${TEST_DATE:-20260909_120000}"; }
getent() { printf '10.0.0.1 node0\n'; }
"""
        status, stdout, stderr = self.run_shell(stubs + command, env=env)
        self.assertEqual(status, 0, stderr)
        return stdout.rstrip("\0").split("\0")

    def settings(self, args, key):
        return [arg for arg in args if arg.lstrip("+").startswith(key + "=")]

    def test_submission_overrides_do_not_change_cleanup_allocation(self) -> None:
        """Apply custom walltime and segment size only to the main job, leaving cleanup CPU-only and short."""
        _, _, _, calls = self.capture_submission(
            "--config", "benchmark.yaml", env={"SBATCH_TIMELIMIT": "7-00:00:00", "SEGMENT": "4"}
        )
        self.assertEqual(len(calls), 2)
        self.assertIn("--time=7-00:00:00", calls[0])
        self.assertEqual([arg for arg in calls[0] if arg.startswith("--segment=")], ["--segment=4"])
        self.assertIn("--time=00:30:00", calls[1])
        self.assertIn("--nodes=1", calls[1])
        self.assertIn("--partition=cpu", calls[1])
        self.assertIn("--qos=cpu-normal", calls[1])
        self.assertIn("--gres=none", calls[1])
        self.assertFalse(any(arg.startswith("--segment=") for arg in calls[1]))

    def test_walltime_override_and_cleanup_isolation(self) -> None:
        """SBATCH_TIMELIMIT overrides the main job's default; cleanup has its own walltime."""
        for overrides, expected in (
            ({}, "04:00:00"),
            ({"SBATCH_TIMELIMIT": "06:00:00"}, "06:00:00"),
            ({"SBATCH_TIMELIMIT": "7-00:00:00"}, "7-00:00:00"),
            ({"SBATCH_TIMELIMIT": ""}, "04:00:00"),
        ):
            for mode in ("pd", "aggregated"):
                with self.subTest(overrides=overrides, mode=mode):
                    _, _, _, calls = self.capture_submission(
                        "--config", "benchmark.yaml", env={"VLLM_MODE": mode} | overrides
                    )
                    self.assertEqual([arg for arg in calls[0] if arg.startswith("--time=")], [f"--time={expected}"])
                    self.assertIn("--time=00:30:00", calls[1])
                    self.assertFalse(any(arg.startswith("--segment=") for arg in calls[1]))

    def test_serving_only_skips_evaluation_and_cleanup_submission(self) -> None:
        """Run only the serving step without eval arguments and preserve its success or failure status."""
        self.require_batch_bash()
        stubs = r"""
scontrol() { printf '%s\n' node0 node1 node2 node3 node4 node5 node6 node7; }
srun() { printf '%s\0' "$@"; printf '\0'; return "$TEST_SERVER_STATUS"; }
"""
        for mode in ("independent", "coupled"):
            for server_status in (0, 7):
                with self.subTest(mode=mode, server_status=server_status):
                    _, _, batch_command, calls = self.capture_submission(
                        env={"VLLM_PD_DEPLOYMENT_MODE": mode, "EXPERIMENT_NAME": "", "EXPORT_TO_CSV": "1"}
                    )
                    self.assertEqual(len(calls), 1)
                    self.assertIn("--job-name=gym-vllm_only-launcher-test", calls[0])
                    status, stdout, stderr = self.run_shell(
                        stubs + batch_command,
                        env={
                            "SLURM_JOB_NODELIST": "test-nodes",
                            "SLURM_SUBMIT_DIR": "/test",
                            "SLURM_CPUS_ON_NODE": "64",
                            "vllm_command": "fake-serving-command",
                            "eval_command": "fake-eval-command",
                            "TEST_SERVER_STATUS": str(server_status),
                        },
                    )
                    self.assertEqual(status, server_status, stderr)
                    steps = stdout.removesuffix("\0\0").split("\0\0")
                    self.assertEqual(len(steps), 1)
                    args = steps[0].split("\0")
                    self.assertIn("--nodes=8", args)
                    self.assertIn("--ntasks=8", args)
                    self.assertIn("--kill-on-bad-exit=1", args)
                    self.assertIn("fake-serving-command", args)
                    self.assertNotIn("--overlap", args)

    def test_default_evaluation_settings_are_unchanged(self):
        """Leave concurrency and resume to Gym and preserve timestamped output naming by default."""
        args = self.eval_arguments()
        self.assertEqual(self.settings(args, "num_samples_in_parallel"), [])
        self.assertEqual(self.settings(args, "resume_from_cache"), [])
        self.assertIn("++output_jsonl_fpath=results/launcher-test/slurm_job_id_12345/date_20260909_120000.jsonl", args)

    def test_continuation_settings_apply_only_to_main_job(self) -> None:
        for maximum in (0, 2):
            with self.subTest(maximum=maximum):
                _, _, _, calls = self.capture_submission(
                    "--config",
                    "benchmark.yaml",
                    env={"GYM_MAX_AUTO_CONTINUATIONS": str(maximum), "ROLLOUTS_FPATH": "results/fixed.jsonl"},
                )
                self.assertEqual("--requeue" in calls[0], maximum > 0)
                self.assertEqual("--signal=B:USR1@600" in calls[0], maximum > 0)
                self.assertNotIn("--requeue", calls[1])
                self.assertFalse(any(arg.startswith("--signal=") for arg in calls[1]))

    def test_continuation_runs_coverage_helper_and_reuses_fixed_output(self) -> None:
        Path(self.workdir, "benchmarks").symlink_to(SCRIPT.parent.parent, target_is_directory=True)
        Path(self.workdir, "fixed_materialized_inputs.jsonl").write_text("{}\n" * 100)
        for successful, expected_status in ((90, 75), (90, 77), (100, 0)):
            with self.subTest(successful=successful, status=expected_status):
                Path(self.workdir, "fixed.jsonl").write_text("{}\n" * successful)
                command, _ = self.generate_commands(
                    env={
                        "GYM_MAX_AUTO_CONTINUATIONS": "2",
                        "ROLLOUTS_FPATH": "fixed.jsonl",
                    }
                )
                command = command.replace("source /opt/Gym_venv/bin/activate", ":").replace("cd /opt/Gym\n", ":\n")
                stubs = r"""
GYM_MODEL_PARAMS=(++example=true)
gym() { if [[ $2 == run ]]; then printf '%s\0' "$@"; fi; }
getent() { printf '10.0.0.1 node0\n'; }
python() { "$TEST_PYTHON" "$@"; }
"""
                status, stdout, stderr = self.run_shell(
                    stubs + command, env={"TEST_PYTHON": sys.executable, "ROLLOUTS_FPATH": "fixed.jsonl"}
                )
                self.assertEqual(status, expected_status, stderr)
                args = stdout.rstrip("\0").split("\0")
                self.assertEqual(self.settings(args, "resume_from_cache"), ["++resume_from_cache=true"])
                self.assertIn("++output_jsonl_fpath=fixed.jsonl", args)

    def test_invalid_continuation_never_submits(self) -> None:
        base = {"GYM_MAX_AUTO_CONTINUATIONS": "2", "ROLLOUTS_FPATH": "results/fixed.jsonl"}
        for overrides, arguments in (
            ({"GYM_MAX_AUTO_CONTINUATIONS": "-1"}, ["--config", "benchmark.yaml"]),
            ({"GYM_MAX_AUTO_CONTINUATIONS": "02"}, ["--config", "benchmark.yaml"]),
            ({"GYM_MAX_AUTO_RETRY_FAILURE_PERCENT": "101"}, ["--config", "benchmark.yaml"]),
            ({"ROLLOUTS_FPATH": ""}, ["--config", "benchmark.yaml"]),
            ({}, []),
            ({}, ["--config", "benchmark.yaml", "++resume_from_cache=false"]),
            ({}, ["--config", "benchmark.yaml", "+resume_from_cache=true"]),
        ):
            with self.subTest(overrides=overrides, arguments=arguments):
                status, stdout, _ = self.run_shell(
                    'sbatch() { printf "unexpected-submission"; }; launcher=$1; shift; source "$launcher" "$@"',
                    str(SCRIPT),
                    *arguments,
                    env=base | overrides,
                )
                self.assertEqual(status, 1)
                self.assertNotIn("unexpected-submission", stdout)

    def test_batch_continuation_is_bounded_and_preserves_failures(self) -> None:
        self.require_batch_bash()
        stubs = r"""
scontrol() {
    if [[ $1 == requeue ]]; then
        printf 'requeue=%s\n' "$2"
        return "$TEST_REQUEUE_STATUS"
    fi
    printf '%s\n' node0 node1
}
srun() {
    if [[ $1 == --overlap ]]; then
        while [[ ! -f server-ready ]]; do "$TEST_SLEEP" 0.01; done
        if [[ $TEST_SIGNAL == 1 ]]; then
            trap 'exit 0' TERM
            kill -USR1 "$$"
            while true; do "$TEST_SLEEP" 0.01; done
        fi
        return "$TEST_EVAL_STATUS"
    fi
    trap 'exit 0' TERM
    touch server-ready
    while true; do "$TEST_SLEEP" 0.01; done
}
"""
        for maximum, restart, eval_status, send_signal, requeue_status, expected in (
            (2, 0, 75, 0, 0, 0),
            (2, 1, 75, 0, 0, 0),
            (2, 2, 75, 0, 0, 124),
            (0, 0, 75, 0, 0, 75),
            (2, 0, 76, 0, 0, 76),
            (2, 0, 77, 0, 0, 77),
            (2, 0, 78, 0, 0, 78),
            (2, 0, 1, 0, 0, 1),
            (2, 0, 0, 0, 0, 0),
            (2, 0, 75, 0, 9, 9),
            (2, 0, 0, 1, 0, 0),
            (2, 2, 0, 1, 0, 124),
        ):
            with self.subTest(maximum=maximum, restart=restart, status=eval_status, signal=send_signal):
                Path(self.workdir, "server-ready").unlink(missing_ok=True)
                _, _, command, _ = self.capture_submission(
                    "--config",
                    "benchmark.yaml",
                    env={"GYM_MAX_AUTO_CONTINUATIONS": str(maximum), "ROLLOUTS_FPATH": "results/fixed.jsonl"},
                )
                status, stdout, stderr = self.run_shell(
                    stubs + command,
                    env={
                        "SLURM_JOB_NODELIST": "test-nodes",
                        "SLURM_SUBMIT_DIR": "/test",
                        "SLURM_CPUS_ON_NODE": "64",
                        "vllm_command": "unused",
                        "eval_command": "unused",
                        "SLURM_RESTART_COUNT": str(restart),
                        "TEST_EVAL_STATUS": str(eval_status),
                        "TEST_SIGNAL": str(send_signal),
                        "TEST_REQUEUE_STATUS": str(requeue_status),
                    },
                )
                self.assertEqual(status, expected, stderr)
                should_requeue = maximum > restart and (eval_status == 75 or send_signal)
                self.assertEqual(stdout.count("requeue=12345\n"), int(bool(should_requeue)))

    def test_explicit_concurrency_arguments_are_preserved(self):
        """Pass explicit concurrency values through to Gym without replacing or duplicating them."""
        for prefix in ("", "+", "++"):
            for value in (16, 32, 512):
                with self.subTest(prefix=prefix, value=value):
                    override = f"{prefix}num_samples_in_parallel={value}"
                    args = self.eval_arguments(override)
                    self.assertEqual(self.settings(args, "num_samples_in_parallel"), [override])

    def test_explicit_resume_arguments_are_preserved(self):
        """Pass explicit resume values through to Gym without changing the supplied output path."""
        for prefix in ("", "+", "++"):
            for value in ("true", "false"):
                with self.subTest(prefix=prefix, value=value):
                    override = prefix + "resume_from_cache=" + value
                    args = self.eval_arguments(override, env={"ROLLOUTS_FPATH": "results/existing/resumable.jsonl"})
                    self.assertEqual(self.settings(args, "resume_from_cache"), [override])
                    self.assertIn("++output_jsonl_fpath=results/existing/resumable.jsonl", args)

    def test_resume_without_fixed_path_uses_restart_timestamp(self):
        """Show that enabling resume alone still selects a new output path when the same job restarts."""
        for timestamp in ("20260909_120000", "20260909_180000"):
            with self.subTest(timestamp=timestamp):
                args = self.eval_arguments("++resume_from_cache=true", env={"TEST_DATE": timestamp})
                self.assertEqual(self.settings(args, "resume_from_cache"), ["++resume_from_cache=true"])
                self.assertIn(
                    f"++output_jsonl_fpath=results/launcher-test/slurm_job_id_12345/date_{timestamp}.jsonl", args
                )

    def test_explicit_resume_keeps_output_path_across_restarts(self):
        """Keep the supplied resume path across requeues and new job IDs, with timestamped logs and W&B names."""
        for job_id in ("12345", "12346"):
            for timestamp in ("20260909_120000", "20260909_180000"):
                with self.subTest(job_id=job_id, timestamp=timestamp):
                    args = self.eval_arguments(
                        "++resume_from_cache=true",
                        env={
                            "ROLLOUTS_FPATH": "results/existing/resumable.jsonl",
                            "SLURM_JOB_ID": job_id,
                            "TEST_DATE": timestamp,
                        },
                    )
                    self.assertEqual(self.settings(args, "resume_from_cache"), ["++resume_from_cache=true"])
                    self.assertIn("++output_jsonl_fpath=results/existing/resumable.jsonl", args)
                    experiment_name = f"launcher-test/slurm_job_id_{job_id}/date_{timestamp}"
                    self.assertIn(f"+wandb_name={experiment_name}", args)
                    self.assertIn(f"+nemo_gym_log_dir=results/{experiment_name}/logs", args)
