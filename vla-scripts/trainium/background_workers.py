"""
Background workers for non-blocking checkpointing and validation during Trainium training.
"""

import multiprocessing as mp
import queue
import gc
import warnings
from pathlib import Path
from typing import Dict, Optional, Any
from dataclasses import dataclass

import torch


@dataclass 
class WorkerMessage:
    """Message passed to background workers."""
    step: int
    state_dict: Dict[str, torch.Tensor]
    run_dir: Path
    extra: Optional[Dict[str, Any]] = None


class BackgroundWorker:
    """Base class for background workers using multiprocessing."""
    
    def __init__(self, name: str):
        self.name = name
        self._process: Optional[mp.Process] = None
        self._queue: Optional[mp.Queue] = None
        self._cancel_event: Optional[mp.Event] = None
        self._busy_event: Optional[mp.Event] = None
    
    def start(self):
        self._queue = mp.Queue(maxsize=1)
        self._cancel_event = mp.Event()
        self._busy_event = mp.Event()
        self._process = mp.Process(target=self._run_loop, daemon=True)
        self._process.start()
    
    def stop(self):
        if self._process and self._process.is_alive():
            self._queue.put(None)
            self._process.join(timeout=5)
            if self._process.is_alive():
                self._process.terminate()
    
    def submit(self, msg: WorkerMessage) -> bool:
        """Submit work, cancelling any in-progress work with warning."""
        if self._busy_event.is_set():
            warnings.warn(f"[{self.name}] Cancelling previous work - consider increasing interval")
            self._cancel_event.set()
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
        
        self._cancel_event.clear()
        self._queue.put(msg)
        return True
    
    def is_busy(self) -> bool:
        return self._busy_event.is_set()
    
    def _run_loop(self):
        while True:
            try:
                msg = self._queue.get()
                if msg is None:
                    break
                self._busy_event.set()
                try:
                    self._process_work(msg, self._cancel_event)
                finally:
                    self._busy_event.clear()
            except Exception as e:
                print(f"[{self.name}] Error: {e}")
                self._busy_event.clear()
    
    def _process_work(self, msg: WorkerMessage, cancel_event: mp.Event):
        raise NotImplementedError


class BackgroundCheckpointer(BackgroundWorker):
    """Saves checkpoints in background."""
    
    def __init__(self):
        super().__init__("Checkpointer")
    
    def _process_work(self, msg: WorkerMessage, cancel_event: mp.Event):
        checkpoint_dir = msg.run_dir / f"checkpoint-{msg.step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        torch.save(msg.state_dict, checkpoint_dir / "pytorch_model.bin")
        torch.save({"step": msg.step, **(msg.extra or {})}, checkpoint_dir / "training_state.pt")
        
        print(f"[Checkpointer] Saved step {msg.step} to {checkpoint_dir}")


class BackgroundValidator(BackgroundWorker):
    """Runs LIBERO validation rollouts in background on CPU."""
    
    def __init__(self, vla_path: str, task_suites: list, unnorm_keys: Dict[str, str],
                 num_episodes: int = 1, num_videos: int = 1, center_crop: bool = True):
        super().__init__("Validator")
        self.vla_path = vla_path
        self.task_suites = task_suites
        self.unnorm_keys = unnorm_keys
        self.num_episodes = num_episodes
        self.num_videos = num_videos
        self.center_crop = center_crop
        self._results_queue: Optional[mp.Queue] = None
    
    def start(self):
        super().start()
        self._results_queue = mp.Queue()
    
    def get_results(self) -> Optional[Dict]:
        try:
            return self._results_queue.get_nowait()
        except queue.Empty:
            return None
    
    def _process_work(self, msg: WorkerMessage, cancel_event: mp.Event):
        from transformers import AutoModelForVision2Seq, AutoProcessor
        from experiments.robot.libero.libero_eval_utils import run_libero_rollouts
        
        step = msg.step
        print(f"[Validator] Starting validation at step {step}...")
        
        model = AutoModelForVision2Seq.from_pretrained(
            self.vla_path, torch_dtype=torch.float32,
            low_cpu_mem_usage=True, trust_remote_code=True,
        )
        model.load_state_dict(msg.state_dict, strict=False)
        model.eval()
        
        if msg.extra and "dataset_stats" in msg.extra:
            model.norm_stats = msg.extra["dataset_stats"]
        
        processor = AutoProcessor.from_pretrained(self.vla_path, trust_remote_code=True)
        
        results = {"step": step, "success_rates": {}, "videos": {}}
        
        for suite in self.task_suites:
            if cancel_event.is_set():
                print(f"[Validator] Cancelled at step {step}")
                return
            
            success_rate, videos_dict = run_libero_rollouts(
                model, processor, suite, self.unnorm_keys.get(suite, suite),
                num_episodes=self.num_episodes, num_videos=self.num_videos,
                center_crop=self.center_crop,
            )
            results["success_rates"][suite] = success_rate
            results["videos"][suite] = videos_dict
            print(f"[Validator] {suite}: {success_rate:.1%}")
        
        if any(results["videos"].values()):
            self._save_videos(results["videos"], msg.run_dir, step)
        
        self._results_queue.put(results)
        del model, processor
        gc.collect()
        print(f"[Validator] Completed step {step}")
    
    def _save_videos(self, videos_dict: Dict, run_dir: Path, step: int):
        try:
            import imageio
        except ImportError:
            return
        
        video_dir = run_dir / "videos" / f"step-{step}"
        video_dir.mkdir(parents=True, exist_ok=True)
        
        for suite, task_videos in videos_dict.items():
            for task_name, rollouts in task_videos.items():
                for idx, (frames, success) in enumerate(rollouts):
                    safe_name = task_name.replace(" ", "_")[:50]
                    status = "success" if success else "fail"
                    path = video_dir / f"{suite}_{safe_name}_{idx}_{status}.mp4"
                    try:
                        imageio.mimwrite(str(path), frames, fps=30)
                    except Exception as e:
                        print(f"[Validator] Video save failed: {e}")


def gather_weights_to_cpu(model, tp_size: int) -> Dict[str, torch.Tensor]:
    """Gather TP-sharded weights to CPU on rank 0."""
    import torch_xla.core.xla_model as xm
    import torch_xla.runtime as xr
    
    rank = xr.global_ordinal()
    
    if tp_size == 1:
        return {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    cpu_state_dict = {}
    
    for name, param in model.named_parameters():
        param_cpu = param.detach().cpu()
        
        is_tp_sharded = any(x in name for x in [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj", "embed_tokens", "lm_head"
        ]) and "language_model" in name
        
        if is_tp_sharded:
            gathered = [torch.zeros_like(param_cpu) for _ in range(tp_size)]
            torch.distributed.all_gather(gathered, param_cpu)
            
            if any(x in name for x in ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "embed_tokens", "lm_head"]):
                full_param = torch.cat(gathered, dim=0)
            else:  # o_proj, down_proj - row parallel
                full_param = torch.cat(gathered, dim=1)
            
            if rank == 0:
                cpu_state_dict[name] = full_param
        elif rank == 0:
            cpu_state_dict[name] = param_cpu
    
    for name, buf in model.named_buffers():
        if rank == 0:
            cpu_state_dict[name] = buf.detach().cpu()
    
    xm.rendezvous("gather_weights")
    return cpu_state_dict if rank == 0 else {}


class BackgroundWorkersManager:
    """Manages background checkpointing and validation."""
    
    def __init__(self, run_dir: Path, vla_path: str, task_suites: list,
                 unnorm_keys: Dict[str, str], dataset_stats: Dict,
                 val_episodes: int = 1, val_videos: int = 1, center_crop: bool = True):
        self.run_dir = Path(run_dir)
        self.dataset_stats = dataset_stats
        
        self.checkpointer = BackgroundCheckpointer()
        self.validator = BackgroundValidator(
            vla_path=vla_path, task_suites=task_suites, unnorm_keys=unnorm_keys,
            num_episodes=val_episodes, num_videos=val_videos, center_crop=center_crop,
        )
    
    def start(self):
        self.checkpointer.start()
        self.validator.start()
    
    def stop(self):
        self.checkpointer.stop()
        self.validator.stop()
    
    def submit(self, model, step: int, tp_size: int):
        """Gather weights and submit to background workers (rank 0 only submits)."""
        import torch_xla.core.xla_model as xm
        
        cpu_state_dict = gather_weights_to_cpu(model, tp_size)
        
        if not xm.is_master_ordinal():
            return
        
        msg = WorkerMessage(
            step=step, state_dict=cpu_state_dict, run_dir=self.run_dir,
            extra={"dataset_stats": self.dataset_stats},
        )
        self.checkpointer.submit(msg)
        self.validator.submit(msg)
    
    def get_validation_results(self) -> Optional[Dict]:
        return self.validator.get_results()
