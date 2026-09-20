"""Retriever-driven kitchen motion with a compact, recording-friendly console."""

import argparse
from collections import deque
from dataclasses import dataclass
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import signal
import sys
import threading
import time
from urllib.parse import urlsplit

import numpy as np
import mujoco
import viser
from retriever.flow import Flow, Latest, Pipeline, Rate, Trigger, io

ROOT = Path(__file__).resolve().parent
from preview_support.embodied import EmbodiedGoal, SkillPlan, SkillStep
from preview_support.runtime import ReplayControls
from preview_support.web_console import RetrieverWebConsole, snapshot_payload

from motion_demo import CupTransfer
from seasoning_demo import SeasoningDemo
from drawer_motion import DrawerPull
from organizer_search import OrganizerSearch
from simulation_visuals import KitchenMujocoScene
from drawer_inspection import DrawerInspection


@io
@dataclass
class Tick:
    frame: int = 0
    cycle: int = 0
    active: bool = False


@io
@dataclass
class Command:
    frame: int = 0
    cycle: int = 0
    active: bool = False
    joints: np.ndarray | None = None
    opening: float = 0.04


@io
@dataclass
class Observation:
    frame: int = 0
    cycle: int = 0
    active: bool = False
    phase: int = 0
    progress: float = 0.0
    finished: bool = False
    success: bool = False


class Runtime:
    def __init__(self, task="seasoning", viewer_port=8106, output_dir=None):
        self.lock = threading.RLock()
        self.viewer_port = viewer_port
        self.output_dir = Path(output_dir or Path.cwd() / "runs").resolve()
        if self.output_dir.is_relative_to(ROOT):
            raise ValueError("Runtime outputs must be outside the installed scene resources")
        self.demo = {"seasoning": SeasoningDemo, "cup": CupTransfer, "drawer": DrawerPull, "search": OrganizerSearch}[task]()
        self.task_name = task
        self.inspection = DrawerInspection(self.demo.model, self.demo.data)
        title = {"seasoning": "Seasoning", "cup": "Cup transfer", "drawer": "Drawer opening", "search": "Drawer search"}[task]
        self.controls = ReplayControls(task=title, episode=0)
        goal = EmbodiedGoal(text={"seasoning": "Season the food", "cup": "Move the cup beside the sink", "drawer": "Open the top-left drawer", "search": "Find the herb seasoning and place it on the counter"}[task], task=title,
                            execution_mode="demonstration", planner="scripted")
        skills = (("locate", "locate", "pick", "pick", "place", "execute_demo", "execute_demo",
                   "execute_demo", "place", "place", "place", "place", "verify") if task == "seasoning" else
                   ("locate", "locate", "pick", "pick", "place", "place", "place", "place", "verify"))
        if task == "drawer":
            skills = ("locate", "locate", "pick", "execute_demo", "place", "place", "verify")
        if task == "search":
            skills = tuple("verify" if "Verify" in label else "locate" if "Inspect" in label else "execute_demo" for label in self.demo.labels)
        count = len(self.demo.labels)
        plan = SkillPlan(goal=goal, source="scripted", steps=tuple(
            SkillStep(step_id=f"skill-{i}", skill=skill, label=label,
                      stage_id=task, stage_label=title,
                      start_fraction=i / count, end_fraction=(i + 1) / count,
                      depends_on=(f"skill-{i - 1}",) if i else ())
            for i, (skill, label) in enumerate(zip(skills, self.demo.labels))))
        self.controls.configure_execution(goal, plan)
        self.goal = goal
        self.controls.set_total_steps(1001)
        self.controls.set_paused(True)
        if task == "search":
            self.controls.set_speed(4)
        self.trace = {name: {"count": 0, "last_ms": 0.0} for name in
                      ("skill_dispatcher", "inverse_kinematics", "mujoco_simulator", "task_verifier", "event_sink")}
        self.scene = None
        self.last_scene_update = 0.0
        self.clients = {}
        self.cycle = 0
        self.timeline = deque(maxlen=160)
        self.event_serial = 0
        self.logged_phase = None
        self.logged_ik_phase = None
        self.observation_count = 0
        self.camera_name = "Overview" if task == "search" else "Agent"
        self.intro_started = None
        self.tour_active = False
        self.tour_finished_at = None
        self.last_camera_update = 0.0
        self.console = RetrieverWebConsole(self.controls, f"http://127.0.0.1:{viewer_port}", camera_handler=self.camera)

    def emit(self, source, message):
        self.event_serial += 1
        self.timeline.append(dict(id=self.event_serial, source=source, message=message,
                                  time=round(float(self.demo.data.time), 2)))

    def reset_timeline(self):
        self.timeline.clear()
        self.logged_phase = self.logged_ik_phase = None
        self.observation_count = 0
        if self.task_name == "search" and self.demo.memory_used:
            self.emit("memory", "Recall: herb seasoning in " + self.demo.drawer_locations[self.demo.target_index])

    def refresh_search_plan(self):
        count = len(self.demo.labels)
        paused = self.controls.snapshot().paused
        self.controls.configure_execution(self.goal, SkillPlan(goal=self.goal, source="scripted", steps=tuple(
            SkillStep(step_id=f"skill-{i}", skill="verify" if "Verify" in label else "locate" if "Inspect" in label else "execute_demo",
                      label=label, stage_id="search", stage_label="Drawer search",
                      start_fraction=i / count, end_fraction=(i + 1) / count,
                      depends_on=(f"skill-{i - 1}",) if i else ())
            for i, label in enumerate(self.demo.labels))))
        self.controls.set_paused(paused)

    def mark(self, name, started):
        self.trace[name]["count"] += 1
        self.trace[name]["last_ms"] = round((time.perf_counter() - started) * 1000, 2)

    def state(self):
        with self.lock:
            payload = snapshot_payload(self.controls.snapshot())
            for event in payload["events"]:
                if event["kind"] == "verification" and event["status"] == "verified":
                    event["message"] = {"seasoning": "Seasoning motion and shaker return verified",
                                        "cup": "Cup placement verified by MuJoCo state",
                                        "drawer": "Drawer opening verified with maintained finger contact",
                                        "search": "Herb seasoning placed on counter; all drawers closed"}[self.task_name]
            return {**payload, "motion": self.demo.snapshot(),
                    "timeline": list(self.timeline), "run_cycle": self.cycle,
                    "trace": {name: values.copy() for name, values in self.trace.items()}, "pipeline": "kitchen_" + self.task_name,
                    "engine": "Retriever / Mink IK / MuJoCo", "camera": self.camera_name,
                    "camera_opening": self.intro_started is not None,
                    "camera_tour": self.tour_active,
                    "inspection": self.inspection.snapshot()}

    def inspect_drawer(self, name, fraction=None):
        self.inspection.move(name, fraction)
        self.intro_started = None
        self.tour_active = False
        self.tour_finished_at = None
        self.controls.set_paused(True)

    def update_inspection(self):
        if self.inspection.update() and self.scene:
            self.scene.update_from_mjdata(self.demo.data)

    def camera_pose(self, name):
        cameras = {
            "Agent": ((3.7, -4.7, 2.7), (-0.25, 0.9, 1.1)),
            "Robot": ((2.0, -0.25, 2.05), (0.40, 1.30, 1.02)),
            "Overview": ((4.3, -4.6, 3.7), (-0.15, 0.7, 1.05)),
            "Aloha": ((1.45, 1.65, 1.90), (-0.92, 2.40, 0.78)),
        }
        if self.task_name == "drawer":
            cameras["Robot"] = ((1.30, 0.55, 1.55), (-0.10, 2.35, 0.80))
        if self.task_name == "search":
            cameras["Agent"] = ((3.40, -2.80, 2.35), (0.35, 1.05, 1.0))
            cameras["Robot"] = ((2.65, -0.60, 2.10), (0.75, 0.85, 1.12))
        return cameras[name]

    def apply_camera(self, eye, target):
        for client in tuple(self.clients.values()):
            aspect = client.camera.aspect or 1.6
            factor = max(1, (1.2 / aspect) ** 0.7)
            with client.atomic():
                client.camera.position = np.array(target) + (np.array(eye) - target) * factor
                client.camera.look_at = target
                client.camera.up_direction = (0, 0, 1)
                client.camera.fov = np.deg2rad(48)

    def camera(self, name):
        eye, target = self.camera_pose(name)
        self.tour_active = False
        self.tour_finished_at = None
        self.intro_started = None
        self.camera_name = name
        self.apply_camera(eye, target)
        self.controls.set_camera_preset(name)

    def start_presentation(self):
        self.inspection.clear()
        self.controls.request_restart()
        self.controls.set_paused(True)
        self.camera("Overview")
        self.intro_started = time.monotonic()
        self.last_camera_update = 0.0

    def start_tour(self):
        if self.task_name == "search":
            self.demo.use_memory = False
        self.start_presentation()
        self.controls.set_speed(1)
        self.tour_active = True

    def tour_pose(self, elapsed):
        # Stay on the room's open sides so walls never hide the manipulation.
        points = [
            (0, *self.camera_pose("Overview")),
            (3, *self.camera_pose("Overview")),
            (10, (4.6, -2.7, 3.0), (0.15, 0.9, 1.05)),
            (24, *self.camera_pose("Agent")),
            (40, (2.9, -1.05, 2.25), (0.7, 0.95, 1.12)),
            (56, *self.camera_pose("Robot")),
            (10000, *self.camera_pose("Robot")),
        ]
        for (start, eye_a, target_a), (end, eye_b, target_b) in zip(points, points[1:]):
            if elapsed <= end:
                t = float(np.clip((elapsed - start) / (end - start), 0, 1))
                blend = t * t * (3 - 2 * t)
                return tuple(np.asarray(a) * (1 - blend) + np.asarray(b) * blend
                             for a, b in ((eye_a, eye_b), (target_a, target_b)))
        return self.camera_pose("Robot")

    def update_tour(self, now):
        if self.intro_started is not None:
            elapsed = now - self.intro_started
            self.apply_camera(*self.tour_pose(min(elapsed, 10)))
            if elapsed >= 10:
                self.intro_started = None
                self.controls.set_paused(False)
            return
        if self.controls.snapshot().paused:
            return
        if not self.demo.finished:
            self.apply_camera(*self.tour_pose(10 + float(self.demo.data.time)))
            return
        if self.tour_finished_at is None:
            self.tour_finished_at = now
        t = float(np.clip((now - self.tour_finished_at - 4) / 8, 0, 1))
        blend = t * t * (3 - 2 * t)
        close = self.tour_pose(10 + float(self.demo.data.time))
        wide = self.camera_pose("Overview")
        self.apply_camera(*(np.asarray(a) * (1 - blend) + np.asarray(b) * blend
                            for a, b in zip(close, wide)))
        if now - self.tour_finished_at >= 15:
            self.camera("Overview")

    def update_presentation(self, now=None):
        if self.intro_started is None and not self.tour_active:
            return
        now = time.monotonic() if now is None else now
        if now - self.last_camera_update < 1 / 30:
            return
        self.last_camera_update = now
        if self.tour_active:
            self.update_tour(now)
            return
        t = np.clip((now - self.intro_started - 0.8) / 2.4, 0, 1)
        blend = t * t * (3 - 2 * t)
        wide, close = self.camera_pose("Overview"), self.camera_pose("Robot")
        eye, target = [np.asarray(a) * (1 - blend) + np.asarray(b) * blend
                       for a, b in zip(wide, close)]
        self.apply_camera(eye, target)
        if t >= 1:
            self.camera("Robot")
            self.controls.set_paused(False)


class Dispatch(Flow[None, Tick]):
    def __init__(self, runtime):
        self.runtime = runtime

    def step(self, _=None):
        runtime = self.runtime
        started = time.perf_counter()
        active, reset = runtime.controls.claim_next_action()
        if reset is not None:
            runtime.inspection.clear()
            runtime.demo.reset()
            if runtime.task_name == "search":
                runtime.refresh_search_plan()
            runtime.cycle = reset
            runtime.reset_timeline()
            if runtime.scene:
                runtime.scene.update_from_mjdata(runtime.demo.data)
        active = active and not runtime.demo.finished and runtime.intro_started is None and not runtime.inspection.active
        if reset is not None and runtime.controls.snapshot().paused:
            active = False
        if active:
            if runtime.logged_phase != runtime.demo.phase:
                runtime.logged_phase = runtime.demo.phase
                runtime.emit("dispatch", runtime.demo.labels[runtime.demo.phase])
            runtime.mark("skill_dispatcher", started)
        return Tick(runtime.demo.frame, runtime.cycle, active)


class InverseKinematics(Flow[Tick, Command]):
    def __init__(self, runtime):
        self.runtime = runtime

    def step(self, tick):
        if not tick.active:
            return Command(tick.frame, tick.cycle)
        started = time.perf_counter()
        joints, opening = self.runtime.demo.prepare()
        if self.runtime.logged_ik_phase != self.runtime.demo.phase:
            self.runtime.logged_ik_phase = self.runtime.demo.phase
            self.runtime.emit("motion", "Joint target solved for " + self.runtime.demo.labels[self.runtime.demo.phase].lower())
        self.runtime.mark("inverse_kinematics", started)
        return Command(tick.frame, tick.cycle, True, joints, opening)


class Simulation(Flow[Command, Observation]):
    def __init__(self, runtime):
        self.runtime = runtime

    def step(self, command):
        if not command.active:
            return Observation(command.frame, command.cycle)
        started = time.perf_counter()
        previous_phase = self.runtime.demo.phase
        was_placed = getattr(self.runtime.demo, "object_placed", False)
        state = self.runtime.demo.integrate(command.joints, command.opening)
        if state["phase"] != previous_phase or state["finished"]:
            outcome = "Stopped: " if state["finished"] and not state["success"] else "Completed: "
            self.runtime.emit("simulation", outcome + self.runtime.demo.labels[previous_phase])
        for observed in state.get("observations", [])[self.runtime.observation_count:]:
            item = observed["item"] or "contents not visible"
            self.runtime.emit("simulation", f'{observed["drawer"].capitalize()} open {observed["opening_m"] * 100:.1f} cm: {item}')
            if observed["item"]:
                self.runtime.emit("memory", f'Stored {observed["drawer"]}: {item}')
        self.runtime.observation_count = len(state.get("observations", []))
        if state.get("object_placed") and not was_placed:
            self.runtime.emit("memory", "Herb seasoning location updated: on counter")
        if self.runtime.scene and (state["finished"] or state["phase"] != previous_phase or
                                   time.perf_counter() - self.runtime.last_scene_update >= 1 / 30):
            self.runtime.scene.update_from_mjdata(self.runtime.demo.data)
            self.runtime.last_scene_update = time.perf_counter()
        self.runtime.mark("mujoco_simulator", started)
        return Observation(state["frame"], command.cycle, True, state["phase"],
                           state["progress"], state["finished"], state["success"])


class Verify(Flow[Observation, Observation]):
    def __init__(self, runtime):
        self.runtime = runtime

    def step(self, observation):
        if observation.active:
            started = time.perf_counter()
            self.runtime.controls.update_observation(
                episode_step=round(observation.progress * 1000), cycle=observation.cycle,
                progress=observation.progress, reward=float(observation.success),
                success=observation.success, action_norm=float(np.linalg.norm(self.runtime.demo.data.ctrl)))
            if observation.finished:
                self.runtime.controls.mark_complete()
                message = ("Retrieval verified: seasoning released on counter, all drawers closed."
                           if self.runtime.task_name == "search" else "Task complete")
                self.runtime.emit("verification", message if observation.success else self.runtime.demo.error or "Task verification failed")
            self.runtime.mark("task_verifier", started)
        return Observation(observation.frame, observation.cycle, observation.active,
                           observation.phase, observation.progress, observation.finished,
                           observation.success)


class Events(Flow[Observation, None]):
    def __init__(self, runtime):
        self.runtime = runtime

    def step(self, observation):
        if observation.active:
            started = time.perf_counter()
            self.runtime.mark("event_sink", started)


def make_pipeline(runtime):
    pipeline = Pipeline("kitchen_" + runtime.task_name, on_lag="catch_up")
    with pipeline:
        source = (Dispatch(runtime) @ Rate(hz=50)).named("skill_dispatcher")
        ik = (InverseKinematics(runtime) @ Trigger("frame", "cycle", "active")).named("inverse_kinematics")
        sim = (Simulation(runtime) @ Trigger("frame", "cycle", "active")).named("mujoco_simulator")
        verify = (Verify(runtime) @ Trigger("frame", "cycle", "active")).named("task_verifier")
        events = (Events(runtime) @ Trigger("frame", "cycle", "active")).named("event_sink")
        source.then(ik, sync=Latest())
        ik.then(sim, sync=Latest())
        sim.then(verify, sync=Latest())
        verify.then(events, sync=Latest())
    return pipeline


def serve(runtime, port):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, status, content, content_type="application/json"):
            encoded = json.dumps(content).encode() if content_type == "application/json" else content
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self):
            if self.path == "/":
                html = (ROOT / "teaser.html").read_text().replace("__TOKEN__", runtime.console.control_token)
                html = html.replace("__VIEWER_PORT__", str(runtime.viewer_port))
                self.send(200, html.encode(), "text/html; charset=utf-8")
            elif self.path == "/api/state":
                self.send(200, runtime.state())
            elif urlsplit(self.path).path == "/viewer":
                # Reuse the installed, self-contained Viser client without altering it.
                client = Path(viser.__file__).parent / "client/build/index.html"
                style = "<style>.mantine-Paper-root,[data-floating-window]{display:none!important}</style>"
                html = client.read_text().replace("</head>", style + "</head>")
                self.send(200, html.encode(), "text/html; charset=utf-8")
            elif self.path == "/lucide.js":
                self.send(200, (ROOT / "vendor/lucide.js").read_bytes(), "text/javascript")
            elif self.path == "/html2canvas.js":
                self.send(200, (ROOT / "vendor/html2canvas.js").read_bytes(), "text/javascript")
            elif self.path == "/record.js":
                self.send(200, (ROOT / "record.js").read_bytes(), "text/javascript")
            elif self.path in ("/console.js", "/console.css"):
                self.send(200, (ROOT / self.path[1:]).read_bytes(),
                          "text/javascript" if self.path.endswith(".js") else "text/css")
            else:
                self.send(404, {"error": "Not found"})

        def do_POST(self):
            host = self.headers.get("Host", "")
            origin = self.headers.get("Origin", "")
            if host not in {f"127.0.0.1:{port}", f"localhost:{port}"} or (origin and origin != f"http://{host}"):
                self.send(403, {"error": "Local same-origin commands only"})
                return
            if not hmac.compare_digest(self.headers.get("X-Retriever-Token", ""), runtime.console.control_token):
                self.send(403, {"error": "Invalid control token"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if self.path == "/api/recording":
                    if not 1000 <= length <= 100_000_000:
                        raise ValueError("Invalid recording size")
                    content_type = self.headers.get("Content-Type", "").split(";")[0]
                    if content_type not in ("video/webm", "video/mp4"):
                        raise ValueError("Unsupported recording format")
                    suffix = ".mp4" if content_type == "video/mp4" else ".webm"
                    runtime.output_dir.mkdir(parents=True, exist_ok=True)
                    path = runtime.output_dir / ("retriever-" + runtime.task_name + "-" + time.strftime("%Y%m%d-%H%M%S") + suffix)
                    metadata = json.loads(self.headers.get("X-Recording-Metadata", "{}"))
                    if not isinstance(metadata, dict):
                        raise ValueError("Invalid recording metadata")
                    content = self.rfile.read(length)
                    if len(content) != length:
                        raise ValueError("Incomplete recording")
                    with path.open("xb") as output:
                        output.write(content)
                    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
                    self.send(200, {"ok": True, "file": path.name})
                    return
                if length < 0 or length > 65536:
                    raise ValueError("Invalid request size")
                payload = json.loads(self.rfile.read(length) or b"{}")
                if not isinstance(payload, dict):
                    raise ValueError("Expected an object")
                if self.path == "/api/speed" and (type(payload.get("speed")) not in (int, float) or
                                                  payload["speed"] not in (0.5, 1, 1.5, 2, 3, 4, 6)):
                    raise ValueError("Unsupported speed")
                with runtime.lock:
                    if self.path in ("/api/pause", "/api/restart", "/api/step", "/api/search-mode", "/api/forget-memory"):
                        runtime.intro_started = None
                        runtime.tour_active = False
                        runtime.tour_finished_at = None
                    if self.path == "/api/drawer":
                        runtime.inspect_drawer(payload.get("id"), payload.get("fraction"))
                        result = {"ok": True}
                    elif self.path == "/api/drawers-close":
                        for name in runtime.inspection.drawers:
                            runtime.inspect_drawer(name, 0)
                        result = {"ok": True}
                    elif self.path == "/api/resume" and runtime.inspection.active:
                        runtime.start_presentation()
                        result = {"ok": True}
                    elif self.path == "/api/tour":
                        runtime.start_tour()
                        result = {"ok": True}
                    elif self.path == "/api/presentation":
                        runtime.start_presentation()
                        result = {"ok": True}
                    elif self.path == "/api/search-mode" and runtime.task_name == "search":
                        use_memory = payload.get("use_memory")
                        if type(use_memory) is not bool:
                            raise ValueError("Invalid search mode")
                        runtime.demo.use_memory = use_memory
                        runtime.controls.request_restart()
                        runtime.controls.set_paused(True)
                        result = {"ok": True}
                    elif self.path == "/api/forget-memory" and runtime.task_name == "search":
                        runtime.demo.memory.clear()
                        runtime.demo.run_memory.clear()
                        runtime.controls.request_restart()
                        runtime.controls.set_paused(True)
                        result = {"ok": True}
                    else:
                        result = runtime.console.handle_command(self.path, payload)
                        if self.path == "/api/restart":
                            runtime.controls.set_paused(True)
                self.send(200, result)
            except (ValueError, KeyError, RuntimeError) as exc:
                self.send(400, {"error": str(exc)})
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def start_viewer(runtime):
    server = viser.ViserServer(host="127.0.0.1", port=runtime.viewer_port, label="SRC Kitchen")
    server.gui.configure_theme(show_logo=False, show_share_button=False)
    server.gui.main_panel.minimize()
    server.scene.set_up_direction("+z")
    runtime.scene = KitchenMujocoScene(server, runtime.demo.model, num_envs=1)
    runtime.scene.camera_tracking_enabled = False
    runtime.scene.geom_groups_visible[:] = [False, True, False, False, False, False]
    runtime.scene.site_groups_visible[:] = [False] * 6
    runtime.scene._sync_visibilities()
    runtime.scene.update_from_mjdata(runtime.demo.data)
    drawer_bodies = {d["body"]: name for name, d in runtime.inspection.drawers.items()}
    def clicked_drawer(event):
        with runtime.lock:
            geom = np.array([-1], dtype=np.int32)
            mujoco.mj_ray(runtime.demo.model, runtime.demo.data,
                          np.asarray(event.ray_origin), np.asarray(event.ray_direction),
                          np.array([0, 1, 0, 0, 0, 0], dtype=np.uint8), True, -1, geom)
            if geom[0] >= 0:
                body = int(runtime.demo.model.geom_bodyid[geom[0]])
                if body in drawer_bodies:
                    runtime.inspect_drawer(drawer_bodies[body])
    for group in runtime.scene._mesh_groups:
        if group.group_id == 1 and any(int(body) in drawer_bodies for body in group.body_ids):
            group.handle.on_click(clicked_drawer)
    server.scene.configure_environment_map("apartment", background=False, environment_intensity=0.55)
    server.scene.configure_default_lights(enabled=False)
    server.scene.add_light_ambient("/lighting/ambient", intensity=0.25)
    for i, x in enumerate((-1.7, 1.5)):
        server.scene.add_light_rectarea(f"/lighting/ceiling{i}", position=(x, 0, 3.08),
                                        width=1.0, height=3.6, intensity=1.0)
    @server.on_client_connect
    def connected(client):
        runtime.clients[client.client_id] = client
        runtime.apply_camera(*runtime.camera_pose(runtime.camera_name))
        last_aspect = [client.camera.aspect]
        @client.camera.on_update
        def resized(_):
            if client.camera.aspect != last_aspect[0]:
                last_aspect[0] = client.camera.aspect
                if runtime.intro_started is None and not runtime.tour_active:
                    runtime.camera(runtime.camera_name)
    @server.on_client_disconnect
    def disconnected(client):
        runtime.clients.pop(client.client_id, None)
    return server


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--task", choices=("seasoning", "cup", "drawer", "search"), default="search")
    parser.add_argument("--port", type=int, default=8105)
    parser.add_argument("--viewer-port", type=int, default=8106)
    parser.add_argument("--output-dir", type=Path, default=Path.cwd() / "runs")
    args = parser.parse_args(argv)
    if args.port == args.viewer_port or not all(1 <= p <= 65535 for p in (args.port, args.viewer_port)):
        parser.error("Choose distinct valid HTTP and viewer ports")
    runtime = Runtime(args.task, viewer_port=args.viewer_port, output_dir=args.output_dir)
    pipeline = make_pipeline(runtime)
    if args.validate:
        for _ in range(10):
            pipeline.step(dt=0.02)
        assert runtime.demo.frame == 0
        runtime.controls.set_paused(False)
        for _ in range(200):
            pipeline.step(dt=0.02)
        runtime.controls.set_paused(True)
        frozen = runtime.demo.data.qpos.copy()
        frame = runtime.demo.frame
        for _ in range(10):
            pipeline.step(dt=0.02)
        np.testing.assert_array_equal(runtime.demo.data.qpos, frozen)
        assert runtime.demo.frame == frame
        runtime.controls.request_step()
        pipeline.step(dt=0.02)
        assert runtime.demo.frame == frame + 1
        runtime.controls.set_paused(False)
        for _ in range(4500):
            pipeline.step(dt=0.02)
            if runtime.demo.finished:
                break
        report = runtime.state()
        report["pause_and_single_step_verified"] = True
        assert runtime.demo.success
        final_qpos = runtime.demo.data.qpos.copy()
        if args.task == "search":
            assert [o["drawer"] for o in runtime.demo.observations] == ["top left", "top right", "bottom left"]
            runtime.demo.use_memory = True
        runtime.controls.request_restart()
        pipeline.step(dt=0.02)
        assert runtime.demo.frame == 1 and not runtime.demo.finished
        runtime.controls.set_paused(False)
        for _ in range(3500):
            pipeline.step(dt=0.02)
            if runtime.demo.finished:
                break
        assert runtime.demo.success
        if args.task == "search":
            assert runtime.demo.memory_used and len(runtime.demo.observations) == 1
            assert runtime.demo.frame < report["motion"]["frame"]
            report["memory_recall"] = runtime.state()
            report["memory_recall_verified"] = True
            first_events = report["timeline"]
            recall_events = report["memory_recall"]["timeline"]
            assert {event["source"] for event in first_events} == {
                "dispatch", "motion", "simulation", "memory", "verification"}
            assert len([event for event in first_events if event["source"] == "memory"]) == 4
            assert first_events[-1]["source"] == "verification"
            assert recall_events[0]["source"] == "memory" and "Recall:" in recall_events[0]["message"]
            assert recall_events[0]["id"] > first_events[-1]["id"]
            assert all(a["time"] <= b["time"] for a, b in zip(first_events, first_events[1:]))
            report["timeline_verified"] = True
        else:
            np.testing.assert_allclose(runtime.demo.data.qpos, final_qpos, atol=1e-10)
            report["restart_repeat_verified"] = True
        runtime.output_dir.mkdir(parents=True, exist_ok=True)
        (runtime.output_dir / f"teaser-{args.task}-validation.json").write_text(json.dumps(report, indent=2) + "\n")
        pipeline.close_stepper()
        print(json.dumps(report, indent=2))
        raise SystemExit(0 if runtime.demo.success else 1)
    viewer = web = None
    try:
        viewer = start_viewer(runtime)
        web = serve(runtime, args.port)
        stopped = threading.Event()
        signal.signal(signal.SIGINT, lambda *_: stopped.set())
        signal.signal(signal.SIGTERM, lambda *_: stopped.set())
        print(f"TEASER_READY http://127.0.0.1:{args.port}", flush=True)
        while not stopped.is_set():
            started = time.monotonic()
            with runtime.lock:
                pipeline.step(dt=0.02)
                runtime.update_presentation()
                runtime.update_inspection()
            period = 0.02 / runtime.controls.snapshot().speed
            stopped.wait(max(0, period - (time.monotonic() - started)))
    finally:
        try:
            pipeline.close_stepper()
        finally:
            if web is not None:
                web.shutdown()
                web.server_close()
            if viewer is not None:
                viewer.stop()


if __name__ == "__main__":
    main()
