import argparse
import json
import math
import random
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import pybullet as p
import pybullet_data
import torch
from ultralytics import YOLO


WORKSPACE = Path(__file__).resolve().parent
TIME_STEP = 1 / 240
TABLETOP_Z = 0.8
TRAY_FLOOR_TOP = TABLETOP_Z + 0.024
END_EFFECTOR = 11
ARM_JOINTS = list(range(7))
GRIPPER_JOINTS = [9, 10]
HOME_CONFIGURATION = [0, -0.3, 0, -2.2, 0, 2.0, 0.8]
DOWNWARD = p.getQuaternionFromEuler([3.14159, 0, 0])
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
INFERENCE_INTERVAL = 0.75
CUP_CLASS = 41
TABLETOP_ROI = (130, 195, 500, 375)
COLORS = ("red", "green", "blue")
COLOR_RGBA = {
	"red": [0.95, 0.04, 0.04, 1],
	"green": [0.04, 0.85, 0.06, 1],
	"blue": [0.04, 0.10, 0.98, 1],
}
COLOR_BGR = {
	"red": (40, 40, 245),
	"green": (40, 220, 40),
	"blue": (245, 70, 30),
}
TRAY_X = {"red": 0.25, "green": 0.48, "blue": 0.71}
TRAY_Y = 0.18


def make_box(center, half_extents, color, collidable=True):
	collision_shape = -1
	if collidable:
		collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
	visual = p.createVisualShape(
		p.GEOM_BOX,
		halfExtents=half_extents,
		rgbaColor=color,
	)
	return p.createMultiBody(
		baseMass=0,
		baseCollisionShapeIndex=collision_shape,
		baseVisualShapeIndex=visual,
		basePosition=center,
	)


def make_tray(color):
	center_x = TRAY_X[color]
	center_y = TRAY_Y
	tray_color = COLOR_RGBA[color]
	make_box(
		[center_x, center_y, TABLETOP_Z + 0.012],
		[0.09, 0.075, 0.012],
		tray_color,
	)
	wall_z = TRAY_FLOOR_TOP + 0.034
	for side in (-1, 1):
		make_box(
			[center_x + side * 0.078, center_y, wall_z],
			[0.012, 0.075, 0.034],
			tray_color,
			collidable=False,
		)
		make_box(
			[center_x, center_y + side * 0.063, wall_z],
			[0.078, 0.012, 0.034],
			tray_color,
			collidable=False,
		)
	return [center_x, center_y, TRAY_FLOOR_TOP]


def color_from_crop(frame, bounds):
	x1, y1, x2, y2 = bounds
	cup_crop = frame[y1:y2, x1:x2]
	if cup_crop.size == 0:
		return None
	height, width = cup_crop.shape[:2]
	body = cup_crop[
		max(0, int(height * 0.14)):max(1, int(height * 0.86)),
		max(0, int(width * 0.18)):max(1, int(width * 0.82)),
	]
	hsv = cv2.cvtColor(body, cv2.COLOR_BGR2HSV)
	hues = hsv[:, :, 0]
	saturated = (hsv[:, :, 1] > 75) & (hsv[:, :, 2] > 45)
	if not np.any(saturated):
		return None
	hues = hues[saturated]
	scores = {
		"red": float(np.mean((hues <= 10) | (hues >= 170))),
		"green": float(np.mean((hues >= 35) & (hues <= 85))),
		"blue": float(np.mean((hues >= 95) & (hues <= 135))),
	}
	color = max(scores, key=scores.get)
	if scores[color] < 0.48:
		return None
	return color, float(np.median(hues)), scores


def pixel_to_world(pixel_x, pixel_y, depth_frame, inverse_view, inverse_projection):
	column = int(np.clip(round(pixel_x), 0, IMAGE_WIDTH - 1))
	row = int(np.clip(round(pixel_y), 0, IMAGE_HEIGHT - 1))
	depth = float(depth_frame[row, column])
	if not math.isfinite(depth) or depth >= 0.999:
		return None
	clip = np.array(
		[
			2 * (pixel_x + 0.5) / IMAGE_WIDTH - 1,
			1 - 2 * (pixel_y + 0.5) / IMAGE_HEIGHT,
			2 * depth - 1,
			1,
		],
		dtype=np.float64,
	)
	eye = inverse_projection @ clip
	if abs(eye[3]) < 1e-9:
		return None
	eye /= eye[3]
	world = inverse_view @ eye
	if abs(world[3]) < 1e-9:
		return None
	world /= world[3]
	return [float(world[0]), float(world[1]), TABLETOP_Z]


def start_laya_worker():
	print("Starting Laya. The first run downloads about 1.7 GB of model weights.")
	worker = subprocess.Popen(
		["node_modules/.bin/tsx", "laya_sorter.ts"],
		cwd=WORKSPACE,
		stdin=subprocess.PIPE,
		stdout=subprocess.PIPE,
		text=True,
		bufsize=1,
	)
	if worker.stdout is None:
		raise RuntimeError("Could not open Laya worker output")
	ready_line = worker.stdout.readline()
	if not ready_line:
		worker.wait()
		raise RuntimeError("Laya worker exited before loading its model")
	ready = json.loads(ready_line)
	if ready.get("ready") is not True:
		raise RuntimeError(f"Laya worker failed to start: {ready}")
	return worker


def ask_laya(worker, cup):
	request = {
		"kind": "destination",
		"cupColor": cup["color"],
		"hsvHue": cup["hsv_hue"],
		"colorScores": cup["color_scores"],
		"visionConfidence": cup["vision_confidence"],
	}
	return send_laya_request(worker, request)


def ask_laya_verification(worker, placements):
	return send_laya_request(
		worker,
		{"kind": "verification", "placements": placements},
	)


def send_laya_request(worker, request):
	if worker.stdin is None or worker.stdout is None:
		raise RuntimeError("Laya worker is not available")
	worker.stdin.write(json.dumps(request) + "\n")
	worker.stdin.flush()
	line = worker.stdout.readline()
	if not line:
		raise RuntimeError("Laya worker stopped while choosing a destination")
	response = json.loads(line)
	if "error" in response:
		raise RuntimeError(response["error"])
	return response


def close_laya_worker(worker):
	if worker.stdin is not None and not worker.stdin.closed:
		worker.stdin.close()
	try:
		worker.wait(timeout=15)
	except subprocess.TimeoutExpired:
		worker.terminate()
		worker.wait(timeout=5)


def main():
	parser = argparse.ArgumentParser(description="Sort three colored cups with YOLO and Laya.")
	parser.add_argument("--seed", type=int, help="Repeat a randomized layout with this seed")
	args = parser.parse_args()
	rng = random.Random(args.seed)

	worker = start_laya_worker()
	client = None
	try:
		print("Loading YOLO11n.")
		torch.set_num_threads(2)
		model = YOLO("yolo11n.pt")
		client = p.connect(p.DIRECT)
		p.setAdditionalSearchPath(pybullet_data.getDataPath())
		p.setGravity(0, 0, -9.81)
		p.setTimeStep(TIME_STEP)
		p.setPhysicsEngineParameter(numSolverIterations=150)

		p.loadURDF("plane.urdf")
		make_box([0.35, 0, TABLETOP_Z - 0.04], [0.65, 0.30, 0.04], [0.35, 0.38, 0.42, 1])
		for x in (-0.20, 0.90):
			for y in (-0.23, 0.23):
				make_box([x, y, 0.38], [0.035, 0.035, 0.38], [0.28, 0.30, 0.32, 1])

		tray_positions = {f"{color}_box": make_tray(color) for color in COLORS}
		robot = p.loadURDF("franka_panda/panda.urdf", [0, 0, TABLETOP_Z], useFixedBase=True)
		for joint, angle in enumerate(HOME_CONFIGURATION):
			p.resetJointState(robot, joint, angle)
		p.setJointMotorControlArray(
			robot,
			ARM_JOINTS,
			p.POSITION_CONTROL,
			targetPositions=HOME_CONFIGURATION,
			forces=[87, 87, 87, 87, 12, 12, 12],
		)

		depth_rows = rng.sample((-0.18, -0.12, -0.06), 3)
		cup_slots = list(zip((0.24, 0.48, 0.72), depth_rows))
		cup_colors = list(COLORS)
		rng.shuffle(cup_colors)
		cups = []
		for color, (x, y) in zip(cup_colors, cup_slots):
			orientation = p.getQuaternionFromEuler([0, 0, rng.uniform(-math.pi, math.pi)])
			body = p.loadURDF("objects/mug.urdf", [x, y, TABLETOP_Z], orientation)
			p.changeVisualShape(body, -1, rgbaColor=COLOR_RGBA[color])
			p.changeDynamics(body, -1, mass=0.12, lateralFriction=1.0)
			cups.append(
				{
					"body": body,
					"spawn_xy": [x, y],
					"spawn_orientation": orientation,
				}
			)

		view = p.computeViewMatrix(
			cameraEyePosition=[1.30, -1.35, 1.75],
			cameraTargetPosition=[0.43, 0.0, 0.85],
			cameraUpVector=[0, 0, 1],
		)
		projection = p.computeProjectionMatrixFOV(
			fov=55,
			aspect=IMAGE_WIDTH / IMAGE_HEIGHT,
			nearVal=0.1,
			farVal=4.0,
		)
		inverse_view = np.linalg.inv(np.array(view).reshape((4, 4), order="F"))
		inverse_projection = np.linalg.inv(np.array(projection).reshape((4, 4), order="F"))
		window_name = "YOLO11 + Laya cup sorter (Q to quit)"
		cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

		latest_cups = []
		status_text = "Detecting cups"
		last_inference = 0.0
		keep_running = True

		def update_camera():
			nonlocal latest_cups, status_text, last_inference, keep_running
			_, _, rgba, depth, _ = p.getCameraImage(
				IMAGE_WIDTH,
				IMAGE_HEIGHT,
				viewMatrix=view,
				projectionMatrix=projection,
				renderer=p.ER_TINY_RENDERER,
			)
			rgb = np.asarray(rgba, dtype=np.uint8).reshape(
				IMAGE_HEIGHT,
				IMAGE_WIDTH,
				4,
			)[:, :, :3]
			frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
			depth_frame = np.asarray(depth, dtype=np.float32).reshape(IMAGE_HEIGHT, IMAGE_WIDTH)
			left, top, right, bottom = TABLETOP_ROI
			result = model.predict(
				frame[top:bottom, left:right],
				imgsz=640,
				conf=0.08,
				classes=[CUP_CLASS],
				verbose=False,
			)[0]
			annotated = frame.copy()
			current_cups = []
			for detection in result.boxes:
				box_left, box_top, box_right, box_bottom = (
					int(value) for value in detection.xyxy[0].cpu().numpy()
				)
				bounds = (
					left + box_left,
					top + box_top,
					left + box_right,
					top + box_bottom,
				)
				classified = color_from_crop(frame, bounds)
				center_x = (bounds[0] + bounds[2]) / 2
				center_y = (bounds[1] + bounds[3]) / 2
				world = pixel_to_world(
					center_x,
					center_y,
					depth_frame,
					inverse_view,
					inverse_projection,
				)
				if world is None:
					continue
				color, hue, color_scores = classified or ("unknown", 0.0, {})
				confidence = float(detection.conf[0])
				current_cups.append(
					{
						"color": color,
						"hsv_hue": hue,
						"color_scores": color_scores,
						"vision_confidence": confidence,
						"world": world,
						"bounds": bounds,
					}
				)
				cv2.rectangle(
					annotated,
					(bounds[0], bounds[1]),
					(bounds[2], bounds[3]),
					COLOR_BGR.get(color, (230, 230, 230)),
					2,
				)
				cv2.putText(
					annotated,
					f"{color} cup {confidence:.2f}",
					(bounds[0], max(18, bounds[1] - 7)),
					cv2.FONT_HERSHEY_SIMPLEX,
					0.5,
					COLOR_BGR.get(color, (230, 230, 230)),
					2,
				)
			latest_cups = current_cups
			cv2.putText(
				annotated,
				f"YOLO11 cups | Laya destination | {status_text}",
				(12, 25),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.52,
				(40, 240, 40),
				2,
			)
			cv2.putText(
				annotated,
				"Matching trays: RED   GREEN   BLUE",
				(12, IMAGE_HEIGHT - 14),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.52,
				(245, 245, 245),
				2,
			)
			cv2.imshow(window_name, annotated)
			if cv2.waitKey(1) & 0xFF == ord("q"):
				keep_running = False
				return False
			last_inference = time.monotonic()
			return True

		def step_for(seconds):
			nonlocal last_inference
			for _ in range(round(seconds / TIME_STEP)):
				p.stepSimulation()
				if time.monotonic() - last_inference >= INFERENCE_INTERVAL:
					if not update_camera():
						return False
				time.sleep(TIME_STEP)
			return True

		def move_arm(position, duration):
			joint_positions = p.calculateInverseKinematics(
				robot,
				END_EFFECTOR,
				position,
				DOWNWARD,
				maxNumIterations=200,
				residualThreshold=0.001,
			)
			p.setJointMotorControlArray(
				robot,
				ARM_JOINTS,
				p.POSITION_CONTROL,
				targetPositions=joint_positions[:7],
				forces=[87, 87, 87, 87, 12, 12, 12],
			)
			if not step_for(duration):
				return False
			elapsed = 0.0
			while elapsed < 3.0:
				current = p.getLinkState(
					robot,
					END_EFFECTOR,
					computeForwardKinematics=True,
				)[4]
				if math.dist(current, position) < 0.025:
					return True
				if not step_for(0.1):
					return False
				elapsed += 0.1
			return False

		def place_cup(cup, destination):
			body = cup["body"]
			position = cup["world"]
			pick = [position[0], position[1], TABLETOP_Z + 0.10]
			pick_approach = [position[0], position[1], TABLETOP_Z + 0.25]
			place_base = tray_positions[destination]
			place = [place_base[0], place_base[1], place_base[2] + 0.10]
			place_approach = [place[0], place[1], place[2] + 0.16]

			p.setJointMotorControlArray(
				robot,
				GRIPPER_JOINTS,
				p.POSITION_CONTROL,
				targetPositions=[0.04, 0.04],
				forces=[20, 20],
			)
			if not move_arm(pick_approach, 1.2) or not move_arm(pick, 0.8):
				return False
			p.setJointMotorControlArray(
				robot,
				GRIPPER_JOINTS,
				p.POSITION_CONTROL,
				targetPositions=[0, 0],
				forces=[20, 20],
			)
			if not step_for(0.5):
				return False

			tool = p.getLinkState(robot, END_EFFECTOR, computeForwardKinematics=True)
			inverse_position, inverse_orientation = p.invertTransform(tool[4], tool[5])
			cup_position, cup_orientation = p.getBasePositionAndOrientation(body)
			local_position, local_orientation = p.multiplyTransforms(
				inverse_position,
				inverse_orientation,
				cup_position,
				cup_orientation,
			)
			grasp = p.createConstraint(
				robot,
				END_EFFECTOR,
				body,
				-1,
				p.JOINT_FIXED,
				[0, 0, 0],
				local_position,
				[0, 0, 0],
				parentFrameOrientation=local_orientation,
			)

			if not move_arm(pick_approach, 0.8):
				return False
			if not move_arm(place_approach, 1.5) or not move_arm(place, 0.8):
				return False
			p.removeConstraint(grasp)
			if not step_for(0.35):
				return False
			p.setJointMotorControlArray(
				robot,
				GRIPPER_JOINTS,
				p.POSITION_CONTROL,
				targetPositions=[0.04, 0.04],
				forces=[20, 20],
			)
			if not move_arm(place_approach, 0.8):
				return False

			landed = p.getBasePositionAndOrientation(body)[0]
			distance = math.hypot(landed[0] - place_base[0], landed[1] - place_base[1])
			return distance < 0.10 and landed[2] >= TABLETOP_Z

		p.setJointMotorControlArray(
			robot,
			GRIPPER_JOINTS,
			p.POSITION_CONTROL,
			targetPositions=[0.04, 0.04],
			forces=[20, 20],
		)
		if not step_for(1.0):
			return
		deadline = time.monotonic() + 8
		while len(latest_cups) != 3 and time.monotonic() < deadline:
			status_text = f"Waiting for 3 cups ({len(latest_cups)}/3)"
			if not step_for(0.2):
				return

		placement_results = {color: False for color in COLORS}
		cups_by_color = {}

		def log_destination_question(cup):
			color = cup["color"]
			question = (
				f"Which box has exactly the same color as this {color} cup? "
				"Choose the matching color box."
			)
			print(f"\nLaya question: {question}", flush=True)
			print(
				"Laya state: "
				+ json.dumps(
					{
						"cup_color": color,
						"HSV hue": round(cup["hsv_hue"], 1),
						"YOLO confidence": round(cup["vision_confidence"], 3),
						"available boxes": [f"{item}_box" for item in COLORS],
					},
					ensure_ascii=False,
				),
				flush=True,
			)
			decision = ask_laya(worker, cup)
			print(
				"Laya response: "
				+ json.dumps(decision, ensure_ascii=False),
				flush=True,
			)
			return decision

		def ask_completion():
			question = (
				"Are all three cups physically confirmed inside their matching-color "
				"boxes? Choose complete only if every placement succeeded; otherwise choose retry."
			)
			placements = [
				{
					"cupColor": color,
					"boxId": f"{color}_box",
					"placed": placement_results[color],
				}
				for color in COLORS
			]
			print(f"\nLaya final question: {question}", flush=True)
			print(
				"Laya verification state: "
				+ json.dumps(placements, ensure_ascii=False),
				flush=True,
			)
			response = ask_laya_verification(worker, placements)
			print(
				"Laya final response: "
				+ json.dumps(response, ensure_ascii=False),
				flush=True,
			)
			return response

		colors_found = {cup["color"] for cup in latest_cups}
		if (
			len(latest_cups) != 3
			or colors_found != set(COLORS)
			or any(cup["color"] == "unknown" for cup in latest_cups)
		):
			status_text = "Need exactly one clearly detected cup of each color; no sorting"
			print(status_text)
		else:
			used_bodies = set()
			for cup in latest_cups:
				if not keep_running:
					break
				color = cup["color"]
				nearest = min(
					(
						item
						for item in cups
						if item["body"] not in used_bodies
					),
					key=lambda item: math.dist(item["spawn_xy"], cup["world"][:2]),
				)
				if math.dist(nearest["spawn_xy"], cup["world"][:2]) > 0.10:
					status_text = "Cup position did not match a reachable object; skipped"
					print(status_text)
					continue
				used_bodies.add(nearest["body"])
				cup["body"] = nearest["body"]
				cup["spawn_xy"] = nearest["spawn_xy"]
				cup["spawn_orientation"] = nearest["spawn_orientation"]
				cups_by_color[color] = cup
				try:
					decision = log_destination_question(cup)
				except (RuntimeError, json.JSONDecodeError) as error:
					status_text = f"Laya error; skipped {color} cup"
					print(f"{status_text}: {error}")
					continue
				expected = f"{color}_box"
				if (
					decision.get("safeToSort") is not True
					or decision.get("destination") != expected
					or decision.get("probability", 0) < 0.6
				):
					status_text = f"Laya was uncertain; skipped {color} cup"
					print(f"{status_text}: {decision}")
					continue
				destination = decision["destination"]
				status_text = f"Moving {color} cup to {destination}"
				print(status_text, flush=True)
				placement_results[color] = place_cup(cup, destination)
				print(
					f"{color} cup "
					f"{'placed' if placement_results[color] else 'placement needs checking'}",
					flush=True,
				)
				if not keep_running:
					break

		completion = ask_completion()
		failed_colors = [color for color in COLORS if not placement_results[color]]
		if failed_colors and keep_running:
			print(
				"Restarting failed placements once: " + ", ".join(failed_colors),
				flush=True,
			)
			for color in failed_colors:
				cup = cups_by_color.get(color)
				if cup is None or not keep_running:
					continue
				spawn = cup["spawn_xy"]
				p.resetBasePositionAndOrientation(
					cup["body"],
					[spawn[0], spawn[1], TABLETOP_Z],
					cup["spawn_orientation"],
				)
				p.resetBaseVelocity(cup["body"], [0, 0, 0], [0, 0, 0])
				if not step_for(0.35):
					break
				try:
					decision = log_destination_question(cup)
				except (RuntimeError, json.JSONDecodeError) as error:
					print(f"Laya retry failed for {color} cup: {error}", flush=True)
					continue
				expected = f"{color}_box"
				if (
					decision.get("safeToSort") is True
					and decision.get("destination") == expected
					and decision.get("probability", 0) >= 0.6
				):
					placement_results[color] = place_cup(cup, expected)
				print(
					f"{color} retry "
					f"{'placed' if placement_results[color] else 'still needs checking'}",
					flush=True,
				)
			completion = ask_completion()

		physically_complete = all(placement_results.values())
		laya_says_complete = (
			completion.get("choice") == "complete"
			and completion.get("probability", 0) >= 0.6
		)
		if physically_complete:
			status_text = "All three cups verified in matching trays; press Q to quit"
			if not laya_says_complete:
				print("Laya disagreed with the physical tray check; keeping verified placements.")
		else:
			status_text = "Retry incomplete; at least one cup still needs placement"
		print(status_text, flush=True)
		while keep_running:
			if not step_for(0.1):
				break
	finally:
		if client is not None and p.isConnected(client):
			p.disconnect(client)
		cv2.destroyAllWindows()
		close_laya_worker(worker)


if __name__ == "__main__":
	main()