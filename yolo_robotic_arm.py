import time

import cv2
import numpy as np
import pybullet as p
import pybullet_data
import torch
from ultralytics import YOLO


TIME_STEP = 1 / 240
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


def make_box(center, half_extents, color):
	collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
	visual = p.createVisualShape(
		p.GEOM_BOX,
		halfExtents=half_extents,
		rgbaColor=color,
	)
	return p.createMultiBody(
		baseMass=0,
		baseCollisionShapeIndex=collision,
		baseVisualShapeIndex=visual,
		basePosition=center,
	)


def main():
	print("Loading YOLO11n. The first run downloads its model weights.")
	torch.set_num_threads(2)
	model = YOLO("yolo11n.pt")
	client = p.connect(p.DIRECT)
	last_inference = 0.0
	keep_running = True
	try:
		p.setAdditionalSearchPath(pybullet_data.getDataPath())
		p.setGravity(0, 0, -9.81)
		p.setTimeStep(TIME_STEP)
		p.setPhysicsEngineParameter(numSolverIterations=150)
		p.resetDebugVisualizerCamera(
			cameraDistance=1.45,
			cameraYaw=45,
			cameraPitch=-35,
			cameraTargetPosition=[0.45, 0, 0.72],
		)

		p.loadURDF("plane.urdf")
		make_box([0.35, 0, 0.76], [0.65, 0.30, 0.04], [0.35, 0.38, 0.42, 1])
		for x in (-0.20, 0.90):
			for y in (-0.23, 0.23):
				make_box([x, y, 0.38], [0.035, 0.035, 0.38], [0.28, 0.30, 0.32, 1])

		robot = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0.8], useFixedBase=True)
		for joint, angle in enumerate(HOME_CONFIGURATION):
			p.resetJointState(robot, joint, angle)
		p.setJointMotorControlArray(
			robot,
			ARM_JOINTS,
			p.POSITION_CONTROL,
			targetPositions=HOME_CONFIGURATION,
			forces=[87, 87, 87, 87, 12, 12, 12],
		)

		start = [0.52, -0.10, 0.80]
		finish = [0.52, 0.10, 0.80]
		mug = p.loadURDF("objects/mug.urdf", start)
		p.changeDynamics(mug, -1, mass=0.12, lateralFriction=1.0)

		view_matrix = p.computeViewMatrix(
			cameraEyePosition=[1.30, -1.35, 1.75],
			cameraTargetPosition=[0.43, 0.0, 0.85],
			cameraUpVector=[0, 0, 1],
		)
		projection_matrix = p.computeProjectionMatrixFOV(
			fov=55,
			aspect=IMAGE_WIDTH / IMAGE_HEIGHT,
			nearVal=0.1,
			farVal=4.0,
		)
		window_name = "YOLO11n - PyBullet camera (Q to quit)"
		cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

		def step_for(seconds):
			nonlocal last_inference, keep_running
			steps = round(seconds / TIME_STEP)
			for _ in range(steps):
				p.stepSimulation()
				now = time.monotonic()
				if now - last_inference >= INFERENCE_INTERVAL:
					_, _, rgba, _, _ = p.getCameraImage(
						IMAGE_WIDTH,
						IMAGE_HEIGHT,
						viewMatrix=view_matrix,
						projectionMatrix=projection_matrix,
						renderer=p.ER_TINY_RENDERER,
					)
					rgb = np.asarray(rgba, dtype=np.uint8).reshape(
						IMAGE_HEIGHT,
						IMAGE_WIDTH,
						4,
					)[:, :, :3]
					frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
					left, top, right, bottom = TABLETOP_ROI
					crop = frame[top:bottom, left:right]
					result = model.predict(
						crop,
						imgsz=640,
						conf=0.08,
						classes=[CUP_CLASS],
						verbose=False,
					)[0]
					annotated = frame.copy()
					for detection in result.boxes:
						box_left, box_top, box_right, box_bottom = (
							int(value) for value in detection.xyxy[0].cpu().numpy()
						)
						box_left += left
						box_right += left
						box_top += top
						box_bottom += top
						confidence = float(detection.conf[0])
						cv2.rectangle(
							annotated,
							(box_left, box_top),
							(box_right, box_bottom),
							(40, 240, 40),
							2,
						)
						cv2.putText(
							annotated,
							f"cup {confidence:.2f}",
							(box_left, max(18, box_top - 8)),
							cv2.FONT_HERSHEY_SIMPLEX,
							0.55,
							(40, 240, 40),
							2,
						)
					cv2.putText(
						annotated,
						"YOLO11n | cup detection | Q: quit",
						(12, 28),
						cv2.FONT_HERSHEY_SIMPLEX,
						0.65,
						(40, 240, 40),
						2,
					)
					cv2.imshow(window_name, annotated)
					if cv2.waitKey(1) & 0xFF == ord("q"):
						keep_running = False
						return False
					last_inference = time.monotonic()
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
			return step_for(duration)

		def move_and_check(position, duration):
			return keep_running and move_arm(position, duration)

		cv2.waitKey(1)
		p.setJointMotorControlArray(
			robot,
			GRIPPER_JOINTS,
			p.POSITION_CONTROL,
			targetPositions=[0.04, 0.04],
			forces=[20, 20],
		)
		if not step_for(1):
			return

		grasp_position = [start[0], start[1], start[2] + 0.10]
		place_position = [finish[0], finish[1], finish[2] + 0.10]
		approach_height = grasp_position[2] + 0.12
		if not move_and_check([start[0], start[1], approach_height], 1.5):
			return
		if not move_and_check(grasp_position, 1):
			return
		p.setJointMotorControlArray(
			robot,
			GRIPPER_JOINTS,
			p.POSITION_CONTROL,
			targetPositions=[0, 0],
			forces=[20, 20],
		)
		if not step_for(0.7):
			return

		tool_state = p.getLinkState(robot, END_EFFECTOR, computeForwardKinematics=True)
		inverse_position, inverse_orientation = p.invertTransform(tool_state[4], tool_state[5])
		mug_position, mug_orientation = p.getBasePositionAndOrientation(mug)
		local_position, local_orientation = p.multiplyTransforms(
			inverse_position,
			inverse_orientation,
			mug_position,
			mug_orientation,
		)
		grasp = p.createConstraint(
			robot,
			END_EFFECTOR,
			mug,
			-1,
			p.JOINT_FIXED,
			[0, 0, 0],
			local_position,
			[0, 0, 0],
			parentFrameOrientation=local_orientation,
		)

		if not move_and_check([start[0], start[1], approach_height], 1):
			return
		if not move_and_check([finish[0], finish[1], approach_height], 2):
			return
		if not move_and_check(place_position, 1):
			return
		p.removeConstraint(grasp)
		if not step_for(0.5):
			return
		p.setJointMotorControlArray(
			robot,
			GRIPPER_JOINTS,
			p.POSITION_CONTROL,
			targetPositions=[0.04, 0.04],
			forces=[20, 20],
		)
		if not move_and_check([finish[0], finish[1], approach_height], 1):
			return

		while keep_running and p.isConnected(client):
			if not step_for(TIME_STEP):
				break
	finally:
		if p.isConnected(client):
			p.disconnect(client)
		cv2.destroyAllWindows()


if __name__ == "__main__":
	main()