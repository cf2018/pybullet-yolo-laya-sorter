import time

import pybullet as p
import pybullet_data


TIME_STEP = 1 / 240
END_EFFECTOR = 11
ARM_JOINTS = list(range(7))
GRIPPER_JOINTS = [9, 10]
HOME_CONFIGURATION = [0, -0.3, 0, -2.2, 0, 2.0, 0.8]
DOWNWARD = p.getQuaternionFromEuler([3.14159, 0, 0])


def step_for(seconds):
	for _ in range(round(seconds / TIME_STEP)):
		p.stepSimulation()
		time.sleep(TIME_STEP)


def move_arm(robot, position, seconds):
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
	step_for(seconds)


def make_box(center, half_extents, color, mass=0):
	shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
	visual = p.createVisualShape(
		p.GEOM_BOX,
		halfExtents=half_extents,
		rgbaColor=color,
	)
	return p.createMultiBody(
		baseMass=mass,
		baseCollisionShapeIndex=shape,
		baseVisualShapeIndex=visual,
		basePosition=center,
	)


client = p.connect(p.GUI)
try:
	p.setAdditionalSearchPath(pybullet_data.getDataPath())
	p.setGravity(0, 0, -9.81)
	p.setTimeStep(TIME_STEP)
	p.setPhysicsEngineParameter(numSolverIterations=150)
	p.resetDebugVisualizerCamera(
		cameraDistance=1.35,
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
	start = [0.52, -0.10, 0.825]
	finish = [0.52, 0.10, 0.825]
	block = make_box(start, [0.0225, 0.0225, 0.0225], [0.88, 0.28, 0.12, 1], mass=0.12)
	p.changeDynamics(block, -1, lateralFriction=1.2, spinningFriction=0.02)

	p.setJointMotorControlArray(
		robot,
		GRIPPER_JOINTS,
		p.POSITION_CONTROL,
		targetPositions=[0.04, 0.04],
		forces=[20, 20],
	)
	step_for(1)

	grasp_position = [start[0], start[1], start[2] + 0.06]
	place_position = [finish[0], finish[1], finish[2] + 0.06]
	approach_height = grasp_position[2] + 0.12
	move_arm(robot, [start[0], start[1], approach_height], 1.5)
	move_arm(robot, grasp_position, 1)
	p.setJointMotorControlArray(
		robot,
		GRIPPER_JOINTS,
		p.POSITION_CONTROL,
		targetPositions=[0, 0],
		forces=[20, 20],
	)
	step_for(0.7)

	tool_state = p.getLinkState(robot, END_EFFECTOR, computeForwardKinematics=True)
	inverse_position, inverse_orientation = p.invertTransform(tool_state[4], tool_state[5])
	block_position, block_orientation = p.getBasePositionAndOrientation(block)
	local_position, local_orientation = p.multiplyTransforms(
		inverse_position,
		inverse_orientation,
		block_position,
		block_orientation,
	)
	grasp = p.createConstraint(
		robot,
		END_EFFECTOR,
		block,
		-1,
		p.JOINT_FIXED,
		[0, 0, 0],
		local_position,
		[0, 0, 0],
		parentFrameOrientation=local_orientation,
	)

	move_arm(robot, [start[0], start[1], approach_height], 1)
	move_arm(robot, [finish[0], finish[1], approach_height], 2)
	move_arm(robot, place_position, 1)
	p.removeConstraint(grasp)
	step_for(0.5)
	p.setJointMotorControlArray(
		robot,
		GRIPPER_JOINTS,
		p.POSITION_CONTROL,
		targetPositions=[0.04, 0.04],
		forces=[20, 20],
	)
	move_arm(robot, [finish[0], finish[1], approach_height], 1)

	while p.isConnected(client):
		p.stepSimulation()
		time.sleep(TIME_STEP)
finally:
	if p.isConnected(client):
		p.disconnect(client)