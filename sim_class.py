import pybullet as p
import time
import pybullet_data
import math
import logging
import os
import random

#logging.basicConfig(level=logging.INFO)

class Simulation:
    def __init__(self, num_agents, render=True, rgb_array=False):
        self.render = render
        self.rgb_array = rgb_array
        if render:
            mode = p.GUI # for graphical version
        else:
            mode = p.DIRECT # for non-graphical version
        # Set up the simulation
        self.physicsClient = p.connect(mode)
        # Hide the default GUI components
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
        p.setGravity(0,0,-10)
        #p.setPhysicsEngineParameter(contactBreakingThreshold=0.000001)
        # load a texture
        texture_list = os.listdir("textures")
        random_texture = random.choice(texture_list[:-1])
        random_texture_index = texture_list.index(random_texture)
        
        # FIX: Get plates list separately and use safe indexing
        plates_list = os.listdir("textures/_plates")
        if plates_list and random_texture_index < len(plates_list):
            self.plate_image_path = f'textures/_plates/{plates_list[random_texture_index]}'
        elif plates_list:
            self.plate_image_path = f'textures/_plates/{plates_list[0]}'  # Fallback to first plate
        else:
            self.plate_image_path = None  # No plates available
        
        self.textureId = p.loadTexture(f'textures/{random_texture}')
        #print(f'textureId: {self.textureId}')

        # Set the camera parameters
        cameraDistance = 1.1*(math.ceil((num_agents)**0.3)) # Distance from the target (zoom)
        cameraYaw = 90  # Rotation around the vertical axis in degrees
        cameraPitch = -35  # Rotation around the horizontal axis in degrees
        cameraTargetPosition = [-0.2, -(math.ceil(num_agents**0.5)/2)+0.5, 0.1]  # XYZ coordinates of the target position

        # Reset the camera with the specified parameters
        p.resetDebugVisualizerCamera(cameraDistance, cameraYaw, cameraPitch, cameraTargetPosition)

        self.baseplaneId = p.loadURDF("plane.urdf")
        # add collision shape to the plane
        #p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=[30, 305, 0.001])

        # define the pipette offset
        self.pipette_offset = [0.073, 0.0895, 0.0895]
        # dictionary to keep track of the current pipette position per robot
        self.pipette_positions = {}

        # Create the robots
        self.create_robots(num_agents)

        # list of sphere ids
        self.sphereIds = []

        # dictionary to keep track of the droplet positions on specimens key for specimenId, list of droplet positions
        self.droplet_positions = {}

    # method to create n robots in a grid pattern
    def create_robots(self, num_agents):
        spacing = 1  # Adjust the spacing as needed

        # Calculate the grid size to fit all agents
        grid_size = math.ceil(num_agents ** 0.5) 

        self.robotIds = []
        self.specimenIds = []
        agent_count = 0  # Counter for the number of placed agents

        for i in range(grid_size):
            for j in range(grid_size):
                if agent_count < num_agents:  # Check if more agents need to be placed
                    # Calculate position for each robot
                    position = [-spacing * i, -spacing * j, 0.03]
                    robotId = p.loadURDF("ot_2_simulation_v6.urdf", position, [0,0,0,1],
                                        flags=p.URDF_USE_INERTIA_FROM_FILE)
                    start_position, start_orientation = p.getBasePositionAndOrientation(robotId)
                    p.createConstraint(parentBodyUniqueId=robotId,
                                    parentLinkIndex=-1,
                                    childBodyUniqueId=-1,
                                    childLinkIndex=-1,
                                    jointType=p.JOINT_FIXED,
                                    jointAxis=[0, 0, 0],
                                    parentFramePosition=[0, 0, 0],
                                    childFramePosition=start_position,
                                    childFrameOrientation=start_orientation)

                    # Load the specimen with an offset
                    offset = [0.18275-0.00005, 0.163-0.026, 0.057]
                    position_with_offset = [position[0] + offset[0], position[1] + offset[1], position[2] + offset[2]]
                    rotate_90 = p.getQuaternionFromEuler([0, 0, -math.pi/2])
                    planeId = p.loadURDF("custom.urdf", position_with_offset, rotate_90)
                    # Disable collision between the robot and the specimen
                    p.setCollisionFilterPair(robotId, planeId, -1, -1, enableCollision=0)
                    spec_position, spec_orientation = p.getBasePositionAndOrientation(planeId)

                    p.createConstraint(parentBodyUniqueId=planeId,
                                    parentLinkIndex=-1,
                                    childBodyUniqueId=-1,
                                    childLinkIndex=-1,
                                    jointType=p.JOINT_FIXED,
                                    jointAxis=[0, 0, 0],
                                    parentFramePosition=[0, 0, 0],
                                    childFramePosition=spec_position,
                                    childFrameOrientation=spec_orientation)
                    # Load your texture and apply it to the plane
                    p.changeVisualShape(planeId, -1, textureUniqueId=self.textureId)

                    self.robotIds.append(robotId)
                    self.specimenIds.append(planeId)

                    agent_count += 1

                    # calculate the pipette position
                    pipette_position = self.get_pipette_position(robotId)
                    # save the pipette position
                    self.pipette_positions[f'robotId_{robotId}'] = pipette_position

    # method to get the current pipette position for a robot
    def get_pipette_position(self, robotId):
        robot_position = p.getBasePositionAndOrientation(robotId)[0]
        robot_position = list(robot_position)
        joint_states = p.getJointStates(robotId, [0, 1, 2])
        robot_position[0] -= joint_states[0][0]
        robot_position[1] -= joint_states[1][0]
        robot_position[2] += joint_states[2][0]
        x_offset = self.pipette_offset[0]
        y_offset = self.pipette_offset[1]
        z_offset = self.pipette_offset[2]
        pipette_position = [robot_position[0]+x_offset, robot_position[1]+y_offset, robot_position[2]+z_offset]
        return pipette_position

    # method to reset the simulation
    def reset(self, num_agents=1):
        for specimenId in self.specimenIds:
            p.changeVisualShape(specimenId, -1, textureUniqueId=-1)

        for robotId in self.robotIds:
            p.removeBody(robotId)
            self.robotIds.remove(robotId)

        for specimenId in self.specimenIds:
            p.removeBody(specimenId)
            self.specimenIds.remove(specimenId)

        for sphereId in self.sphereIds:
            p.removeBody(sphereId)
            self.sphereIds.remove(sphereId)

        self.pipette_positions = {}
        self.sphereIds = []
        self.droplet_positions = {}

        self.create_robots(num_agents)

        return self.get_states()

    # method to run the simulation for a specified number of steps
    def run(self, actions, num_steps=1):
        start = time.time()
        n = 100
        for i in range(num_steps):
            self.apply_actions(actions)
            p.stepSimulation()

            for specimenId, robotId in zip(self.specimenIds, self.robotIds):
                self.check_contact(robotId, specimenId)

            if self.rgb_array:
                camera_pos = [1, 0, 1]
                camera_target = [-0.3, 0, 0]
                up_vector = [0, 0, 1]
                fov = 50
                aspect = 320/240

                width, height, rgbImg, depthImg, segImg = p.getCameraImage(width=320, height=240, viewMatrix=p.computeViewMatrix(camera_pos, camera_target, up_vector), projectionMatrix=p.computeProjectionMatrixFOV(fov, aspect, 0.1, 100.0))
                
                self.current_frame = rgbImg

            if self.render:
                time.sleep(1./240.)

        return self.get_states()
    
    # method to apply actions to the robots using velocity control
    def apply_actions(self, actions):
        for i in range(len(self.robotIds)):
            p.setJointMotorControl2(self.robotIds[i], 0, p.VELOCITY_CONTROL, targetVelocity=-actions[i][0], force=500)
            p.setJointMotorControl2(self.robotIds[i], 1, p.VELOCITY_CONTROL, targetVelocity=-actions[i][1], force=500)
            p.setJointMotorControl2(self.robotIds[i], 2, p.VELOCITY_CONTROL, targetVelocity=actions[i][2], force=800)
            if actions[i][3] == 1:
                self.drop(robotId=self.robotIds[i])

    # method to drop a simulated droplet on the specimen from the pipette
    def drop(self, robotId):
        robot_position = p.getBasePositionAndOrientation(robotId)[0]
        robot_position = list(robot_position)
        joint_states = p.getJointStates(robotId, [0, 1, 2])
        robot_position[0] -= joint_states[0][0]
        robot_position[1] -= joint_states[1][0]
        robot_position[2] += joint_states[2][0]
        x_offset = self.pipette_offset[0]
        y_offset = self.pipette_offset[1]
        z_offset = self.pipette_offset[2]-0.0015
        specimen_position = p.getBasePositionAndOrientation(self.specimenIds[0])[0]
        sphereRadius = 0.003
        sphereColor = [1, 0, 0, 0.5]
        visualShapeId = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=sphereRadius, rgbaColor=sphereColor)
        collision = p.createCollisionShape(shapeType=p.GEOM_SPHERE, radius=sphereRadius)
        sphereBody = p.createMultiBody(baseMass=0.1, baseVisualShapeIndex=visualShapeId, baseCollisionShapeIndex=collision)
        droplet_position = [robot_position[0]+x_offset, robot_position[1]+y_offset, robot_position[2]+z_offset]
        p.resetBasePositionAndOrientation(sphereBody, droplet_position, [0, 0, 0, 1])
        self.sphereIds.append(sphereBody)
        self.dropped = True
        return droplet_position

    # method to get the states of the robots
    def get_states(self):
        states = {}
        for robotId in self.robotIds:
            raw_joint_states = p.getJointStates(robotId, [0, 1, 2])

            joint_states = {}
            for i, joint_state in enumerate(raw_joint_states):
                joint_states[f'joint_{i}'] = {
                    'position': joint_state[0],
                    'velocity': joint_state[1],
                    'reaction_forces': joint_state[2],
                    'motor_torque': joint_state[3]
                }

            robot_position = p.getBasePositionAndOrientation(robotId)[0]
            robot_position = list(robot_position)

            robot_position[0] -= raw_joint_states[0][0]
            robot_position[1] -= raw_joint_states[1][0]
            robot_position[2] += raw_joint_states[2][0]

            pipette_position = [robot_position[0] + self.pipette_offset[0],
                                robot_position[1] + self.pipette_offset[1],
                                robot_position[2] + self.pipette_offset[2]]
            pipette_position = [round(num, 4) for num in pipette_position]

            states[f'robotId_{robotId}'] = {
                "joint_states": joint_states,
                "robot_position": robot_position,
                "pipette_position": pipette_position
            }

        return states
    
    # method to check contact with the spheres and the specimen and robot
    def check_contact(self, robotId, specimenId):
        for sphereId in self.sphereIds:
            contact_points_specimen = p.getContactPoints(sphereId, specimenId)
            contact_points_robot = p.getContactPoints(sphereId, robotId)

            if contact_points_specimen:
                p.setCollisionFilterPair(sphereId, specimenId, -1, -1, enableCollision=0)
                sphere_position, sphere_orientation = p.getBasePositionAndOrientation(sphereId)
                p.createConstraint(parentBodyUniqueId=sphereId,
                                    parentLinkIndex=-1,
                                    childBodyUniqueId=-1,
                                    childLinkIndex=-1,
                                    jointType=p.JOINT_FIXED,
                                    jointAxis=[0, 0, 0],
                                    parentFramePosition=[0, 0, 0],
                                    childFramePosition=sphere_position,
                                    childFrameOrientation=sphere_orientation)
                if f'specimenId_{specimenId}' in self.droplet_positions:
                    self.droplet_positions[f'specimenId_{specimenId}'].append(sphere_position)
                else:
                    self.droplet_positions[f'specimenId_{specimenId}'] = [sphere_position]

            if contact_points_robot:
                p.removeBody(sphereId)
                self.sphereIds.remove(sphereId)

    def set_start_position(self, x, y, z):
        for robotId in self.robotIds:
            robot_position = p.getBasePositionAndOrientation(robotId)[0]
            adjusted_x = x - robot_position[0] - self.pipette_offset[0]
            adjusted_y = y - robot_position[1] - self.pipette_offset[1]
            adjusted_z = z - robot_position[2] - self.pipette_offset[2]

            p.resetJointState(robotId, 0, targetValue=adjusted_x)
            p.resetJointState(robotId, 1, targetValue=adjusted_y)
            p.resetJointState(robotId, 2, targetValue=adjusted_z)

    # function to return the path of the current plate image
    def get_plate_image(self):
        return self.plate_image_path
    
    # close the simulation
    def close(self):
        p.disconnect()
