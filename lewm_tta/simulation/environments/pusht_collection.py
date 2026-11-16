"""PushT-specific bounded exploratory controls and true body-pair contacts.

Only the collector uses this controller. It changes actions, never object poses,
walls, physics parameters, camera framing or the native task goal.
"""

import numpy as np


def block_vertices(native):
    return np.asarray([tuple(native.block.local_to_world(v))
                       for shape in native.block.shapes for v in shape.get_vertices()], dtype=np.float64)


def body_bounds(native):
    vertices = block_vertices(native)
    radii = [float(shape.radius) for shape in native.block.shapes]
    agent_shapes = list(native.agent.shapes)
    centers = np.asarray([tuple(native.agent.local_to_world(shape.offset)) for shape in agent_shapes])
    agent_radii = np.asarray([shape.radius for shape in agent_shapes])[:, None]
    return np.asarray([np.r_[(centers-agent_radii).min(0), (centers+agent_radii).max(0)],
                       np.r_[vertices.min(0)-max(radii), vertices.max(0)+max(radii)]])


def agent_block_pair(arbiter, native):
    first, second = (shape.body for shape in arbiter.shapes)
    return ((first is native.agent and second is native.block)
            or (second is native.agent and first is native.block))


class BoundedContactPolicy:
    def __init__(self, native, settings):
        self.native, self.settings = native, settings
        self.direction = None
        self.age = self.push_steps = 0

    def choose_direction(self, agent, center, vertices, rng):
        # Prefer a reachable side and a modest inward push. The safety score uses
        # the actual rotated shape, including the source/shifted geometry.
        angles = rng.uniform(0, 2*np.pi, 32)
        directions = np.stack((np.cos(angles), np.sin(angles)), axis=1)
        toward = center-agent
        toward /= max(np.linalg.norm(toward), 1.)
        extent = self.native.window_size
        lower, upper = vertices.min(0), vertices.max(0)
        displaced_lower = lower+directions*45
        displaced_upper = upper+directions*45
        clearance = np.minimum(displaced_lower.min(1), (extent-displaced_upper).min(1))
        new_center = center+directions*45
        radial_improvement = np.linalg.norm(center-extent/2)-np.linalg.norm(new_center-extent/2, axis=1)
        score = directions@toward + .025*radial_improvement
        score -= np.maximum(self.settings["safety_margin"]-clearance, 0)*.25
        return directions[int(np.argmax(score))]

    def action(self, observation, rng):
        native, settings = self.native, self.settings
        agent = np.asarray(native.agent.position, dtype=np.float64)
        vertices = block_vertices(native)
        center = (vertices.min(0)+vertices.max(0))/2
        radius = max(float(shape.radius) for shape in native.agent.shapes)
        relative = vertices-center
        projected = vertices if self.direction is None else vertices+self.direction*settings["push_distance"]
        clearance = min(vertices.min(), native.window_size-vertices.max())
        projected_clearance = min(projected.min(), native.window_size-projected.max())
        # Already-near-edge starts still need a stable inward recovery direction.
        unsafe = projected_clearance < settings["safety_margin"] and projected_clearance < clearance
        if self.direction is None or self.age >= 70 or self.push_steps >= 14 or unsafe:
            self.direction = self.choose_direction(agent, center, vertices, rng)
            self.age = self.push_steps = 0
        self.age += 1
        direction = self.direction
        back_distance = -(relative@direction).min()+radius+8
        behind = center-direction*back_distance
        ray = agent-center
        agent_angle = np.arctan2(ray[1], ray[0])
        behind_angle = np.arctan2(-direction[1], -direction[0])
        angle = (behind_angle-agent_angle+np.pi) % (2*np.pi)-np.pi
        if abs(angle) > .55 and np.linalg.norm(agent-behind) > 16:
            # Circle around the bounding disk, rather than walking through the
            # block while trying to reach its opposite side.
            orbit_radius = np.linalg.norm(relative, axis=1).max()+radius+12
            angle = agent_angle+np.clip(angle, -.4, .4)
            target = center+orbit_radius*np.array([np.cos(angle), np.sin(angle)])
        elif np.linalg.norm(agent-behind) > 12 and self.push_steps == 0:
            target = behind
        else:
            target = agent+direction*settings["push_distance"]
            self.push_steps += 1
        low = radius+settings["safety_margin"]
        target = np.clip(target, low, native.window_size-low)
        command = (target-agent)/native.action_scale
        command *= min(1., settings["max_action_norm"]/max(np.linalg.norm(command), 1e-12))
        return command.astype(np.float32)
