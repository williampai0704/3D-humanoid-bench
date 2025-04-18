import gymnasium as gym
import numpy as np
import argparse
from humanoid_bench.mjx.visualization_utils import save_numpy_as_video
import humanoid_bench

parser = argparse.ArgumentParser(prog="HumanoidBench environment test")
parser.add_argument("--env", help="e.g. h1-walk-v0")
parser.add_argument("--keyframe", default=None)
parser.add_argument("--policy_path", default=None)
parser.add_argument("--mean_path", default=None)
parser.add_argument("--var_path", default=None)
parser.add_argument("--policy_type", default=None)
parser.add_argument("--blocked_hands", default="False")
parser.add_argument("--small_obs", default="False")
parser.add_argument("--obs_wrapper", default="False")
parser.add_argument("--sensors", default="")
parser.add_argument("--render_mode", default="rgb_array")
args = parser.parse_args()

kwargs = vars(args).copy()
kwargs.pop("env")
kwargs.pop("render_mode")
if kwargs["keyframe"] is None:
    kwargs.pop("keyframe")
print(f"arguments: {kwargs}")

# Create the environment
env = gym.make(args.env, render_mode="rgb_array", **kwargs)

# Initialize lists to store frames
frames = []
left_eye_frames = []
right_eye_frames = []

# Run the environment
obs, _ = env.reset()
done = False
if isinstance(obs, dict):
    print(f"ob_space = {env.observation_space}")
    print(f"ob = ")
    for k, v in obs.items():
        print(f"  {k}: {v.shape}")
else:
    print(f"ob_space = {env.observation_space}, ob = {obs.shape}")
print(f"ac_space = {env.action_space.shape}")

while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    
    # Get the rendered frame
    frame = env.render()
    frames.append(frame)
    
    # Only try to access camera images if obs is a dictionary
    if isinstance(obs, dict):
        left_eye_frames.append(obs['image_left_eye'])
        right_eye_frames.append(obs['image_right_eye'])

# Convert frames to numpy arrays
frames = np.array(frames)

# Save the main rendered video
save_numpy_as_video(frames, "output.mp4", fps=20)

# Save camera videos if they exist
if left_eye_frames:
    left_eye_frames = np.array(left_eye_frames)
    right_eye_frames = np.array(right_eye_frames)
    save_numpy_as_video(left_eye_frames, "left_eye.mp4", fps=20)
    save_numpy_as_video(right_eye_frames, "right_eye.mp4", fps=20)

# Close the environment
env.close() 