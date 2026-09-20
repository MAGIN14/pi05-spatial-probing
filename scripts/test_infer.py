from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.policies import libero_policy
from openpi.shared import download

cfg = _config.get_config("pi05_libero")
ckpt = download.maybe_download("gs://openpi-assets/checkpoints/pi05_libero")
policy = policy_config.create_trained_policy(cfg, ckpt)

example = libero_policy.make_libero_example()
for k, v in example.items():
    print(k, "->", getattr(v, "shape", v), getattr(v, "dtype", ""))

result = policy.infer(example)
print("action shape:", result["actions"].shape)
