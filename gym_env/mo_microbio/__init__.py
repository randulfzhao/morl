from gymnasium.envs.registration import register

register(
    id="mo-microbio-v0",
    entry_point="mo_gymnasium.envs.mo_microbio.mo_microbio:MOEvolvingEnv",  # 入口指向 mo_microbio.py 中的环境类
    nondeterministic=True,
)
