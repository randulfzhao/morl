from gymnasium.envs.registration import register

register(
    id="microbio-v0",
    entry_point="mo_gymnasium.envs.microbio.microbio:SingleEvolvingEnv",  # 入口指向 microbio.py 中的环境类
    nondeterministic=True,
)
