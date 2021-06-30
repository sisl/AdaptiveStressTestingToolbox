def make_env(env_name, **kwargs):
	if env_name == 'highway':
		from traffic.scenarios.highway import HighWay
		env = HighWay(**kwargs)
	else:
		raise NotImplementedError
	return env