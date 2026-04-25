from openenv.core.env_server.mcp_types import CallToolObservation
obs = CallToolObservation(tool_name="test")
obs.metadata = {"hello": "world"}
print(obs.model_dump())
print(obs.model_dump(exclude_unset=True))
