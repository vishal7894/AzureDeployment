import docker

client = docker.from_env()
image = client.images.build(path = "./", tag = 'pepsomest.azurecr.io/dash-37-eg:test', platform= 'linux', nocache=True)
resp = client.api.push('pepsomest.azurecr.io/dash-37-eg:test')
print(image)