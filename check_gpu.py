# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())


from tensorflow.python.client import device_lib

devices = device_lib.list_local_devices()
for device in devices:
    print(f"Name: {device.name}, Type: {device.device_type}")
