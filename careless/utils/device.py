import tensorflow as tf

def disable_gpu():
    # Disable all GPUS
    tf.config.set_visible_devices([], 'GPU')
    visible_devices = tf.config.get_visible_devices()
    status = True
    for device in visible_devices:
        status &= device.device_type != 'GPU'
    return status
