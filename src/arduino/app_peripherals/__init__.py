from pathlib import Path
import threading

# Base IPC key for ALSA devices. Newly defined PCMs need unique IPC keys.
_alsa__device_ipc_key = 1024
_alsa__device_ipc_cache = {}
_alsa__configuration_lock = threading.Lock()


def _get_next_alsa_ipc_key(hw_device: str) -> int:
    """Get a unique ALSA IPC key for the given hardware device identifier.
    Args:
        hw_device (str): The hardware device identifier.
        Returns:
        int: A unique IPC key for the ALSA device.
        Raises:
        ValueError: If hw_device is None or an empty string.
    """
    if hw_device is None or hw_device == "":
        raise ValueError("Hardware device identifier must be provided for ALSA IPC key generation.")

    global _alsa__device_ipc_key, _alsa__device_ipc_cache, _alsa__configuration_lock

    with _alsa__configuration_lock:
        if hw_device in _alsa__device_ipc_cache:
            return _alsa__device_ipc_cache[hw_device]

        # Get the next available IPC key
        ipc_key = _alsa__device_ipc_key
        _alsa__device_ipc_cache[hw_device] = ipc_key
        _alsa__device_ipc_key += 1  # Increment for next use
        return ipc_key


def _write_alsa_config(device_conf_file_name: str, alsa_wrapper_file_template: str):
    # Avoid race conditions when multiple processes try to write ALSA config
    global _alsa__configuration_lock
    with _alsa__configuration_lock:
        home_directory = Path.home()
        alsa_sound_src_path = home_directory / ".asoundrc"
        alsa_conf_path = home_directory / device_conf_file_name

        write_alsa_conf = True
        if Path.exists(alsa_conf_path):
            with open(alsa_conf_path, "r") as f:
                file_content = f.read()
                if alsa_wrapper_file_template == file_content:
                    write_alsa_conf = False

        if write_alsa_conf:
            print(f"---> Writing ALSA config to {alsa_conf_path}")
            with open(alsa_conf_path, "w") as f:
                f.write(alsa_wrapper_file_template)

        if Path.exists(alsa_sound_src_path):
            with open(alsa_sound_src_path, "r") as f:
                file_content = f.read()
                if f"<{alsa_conf_path}>" not in file_content:
                    with open(alsa_sound_src_path, "a") as f:
                        f.write(f"\n<{alsa_conf_path}>\n")
        else:
            with open(alsa_sound_src_path, "w") as f:
                f.write(f"<{alsa_conf_path}>\n")
