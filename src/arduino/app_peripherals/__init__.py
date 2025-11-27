
def _resolve_alsa_device_index(device: str) -> tuple[str, int, int]:
    """Resolve the ALSA device name to (card_index, device_index).
    Args:
        device (str): The ALSA device name.
    Returns:
        tuple[int, int]: (card_index, device_index) or None if not resolvable.
    """

    import alsaaudio
    import re

    match = re.search(r"(plughw|hw):CARD=([\w]+),DEV=([\w])", device)
    if match:
        card = match.group(2)
        device_num = match.group(3)
        card_names = alsaaudio.cards()
        try:
            card_index = card_names.index(card)
            return card, card_index, int(device_num)
        except ValueError:
            print(f"Card '{card}' not found.")
            return None, None, None

    return None, None, None
