import socket


def find_available_port(start_port: int) -> int:
    port = start_port
    while port <= 65_535:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            try:
                server.bind(("0.0.0.0", port))
            except OSError:
                port += 1
                continue
        return port

    raise RuntimeError(f"no available port found from {start_port}")
