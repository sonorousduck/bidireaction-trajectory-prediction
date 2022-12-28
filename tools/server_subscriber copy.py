import time
import socket
import json

class TrajectoryPredictionSubscriber:
    def __init__(self):
        pass

    def get_images(self):
        pass


if __name__ == "__main__":
    
    trajectory_prediction_sub = TrajectoryPredictionSubscriber()

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    host_name = socket.gethostname()
    host_ip = socket.gethostbyname(host_name)
    print(f"HOST IP: {host_name}")
    port = 5679
    socket_address = (host_ip, port)

    server_socket.bind(socket_address)

    server_socket.listen(5)
    print(f"Listening at: {socket_address}")
    client_socket, addr = server_socket.accept()
    print(f"Got Connection From: {addr}")

    try:
        while True:
            if client_socket:
                data = client_socket.recv(1024)
                if not data:
                    print("No data was found... Ending")
                    break
    except Exception as e:
        print(f"Terminating program due to {e}")
        server_socket.close()

    finally:
        print("Terminating Server")
        server_socket.close()
