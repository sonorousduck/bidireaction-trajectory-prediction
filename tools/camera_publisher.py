import socket
import cv2
import pickle
import struct

if __name__ == "__main__":
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    host_name = socket.gethostname()
    host_ip = socket.gethostbyname(host_name)
    print('HOST IP:',host_ip)
    port = 5678
    socket_address = (host_ip, port)
    
    server_socket.bind(socket_address)

    server_socket.listen(5)
    print("Listening at:", socket_address)

    client_socket, addr = server_socket.accept()
    print("Got connection from:", addr)
    
    while True:
        if client_socket:
            vid = cv2.VideoCapture(0)
            while vid.isOpened():
                img, frame = vid.read()

                a = pickle.dumps(frame)
                message = struct.pack("Q", len(a)) + a
                client_socket.sendall(message)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    client_socket.close()
