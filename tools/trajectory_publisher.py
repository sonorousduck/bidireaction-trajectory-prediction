import time
import socket
import json
import struct
import pickle
import cv2
from object_tracking import *



class TrajectoryPredictionPublisher:
    def __init__(self, image_ip_port):
        self.image_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Other sockets here
        
        self.image_socket.connect(image_ip_port)

        self.data = b""
        self.payload_size = struct.calcsize("Q")
        self.frames = []


    def predict(self, frame):
        results = yolo(frame)
        detections = results.pred[0].cpu().numpy()
        people = []

        for i, prediction in enumerate(detections):
            if prediction[5] == 0:
                people.append(prediction)
        if len(people) > 0:
            track_bbs_ids = mot_tracker.update(np.array(people))



            for j, coords in enumerate(track_bbs_ids.tolist()):
                x1, y1, x2, y2, name_idx = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3]), int(coords[4])
                name = f"ID: {name_idx}"

                bbox = np.array([[coords[0], coords[1], coords[2], coords[3]]])


                reconstruction = np.array([[0., 0., 0., 0.]])
                reconstruction[..., [0, 1]] = bbox[..., [0, 1]]

                bbox[..., [2, 3]] -= reconstruction[..., [0, 1]]
                reconstruction[..., [2, 3]] = bbox[..., [2, 3]] / 2
                bbox[..., [0, 1]] += reconstruction[..., [2, 3]]



                bbox = (bbox - _min) / (_max - _min)

                
                bbox = bbox[0]
                if name_idx not in ids.keys():
                    ids[name_idx] = Person(name_idx)                    
                
                if ids[name_idx].add_frame(coords):
                    ids_to_update.append(name_idx)


                bbox = bbox * (_max - _min) + _min
                bbox[..., [0, 1]] -= reconstruction[..., [2, 3]]
                bbox[..., [2, 3]] += reconstruction[..., [0, 1]]

                x, y, x2, y2 = int(bbox[0][0]), int(bbox[0][1]), int(bbox[0][2]), int(bbox[0][3])
                cv2.rectangle(frame, (x, y), (int(x2) , int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                
        for person_id in ids_to_update:
            data_to_predict_on = []
            reconstruction = np.array([[0., 0., 0., 0.]])

            bounding_boxes = ids[person_id].get_prediction_frames()
            for j, row in enumerate(bounding_boxes):

                bbox = np.array([[row[0], row[1], row[2], row[3]]])

                reconstruction[..., [0, 1]] = bbox[..., [0, 1]]

                bbox[..., [2, 3]] -= reconstruction[..., [0, 1]]
                reconstruction[..., [2, 3]] = bbox[..., [2, 3]] / 2
                bbox[..., [0, 1]] += reconstruction[..., [2, 3]]

                bbox = (bbox - _min) / (_max - _min)
                data_to_predict_on.append(bbox[0])
                
            reconstruction = reconstruction[0]
            # Should have length greater than 15, due to it only being for looped if that holds true
            prediction_data = torch.FloatTensor(np.array(data_to_predict_on[:15])).to(device)
            prediction_data = prediction_data.unsqueeze(0)
            cur_pos = prediction_data[:, -1, :cfg.MODEL.DEC_OUTPUT_DIM]
            pred_traj, pred_goal = predict(model, prediction_data, cur_pos=prediction_data[-1, :cfg.MODEL.DEC_OUTPUT_DIM])
            pred_goal = pred_goal.detach().to('cpu').numpy() * (_max - _min) + _min
            pred_traj = pred_traj.detach().to('cpu').numpy() * (_max - _min) + _min

            # Structure of Pred Traj
            # 4 potential trajectories
            # 45 frames of predictions
            # In each 45, there are 20 predictions, not really sure why.
            trajs = []
            for traj in pred_traj:
                for i, frames in enumerate(traj):
                    box = frames[0]
                    box[0] -= reconstruction[2]
                    box[1] -= reconstruction[3]
                    box[2] += reconstruction[0]
                    box[3] += reconstruction[1]


                    if i < len(traj) - 1:
                        
                    # Compute the center of each bounding box
                        centerX = box[2] + box[0] / 2
                        centerY = box[3] + box[1] / 2

                        nextBox = traj[i + 1][0]
                        nextCenterX = nextBox[2] + nextBox[0] / 2
                        nextCenterY = nextBox[3] + nextBox[1] / 2
                        x, y, w, h = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                        trajs.append((x, y, w, h))
                        # Below is the full bounding box. What is used it prettier to look at
                        # since it will just be a single line on the floor of the bounding box. Much better looking
                        # cv2.rectangle(frame, (x, y), (w, h), (255, 0, 0), 1)
                        cv2.rectangle(frame, (x, h), (w, h + 1), (255, 0, 0), 1)

            goals = []
            for goal in pred_goal:
                location = goal[0]
                location[0] -= reconstruction[2]
                location[1] -= reconstruction[3]
                location[2] += reconstruction[0]
                location[3] += reconstruction[1]
                x, y, w, h = int(location[0]), int(location[1]), int(location[2]), int(location[3])
                goals.append((x, y, w, h))
                cv2.rectangle(frame, (x, y), (int(x + w) , int(y + h)), (0, 255, 0), 2)
        
        bounding_boxes = {"trajectories": trajs, "goals": goals}
        ids_to_update.clear()

        # If you need to display, uncomment this
        # cv2.imshow("Predictions", frame)

        return bounding_boxes
        


    def gather_image(self):
        while len(self.data) < self.payload_size:
            packet = self.image_socket.recv(4 * 1024) # 4K
            if not packet:
                break
            self.data += packet
        packed_msg_size = self.data[:self.payload_size]
        self.data = self.data[self.payload_size:]
        msg_size = struct.unpack("Q", packed_msg_size)[0]

        while len(self.data) < msg_size:
            self.data += self.image_socket.recv(4*1024)
        
        frame_data = self.data[:msg_size]
        self.data = self.data[msg_size:]
        frame = pickle.loads(frame_data)

        return frame

    def shutdown(self):
        self.image_socket.close()

    
if __name__ == "__main__":
    # Create the server that will publish the information that the trajectory publisher creates

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    host_name = socket.gethostname()
    host_ip = socket.gethostbyname(host_name)
    trajectory_prediction_port = 5679

    image_port = 5678
    image_ip_port = (host_ip, image_port)
    trajectory_ip_port = (host_ip, trajectory_prediction_port)

    trajectory_prediction = TrajectoryPredictionPublisher(image_ip_port)

    server_socket.bind(trajectory_ip_port)
    server_socket.listen(5)
    print("LISTENING AT:", trajectory_ip_port)

    model = setup_trajectory_model()
    mot_tracker = Sort()

    yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    cfg.merge_from_file("./configs/bitrap_np_JAAD.yml")
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    stats = {}

    ids_to_update = []


    _min = np.array([0,0,0,0])[None, :]
    _max = np.array([640, 480, 640, 480])[None, :]
    ids = {}


    try:
        while True:
            client_socket,addr = server_socket.accept()

            frame = trajectory_prediction.gather_image()
            bounding_boxes = trajectory_prediction.predict(frame)

            data = pickle.dumps(bounding_boxes)
            message = struct.pack("Q", len(data)) + data
            client_socket.sendall(message)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    except Exception as e:
        print(f"Shutting down because of {e}")
        trajectory_prediction.shutdown()
