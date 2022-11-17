import os
import sys
import torch
from torch.nn import functional as FeatureAlphaDropout
import numpy as np
sys.path.append(os.path.realpath('./'))
from bitrap.modeling import make_model
from configs import cfg
from termcolor import colored 
from torch.utils.data import DataLoader
import cv2
import matplotlib.pyplot as plt
sys.path.append(os.path.realpath('../'))
from sort import *
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setup_trajectory_model():
    cfg.merge_from_file("./configs/bitrap_np_JAAD.yml")
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    model = make_model(cfg)
    model = model.to(cfg.DEVICE)
    checkpoint = "./" + cfg.CKPT_DIR + "best.pth"

    if os.path.isfile(checkpoint):
        model.load_state_dict(torch.load(checkpoint))
        print(colored('Loaded checkpoint:{}'.format(checkpoint), 'blue', 'on_green'))
    else:
        print(colored('The cfg.CKPT_DIR id not a file: {}'.format(checkpoint), 'green', 'on_red'))

    return model

def predict(model, input_x, cur_pos = None):
    gt_goal = None
    cur_pos = input_x[:, -1, :cfg.MODEL.DEC_OUTPUT_DIM] if cur_pos is None else cur_pos
    batch_size, seg_len, _ = input_x.shape

    h_x = model.encoder(input_x, None)
    Z, KLD = model.gaussian_latent_net(h_x, input_x[:, -1, :cfg.MODEL.DEC_OUTPUT_DIM], None, z_mode=False)
    enc_h_and_z = torch.cat([h_x.unsqueeze(1).repeat(1, Z.shape[1], 1), Z], dim=-1)
    pred_goal = model.goal_decoder(enc_h_and_z)
    # dec_h = enc_h_and_z if model.cfg.DEC_WITH_Z else h_x

    pred_goal = model.goal_decoder(enc_h_and_z)

    # dec_h = enc_h_and_z if model.cfg.DEC_WITH_Z else h_x
    pred_traj = model.pred_future_traj(h_x, pred_goal)
    cur_pos = input_x[:, -1, :cfg.MODEL.DEC_OUTPUT_DIM] if cur_pos is None else cur_pos.unsqueeze(1)
    pred_goal = pred_goal + cur_pos
    pred_traj = pred_traj + cur_pos.unsqueeze(1)

    return pred_traj, pred_goal

class Person:
    def __init__(self, id):
        self.id = id
        # Form is x1, y1, x2, y2, id
        self.frames = []
    
    def add_frame(self, frame):
        self.frames.append(frame)
        return len(self.frames) >= 15

    def get_prediction_frames(self):
        frames = self.frames[:15]
        self.frames.pop(0)
        return frames

    def __len__(self):
        return len(self.frames)

    def __str__(self):
        return f"Person {self.id}, with {len(self.frames)} frames"
        



if __name__ == "__main__":

    model = setup_trajectory_model()

    mot_tracker = Sort() 
    video = cv2.VideoCapture(0) # THIS MAKES IT USE THE REALSENSE INSTEAD
    
    video.set(3, 640)
    video.set(4, 480)

    if video.isOpened() == False:
        print("Error reading video file")
    
    
    yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    cfg.merge_from_file("./configs/bitrap_np_JAAD.yml")
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    stats = {}

    ids_to_update = []


    _min = np.array([0,0,0,0])[None, :]
    _max = np.array([640, 480, 640, 480])[None, :]
    # _max = np.array([video.get(cv2.CAP_PROP_FRAME_WIDTH), video.get(cv2.CAP_PROP_FRAME_HEIGHT), video.get(cv2.CAP_PROP_FRAME_WIDTH), video.get(cv2.CAP_PROP_FRAME_HEIGHT)])[None, :]
    ids = {}
    print(video.get(cv2.CAP_PROP_FRAME_WIDTH), video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(video.get(cv2.CAP_PROP_FPS))
    test = 0

    while True:
        ret, frame = video.read()
        
        if test < 15:
            if cv2.imwrite(f"image_{test}.png", frame):
                test += 1
        else:
            print("COMPLETE!")
        results = yolo(frame)
        detections = results.pred[0].cpu().numpy()
        people = []

        for i, prediction in enumerate(detections):
            if prediction[5] == 0:
                people.append(prediction)
        if len(people) > 0:
        # detections = np.where(detections[i][5] == 0 for i in range(len(detections)))
            track_bbs_ids = mot_tracker.update(np.array(people))



            for j, coords in enumerate(track_bbs_ids.tolist()):
                x1, y1, x2, y2, name_idx = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3]), int(coords[4])
                name = f"ID: {name_idx}"

                bbox = np.array([[coords[0], coords[1], coords[2], coords[3]]])
                # test = np.array([[coords[0], coords[1], coords[2], coords[3]]])


                reconstruction = np.array([[0., 0., 0., 0.]])
                reconstruction[..., [0, 1]] = bbox[..., [0, 1]]

                # test[..., [2, 3]] -= reconstruction[..., [0, 1]]
                # test[..., [0, 1]] += reconstruction[..., [2, 3]]

                bbox[..., [2, 3]] -= reconstruction[..., [0, 1]]
                reconstruction[..., [2, 3]] = bbox[..., [2, 3]] / 2
                bbox[..., [0, 1]] += reconstruction[..., [2, 3]]

                # print(bbox[0][0] == test[0][0])


                bbox = (bbox - _min) / (_max - _min)

                
                bbox = bbox[0]
                if name_idx not in ids.keys():
                    ids[name_idx] = Person(id)                    
                
                if ids[name_idx].add_frame(coords):
                    ids_to_update.append(name_idx)


                bbox = bbox * (_max - _min) + _min
                bbox[..., [0, 1]] -= reconstruction[..., [2, 3]]
                bbox[..., [2, 3]] += reconstruction[..., [0, 1]]
                # bbox[..., [0, 1]] -= bbox[..., [2, 3]] * 2
                # bbox[..., [2, 3]] = bbox[..., [2, 3]] + bbox[..., [0, 1]]

                x, y, x2, y2 = int(bbox[0][0]), int(bbox[0][1]), int(bbox[0][2]), int(bbox[0][3])
                # bbox[0], bbox[1], bbox[2], bbox[3] = x, y, int(x + w), int(y + h)
                cv2.rectangle(frame, (x, y), (int(x2) , int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                
        for person_id in ids_to_update:
            data_to_predict_on = []
            reconstruction = np.array([[0., 0., 0., 0.]])

            bounding_boxes = ids[person_id].get_prediction_frames()
            # reconstruction_array = []
            for j, row in enumerate(bounding_boxes):

                bbox = np.array([[row[0], row[1], row[2], row[3]]])

                reconstruction[..., [0, 1]] = bbox[..., [0, 1]]

                bbox[..., [2, 3]] -= reconstruction[..., [0, 1]]
                reconstruction[..., [2, 3]] = bbox[..., [2, 3]] / 2
                bbox[..., [0, 1]] += reconstruction[..., [2, 3]]

                bbox = (bbox - _min) / (_max - _min)
                data_to_predict_on.append(bbox[0])
                # reconstruction_array.append(reconstruction)
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
                        cv2.rectangle(frame, (x, y), (w, h), (255, 0, 0), 1)
                        # cv2.line(frame, (int(centerX), int(centerY)), (int(nextCenterX), int(nextCenterY)), (0, 0, 255), 1)

            # for traj in pred_traj:
            #     for j, traj in enumerate(test):
            #         for i, box in enumerate(traj):
            #             reconstruction = reconstruction_array[j][0]
            #             # if i == len(traj) - 1:
            #             # rect = cv2.boundingRect(testArray)
            #             # print(reconstruction)
            #             box[0] -= reconstruction[2]
            #             box[1] -= reconstruction[3]
            #             box[2] += reconstruction[0]
            #             box[3] += reconstruction[1]
            #             x, y, w, h = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            #             cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)
            #             break

            for goal in pred_goal:
                location = goal[0]
                # for i, location in enumerate(goal):
                    # if i == len(goal) - 1:
                    # reconstruction = reconstruction_array[i][0]

                location[0] -= reconstruction[2]
                location[1] -= reconstruction[3]
                location[2] += reconstruction[0]
                location[3] += reconstruction[1]
                x, y, w, h = int(location[0]), int(location[1]), int(location[2]), int(location[3])
                cv2.rectangle(frame, (x, y), (int(x + w) , int(y + h)), (0, 255, 0), 2)
                    # break
        
        ids_to_update.clear()



        # for j, prediction in enumerate(results.pandas().xyxy):
        #     df = results.pandas().xyxy[j]
        #     bounding_boxes_pedestrians = df.loc[df['class'] == 0]
    
        #     for i, row in bounding_boxes_pedestrians.iterrows():
        #         cv2.rectangle(frame, (int(row['xmin']), int(row['ymin'])), (int(row['xmax']), int(row['ymax'])), (0, 255, 0), 1)
                # bbox = np.array([[row['xmin'], row['ymin'], row['xmax'], row['ymax']]])
                # bbox[..., [2, 3]] = bbox[..., [2, 3]] - bbox[..., [0, 1]]
                # bbox[..., [0, 1]] += bbox[..., [2, 3]]/2

                # bbox = (bbox - _min) / (_max - _min)
                # data_for_prediction.append(bbox[0])
        
        # if (len(data_for_prediction) > 15):
        #     # Predict trajectory with 15 images
        #     # Time speed of this, then see if fast enough, make a rolling calculation instead.
        #     # For now, remove 15 values.

        #     prediction_data = torch.FloatTensor(np.array(data_for_prediction[:15]))
        #     prediction_data = prediction_data.unsqueeze(0)
        #     cur_pos = prediction_data[:, -1, :cfg.MODEL.DEC_OUTPUT_DIM]
        #     pred_traj, pred_goal = predict(model, prediction_data, cur_pos=prediction_data[-1, :cfg.MODEL.DEC_OUTPUT_DIM])
        #     pred_traj = pred_traj.detach().to('cpu').numpy()
        #     pred_goal = pred_goal.detach().to('cpu').numpy()






        # for index, person in enumerate(result):

        #     if not stats.__contains__(index):
        #         stats[index] = []
        #     stats[index].append(person['emotions'])
        #     bounding_box = person['box']
        #     cv2.rectangle(frame, (bounding_box[0], bounding_box[1]), (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]), (255, 0, 0), 4)

        #     max_value = 0
        #     for value in person['emotions'].values():
        #         if value > max_value:
        #             max_value = value

        #     cv2.putText(frame, f"Person {index}", (bounding_box[0], bounding_box[1] - TEXT_SPACING), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        #     for i, (emotion, value) in enumerate(person['emotions'].items()):
        #         if value == max_value:
        #             color = (0, 255, 0)
        #         else:
        #             color = (0, 0, 255)
        #         cv2.putText(frame, f"{emotion}: {value}", (bounding_box[0], bounding_box[1] + bounding_box[3] + TEXT_SPACING * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        if ret == True:
            # Flipping the display because laptop camera dumb
            # frame = cv2.flip(frame, 1)
            cv2.namedWindow("Press q to close", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Press q to close", 640, 480)

            cv2.imshow('Press q to close', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            break

    video.release()
    cv2.destroyAllWindows()

