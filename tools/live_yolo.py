import os
import sys
import torch
from torch.nn import functional as FeatureAlphaDropout
sys.path.append(os.path.realpath('../'))
import numpy as np
# from bitrap.modeling import make_model
# from configs import cfg
# from termcolor import colored 
from torch.utils.data import DataLoader
import cv2
import matplotlib.pyplot as plt

# def setup_trajectory_model():
#     cfg.merge_from_file("../configs/bitrap_np_JAAD.yml")
#     os.environ['CUDA_VISIBLE_DEVICES'] = "0"

#     model = make_model(cfg)
#     model = model.to(cfg.DEVICE)
#     checkpoint = "../" + cfg.CKPT_DIR + "best.pth"

#     if os.path.isfile(checkpoint):
#         model.load_state_dict(torch.load(checkpoint))
#         print(colored('Loaded checkpoint:{}'.format(checkpoint), 'blue', 'on_green'))
#     else:
#         print(colored('The cfg.CKPT_DIR id not a file: {}'.format(checkpoint), 'green', 'on_red'))

#     return model

# def predict(model, input_x, cur_pos = None):
#     gt_goal = None
#     cur_pos = input_x[:, -1, :cfg.MODEL.DEC_OUTPUT_DIM] if cur_pos is None else cur_pos
#     batch_size, seg_len, _ = input_x.shape

#     h_x = model.encoder(input_x, None)
#     Z, KLD = model.gaussian_latent_net(h_x, input_x[:, -1, :cfg.MODEL.DEC_OUTPUT_DIM], None, z_mode=False)
#     enc_h_and_z = torch.cat([h_x.unsqueeze(1).repeat(1, Z.shape[1], 1), Z], dim=-1)
#     pred_goal = model.goal_decoder(enc_h_and_z)
#     # dec_h = enc_h_and_z if model.cfg.DEC_WITH_Z else h_x

#     pred_goal = model.goal_decoder(enc_h_and_z)

#     # dec_h = enc_h_and_z if model.cfg.DEC_WITH_Z else h_x
#     pred_traj = model.pred_future_traj(h_x, pred_goal)
#     cur_pos = input_x[:, -1, :cfg.MODEL.DEC_OUTPUT_DIM] if cur_pos is None else cur_pos.unsqueeze(1)
#     pred_goal = pred_goal + cur_pos
#     pred_traj = pred_traj + cur_pos.unsqueeze(1)

#     return pred_traj, pred_goal


if __name__ == "__main__":

    # model = setup_trajectory_model()


    video = cv2.VideoCapture(0)

    if video.isOpened() == False:
        print("Error reading video file")
    yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    stats = {}

    data_for_prediction = []

    _min = np.array([0,0,0,0])[None, :]
    _max = np.array([1920, 1080, 1920, 1080])[None, :]


    while True:
        ret, frame = video.read()

        results = yolo(frame)
        # _, frame = cv2.imencode('.jpeg', frame)
        


        for j, prediction in enumerate(results.pandas().xyxy):
            df = results.pandas().xyxy[j]
            bounding_boxes_pedestrians = df.loc[df['class'] == 0]

            for i, row in bounding_boxes_pedestrians.iterrows():
                cv2.rectangle(frame, (int(row['xmin']), int(row['ymin'])), (int(row['xmax']), int(row['ymax'])), (0, 255, 0), 1)
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
            frame = cv2.flip(frame, 1)
            cv2.imshow('Press q to close', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            break

    video.release()
    cv2.destroyAllWindows()

