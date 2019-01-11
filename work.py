import os
import cv2
import time
import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
import argparse

CWD_PATH = os.getcwd()

# Path to frozen detection graph. This is the actual model that is used for the object detection.
MODEL_NAME = 'ssd_mobilenet_v1_lfp_2019_01_11'
PATH_TO_CKPT = os.path.join(CWD_PATH, 'train_model', MODEL_NAME, 'frozen_inference_graph.pb')

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(CWD_PATH, 'data', 'light_face_person.pbtxt')

NUM_CLASSES = 4

# The size of a frame in the video
FRAME_SHAPE = (668, 1130, 3)

# Batch size of a dataSet, modify this according to your CPU
BATCH_SIZE = 4
PREFETCH = 1

# Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def add_fps(image_np, start, end):
    cv2.putText(image_np, "FPS {0}".format(str(1.0 / (end - start))), (10, 230), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                (255, 0, 255), 2)


def show(image_np):
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    cv2.imshow("capture", image_np)
    cv2.waitKey(1)


def gen():
    while True:
        ret, frame = video_capture.read()
        if ret is None or frame is None:
            break
        else:
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _draw_boxes(image_np, boxes, classes, scores):
    vis_util.visualize_boxes_and_labels_on_image_array(
        np.asarray(image_np),
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=2,
        min_score_thresh=0.05)
    return image_np


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-src', '--source', dest='video_source', type=str,
                        default='C:\Downloads\\test.mp4', help='Path of the source video.')
    parser.add_argument('-rt', '--real_time', dest='real_time', type=int,
                        default=1, help='In real time mode or file mode')
    parser.add_argument('-b', '--boxes', dest='draw_boxes', type=int,
                        default=1, help='draw boxes or not')
    args = parser.parse_args()

    video_capture = cv2.VideoCapture(args.video_source)
    detection_graph = tf.Graph()
    fps = 0
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)

                if args.real_time == 1:
                    tf.import_graph_def(od_graph_def, name='')
                    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                else:
                    dataset = tf.data.Dataset.from_generator(gen, tf.uint8, tf.TensorShape([FRAME_SHAPE[0], FRAME_SHAPE[1], FRAME_SHAPE[2]]))
                    batch = dataset.batch(BATCH_SIZE)
                    batch = batch.prefetch(PREFETCH)
                    it = tf.data.Iterator.from_structure(batch.output_types, batch.output_shapes)
                    sess.run(it.make_initializer(batch))
                    image_tensor = it.get_next()
                    tf.import_graph_def(od_graph_def, name='', input_map={'image_tensor:0': image_tensor})
                boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                scores = detection_graph.get_tensor_by_name('detection_scores:0')
                classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')

                while True:
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    start = time.clock()
                    # Actual detection.
                    if args.real_time == 1:
                        # Load video by frame
                        ret, frame = video_capture.read()
                        if ret is None or frame is None:
                            break
                        image_np = frame
                        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
                        #image_resize = cv2.resize(image_np, (334, 565))
                        image_np_expanded = np.expand_dims(image_np, axis=0)

                        (detection_boxes, detection_scores, detection_classes, detection_num_detections) = sess.run(
                            [boxes, scores, classes, num_detections],
                            feed_dict={image_tensor: image_np_expanded}
                        )
                        if args.draw_boxes:
                            _draw_boxes(image_np, detection_boxes, detection_classes, detection_scores)
                        add_fps(image_np, start, time.clock())
                        show(image_np)
                    else:
                        try:
                            (images, detection_boxes, detection_scores, detection_classes, detection_num_detections) = sess.run(
                                [image_tensor, boxes, scores, classes, num_detections])
                            for i in range(0, images.shape[0]):
                                image_np = images[i]
                                if args.draw_boxes:
                                    _draw_boxes(image_np, detection_boxes[i], detection_classes[i], detection_scores[i])
                                show(image_np)
                            end = time.clock()
                            print("average_batch_fps: ", images.shape[0] * 1.0 / (end - start))
                        except tf.errors.OutOfRangeError:
                            break
