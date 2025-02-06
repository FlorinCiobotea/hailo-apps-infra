import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import os
import argparse
import multiprocessing
import numpy as np
import setproctitle
import cv2
import time
import hailo
from hailo_apps_infra.hailo_rpi_common import (
    detect_hailo_arch,
)
from hailo_apps_infra.gstreamer_helper_pipelines import(
    SOURCE_PIPELINE,
    TILE_CROPPER_PIPELINE,
    TILE_AGGREGATOR_PIPELINE,
    INFERENCE_PIPELINE,
    USER_CALLBACK_PIPELINE,
    DISPLAY_PIPELINE,
)
from hailo_apps_infra.gstreamer_app import (
    GStreamerApp,
    app_callback_class,
    dummy_callback
)



# -----------------------------------------------------------------------------------------------
# User Gstreamer Application
# -----------------------------------------------------------------------------------------------

# This class inherits from the hailo_rpi_common.GStreamerApp class
class GStreamerDetectionApp(GStreamerApp):
    def __init__(self, app_callback, user_data):
        # Call the parent class constructor
        super().__init__(user_data)
        # Additional initialization code can be added here
        # Set Hailo parameters these parameters should be set based on the model used
        self.batch_size = 1
        nms_score_threshold = 0.3
        nms_iou_threshold = 0.45

        # Determine the architecture if not specified
        detected_arch = detect_hailo_arch()
        if detected_arch is None:
            raise ValueError("Could not auto-detect Hailo architecture. Please specify --arch manually.")
        self.arch = detected_arch
        print(f"Auto-detected Hailo architecture: {self.arch}")
        
        self.hef_path = "../resources/yolov5m6_6.1.hef"
        
        # Set the post-processing shared object file
        self.post_process_so = os.path.join(self.current_path, '../resources/libyolo_hailortpp_postprocess.so')
        self.post_function_name="filter_letterbox"
        self.labels_json = None

        self.app_callback = app_callback

        self.thresholds_str = (
            f"nms-score-threshold={nms_score_threshold} "
            f"nms-iou-threshold={nms_iou_threshold} "
            f"output-format-type=HAILO_FORMAT_TYPE_FLOAT32"
        )

    
        self.show_fps=False
        # Set the process title
        setproctitle.setproctitle("Hailo Detection App")


    def get_pipeline_string(self,disable_inference):
        source_pipeline = SOURCE_PIPELINE(self.video_source, self.video_width, self.video_height)
        cropper_name="cropper"
        tile_cropper_pipeline=TILE_CROPPER_PIPELINE(name=cropper_name)
        agg_name="agg"
        tile_aggregator_pipeline=TILE_AGGREGATOR_PIPELINE(name=agg_name,cropper_name=cropper_name)
        detection_pipeline = INFERENCE_PIPELINE(
            hef_path=self.hef_path,
            post_process_so=self.post_process_so,
            post_function_name=self.post_function_name,
            batch_size=self.batch_size,
            config_json=self.labels_json,
            additional_params=self.thresholds_str)
        user_callback_pipeline = USER_CALLBACK_PIPELINE()
        display_pipeline = DISPLAY_PIPELINE(video_sink=self.video_sink, sync=self.sync, show_fps=self.show_fps)

        pipeline_string = (
            f'{source_pipeline} '
            f'{tile_cropper_pipeline} '
            f'{tile_aggregator_pipeline} ! '
            f'{detection_pipeline} ! {agg_name}. {agg_name}. ! '
            f'{user_callback_pipeline} ! '
            f'{display_pipeline}'
        ) if not disable_inference else (
            f'{source_pipeline} '
            f'{display_pipeline}'
        )
        
        print(pipeline_string)
        return pipeline_string

if __name__ == "__main__":
    # Create an instance of the user app callback class
    user_data = app_callback_class()
    app_callback = dummy_callback
    app = GStreamerDetectionApp(app_callback, user_data)
    app.run()
