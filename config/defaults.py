import os.path as osp

DEFAULT_NUM_INIT_ITERS = 70
DEFAULT_NUM_JOINT_ITERS = 500

NUM_THREADS = 8

# "kaolin" (https://github.com/NVIDIAGameWorks/kaolin/) or "nmr" (https://github.com/daniilidis-group/neural_renderer/)
RENDERER = "kaolin"
USE_OPTUNA = False
# when using initial estimation, keep DEFAULT_USE_ADAPTIVE_ITERS = False for faster estimation
DEFAULT_USE_ADAPTIVE_ITERS = True
DEFAULT_USE_ADAPTIVE_ITERS_INIT = False
ADAPTIVE_ITERS_CONVERGE_THRESH = 2.
ADAPTIVE_ITERS_STOPPING_TOLERANCE = 100
ADAPTIVE_ITERS_PRUNE_CANDIDATES = False
ADAPTIVE_ITERS_PRUNE_RATE = 0.1

PROJECT_ROOT_DIR = osp.abspath(osp.dirname(osp.dirname(__file__)))
DATASET_ROOT_DIR = osp.join(PROJECT_ROOT_DIR, "data", "original")

BEHAVE_DATASET_DIR = osp.join(DATASET_ROOT_DIR, "BEHAVE")
INTERCAP_DATASET_DIR = osp.join(DATASET_ROOT_DIR, "InterCap")

DEFAULT_OUTPUT_DIR = osp.join(PROJECT_ROOT_DIR, "data", "outputs", "{dataset}")
DEFAULT_CROP_INFO_DIR = osp.join(PROJECT_ROOT_DIR, "data", "preprocessed", "{dataset}", "crop_info")
DEFAULT_PARE_SAVE_DIR = osp.join(PROJECT_ROOT_DIR, "data", "preprocessed", "{dataset}", "person")
DEFAULT_MASK_DIR =  osp.join(PROJECT_ROOT_DIR, "data", "preprocessed", "{dataset}", "instances")
DEFAULT_PERSON_DEPTH_DIR = osp.join(PROJECT_ROOT_DIR, "data", "preprocessed", "{dataset}", "cropped_depth")
DEFAULT_OBJECT_DEPTH_DIR = osp.join(PROJECT_ROOT_DIR, "data", "preprocessed", "{dataset}", "cropped_depth")

CONTACT_PATH = osp.join(PROJECT_ROOT_DIR, "labels", "behave_bodypart_contact_gt_hot_segmentation.pkl")  # GT contact
INTERCAP_KEYFRAME_PATH = osp.join(PROJECT_ROOT_DIR, "labels", "intercap_keyframes.json")  # GT contact

PERSON_LABEL_PATH = osp.join(PROJECT_ROOT_DIR, "labels", "smpl_seg_hot_lang.json")

DEFAULT_PARE_CONFIG_PATH = osp.join(PROJECT_ROOT_DIR, "external", "pare", "data", "pare", "checkpoints", "pare_w_3dpw_config.yaml")
DEFAULT_PARE_CHECKPOINT_PATH = osp.join(PROJECT_ROOT_DIR, "external", "pare", "data", "pare", "checkpoints", "pare_w_3dpw_checkpoint.ckpt")

BEHAVE_GT_ORIGIN_PATH = osp.join(PROJECT_ROOT_DIR, "models", "gt_origin", "behave", "{cls}.ply")
INTERCAP_GT_ORIGIN_PATH = osp.join(PROJECT_ROOT_DIR, "models", "gt_origin", "intercap", "{cls}.ply")

PARE_DIR = osp.join(PROJECT_ROOT_DIR, "external", "pare")

# Configurations for PointRend
POINTREND_CONFIG = osp.join(
    PROJECT_ROOT_DIR, "external", "pointrend", "configs", "InstanceSegmentation", "pointrend_rcnn_R_50_FPN_3x_coco.yaml"
)
POINTREND_MODEL_WEIGHTS =  osp.join(PROJECT_ROOT_DIR, "external", "pointrend", "model_final_3c3198.pkl")

DEFAULT_JOINT_LR = 2e-3

DEFAULT_IMAGE_SIZE = 640
FOCAL_LENGTH = 1.0
REND_SIZE = 256  # Size of target masks for silhouette loss.
BBOX_EXPANSION_FACTOR = 0.3  # Amount to pad the target masks for silhouette loss.
SMPL_FACES_PATH = osp.join(PROJECT_ROOT_DIR, "external", "smpl", "smpl_faces.npy")

MAX_BBOX_CENTER_MERGING_DIST = 200.0  # FALLBACK strategy, use depth-based selection when possible

PERSON_DEPTH_MERGING_THRESHOLD = 1.5
OBJECT_DEPTH_MERGING_THRESHOLD = 1.5

# Empirical intrinsic scales learned by our method. To convert from scale to size in
# meters, multiply by 2 (i.e. scale of 1 corresponds to size of 2 meters).
# units are 2 meters

MESH_DIR = "models/default/"
MINIMAL_MESH_DIR = "models/minimal/"

INSTANCES_MAX_IMG_WIDTH = 1280
INSTANCES_MAX_IMG_HEIGHT = 720

MESH_MAP = {  # Class name -> list of paths to objs.
    "bicycle": [osp.join(MESH_DIR, "bicycle_01.obj")],
    "laptop": [osp.join(MESH_DIR, "laptop.obj")],
    "chair": [osp.join(MESH_DIR, "chairblack.obj")],
    "handbag": [osp.join(MESH_DIR, "handbag.obj")],
    "scissors": [osp.join(MESH_DIR, "scissors.obj")],
    "dining table": [osp.join(MESH_DIR, "table.obj")],
    "bed": [osp.join(MESH_DIR, "bed.obj")],
    "bowl": [osp.join(MESH_DIR, "bowl.obj")],
    "table": [osp.join(MESH_DIR, "table.obj")],
    "sports ball" : [osp.join(MESH_DIR, "ball.obj")],
    "umbrella": [osp.join(MESH_DIR, "umbrella.obj")],
    "skateboard" : [osp.join(MESH_DIR, "skateboard.obj")],
    
    # BEHAVE
    # Minimal meshes (for initial estimation):

    "chairblack": [osp.join(MINIMAL_MESH_DIR, "chairblack_minimal.obj")],
    "chairwood": [osp.join(MINIMAL_MESH_DIR, "chairwood_minimal.obj")],
    "basketball": [osp.join(MINIMAL_MESH_DIR, "basketball_minimal.obj")],
    "suitcase": [osp.join(MINIMAL_MESH_DIR, "suitcase_minimal_unflip.obj")],
    "boxtiny": [osp.join(MINIMAL_MESH_DIR, "boxtiny_minimal.obj")],
    "boxsmall": [osp.join(MINIMAL_MESH_DIR, "boxsmall_minimal.obj")],
    "boxmedium": [osp.join(MINIMAL_MESH_DIR, "boxmedium_minimal.obj")],
    "boxlarge": [osp.join(MINIMAL_MESH_DIR, "boxlarge_minimal.obj")],
    "boxlong": [osp.join(MINIMAL_MESH_DIR, "boxlong_minimal.obj")],
    "monitor": [osp.join(MINIMAL_MESH_DIR, "monitor_minimal.obj")],
    "keyboard": [osp.join(MINIMAL_MESH_DIR, "keyboard_minimal.obj")],
    "plasticcontainer": [osp.join(MINIMAL_MESH_DIR, "plasticcontainer_minimal.obj")],
    "backpack": [osp.join(MINIMAL_MESH_DIR, "backpack_minimal.obj")],
    "yogamat": [osp.join(MINIMAL_MESH_DIR, "yogamat_minimal.obj")],
    "yogaball": [osp.join(MINIMAL_MESH_DIR, "yogaball_minimal_unflip.obj")],
    "toolbox": [osp.join(MINIMAL_MESH_DIR, "toolbox_minimal.obj")],
    "trashbin": [osp.join(MINIMAL_MESH_DIR, "trashbin_minimal.obj")],
    "tablesmall": [osp.join(MINIMAL_MESH_DIR, "tablesmall_minimal.obj")],
    "tablesquare": [osp.join(MINIMAL_MESH_DIR, "tablesquare_minimal.obj")],
    "stool": [osp.join(MINIMAL_MESH_DIR, "stool_minimal.obj")],
    
    # InterCap
    "bottle_intercap": [osp.join(MESH_DIR, "bottle_intercap_centered.obj")],
    "cup_intercap": [osp.join(MESH_DIR, "cup_intercap_centered.obj")],
    "chair_intercap": [osp.join(MESH_DIR, "chair_intercap_centered.obj")],
    "stool_intercap": [osp.join(MESH_DIR, "stool_intercap_centered.obj")],
    "firstaidkit_intercap": [osp.join(MESH_DIR, "firstaidkit_intercap_centered.obj")],
    "skateboard_intercap": [osp.join(MESH_DIR, "skateboard_intercap_centered.obj")],
    "umbrella_intercap": [osp.join(MESH_DIR, "umbrella_intercap_centered.obj")],
    "tennisracket_intercap": [osp.join(MESH_DIR, "tennisracket_intercap_centered.obj")],
    "suitcase_intercap": [osp.join(MESH_DIR, "suitcase_intercap_centered.obj")],
    "soccerball_intercap": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],


    # custom images
    "custom_type_on_laptop_013893.png": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_bench_0001.jpg": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_bicycle_0001.jpg": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_bicycle_0002.jpg": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_bicycle_0003.jpg": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_bicycle_0004.jpg": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_bicycle_0005.jpg": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_bicycle_0006.jpg": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_bicycle_0007.jpg": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_bicycle_0008.jpg": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_bicycle_0009.jpg": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_bowl_0001.jpg": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_bowl_0002.jpg": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_bowl_0003.jpg": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_cap_0001.jpg": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_cap_0002.jpg": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_cap_0003.jpg": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_cap_0004.jpg": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_cap_0005.jpg": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_cap_0006.jpg": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_cap_0007.jpg": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_cap_0008.jpg": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_cap_0009.jpg": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_cart_0001.jpg": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_cart_0002.jpg": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_cart_0003.jpg": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_cart_0004.jpg": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_cart_0005.jpg": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_cut_with_knife_003370.png": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_cut_with_knife_007146.png": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_cut_with_knife_007314.png": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_cut_with_knife_008552.png": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_cut_with_knife_009024.png": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_globe_0001.jpg": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_globe_0002.jpg": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_globe_0003.jpg": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_globe_0004.jpg": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_globe_0005.jpg": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_guitar_0001.jpg": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_guitar_0002.jpg": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_guitar_0003.jpg": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_guitar_0004.jpg": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_guitar_0005.jpg": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_guitar_0006.jpg": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_guitar_0007.jpg": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_helmet_0001.jpg": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_helmet_0002.jpg": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_helmet_0003.jpg": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_helmet_0004.jpg": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_helmet_0005.jpg": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_helmet_0006.jpg": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_helmet_0007.jpg": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_helmet_0008.jpg": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_helmet_0009.jpg": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_keyboard_0001.jpg": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_keyboard_0002.jpg": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_keyboard_0003.jpg": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_keyboard_0004.jpg": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_keyboard_0005.jpg": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_laptop_0001.jpg": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_laptop_0002.jpg": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_laptop_0003.jpg": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_laptop_0004.jpg": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_laptop_0005.jpg": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_laptop_0006.jpg": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_laptop_0007.jpg": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_lie_on_bed_000236.png": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_lie_on_bench_004927.png": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_lie_on_bench_006191.png": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_lie_on_bench_007708.png": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_lie_on_couch_006926.png": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_lie_on_couch_007356.png": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_microphone_0001.jpg": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_microphone_0002.jpg": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_microphone_0003.jpg": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_microphone_0004.jpg": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_microphone_0006.jpg": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_microphone_0007.jpg": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_microphone_0008.jpg": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_microphone_0009.jpg": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_microphone_0010.jpg": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_open_book_002530.png": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_pot_0001.jpg": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_pot_0002.jpg": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_pot_0003.jpg": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_pot_0004.jpg": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_pot_0005.jpg": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_pot_0006.jpg": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_pot_0007.jpg": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_sit_on_bench_001018.png": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_sit_on_bench_002214.png": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_sit_on_bench_002624.png": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_sit_on_bench_003306.png": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_sit_on_bench_004320.png": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_sit_on_bench_005115.png": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_sit_on_bench_013921.png": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_sit_on_chair_001167.png": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_sit_on_chair_001980.png": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_sit_on_chair_003064.png": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_sit_on_chair_003719.png": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_sit_on_chair_004116.png": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_sit_on_chair_007662.png": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_sit_on_chair_007986.png": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_sit_on_chair_011002.png": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_sit_on_chair_011685.png": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_sit_on_chair_012784.png": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_sit_on_chair_013642.png": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_sit_on_couch_001731.png": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_sit_on_couch_002577.png": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_sit_on_couch_002629.png": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_sit_on_couch_004640.png": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_sit_on_couch_004922.png": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_sit_on_couch_008322.png": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_sunglasses_0001.jpg": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_sunglasses_0002.jpg": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_sunglasses_0003.jpg": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_sunglasses_0004.jpg": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_sunglasses_0005.jpg": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_sunglasses_0006.jpg": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_sunglasses_0007.jpg": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_sunglasses_0008.jpg": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_take_photo_camera_001965.png": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_take_photo_camera_004511.png": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_take_photo_camera_006149.png": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_take_photo_camera_006733.png": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_take_photo_camera_011088.png": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_take_photo_camera_011158.png": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_take_photo_camera_013438.png": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_take_photo_phone_000566.png": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_take_photo_phone_003856.png": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_take_photo_phone_011473.png": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_text_on_cell_phone_002730.png": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_text_on_cell_phone_012717.png": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_type_on_laptop_002748.png": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_type_on_laptop_003987.png": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_type_on_laptop_005411.png": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_type_on_laptop_007534.png": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_type_on_laptop_007823.png": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_type_on_laptop_008680.png": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_type_on_laptop_009269.png": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_type_on_laptop_010664.png": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")],
    "custom_type_on_laptop_011770.png": [osp.join(MESH_DIR, "soccerball_intercap_centered.obj")]
}

COCO_REMAPPINGS = {  # from our categories to COCO categories
    "chairblack": "chair",
    "chairwood": "chair"
}

# TODO: change to different interactions!
# not used anymore
INTERACTION_MAPPING = {
    "backpack": ["back"],
    "bowl": ["lhand", "rhand"],
    "bat": ["lpalm", "rpalm"],
    "bench": ["back", "butt"],
    "bicycle": ["lhand", "rhand", "butt"],
    "laptop": ["lhand", "rhand"],
    "motorcycle": ["lhand", "rhand", "butt"],
    "skateboard": ["lfoot", "rfoot", "lhand", "rhand"],
    "surfboard": ["lfoot", "rfoot", "lhand", "rhand"],
    "tennis": ["lpalm", "rpalm"],
    "chair": ["back", "butt"],
    "table": ["lhand", "rhand"],
    "dining table": ["lhand", "rhand"],
    "suitcase": ["lhand", "rhand"],
    "keyboard": ["lhand", "rhand"],
    "bed": ["back"],
    "handbag": ["lhand", "rhand"],
    "scissors": ["lhand", "rhand"],
    "sports ball": ["lhand", "rhand"],
    "umbrella": ["lhand", "rhand"],
}

# many of these values were set in an arbitrary manner
# related to size of object in image
BBOX_EXPANSION = {
    "bat": 0.5,
    "bench": 0.3,
    "bicycle": 0.0,
    "bottle": 0.3,
    "chair": 0.3,
    "couch": 0.3,
    "cup": 0.3,
    "horse": 0.0,
    "laptop": 0.2,
    "motorcycle": 0.0,
    "skateboard": 0.8,
    "surfboard": 0,
    "tennis": 0.4,
    "wineglass": 0.3,
    "table": 0.3,
    "dining table": 0.3,
    "bed": 0.3,
    "bowl": 0.3,
    "scissors": 0.3,
    "handbag": 0.3,
    "sports ball": 0.5,
    "umbrella": 0.5,

    "chairwood": 0.3,
    "chairblack": 0.3,
    "backpack": 0.3,
    "suitcase": 0.3,
    "basketball": 0.5,
    "boxlarge": 0.3,
    "boxlong": 0.5,
    "boxmedium": 0.3,
    "boxsmall": 0.3,
    "boxtiny": 0.7,
    "keyboard": 0.7,
    "monitor": 0.3,
    "plasticcontainer": 0.4,
    "stool": 0.3,
    "suitcase": 0.5,
    "tablesmall": 0.3,
    "tablesquare": 0.3,
    "toolbox": 0.4,
    "trashbin": 0.4,
    "yogaball": 0.3,
    "yogamat": 0.5,

    # InterCap
    "suitcase_intercap": 0.3,
    "skateboard_intercap": 0.8,
    "soccerball_intercap": 0.5,
    "umbrella_intercap": 0.5,
    "tennisracket_intercap": 0.4,
    "firstaidkit_intercap": 0.3,
    "chair_intercap": 0.3,
    "bottle_intercap": 0.4,
    "cup_intercap": 0.4,
    "stool_intercap": 0.3
}

# many of these values were set in an arbitrary manner
BBOX_EXPANSION_PARTS = {
    "backpack":0.3,
    "bat": 2.5,
    "bowl": 2.5,
    "bench": 0.5,
    "bicycle": 0.7,
    "bottle": 0.3,
    "chair": 0.3,
    "couch": 0.3,
    "cup": 0.3,
    "scissors": 0.3,
    "horse": 0.3,
    "laptop": 0.0,
    "motorcycle": 0.7,
    "skateboard": 0.5,
    "surfboard": 0.2,
    "tennis": 2,
    "wineglass": 0.3,
    "table": 0.3,
    "table": 0.3,
    "table": 0.3,
    "dining table": 0.3,
    "suitcase": 0.3,
    "bed": 0.3,
    "handbag":0.3,
    "keyboard": 1,
    "sports ball": 0.5,
    "umbrella": 0.5,
    
    "chairwood": 0.3,
    "chairblack": 0.3,
    "backpack": 0.3,
    "suitcase": 0.3,
    "basketball": 0.5,
    "boxlarge": 0.3,
    "boxlong": 0.5,
    "boxmedium": 0.3,
    "boxsmall": 0.3,
    "boxtiny": 0.7,
    "keyboard": 0.7,
    "monitor": 0.3,
    "plasticcontainer": 0.4,
    "stool": 0.3,
    "suitcase": 0.5,
    "tablesmall": 0.3,
    "tablesquare": 0.3,
    "toolbox": 0.4,
    "trashbin": 0.4,
    "yogaball": 0.3,
    "yogamat": 0.5,

    # InterCap
    "suitcase_intercap": 0.3,
    "skateboard_intercap": 0.8,
    "soccerball_intercap": 0.5,
    "umbrella_intercap": 0.5,
    "tennisracket_intercap": 0.4,
    "firstaidkit_intercap": 0.3,
    "chair_intercap": 0.3,
    "bottle_intercap": 0.4,
    "cup_intercap": 0.4,
    "stool_intercap": 0.3
}

# many of these values were set in an arbitrary manner
DEFAULT_INTERACTION_THRESHOLD = 3
INTERACTION_THRESHOLD = {
    "bat": 5,
    "bicycle": 2,
    "laptop": 2.5,
    "motorcycle": 5,
    "surfboard": 5,
    "tennis": 5,
    "bench": DEFAULT_INTERACTION_THRESHOLD,
    "skateboard": DEFAULT_INTERACTION_THRESHOLD,
    "chair": DEFAULT_INTERACTION_THRESHOLD,
    "chairblack": DEFAULT_INTERACTION_THRESHOLD,
    "chairwood": DEFAULT_INTERACTION_THRESHOLD,
    "table": DEFAULT_INTERACTION_THRESHOLD,
    "dining table": DEFAULT_INTERACTION_THRESHOLD,
    "backpack": DEFAULT_INTERACTION_THRESHOLD,
    "suitcase": DEFAULT_INTERACTION_THRESHOLD,
    "bed": DEFAULT_INTERACTION_THRESHOLD,
    "scissors": DEFAULT_INTERACTION_THRESHOLD,
    "bowl": DEFAULT_INTERACTION_THRESHOLD,
    "handbag": DEFAULT_INTERACTION_THRESHOLD,
    "keyboard": DEFAULT_INTERACTION_THRESHOLD,
    "sports ball": DEFAULT_INTERACTION_THRESHOLD,
    "umbrella": DEFAULT_INTERACTION_THRESHOLD
}

DEFAULT_LOSS_WEIGHTS = {  # Loss weights.
    "default": {
        "lw_inter": 0.0,  # 30,
        "lw_depth": 0,  # 0.1,
        "lw_inter_part": 0, ##50,
        "lw_sil": 0.0, ##50.0,
        "lw_collision": 0.0,##10.0,
        "lw_scale": 1000.0,##1,  # 1000,
        "lw_scale_person": 1000.0,##1,  # 100,
        "lw_ground_contact": 0.0,##10
        "lw_human_obj_contact": 1e5,#5.0,
        "lw_penetration": 1e3,
        "lw_reldepth": 0.0,
    },
    "bat": {
        "lw_inter": 30,
        "lw_depth": 0.01,
        "lw_inter_part": 100,
        "lw_sil": 20.0,
        "lw_collision": 1.0,
        "lw_scale": 10000,
        "lw_scale_person": 1000,
    },
    "bench": {
        "lw_inter": 30,
        "lw_depth": 0.1,
        "lw_inter_part": 50,
        "lw_sil": 50.0,
        "lw_collision": 10.0,
        "lw_scale": 1000,
        "lw_scale_person": 100,
    },
    "bicycle": {
        "lw_inter": 20,
        "lw_depth": 1,
        "lw_inter_part": 50,
        "lw_sil": 10.0,
        "lw_collision": 2.0,
        "lw_scale": 100,
        "lw_scale_person": 100,
    },
    "laptop": {
        "lw_inter": 20,
        "lw_depth": 0.01,
        "lw_inter_part": 20,
        "lw_sil": 10.0,
        "lw_collision": 10,
        "lw_scale": 1e3,
        "lw_scale_person": 1e3,
    },
    "motorcycle": {
        "lw_inter": 0,
        "lw_depth": 1.0,
        "lw_inter_part": 100,
        "lw_sil": 20.0,
        "lw_collision": 2.0,
        "lw_scale": 100,
        "lw_scale_person": 100,
    },
    "surfboard": {
        "lw_inter": 50,
        "lw_depth": 10,
        "lw_inter_part": 100,
        "lw_sil": 10.0,
        "lw_collision": 20,
        "lw_scale": 1e3,
        "lw_scale_person": 1e3,
    },
    "tennis": {
        "lw_inter": 30,
        "lw_depth": 0.01,
        "lw_inter_part": 500,
        "lw_sil": 10.0,
        "lw_collision": 10,
        "lw_scale": 1e4,
        "lw_scale_person": 100,
    },
    "chair": {
        "lw_inter": 0.0, 
        "lw_depth": 0,  
        "lw_inter_part": 0,
        "lw_sil": 0.0, 
        "lw_collision": 0.0,
        "lw_scale": 1000.0,
        "lw_scale_person": 1000.0,
        "lw_ground_contact": 0.0,
        "lw_human_obj_contact": 1e5,
        "lw_penetration": 1e3,
    },
    "chairblack": {
        "lw_inter": 0.0,
        "lw_depth": 0,
        "lw_inter_part": 0,
        "lw_sil": 0.0,
        "lw_collision": 0.0,
        "lw_scale": 1000.0,
        "lw_scale_person": 1000.0,
        "lw_ground_contact": 0.0,
        "lw_human_obj_contact": 1e5,
        "lw_penetration": 1e3,
    },
    "boxsmall": {
        "lw_inter": 0.0,
        "lw_depth": 0,
        "lw_inter_part": 0,
        "lw_sil": 0.0,
        "lw_collision": 0.0,
        "lw_scale": 1000.0,
        "lw_scale_person": 1000.0,
        "lw_ground_contact": 0.0,
        "lw_human_obj_contact": 1e5,
        "lw_penetration": 1e3,
    },
    "chairwood": {
        "lw_inter": 0.0,
        "lw_depth": 0,
        "lw_inter_part": 0,
        "lw_sil": 0.0,
        "lw_collision": 0.0,
        "lw_scale": 1000.0,
        "lw_scale_person": 1000.0,
        "lw_ground_contact": 0.0,
        "lw_human_obj_contact": 1e5,
        "lw_penetration": 1e3,
    },
    "basketball": {
        "lw_inter": 0.0,
        "lw_depth": 0,
        "lw_inter_part": 0,
        "lw_sil": 0.0,
        "lw_collision": 0.0,
        "lw_scale": 1000.0,
        "lw_scale_person": 1000.0,
        "lw_ground_contact": 0.0,
        "lw_human_obj_contact": 1e5,
        "lw_penetration": 1e3,
    },
    "keyboard": {
        "lw_inter": 0.0,
        "lw_depth": 0,
        "lw_inter_part": 0,
        "lw_sil": 0.0,
        "lw_collision": 0.0,
        "lw_scale": 1000.0,
        "lw_scale_person": 1000.0,
        "lw_ground_contact": 0.0,
        "lw_human_obj_contact": 1e5,
        "lw_penetration": 1e3,
    },
    "table": {
        "lw_inter": 0.0,
        "lw_depth": 0,
        "lw_inter_part": 0,
        "lw_sil": 0.0,
        "lw_collision": 0.0,
        "lw_scale": 1e3,
        "lw_scale_person": 1e3,
        "lw_ground_contact": 0.0,
        "lw_human_obj_contact": 1e5,
        "lw_penetration": 1e3,
    },
    "suitcase": {
        "lw_inter": 0.0,
        "lw_depth": 0,
        "lw_inter_part": 0,
        "lw_sil": 0.0,
        "lw_collision": 0.0,
        "lw_scale": 1000.0,
        "lw_scale_person": 1000.0,
        "lw_ground_contact": 0.0,
        "lw_human_obj_contact": 1e5,
        "lw_penetration": 1e3,
    },
    "backpack": {
        "lw_inter": 0,
        "lw_depth": 0,
        "lw_inter_part": 0,
        "lw_sil": 0.0,
        "lw_collision": 0.0,
        "lw_scale": 1000.0,
        "lw_scale_person": 1000.0,
        "lw_ground_contact": 0.0,
        "lw_human_obj_contact": 1e5,
        "lw_penetration": 1e3,
    },
    "sports ball": {
        "lw_inter": 0,
        "lw_depth": 0,
        "lw_inter_part": 0,
        "lw_sil": 0.0,
        "lw_collision": 0.0,
        "lw_scale": 1000.0,
        "lw_scale_person": 1000.0,
        "lw_ground_contact": 0.0,
        "lw_human_obj_contact": 1e5,
        "lw_penetration": 1e3,
    },
    "umbrella": {
        "lw_inter": 0,
        "lw_depth": 0,
        "lw_inter_part": 0,
        "lw_sil": 0.0,
        "lw_collision": 0.0,
        "lw_scale": 1000.0,
        "lw_scale_person": 1000.0,
        "lw_ground_contact": 0.0,
        "lw_human_obj_contact": 1e5,
        "lw_penetration": 1e3,
    },
    "skateboard": {
        "lw_inter": 0,
        "lw_depth": 0,
        "lw_inter_part": 0,
        "lw_sil": 0.0,
        "lw_collision": 0.0,
        "lw_scale": 1000.0,
        "lw_scale_person": 1000.0,
        "lw_ground_contact": 0.0,
        "lw_human_obj_contact": 1e5,
        "lw_penetration": 1e3,
    },
}

# format:
# path_component: (obj_name, word_name, dataset)
# path_component must be unique for detection
PATH_CLASS_NAME_DICT = {
    # change entries here such that the categories get recognized by the category selection
    # and mesh selection in preprocess.py and reconstruct.py

    # BEHAVE
    "chairwood": ("chairwood", "wooden chair", "behave"),
    "chairblack": ("chairblack", "black chair", "behave"),
    "backpack": ("backpack", "backpack", "behave"),
    "suitcase": ("suitcase", "suitcase", "behave"),
    "basketball": ("basketball", "basketball", "behave"),
    "boxlarge": ("boxlarge", "box", "behave"),
    "boxlong": ("boxlong", "box", "behave"),
    "boxmedium": ("boxmedium", "box", "behave"),
    "boxsmall": ("boxsmall", "box", "behave"),
    "boxtiny": ("boxtiny", "box", "behave"),
    "keyboard": ("keyboard", "keyboard", "behave"),
    "monitor": ("monitor", "monitor", "behave"),
    "plasticcontainer": ("plasticcontainer", "plastic container", "behave"),
    "stool": ("stool", "stool", "behave"),
    "suitcase": ("suitcase", "suitcase", "behave"),
    "tablesmall": ("tablesmall", "table", "behave"),
    "tablesquare": ("tablesquare", "table", "behave"),
    "toolbox": ("toolbox", "toolbox", "behave"),
    "trashbin": ("trashbin", "trash bin", "behave"),
    "yogaball": ("yogaball", "yoga ball", "behave"),
    "yogamat": ("yogamat", "yoga mat", "behave"),

    # InterCap
    "/01/Seg_": ("suitcase_intercap", "suitcase", "intercap"),
    "/02/Seg_": ("skateboard_intercap", "skateboard", "intercap"),
    "/03/Seg_": ("soccerball_intercap", "soccer ball", "intercap"),
    "/04/Seg_": ("umbrella_intercap", "umbrella", "intercap"),
    "/05/Seg_": ("tennisracket_intercap", "tennis racket", "intercap"),
    "/06/Seg_": ("firstaidkit_intercap", "first aid kit", "intercap"),
    "/07/Seg_": ("chair_intercap", "chair", "intercap"),
    "/08/Seg_": ("bottle_intercap", "bottle", "intercap"),
    "/09/Seg_": ("cup_intercap", "cup", "intercap"),
    "/10/Seg_": ("stool_intercap", "stool", "intercap"),
}
