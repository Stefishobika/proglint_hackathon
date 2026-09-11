import cv2
import math
import time
from collections import deque
from ultralytics import YOLO

VIDEO_PATH = r"C:\Users\stefi\Downloads\pose.mp4"

OBJECT_MODEL = "yolo11n.pt"
POSE_MODEL = "yolo11n-pose.pt"

OUTPUT_VIDEO = r"C:\Users\stefi\Downloads\bag_status_accurate.avi"

CONF = 0.25
POSE_CONF = 0.30
POSE_EVERY_N_FRAMES = 1

HAND_DISTANCE_RATIO = 0.30

LIFT_HEIGHT_RATIO = 0.08

MOVE_FROM_REGION_RATIO = 0.12


# ------------------------------------------------------------
# Region
# ------------------------------------------------------------

# The original bag box is expanded by this factor.
# This is the "safe" area around the original position.
REGION_PADDING_RATIO = 1.5


# ------------------------------------------------------------
# Stability / debounce
# ------------------------------------------------------------

# Small debounce instead of 5-10 frames.
# This makes commands appear much closer to the actual action.
PICKING_CONFIRM_FRAMES = 2
PICKED_CONFIRM_FRAMES = 2
MOVED_CONFIRM_FRAMES = 2
KEPT_CONFIRM_FRAMES = 4


# Bag movement smoothing.
# Keep small so status does not become delayed.
MOVEMENT_HISTORY_SIZE = 3


# A bag is considered stationary when its normalized
# movement is below this value.
STATIONARY_RATIO = 0.012


# Event message display time.
EVENT_DISPLAY_SECONDS = 1.5


# ============================================================
# COCO CLASS NAMES
# ============================================================

PERSON_CLASS = "person"

BAG_CLASS_NAMES = {
    "backpack",
    "handbag",
    "suitcase"
}


# ============================================================
# STATUS TEXT
# ============================================================

STATUS_IN_REGION = "BAG IS IN ITS REGION"
STATUS_PICKING = "BAG IS BEING PICKED UP"
STATUS_PICKED = "BAG IS PICKED UP"
STATUS_MOVED = "BAG HAS BEEN MOVED FROM ITS REGION"
STATUS_KEPT = "BAG IS KEPT IN ITS REGION"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def center_of_box(box):
    x1, y1, x2, y2 = box

    return (
        (x1 + x2) / 2.0,
        (y1 + y2) / 2.0
    )


def distance(p1, p2):
    if p1 is None or p2 is None:
        return float("inf")

    return math.hypot(
        p1[0] - p2[0],
        p1[1] - p2[1]
    )


def box_area(box):
    x1, y1, x2, y2 = box

    return max(0.0, x2 - x1) * max(
        0.0,
        y2 - y1
    )


def calculate_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)

    intersection = iw * ih

    union = (
        box_area(box_a)
        + box_area(box_b)
        - intersection
    )

    if union <= 0:
        return 0.0

    return intersection / union


def point_inside_region(point, region):
    if point is None or region is None:
        return False

    x, y = point

    rx1, ry1, rx2, ry2 = region

    return (
        rx1 <= x <= rx2
        and
        ry1 <= y <= ry2
    )


def create_region(
    bag_box,
    frame_width,
    frame_height
):
    x1, y1, x2, y2 = bag_box

    bag_width = max(1.0, x2 - x1)
    bag_height = max(1.0, y2 - y1)

    pad_x = int(
        bag_width * REGION_PADDING_RATIO
    )

    pad_y = int(
        bag_height * REGION_PADDING_RATIO
    )

    return (
        max(0, int(x1 - pad_x)),
        max(0, int(y1 - pad_y)),
        min(
            frame_width - 1,
            int(x2 + pad_x)
        ),
        min(
            frame_height - 1,
            int(y2 + pad_y)
        )
    )


def get_wrist_points(
    pose_result,
    person_box
):
    """
    YOLO COCO pose keypoints:

    9  = left wrist
    10 = right wrist
    """

    if pose_result.keypoints is None:
        return None, None

    if (
        pose_result.boxes is None
        or
        len(pose_result.boxes) == 0
    ):
        return None, None

    pose_boxes = (
        pose_result
        .boxes
        .xyxy
        .cpu()
        .numpy()
    )

    pose_keypoints = (
        pose_result
        .keypoints
        .xy
        .cpu()
        .numpy()
    )

    pose_conf = None

    if pose_result.keypoints.conf is not None:
        pose_conf = (
            pose_result
            .keypoints
            .conf
            .cpu()
            .numpy()
        )

    best_iou = 0.0
    best_index = -1

    for i, pose_box in enumerate(
        pose_boxes
    ):

        iou = calculate_iou(
            person_box,
            pose_box
        )

        if iou > best_iou:
            best_iou = iou
            best_index = i

    if (
        best_index == -1
        or
        best_iou < 0.10
    ):
        return None, None

    left_wrist = (
        pose_keypoints[
            best_index
        ][9]
    )

    right_wrist = (
        pose_keypoints[
            best_index
        ][10]
    )

    if pose_conf is not None:

        if (
            pose_conf[
                best_index
            ][9]
            < POSE_CONF
        ):
            left_wrist = None

        if (
            pose_conf[
                best_index
            ][10]
            < POSE_CONF
        ):
            right_wrist = None

    return (
        left_wrist,
        right_wrist
    )


def put_text(
    frame,
    text,
    position,
    scale=0.7,
    thickness=2
):

    x, y = position

    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        thickness + 3,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA
    )


def draw_label(
    frame,
    text,
    x,
    y
):

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.50
    thickness = 2

    (tw, th), baseline = (
        cv2.getTextSize(
            text,
            font,
            scale,
            thickness
        )
    )

    y = max(
        th + 8,
        int(y)
    )

    cv2.rectangle(
        frame,
        (
            int(x),
            y - th - 8
        ),
        (
            int(x + tw + 10),
            y + baseline
        ),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        text,
        (
            int(x + 5),
            y
        ),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA
    )


# ============================================================
# LOAD MODELS
# ============================================================

print("--------------------------------")
print("LOADING MODELS")
print("--------------------------------")

object_model = YOLO(
    OBJECT_MODEL
)

pose_model = YOLO(
    POSE_MODEL
)

print("Object model loaded.")
print("Pose model loaded.")


# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(
    VIDEO_PATH
)

if not cap.isOpened():

    raise RuntimeError(
        f"ERROR: Could not open video:\n"
        f"{VIDEO_PATH}"
    )


fps = cap.get(
    cv2.CAP_PROP_FPS
)

if (
    fps <= 0
    or
    math.isnan(fps)
):
    fps = 30.0


width = int(
    cap.get(
        cv2.CAP_PROP_FRAME_WIDTH
    )
)

height = int(
    cap.get(
        cv2.CAP_PROP_FRAME_HEIGHT
    )
)

total_frames = int(
    cap.get(
        cv2.CAP_PROP_FRAME_COUNT
    )
)


print("--------------------------------")
print("INPUT VIDEO")
print("--------------------------------")
print(
    f"Resolution : "
    f"{width} x {height}"
)
print(
    f"FPS        : "
    f"{fps:.2f}"
)
print(
    f"Frames     : "
    f"{total_frames}"
)
print("--------------------------------")


# ============================================================
# OUTPUT VIDEO
# ============================================================

# MJPG is used because your OpenCV installation previously
# failed with H.264 and mp4v.

fourcc = cv2.VideoWriter_fourcc(
    *"MJPG"
)

out = cv2.VideoWriter(
    OUTPUT_VIDEO,
    fourcc,
    fps,
    (width, height)
)


if not out.isOpened():

    cap.release()

    raise RuntimeError(
        "ERROR: Could not create output video "
        "using MJPG."
    )


print(
    f"Output: {OUTPUT_VIDEO}"
)

print("--------------------------------")


# ============================================================
# STATE VARIABLES
# ============================================================

current_status = STATUS_IN_REGION

status_number = 1


# Original region of the bag
bag_region = None


# Original bag center
initial_bag_center = None


# Track ID of original bag
bag_track_id = None


# Track ID of main person
person_track_id = None


# Whether bag has been physically lifted
bag_is_picked = False


# Whether bag has already left its region
bag_has_left_region = False


# Counters
picking_counter = 0
picked_counter = 0
moved_counter = 0
kept_counter = 0


# Previous bag center
previous_bag_center = None


# Recent bag movement values
movement_history = deque(
    maxlen=MOVEMENT_HISTORY_SIZE
)


# Previous wrist positions
previous_left_wrist = None
previous_right_wrist = None


# Current wrist positions
left_wrist = None
right_wrist = None


# Event display
event_message = ""
event_message_until = 0


# Frame counter
frame_number = 0


# ============================================================
# FUNCTION TO CHANGE STATUS
# ============================================================

def change_status(
    new_status,
    new_number
):

    global current_status
    global status_number
    global event_message
    global event_message_until

    if current_status == new_status:
        return

    current_status = new_status
    status_number = new_number

    event_message = new_status

    event_message_until = (
        frame_number
        +
        int(
            fps
            *
            EVENT_DISPLAY_SECONDS
        )
    )

    print(
        f"[Frame {frame_number}] "
        f"{new_status}"
    )


# ============================================================
# MAIN VIDEO LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1


    # ========================================================
    # STEP 1
    # OBJECT DETECTION + BYTETRACK
    # ========================================================

    results = object_model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=CONF,
        verbose=False
    )


    person_data = []
    bag_data = []


    if (
        results
        and
        len(results) > 0
    ):

        result = results[0]


        if (
            result.boxes is not None
            and
            len(result.boxes) > 0
        ):

            boxes = (
                result.boxes
                .xyxy
                .cpu()
                .numpy()
            )

            classes = (
                result.boxes
                .cls
                .cpu()
                .numpy()
            )

            confidences = (
                result.boxes
                .conf
                .cpu()
                .numpy()
            )


            if result.boxes.id is not None:

                track_ids = (
                    result.boxes
                    .id
                    .cpu()
                    .numpy()
                    .astype(int)
                )

            else:

                track_ids = (
                    [-1] * len(boxes)
                )


            for (
                box,
                cls,
                confidence,
                track_id
            ) in zip(
                boxes,
                classes,
                confidences,
                track_ids
            ):

                class_name = (
                    object_model.names[
                        int(cls)
                    ]
                )


                # ============================================
                # PERSON
                # ============================================

                if (
                    class_name
                    ==
                    PERSON_CLASS
                ):

                    person_data.append({

                        "box": box,

                        "id": int(
                            track_id
                        ),

                        "confidence":
                            float(
                                confidence
                            )
                    })


                # ============================================
                # BAG
                # ============================================

                elif (
                    class_name
                    in
                    BAG_CLASS_NAMES
                ):

                    bag_data.append({

                        "box": box,

                        "id": int(
                            track_id
                        ),

                        "class":
                            class_name,

                        "confidence":
                            float(
                                confidence
                            )
                    })


    # ========================================================
    # STEP 2
    # SELECT PERSON
    # ========================================================

    selected_person = None


    if person_data:

        # If we already have a person ID,
        # continue following that person.
        if person_track_id is not None:

            same_person = [
                p
                for p in person_data
                if p["id"]
                ==
                person_track_id
            ]

            if same_person:

                selected_person = max(
                    same_person,
                    key=lambda p:
                    box_area(p["box"])
                )


        # Otherwise select the largest person
        if selected_person is None:

            selected_person = max(
                person_data,
                key=lambda p:
                box_area(p["box"])
            )

            person_track_id = (
                selected_person["id"]
            )


    # ========================================================
    # STEP 3
    # SELECT BAG
    # ========================================================

    selected_bag = None


    if bag_data:

        # Continue following the original bag ID.
        if bag_track_id is not None:

            same_bag = [
                b
                for b in bag_data
                if b["id"]
                ==
                bag_track_id
            ]

            if same_bag:

                selected_bag = max(
                    same_bag,
                    key=lambda b:
                    box_area(b["box"])
                )


        # If this is the first detection,
        # select the largest bag.
        if selected_bag is None:

            selected_bag = max(
                bag_data,
                key=lambda b:
                box_area(b["box"])
            )


            bag_track_id = (
                selected_bag["id"]
            )


    # ========================================================
    # STEP 4
    # CREATE ORIGINAL BAG REGION
    # ========================================================

    if (
        bag_region is None
        and
        selected_bag is not None
    ):

        bag_region = create_region(
            selected_bag["box"],
            width,
            height
        )

        initial_bag_center = (
            center_of_box(
                selected_bag["box"]
            )
        )

        previous_bag_center = (
            initial_bag_center
        )

        print(
            f"[Frame {frame_number}] "
            "Original bag region created."
        )


    # ========================================================
    # STEP 5
    # PERSON HEIGHT
    # ========================================================

    person_height = float(
        height
    )


    if selected_person is not None:

        px1, py1, px2, py2 = (
            selected_person["box"]
        )

        person_height = max(
            1.0,
            py2 - py1
        )


    # ========================================================
    # STEP 6
    # POSE
    # ========================================================

    # Pose is calculated every frame by default.
    # This is important because hand movement is fast.

    if (
        selected_person is not None
        and
        frame_number
        %
        POSE_EVERY_N_FRAMES
        ==
        0
    ):

        pose_results = pose_model(
            frame,
            conf=POSE_CONF,
            verbose=False
        )


        if pose_results:

            (
                new_left_wrist,
                new_right_wrist
            ) = get_wrist_points(
                pose_results[0],
                selected_person["box"]
            )


            if new_left_wrist is not None:

                left_wrist = (
                    float(
                        new_left_wrist[0]
                    ),
                    float(
                        new_left_wrist[1]
                    )
                )


            if new_right_wrist is not None:

                right_wrist = (
                    float(
                        new_right_wrist[0]
                    ),
                    float(
                        new_right_wrist[1]
                    )
                )


    # ========================================================
    # STEP 7
    # BAG POSITION + MOVEMENT
    # ========================================================

    current_bag_center = None


    if selected_bag is not None:

        current_bag_center = (
            center_of_box(
                selected_bag["box"]
            )
        )


        if previous_bag_center is not None:

            raw_movement = distance(
                previous_bag_center,
                current_bag_center
            )


            normalized_movement = (
                raw_movement
                /
                person_height
            )


            movement_history.append(
                normalized_movement
            )


        previous_bag_center = (
            current_bag_center
        )


    else:

        # Do not compare a reappearing bag
        # with an old position.
        previous_bag_center = None

        movement_history.clear()


    if movement_history:

        average_movement = (
            sum(movement_history)
            /
            len(movement_history)
        )

    else:

        average_movement = 0.0


    bag_moving = (
        average_movement
        >
        STATIONARY_RATIO
    )


    bag_stopped = not bag_moving


    # ========================================================
    # STEP 8
    # IS HAND NEAR BAG?
    # ========================================================

    hand_near_bag = False


    closest_hand = None


    if current_bag_center is not None:

        left_distance = distance(
            left_wrist,
            current_bag_center
        )

        right_distance = distance(
            right_wrist,
            current_bag_center
        )


        closest_distance = min(
            left_distance,
            right_distance
        )


        allowed_hand_distance = (
            HAND_DISTANCE_RATIO
            *
            person_height
        )


        if (
            closest_distance
            <=
            allowed_hand_distance
        ):

            hand_near_bag = True


            if (
                left_distance
                <=
                right_distance
            ):

                closest_hand = (
                    "LEFT",
                    left_wrist
                )

            else:

                closest_hand = (
                    "RIGHT",
                    right_wrist
                )


    # ========================================================
    # STEP 9
    # HAND APPROACH DIRECTION
    # ========================================================

    # We check whether the hand is moving toward the bag.
    # This helps distinguish "person is standing near bag"
    # from "person is actually reaching for bag".

    hand_approaching = False


    if closest_hand is not None:

        hand_name, current_hand = (
            closest_hand
        )


        if hand_name == "LEFT":

            previous_hand = (
                previous_left_wrist
            )

        else:

            previous_hand = (
                previous_right_wrist
            )


        if (
            previous_hand is not None
            and
            current_hand is not None
            and
            current_bag_center is not None
        ):

            old_distance = distance(
                previous_hand,
                current_bag_center
            )


            new_distance = distance(
                current_hand,
                current_bag_center
            )


            # Hand must be moving closer.
            if new_distance < old_distance:
                hand_approaching = True


    # Save wrist positions
    if left_wrist is not None:

        previous_left_wrist = (
            left_wrist
        )


    if right_wrist is not None:

        previous_right_wrist = (
            right_wrist
        )


    # ========================================================
    # STEP 10
    # BAG INSIDE ORIGINAL REGION?
    # ========================================================

    bag_inside_region = False


    if current_bag_center is not None:

        bag_inside_region = (
            point_inside_region(
                current_bag_center,
                bag_region
            )
        )


    # ========================================================
    # STEP 11
    # BAG LIFT HEIGHT
    # ========================================================

    bag_has_risen = False


    if (
        current_bag_center is not None
        and
        initial_bag_center is not None
    ):

        vertical_lift = (
            initial_bag_center[1]
            -
            current_bag_center[1]
        )


        lift_threshold = (
            LIFT_HEIGHT_RATIO
            *
            person_height
        )


        if (
            vertical_lift
            >=
            lift_threshold
        ):

            bag_has_risen = True


    # ========================================================
    # STEP 12
    # BAG DISTANCE FROM ORIGINAL POSITION
    # ========================================================

    bag_far_from_original = False


    if (
        current_bag_center is not None
        and
        initial_bag_center is not None
    ):

        distance_from_original = (
            distance(
                current_bag_center,
                initial_bag_center
            )
        )


        move_threshold = (
            MOVE_FROM_REGION_RATIO
            *
            person_height
        )


        if (
            distance_from_original
            >=
            move_threshold
        ):

            bag_far_from_original = True


    # ========================================================
    # STATE MACHINE
    # ========================================================

    # --------------------------------------------------------
    # STATUS 1:
    # BAG IS IN ITS REGION
    # --------------------------------------------------------

    if (
        not bag_is_picked
        and
        not bag_has_left_region
    ):

        # Person has reached the bag.
        # We don't wait for 5 frames anymore.
        if (
            bag_inside_region
            and
            hand_near_bag
            and
            (
                hand_approaching
                or
                bag_moving
            )
        ):

            picking_counter += 1

        else:

            picking_counter = max(
                0,
                picking_counter - 1
            )


        # ----------------------------------------------------
        # STATUS 2:
        # BAG IS BEING PICKED UP
        # ----------------------------------------------------

        if (
            picking_counter
            >=
            PICKING_CONFIRM_FRAMES
        ):

            change_status(
                STATUS_PICKING,
                2
            )


            # As soon as the bag visibly rises,
            # confirm pickup.
            if bag_has_risen:

                picked_counter += 1

            else:

                picked_counter = 0


            # ------------------------------------------------
            # STATUS 3:
            # BAG IS PICKED UP
            # ------------------------------------------------

            if (
                picked_counter
                >=
                PICKED_CONFIRM_FRAMES
            ):

                bag_is_picked = True

                change_status(
                    STATUS_PICKED,
                    3
                )


                picking_counter = 0
                picked_counter = 0


    # ========================================================
    # AFTER PICKUP
    # ========================================================

    if bag_is_picked:

        # ----------------------------------------------------
        # STATUS 4:
        # BAG HAS BEEN MOVED FROM ITS REGION
        # ----------------------------------------------------

        if (
            not bag_inside_region
            and
            bag_far_from_original
        ):

            moved_counter += 1

        else:

            moved_counter = max(
                0,
                moved_counter - 1
            )


        if (
            moved_counter
            >=
            MOVED_CONFIRM_FRAMES
        ):

            bag_has_left_region = True

            change_status(
                STATUS_MOVED,
                4
            )


        # ----------------------------------------------------
        # STATUS 5:
        # BAG IS KEPT IN ITS REGION
        # ----------------------------------------------------

        if (
            bag_has_left_region
            and
            bag_inside_region
        ):

            # The bag has returned to the original region.
            # We need it to stop and the hand to move away.
            if (
                bag_stopped
                and
                not hand_near_bag
            ):

                kept_counter += 1

            else:

                kept_counter = max(
                    0,
                    kept_counter - 1
                )


            if (
                kept_counter
                >=
                KEPT_CONFIRM_FRAMES
            ):

                change_status(
                    STATUS_KEPT,
                    5
                )


                # Reset for another complete cycle.
                bag_is_picked = False
                bag_has_left_region = False

                picking_counter = 0
                picked_counter = 0
                moved_counter = 0
                kept_counter = 0


    # ========================================================
    # DRAW PERSON
    # ========================================================

    if selected_person is not None:

        x1, y1, x2, y2 = map(
            int,
            selected_person["box"]
        )


        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )


        draw_label(
            frame,
            (
                f"Person ID: "
                f"{selected_person['id']}"
            ),
            x1,
            y1 - 5
        )


    # ========================================================
    # DRAW BAG
    # ========================================================

    if selected_bag is not None:

        x1, y1, x2, y2 = map(
            int,
            selected_bag["box"]
        )


        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )


        draw_label(
            frame,
            (
                f"Bag ID: "
                f"{selected_bag['id']} "
                f"({selected_bag['class']})"
            ),
            x1,
            y1 - 5
        )


        if current_bag_center is not None:

            bx, by = map(
                int,
                current_bag_center
            )


            # Red point = current bag center
            cv2.circle(
                frame,
                (bx, by),
                6,
                (0, 0, 255),
                -1,
                cv2.LINE_AA
            )


    # ========================================================
    # DRAW ORIGINAL BAG REGION
    # ========================================================

    if bag_region is not None:

        rx1, ry1, rx2, ry2 = (
            bag_region
        )


        cv2.rectangle(
            frame,
            (rx1, ry1),
            (rx2, ry2),
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )


        draw_label(
            frame,
            "ORIGINAL BAG REGION",
            rx1,
            ry1 - 5
        )


    # ========================================================
    # DRAW INITIAL BAG CENTER
    # ========================================================

    if initial_bag_center is not None:

        ix, iy = map(
            int,
            initial_bag_center
        )


        cv2.circle(
            frame,
            (ix, iy),
            4,
            (255, 255, 255),
            -1,
            cv2.LINE_AA
        )


    # ========================================================
    # DRAW LEFT WRIST
    # ========================================================

    if left_wrist is not None:

        lx, ly = map(
            int,
            left_wrist
        )


        cv2.circle(
            frame,
            (lx, ly),
            7,
            (255, 255, 255),
            -1,
            cv2.LINE_AA
        )


        put_text(
            frame,
            "L wrist",
            (lx + 8, ly - 8),
            0.45,
            1
        )


    # ========================================================
    # DRAW RIGHT WRIST
    # ========================================================

    if right_wrist is not None:

        rx, ry = map(
            int,
            right_wrist
        )


        cv2.circle(
            frame,
            (rx, ry),
            7,
            (255, 255, 255),
            -1,
            cv2.LINE_AA
        )


        put_text(
            frame,
            "R wrist",
            (rx + 8, ry - 8),
            0.45,
            1
        )


    # ========================================================
    # DRAW HAND -> BAG LINE
    # ========================================================

    if (
        closest_hand is not None
        and
        current_bag_center is not None
    ):

        _, hand_point = closest_hand


        if hand_point is not None:

            hx, hy = map(
                int,
                hand_point
            )

            bx, by = map(
                int,
                current_bag_center
            )


            cv2.line(
                frame,
                (hx, hy),
                (bx, by),
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )


    # ========================================================
    # STATUS PANEL
    # ========================================================

    panel_height = 145

    overlay = frame.copy()


    cv2.rectangle(
        overlay,
        (10, 10),
        (430, panel_height),
        (0, 0, 0),
        -1
    )


    frame = cv2.addWeighted(
        overlay,
        0.65,
        frame,
        0.35,
        0
    )


    put_text(
        frame,
        "BAG STATUS",
        (20, 38),
        0.70,
        2
    )


    put_text(
        frame,
        current_status,
        (20, 68),
        0.55,
        2
    )


    put_text(
        frame,
        f"STATUS {status_number}/5",
        (20, 94),
        0.48,
        1
    )


    put_text(
        frame,
        f"Hand near bag: {hand_near_bag}",
        (20, 117),
        0.45,
        1
    )


    put_text(
        frame,
        f"Bag moving: {bag_moving}",
        (220, 117),
        0.45,
        1
    )


    put_text(
        frame,
        f"Frame: {frame_number}/{total_frames}",
        (20, 138),
        0.40,
        1
    )


    # ========================================================
    # EVENT MESSAGE
    # ========================================================

    if (
        event_message
        and
        frame_number
        <=
        event_message_until
    ):

        font = cv2.FONT_HERSHEY_SIMPLEX

        scale = 0.85

        thickness = 3


        (tw, th), baseline = (
            cv2.getTextSize(
                event_message,
                font,
                scale,
                thickness
            )
        )


        tx = max(
            10,
            (width - tw) // 2
        )


        ty = max(
            180,
            height // 2
        )


        cv2.rectangle(
            frame,
            (
                tx - 15,
                ty - th - 15
            ),
            (
                tx + tw + 15,
                ty + baseline + 15
            ),
            (0, 0, 0),
            -1
        )


        cv2.putText(
            frame,
            event_message,
            (tx, ty),
            font,
            scale,
            (0, 0, 0),
            thickness + 5,
            cv2.LINE_AA
        )


        cv2.putText(
            frame,
            event_message,
            (tx, ty),
            font,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA
        )


    # ========================================================
    # WRITE FRAME
    # ========================================================

    # No resizing.
    # Every processed frame is written at the exact
    # original resolution.
    out.write(frame)


    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.imshow(
        "Accurate Bag Status Detection",
        frame
    )


    # Press Q to stop.
    if (
        cv2.waitKey(1)
        &
        0xFF
    ) == ord("q"):

        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

out.release()

cv2.destroyAllWindows()


print()
print("--------------------------------")
print("PROCESSING COMPLETE")
print("--------------------------------")
print(
    f"Frames processed : "
    f"{frame_number}"
)
print(
    f"Output video     : "
    f"{OUTPUT_VIDEO}"
)
print("--------------------------------")
