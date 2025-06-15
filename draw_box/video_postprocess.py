import json
from google import genai
from google.genai import types
from PIL import Image
import os
import cv2
import numpy as np
import tyro
from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    """Configuration for video processing"""
    video_root: str = "/Users/xiaoli/projects/code/lerobot/demo_data/littledragon/so101_sock_stowing3/videos/chunk-000/observation.images.top"
    start_idx: int = 0  # Start processing from this video index
    end_idx: Optional[int] = None  # End processing at this video index, if None, process all videos
    model_name: str = "gemini-2.5-pro-preview-06-05"
    api_key: str = "AIzaSyCtcbjKF3423ik54zKIETbpvohFpi7IcfU"

    def __post_init__(self):
        # Validate start_idx
        if self.start_idx < 0:
            raise ValueError("start_idx must be non-negative")
            
        # Get the number of videos
        if not os.path.exists(self.video_root):
            raise ValueError(f"Video root directory does not exist: {self.video_root}")
            
        video_files = sorted([f for f in os.listdir(self.video_root) if f.endswith(".mp4")])
        num_videos = len(video_files)
        
        # Set end_idx to num_videos if None
        if self.end_idx is None:
            self.end_idx = num_videos
            
        # Validate end_idx
        if self.end_idx > num_videos:
            print(f"Warning: end_idx {self.end_idx} exceeds number of videos ({num_videos}). Setting to {num_videos}")
            self.end_idx = num_videos
            
        if self.end_idx <= self.start_idx:
            raise ValueError(f"end_idx ({self.end_idx}) must be greater than start_idx ({self.start_idx})")

model_name = "gemini-2.5-pro-preview-06-05"
# model_name = "models/gemini-2.5-flash-preview-05-20"
bounding_box_system_instructions = """
Return bounding boxes as a JSON array with labels. Never return masks or code fencing. Limit to 2 objects.
"""
prompt_vocab = {
    "default": """
    Draw a bounding box around one sock that is outside the bin.
    If there is a similar sock inside the bin, draw another box around that bin.
    If no sock is in the bin, draw a box around an empty bin.
    """,
    "tracking": """
    Given 3 images, the 1st image is before manipulation, the 2nd image is in the middle of manipulation, and the 3rd image is the result after robot manipulation.
    In the 1st image, draw a bounding box around one sock that is outside the bin and has been moved.
    also output the positions of bin that the sock is moving into, or being changed.
    """,
    "sock": """
    Draw a bounding box around one sock that is outside the bin.
    If there is a similar sock inside the bin, draw another box around that sock.
    If no sock is in the bin, draw a box around an empty bin.
    """
}


safety_settings = [
    types.SafetySetting(
        category="HARM_CATEGORY_DANGEROUS_CONTENT",
        threshold="BLOCK_ONLY_HIGH",
    ),
]



def parse_json(json_output: str):
    # Parsing out the markdown fencing
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line == "```json":
            # Remove everything before "```json"
            json_output = "\n".join(lines[i+1:])
            # Remove everything after the closing "```"
            json_output = json_output.split("```")[0]
            break  # Exit the loop once "```json" is found
    return json_output


def get_box(img, prompt = """
    Pick one sock that's outside the bin and draw one box over this sock.
    If there is another similar sock with in the bin, draw a box over that bin,
    otherwise draw one box over an empty bin.
    """):

    # Load and resize image
    # im = Image.open("IMG_7682.jpg")
    # im = Image.open(BytesIO(open("IMG_7682.jpg", "rb").read()))
    # im.thumbnail([1024, 1024], Image.Resampling.LANCZOS)

    print(f"{img.shape=}")
    img_obj = Image.fromarray(img)
    print(f"{img_obj.size=}")
    response = client.models.generate_content(
        model=model_name,
        contents=[prompt, img_obj],
        config=types.GenerateContentConfig(
            system_instruction=bounding_box_system_instructions,
            temperature=0.5,
            safety_settings=safety_settings,
        )
    )

    height, width, _depth = img.shape

    # Parsing out the markdown fencing
    bounding_boxes = parse_json(response.text)

    # Iterate over the bounding boxes
    boxes = []
    for i, bounding_box in enumerate(json.loads(bounding_boxes)):
        # Convert normalized coordinates to absolute coordinates
        abs_y1 = int(bounding_box["box_2d"][0]/1000 * height)
        abs_x1 = int(bounding_box["box_2d"][1]/1000 * width)
        abs_y2 = int(bounding_box["box_2d"][2]/1000 * height)
        abs_x2 = int(bounding_box["box_2d"][3]/1000 * width)

        if abs_x1 > abs_x2:
            abs_x1, abs_x2 = abs_x2, abs_x1

        if abs_y1 > abs_y2:
            abs_y1, abs_y2 = abs_y2, abs_y1

        boxes.append((abs_x1, abs_y1))
        boxes.append((abs_x2, abs_y2))
        # boxes.append((abs_y1, abs_x1))
        # boxes.append((abs_y2, abs_x2))

    print("Drawing boxes", boxes)
    return boxes

def draw_boxes(image, boxes, colors=[(255, 0, 0), (0, 0, 255)], thickness=6):
    """
    Draw bounding boxes on the image with different colors
    Args:
        image: numpy array of the image
        boxes: list of box coordinates [(x1,y1), (x2,y2)]
        colors: list of RGB color tuples (default: [red, blue])
        thickness: thickness of the box lines
    Returns:
        image with boxes drawn
    """
    img_with_boxes = image.copy()
    for i in range(0, len(boxes), 2):
        if i + 1 < len(boxes):
            pt1 = boxes[i]
            pt2 = boxes[i + 1]
            # Use different colors for each box in sequence
            color = colors[i // 2 % len(colors)]
            cv2.rectangle(img_with_boxes, pt1, pt2, color, thickness)
    return img_with_boxes

def get_bbox_with_sequence(img1, img2, img3, prompt_type="tracking"):
    """
    Get bounding boxes by comparing three images in sequence (before, during, after manipulation)
    Args:
        img1: First image as numpy array (RGB) - before manipulation
        img2: Second image as numpy array (RGB) - during manipulation
        img3: Third image as numpy array (RGB) - after manipulation
        prompt_type: Type of prompt to use from prompt_vocab
    Returns:
        List of box coordinates [(x1,y1), (x2,y2)]
    """
    # Create temporary files for all images
    import tempfile
    import os
    
    # Save images to temporary files
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp1:
        temp1_path = temp1.name
        cv2.imwrite(temp1_path, cv2.cvtColor(img1, cv2.COLOR_RGB2BGR))
    
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp2:
        temp2_path = temp2.name
        cv2.imwrite(temp2_path, cv2.cvtColor(img2, cv2.COLOR_RGB2BGR))
        
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp3:
        temp3_path = temp3.name
        cv2.imwrite(temp3_path, cv2.cvtColor(img3, cv2.COLOR_RGB2BGR))
    
    try:
        # Upload the first image
        uploaded_file = client.files.upload(file=temp1_path)
        
        # Prepare the second and third images as inline data
        with open(temp2_path, 'rb') as f:
            img2_bytes = f.read()
            
        with open(temp3_path, 'rb') as f:
            img3_bytes = f.read()
        
        # Get the appropriate prompt
        prompt = prompt_vocab.get(prompt_type)
        
        # Generate content with all three images
        response = client.models.generate_content(
            model=model_name,
            contents=[
                prompt,
                uploaded_file,  # Use the uploaded file reference for first image
                types.Part.from_bytes(
                    data=img2_bytes,
                    mime_type='image/png'
                ),
                types.Part.from_bytes(
                    data=img3_bytes,
                    mime_type='image/png'
                )
            ],
            config=types.GenerateContentConfig(
                system_instruction=bounding_box_system_instructions,
                temperature=0.5,
                safety_settings=safety_settings,
            )
        )

        height, width, _depth = img1.shape

        # Parse the response
        bounding_boxes = parse_json(response.text)

        # Convert normalized coordinates to absolute coordinates
        boxes = []
        for bounding_box in json.loads(bounding_boxes):
            abs_y1 = int(bounding_box["box_2d"][0]/1000 * height)
            abs_x1 = int(bounding_box["box_2d"][1]/1000 * width)
            abs_y2 = int(bounding_box["box_2d"][2]/1000 * height)
            abs_x2 = int(bounding_box["box_2d"][3]/1000 * width)

            if abs_x1 > abs_x2:
                abs_x1, abs_x2 = abs_x2, abs_x1

            if abs_y1 > abs_y2:
                abs_y1, abs_y2 = abs_y2, abs_y1

            boxes.append((abs_x1, abs_y1))
            boxes.append((abs_x2, abs_y2))

        print("Drawing boxes", boxes)
        return boxes
        
    finally:
        # Clean up temporary files
        os.unlink(temp1_path)
        os.unlink(temp2_path)
        os.unlink(temp3_path)

if __name__ == "__main__":
    # Parse command line arguments
    config = tyro.cli(Config)
    
    # Initialize client with config
    client = genai.Client(api_key=config.api_key)
    model_name = config.model_name
    
    video_root = config.video_root
    video_save_dir = video_root.replace("observation.images.top", "observation.images.top.boxes")
    box_save_path = os.path.join(video_save_dir, "boxes.json")
    if not os.path.exists(video_save_dir):
        os.makedirs(video_save_dir)
    
    # Dictionary to store boxes for all videos
    all_boxes = {}
    
    video_files = sorted([f for f in os.listdir(video_root) if f.endswith(".mp4")])
    for video_file in video_files[config.start_idx:config.end_idx]:
        video_path = os.path.join(video_root, video_file)
        video_save_path = os.path.join(video_save_dir, video_file)
        print(f"Processing video: {video_path}")
        
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create video writer with H.264 codec
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Use H.264 codec
        out = cv2.VideoWriter(video_save_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            print("Error: Could not open output video file")
            continue
            
        # Get the first frame for box detection
        ret, first_frame = cap.read()
        if not ret:
            print("Error reading first frame")
            continue
            
        # Get middle frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
        ret, middle_frame = cap.read()
        if not ret:
            print("Error reading middle frame")
            continue
            
        # Get the last frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
        ret, last_frame = cap.read()
        if not ret:
            print("Error reading last frame")
            continue
            
        # Convert frames to RGB for box detection
        first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
        middle_frame_rgb = cv2.cvtColor(middle_frame, cv2.COLOR_BGR2RGB)
        last_frame_rgb = cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB)

        # Get boxes using all three frames for tracking
        boxes = get_bbox_with_sequence(first_frame_rgb, middle_frame_rgb, last_frame_rgb, "tracking")
        print(f"Boxes: {boxes}")
        
        # Store boxes in dictionary
        all_boxes[video_file] = {
            "boxes": boxes,
            "frame_shape": [height, width, 3]
        }
        
        # Draw boxes on first frame and save it
        first_frame_with_boxes = draw_boxes(first_frame, boxes, thickness=6)
        first_frame_save_path = os.path.join(video_save_dir, f"{os.path.splitext(video_file)[0]}_first_frame_boxes.png")
        cv2.imwrite(first_frame_save_path, first_frame_with_boxes)
        print(f"Saved first frame with boxes to: {first_frame_save_path}")
        
        # Reset video capture to beginning
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Process each frame
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Draw boxes on the frame
            frame_with_boxes = draw_boxes(frame, boxes, thickness=6)
            
            # Write the frame
            out.write(frame_with_boxes)
            
            frame_count += 1
            if frame_count % 100 == 0:
                print(f"Processed {frame_count}/{total_frames} frames")
        
        # Release everything
        cap.release()
        out.release()
        print(f"Saved video with boxes to: {video_save_path}")
    
    # Save all boxes to JSON file
    with open(box_save_path, 'w') as f:
        json.dump(all_boxes, f, indent=2)
    print(f"Saved all bounding boxes to: {box_save_path}")
        
        
        
        
