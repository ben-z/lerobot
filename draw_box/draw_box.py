import json
from google import genai
from google.genai import types

client = genai.Client(api_key="AIzaSyDwTSty8LLYSKfaCRQyp7z6CkjZh0iYsGc")

model_name = "gemini-2.5-flash-preview-05-20"
bounding_box_system_instructions = """
Return bounding boxes as a JSON array with labels. Never return masks or code fencing. Limit to 2 objects.
"""

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


def get_box(img):
    prompt = """
    Pick one sock that's outside the bin and draw one box over this sock.
    If there is another similar sock with in the bin, draw a box over that bin,
    otherwise draw one box over an empty bin.
    """


# Load and resize image
# im = Image.open("IMG_7682.jpg")
# im = Image.open(BytesIO(open("IMG_7682.jpg", "rb").read()))
# im.thumbnail([1024, 1024], Image.Resampling.LANCZOS)


# # Run model to find bounding boxes
    response = client.models.generate_content(
        model=model_name,
        contents=[prompt, img],
        config=types.GenerateContentConfig(
            system_instruction=bounding_box_system_instructions,
            temperature=0.5,
            safety_settings=safety_settings,
        )
    )

    width, height = img.size

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

        boxes.append((abs_x1, abs_y1), (abs_x2, abs_y2))
    return boxes
