import torch
from PIL import Image
from numpy import asarray
import os
import numpy as np
import pandas as pd
from PIL import Image
import numpy as np
import transformers
import re
import sys
import os
from huggingface_hub import snapshot_download
from pathlib import Path

from torchvision.transforms import Compose, Resize, ToTensor, CenterCrop

evaluation_dir = os.path.abspath(os.path.dirname(__file__))
radialog_path = os.path.join(evaluation_dir, "RaDialog")
if radialog_path not in sys.path:
    sys.path.append(radialog_path)

# radialog imports 
# from LLAVA_Biovil.llava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria, remap_to_uint8
# from LLAVA_Biovil.llava.model.builder import load_pretrained_model
# from LLAVA_Biovil.llava.conversation import SeparatorStyle, conv_vicuna_v1
# from LLAVA_Biovil.llava.constants import IMAGE_TOKEN_INDEX


MIN_PIXELS = 1024
MAX_PIXELS = 451584

try:
    from io import BytesIO
    from llava.constants import IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
    from llava.model.builder import load_pretrained_model
    from llava.utils import disable_torch_init
    from llava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria
    llava_available = True
except ImportError as e:
    print(f"unable to import llava {e}")
    llava_available = False


def load_model_and_processor(model_name, device_map='cpu'):

    processor = None
    tokenizer = None
    
    if model_name == 'radialog':
        repo_id = "ChantalPellegrini/RaDialog-interactive-radiology-report-generation"

        model_path = snapshot_download(repo_id=repo_id, revision="main")
        model_path = Path(model_path)

        tokenizer, model, _, _ = load_pretrained_model(
            model_path, 
            model_base='liuhaotian/llava-v1.5-7b',
            model_name="llava-v1.5-7b-task-lora_radialog_instruct_llava_biovil_unfrozen_2e-5_5epochs_v5_checkpoint-21000", 
            load_8bit=False,
            device_map=device_map, 
            load_4bit=False
            )
        
    
    elif model_name == 'chexagent':
        model_id = "StanfordAIMI/CheXagent-2-3b"
        dtype = torch.bfloat16
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = transformers.AutoModelForCausalLM.from_pretrained(model_id, device_map=device_map, trust_remote_code=True)
        model = model.to(dtype)
        model.eval()
    elif model_name == 'llavamed':
        from radvlm.evaluation.llava_med_loading import register_llava_med_hf
        model_path = 'microsoft/llava-med-v1.5-mistral-7b'
        register_llava_med_hf()
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            local_files_only=False,
            device_map=device_map,
            trust_remote_code=True,
        )
        processor = transformers.AutoProcessor.from_pretrained(
            model_path, 
            local_files_only=False, 
            trust_remote_code=True
        )

    elif model_name == 'maira2':
        model = transformers.AutoModelForCausalLM.from_pretrained(
            'microsoft/maira-2',
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map=device_map
            )
        processor = transformers.AutoProcessor.from_pretrained(
            'microsoft/maira-2', 
            trust_remote_code=True
            )

    elif model_name == "google/medgemma-4b-pt":
            model = transformers.AutoModelForImageTextToText.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map=device_map,
            )
            processor = transformers.AutoProcessor.from_pretrained(model_name)
    elif model_name == "microsoft/llava-rad":
        disable_torch_init()

        model_path = "microsoft/llava-rad"
        model_base = "lmsys/vicuna-7b-v1.5"
        model_name = "llavarad"
        conv_mode = "v1"

        # tokenizer, model, image_processor, context_len 
        tokenizer, model, processor, _ = load_pretrained_model(model_path, model_base, model_name, device_map=device_map)

    else:
        if 'qwen' in model_name.lower():
            model = transformers.Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name, torch_dtype=torch.float16, device_map=device_map
                )
            
            processor = transformers.AutoProcessor.from_pretrained(model_name, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)

        else:
            # Load llava-ov checkpoint 
            common_kwargs = {
                "torch_dtype": torch.float16,
                "low_cpu_mem_usage": True, 
            }

            if model_name == 'llavaov':
                model_name = 'llava-hf/llava-onevision-qwen2-7b-si-hf'

            model = transformers.LlavaOnevisionForConditionalGeneration.from_pretrained(
                model_name,
                device_map=device_map,
                **common_kwargs
            )
            processor = transformers.AutoProcessor.from_pretrained(model_name)

    return tokenizer, model, processor



def inference_maira2_report(model, processor, image_path, prompt, grounding=False, max_new_tokens=500):
    image = Image.open(image_path).convert('RGB') 
    processed_inputs = processor.format_and_preprocess_reporting_input(
                current_frontal=image,
                current_lateral=None,
                prior_frontal=None,
                indication=None,
                technique=None,
                comparison=None,
                prior_report=None,
                return_tensors="pt",
                get_grounding=False
                ).to(model.device)
    
    output_decoding = model.generate(
                    **processed_inputs,
                    max_new_tokens=max_new_tokens,
                    use_cache=True
                )
    prompt_length = processed_inputs["input_ids"].shape[-1]
    decoded_text = processor.decode(output_decoding[0][prompt_length:], skip_special_tokens=True)
    decoded_text = decoded_text.lstrip()  # Findings generation completions have a single leading space
    generated_text = processor.convert_output_to_plaintext_or_grounded_sequence(decoded_text)

    return generated_text



def inference_maira2_grounding(model, processor, image_path, label, max_new_tokens=500):

    image = Image.open(image_path).convert('RGB') 
    processed_inputs = processor.format_and_preprocess_phrase_grounding_input(
        frontal_image=image,
        phrase=label,
        return_tensors="pt",
    ).to(model.device)
    
    output_decoding = model.generate(
        **processed_inputs, 
        max_new_tokens=max_new_tokens,
        use_cache=True
    )

    prompt_length = processed_inputs["input_ids"].shape[-1]
    decoded_text = processor.decode(output_decoding[0][prompt_length:], skip_special_tokens=True)
    try:
        prediction = processor.convert_output_to_plaintext_or_grounded_sequence(decoded_text)

        
        width, height = image.size
        coordinates = [
            list(processor.adjust_box_for_original_image_size(coord, width=width, height=height))
            for coord in prediction[0][1] if coord is not None
        ]
        coordinates_str = ", ".join(str([round(val, 2) for val in box]) for box in coordinates) if coordinates else ""

    except Exception as e:
        print(f"Error occurred: {e}")
        coordinates_str = ""

    return coordinates_str




def inference_radialog(tokenizer, model, image_path, prompt, chat_history=None, max_new_tokens=500):
    """
    Generate a response in a single-turn or multi-turn conversation for the RaDialog model.
    
    This function always returns the updated chat_history and the model's response.
    If `chat_history` is None or empty, it acts as single-turn but still returns the updated chat_history.

    Args:
        tokenizer: The tokenizer corresponding to the RaDialog model.
        model: The RaDialog model.
        image: The PIL image (or similar) used for visual context.
        prompt: The new user prompt for this turn.
        chat_history: A list of (user_msg, assistant_msg) representing the conversation so far.
                      If None or empty, acts as single-turn but will return the new chat_history.
        max_new_tokens: The maximum number of new tokens to generate.

    Returns:
        chat_history (list): The updated chat_history including this turn's user query and assistant response.
        pred (str): The assistant's response for this turn.
    """

    # Initialize chat_history if not provided
    if chat_history is None:
        chat_history = []

    # Check if this is the first turn (single-turn scenario)
    first_turn = (len(chat_history) == 0)

    # Preprocess the image
    image = Image.open(image_path).convert('RGB')
    image = remap_to_uint8(np.array(image))
    image = Image.fromarray(image).convert("L")

    model.config.tokenizer_padding_side = "left"
    conv = conv_vicuna_v1.copy()

    # Rebuild the conversation from history if it's multi-turn
    for human, assistant in chat_history:
        conv.append_message("USER", human)
        conv.append_message("ASSISTANT", assistant)

    # For the very first turn, prepend "<image>. "
    if first_turn:
        user_prompt = "<image>. " + prompt
    else:
        user_prompt = prompt

    # Add the new user message and a placeholder for the assistant's response
    conv.append_message("USER", user_prompt)
    conv.append_message("ASSISTANT", None)

    # Construct the final prompt text
    text_input = conv.get_prompt()

    # Prepare the image tensor
    vis_transforms_biovil = create_chest_xray_transform_for_inference(512, center_crop_size=448)
    image_tensor = vis_transforms_biovil(image).unsqueeze(0)
    image_tensor = image_tensor.to(model.device, dtype=torch.bfloat16)

    # Tokenize input including the image token
    input_ids = tokenizer_image_token(text_input, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)

    # Stopping criteria
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

    # Generate the response
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=False,
            use_cache=True,
            max_new_tokens=max_new_tokens,
            stopping_criteria=[stopping_criteria],
            pad_token_id=tokenizer.pad_token_id
        )

    # Decode the generated output
    pred = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip().replace("</s>", "")

    # Update the conversation: remove the None placeholder and add the actual assistant response
    conv.messages.pop()  # remove the placeholder None
    conv.append_message("ASSISTANT", pred)

    # Update the external chat_history with the new turn
    chat_history.append((prompt, pred))

    return pred, chat_history


class ExpandChannels:
    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        if data.shape[0] != 1:
            raise ValueError(f"Expected input of shape [1, H, W], found {data.shape}")
        return torch.repeat_interleave(data, 3, dim=0)

def create_chest_xray_transform_for_inference(resize: int, center_crop_size: int) -> Compose:
    transforms = [Resize(resize), CenterCrop(center_crop_size), ToTensor(), ExpandChannels()]
    return Compose(transforms)




def inference_llavamed(model, processor, image_path, prompt, chat_history=None, max_new_tokens=500):
    """
    Unified function for LLaVA-Med inference, supporting both single-turn and multi-turn modes.

    Args:
        model: The LLaVA-Med model.
        processor: The processor for LLaVA-Med (provides apply_chat_template and tokenizer).
        image: The image (PIL or NumPy array) for the first turn, or None for subsequent turns.
        prompt: The user message for this turn.
        chat_history: A list of (user_message, assistant_message) for past turns. If None or empty, single-turn mode is used.
        max_new_tokens: Maximum number of new tokens to generate.

    Returns:
        chat_history: The updated chat_history including this turn's user prompt and the assistant's response.
        response: The assistant's response string for this turn.
    """
    IMAGE_TOKEN_INDEX = -200

    # Initialize chat history if not provided
    if chat_history is None:
        chat_history = []

    # Prepare conversation history
    conversation = []
    for i, (user_text, assistant_text) in enumerate(chat_history):
        if i == 0:
            conversation.append({"role": "user", "content": f"<image>\n{user_text}"})
        else:
            conversation.append({"role": "user", "content": user_text})
        conversation.append({"role": "assistant", "content": assistant_text})

    # Add the current user prompt
    if len(chat_history) == 0:
        # First turn: Add the image token
        user_content = f"<image>\n{prompt}"
    else:
        # Subsequent turns: No image token
        user_content = prompt

    conversation.append({"role": "user", "content": user_content})

    # Apply chat template to prepare the full prompt
    full_prompt = processor.apply_chat_template(
        conversation,
        tokenize=False,
        add_special_tokens=True,
        add_generation_prompt=True,
    )

    # Tokenize the prompt and add image token
    input_ids = tokenizer_image_token(
        full_prompt, processor, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).to(model.device)

    inputs = {"inputs": input_ids.unsqueeze(0)}

    # If an image is provided (first turn), preprocess and include it
    image = Image.open(image_path).convert('RGB')
    if image is not None:
        inputs["images"] = (
            model.get_vision_tower()
            .image_processor.preprocess(image, return_tensors="pt")["pixel_values"]
            .to(model.device, torch.float16)
        )

    # Set attention mask
    inputs["attention_mask"] = torch.ones_like(inputs["inputs"])

    # Load generation config
    generation_config = transformers.GenerationConfig.from_pretrained(
        'microsoft/llava-med-v1.5-mistral-7b',
        local_files_only=False, trust_remote_code=True
    )
    generation_config.pad_token_id = processor.pad_token_id

    # Generate output
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            generation_config=generation_config,
            max_new_tokens=max_new_tokens
        )

    # Decode the model's output
    response = processor._tokenizer.decode(output[0].tolist(), skip_special_tokens=True).strip()

    # Update the chat history
    chat_history.append((prompt, response))

    return response, chat_history

def inference_llava_rad(model, tokenizer, image_processor, image_path, prompt, chat_history=None, max_new_tokens=1024):
    # https://huggingface.co/microsoft/llava-rad
    # you need to install the llava rad repo to run this. A copy is RadVLM-GRPO/repos_deps/LLaVA-Rad
    del prompt
    query = "<image>\nDescribe the findings of the chest x-ray.\n"

    conv_mode = "v1"
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], query)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    image = Image.open(image_path).convert('RGB')
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0].half().unsqueeze(0).cuda()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    stopping_criteria = KeywordsStoppingCriteria(["</s>"], tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=False,
            temperature=0.0,
            max_new_tokens=1024,
            use_cache=True,
            stopping_criteria=[stopping_criteria])

    outputs = tokenizer.batch_decode(output_ids[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
    outputs = outputs.strip()

    return outputs, query


from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

def inference_deepmedix_r1(model, tokenizer, image_path, prompt, chat_history=None):
    del prompt
    # https://huggingface.co/datasets/Qika/xraybench
    # https://huggingface.co/Qika/DeepMedix-R1
    reason_prompt = r"You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. During this reasoning process, prioritize analyzing the local regions of the image by leveraging the bounding box coordinates in the format [x_min, y_min, x_max, y_max]. The final answer MUST BE put in \boxed{}. An example is like: <think> reasoning process 1 with [x_min1, y_min1, x_max1, y_max1]; reasoning process 2 with [x_min2, y_min2, x_max2, y_max2] </think>. The answer is: \boxed{answer}."
    content1 = "Please act as an experienced radiologist and generate the \"FINDINGS\" section of an X-ray report based on the provided image(s). Carefully examine the image(s) and describe all observed anatomical structures and abnormalities in a systematic and objective manner."

    images = [image_path]

    content_list = []
    for image_url in images:
        content_list.append({
            "type": "image",
            "image": image_url,
        })

    mode = 'think'

    if mode == 'think':
        prompt = content1 + '\n' + reason_prompt + '\n'
        content_list.append({"type": "text",
                             "text": prompt})
    else:
        prompt = content1
        content_list.append({"type": "text",
                             "text": prompt})
    messages = [
        {
            "role": "user",
            "content": content_list
        }
    ]

    # Preparation for inference
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # print(text)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = tokenizer(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=4096, do_sample=True, temperature=0.6)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = tokenizer.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    # print(output_text)
    # print(output_text[0])
    full_output = output_text[0]
    final_answer, matched = _extract_answer(full_output, True)
    return final_answer, full_output, prompt

def clean_text(pred: str) -> str:

    # Normalize whitespace and line breaks
    text = re.sub(r'[\r\n]+', ' ', pred)
    text = re.sub(r'\s+', ' ', text)

    # Remove duplicated commas or stray punctuation
    text = re.sub(r'\s*,\s*,+', ', ', text)
    text = re.sub(r'\s*,\s*\.', '.', text)
    text = re.sub(r'\s*\.\s*\.', '.', text)
    text = re.sub(r'\s*,\s*$', '', text)
    text = re.sub(r'^\s*,\s*', '', text)

    # Remove repeated punctuation artifacts
    text = re.sub(r'(,\s*){2,}', ', ', text)
    text = re.sub(r'(\.\s*){2,}', '. ', text)

    # Trim leading/trailing whitespace or punctuation
    text = text.strip(" ,.")

    return text

def inference_medgemma_4b_pt(model, processor, image_path, prompt, chat_history=None, max_new_tokens=100):
    # https://huggingface.co/google/medgemma-4b-pt
    # prompt from https://arxiv.org/abs/2507.05201 page 51 CXR report generation prompt Table A7

    image = Image.open(image_path).convert("RGB")
    prompt = "<start_of_image> findings:"

    inputs = processor(
        text=prompt, images=image, return_tensors="pt"
    ).to(model.device, dtype=torch.bfloat16)

    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generation = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        generation = generation[0][input_len:]

    decoded = processor.decode(generation, skip_special_tokens=True)
    decoded = clean_text(decoded)
    # no chat history for medgemma pre trained
    return decoded, prompt

from qwen_vl_utils import process_vision_info
import torch


import re

def inference_qwen2vl(model, processor, image_path, prompt, chat_history=None, max_new_tokens=1500):
    """
    Generate a response using the Qwen-2-VL model in either single-turn or multi-turn mode,
    taking a local image_path and preserving full conversation history.

    Args:
        model: The Qwen-2-VL model.
        processor: The Qwen-2-VL processor (apply_chat_template, tokenization, batch_decode).
        image_path: Path on disk to your image (e.g. "/path/to/img.jpg").
        prompt: The user prompt for this turn.
        chat_history: A list of (user_msg, assistant_msg) tuples representing prior conversation.
                      If None or empty, single-turn mode is used.
        max_new_tokens: The maximum number of new tokens to generate.

    Returns:
        response (str): The assistant’s response for this turn.
        chat_history (list): The updated chat_history including this turn’s (prompt, response).
    """
    # 1) Initialize history if needed
    if chat_history is None:
        chat_history = []

    # 2) Build the file:// URI for the image
    image_uri = f"file://{image_path}"

    # 3) Construct the messages list
    messages = []
    for turn_idx, (user_txt, assistant_txt) in enumerate(chat_history):
        # user turn
        if turn_idx == 0:
            # first historic turn included the image
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text",  "text": user_txt},
                    {"type": "image", "image": image_uri},
                ],
            })
        else:
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": user_txt},
                ],
            })
        # assistant response
        messages.append({
            "role": "assistant",
            "content": [
                {"type": "text", "text": assistant_txt},
            ],
        })

    # 4) Add the new user turn
    if len(chat_history) == 0:
        # first-ever turn: include image + prompt
        messages.append({
            "role": "user",
            "content": [
                {"type": "image", "image": image_uri, },
                {"type": "text",  "text": prompt},
            ],
        })
    else:
        # subsequent round: only text
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
            ],
        })

    # 5) Extract vision inputs
    image_inputs, video_inputs = process_vision_info(messages)

    # 6) Build the full text prompt
    full_prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # 7) Prepare tensor inputs
    inputs = processor(
        text=[full_prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    # 8) Move all tensors to device, then cast only vision tensors to float16
    for k, v in inputs.items():
        inputs[k] = v.to(model.device)
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].half()
    if "video_inputs" in inputs:
        inputs["video_inputs"] = inputs["video_inputs"].half()
    if "video_frames" in inputs:
        inputs["video_frames"] = inputs["video_frames"].half()

    # 9) Generate
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )

    # 10) Trim off the input tokens from each output
    trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]

    # 11) Decode
    output_texts = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    response = output_texts[0].strip()

    # 12) Update history and return
    chat_history.append((prompt, response))
    return response, chat_history
    

def inference_llavaov(model, processor, image_path, prompt, chat_history=None, max_new_tokens=1500):
    """
    Generate a response using the LLaVA-OV model in either single-turn or multi-turn mode.

    Args:
        model: The LLaVA-OV model.
        processor: The processor for LLaVA-OV (provides apply_chat_template and tokenization).
        image: The PIL or NumPy image. On the first turn, this will be included with the prompt.
        prompt: The user prompt for this turn.
        chat_history: A list of (user_msg, assistant_msg) tuples representing the conversation so far.
                      If None or empty, single-turn mode is used. Even in single-turn mode, 
                      this function returns chat_history so that you can continue in subsequent turns.
        max_new_tokens: The maximum number of new tokens to generate.

    Returns:
        chat_history (list): The updated chat_history including this turn's (prompt, response).
        response (str): The assistant's response for this turn.
    """

    # If no chat_history provided, initialize an empty one (single-turn scenario)
    if chat_history is None:
        chat_history = []

    # Convert image to the expected shape (C, H, W)

    image = Image.open(image_path).convert('RGB')
    image = asarray(image.convert('RGB')).transpose(2, 0, 1)

    # Prepare the conversation from chat_history
    conversation = []
    num_round = 0
    for user_text, assistant_text in chat_history:
        if num_round==0:
            conversation.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {"type": "image"},
                    ],
                }
            )
        else:
            conversation.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                    ],
                }
            )
        # Add assistant response from history     
        conversation.append(
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": assistant_text},
                    ],
                }
            )
        num_round += 1

    # Check if this is the first round of conversation
    if len(chat_history) == 0:
        # First turn: Add the user message with the image token
        conversation.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image"},
                ],
            }
        )
    else:
        # Subsequent turns: Add user message without the image token
        conversation.append(
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        )

    # Generate a response using the model
    full_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    # Prepare model inputs
    inputs = processor(images=image, text=full_prompt, return_tensors="pt", padding=True).to(
        model.device, torch.float16
    )

    # Generate response
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    full_response = processor.decode(output[0], skip_special_tokens=True)
    response = re.split(r"(user|assistant)", full_response)[-1].strip()

    # Update chat_history
    chat_history.append((prompt, response))

    return response, chat_history

def inference_chexagent(model, tokenizer, image_path, prompt, grounding=False, max_new_tokens=500):
    paths = [image_path]
    query = tokenizer.from_list_format([*[{'image': path} for path in paths], {'text': prompt}])
    conv = [{"from": "system", "value": "You are a helpful assistant."}, {"from": "human", "value": query}]
    input_ids = tokenizer.apply_chat_template(conv, add_generation_prompt=True, return_tensors="pt")
    output = model.generate(
        input_ids.to(model.device), do_sample=False, num_beams=1, temperature=1., top_p=1., use_cache=True,
        max_new_tokens=max_new_tokens
    )[0]
    generated_text = tokenizer.decode(output[input_ids.size(1):-1])

    if grounding:
        pattern = re.compile(r"<\|box\|> \((\d+),(\d+)\),\((\d+),(\d+)\) <\|/box\|>")
        # Find all matches in the text
        matches = pattern.findall(generated_text)
        if not matches:
            return ""
        # Transform the coordinates into the desired format
        result = [
            f"[{int(x1)/100:.2f}, {int(y1)/100:.2f}, {int(x2)/100:.2f}, {int(y2)/100:.2f}]"
            for x1, y1, x2, y2 in matches
        ]
        
        generated_text = ", ".join(result)


    return generated_text


DATA_SOURCE = "mimic_grpo"
ABILITY = "radiology"
PROMPT_SUFFIXE = " /think"
PROMPT_PREFIX = "<image>\n"
MARK = "</think>"

import json, argparse, os, re, datasets

def _needs_retry_r1(out):
    """
    Retry only if:
    </think> is missing, OR
    generation ended because it hit the max token limit (i.e. not EOS/stop)
    """
    fr = str(getattr(out.outputs[0], "finish_reason", "")).lower()

    hit_length_limit = fr in {"length", "max_length_exceeded", "max_length"}
    text = out.outputs[0].text

    idx = text.rfind(MARK)

    if idx == -1:
        mark_not_in_text = True
    else:
        mark_not_in_text = False

    retry = mark_not_in_text or hit_length_limit

    return retry

def extract_boxed(text):
    matches = re.findall(r'\\boxed\s*\{([\s\S]*?)\}', text)
    return matches[-1] if matches else None

def _extract_answer(text, r1_enabled):
    if r1_enabled:
        # for deepmedix r1
        boxed_match = extract_boxed(text)

        if boxed_match:
            text = boxed_match
            return text, True

        idx = text.rfind(MARK)
        if idx == -1:
            return text.strip(), False
        return text[idx + len(MARK):].strip(), True

    return text, True

def get_prompt(datapoint, model_name, r1, remove_think_suffixe):

    prompt_prefix = PROMPT_PREFIX
    prompt_suffix = PROMPT_SUFFIXE

    if model_name == "Qika/DeepMedix-R1":
        # https://huggingface.co/Qika/DeepMedix-R1
        reason_prompt = r"You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. During this reasoning process, prioritize analyzing the local regions of the image by leveraging the bounding box coordinates in the format [x_min, y_min, x_max, y_max]. The final answer MUST BE put in \boxed{}. An example is like: <think> reasoning process 1 with [x_min1, y_min1, x_max1, y_max1]; reasoning process 2 with [x_min2, y_min2, x_max2, y_max2] </think>. The answer is: \boxed{answer}."
        prompt_suffix  = '\n' + reason_prompt + '\n'

        # https://huggingface.co/datasets/Qika/xraybench
        prompt = "<image> Please act as an experienced radiologist and generate the \"FINDINGS\" section of an X-ray report based on the provided image(s). Carefully examine the image(s) and describe all observed anatomical structures and abnormalities in a systematic and objective manner."
        prompt = prompt + prompt_suffix
        return prompt

    elif model_name == "google/medgemma-27b-it":
        # https://huggingface.co/google/medgemma-27b-it
        prompt = "<image> You are an expert radiologist. Please succinctly describe the findings for the above chest x-ray."
        return prompt

    elif model_name == "microsoft/llava-rad":
        # https://huggingface.co/microsoft/llava-rad
        prompt = "<image>\nDescribe the findings of the chest x-ray.\n"
        return prompt

    elif model_name == "google/medgemma-4b-pt":
        assert False, "use the huggingface code"
        prompt = "<image> findings:"
        return prompt

    elif model_name == "Qwen/Qwen3-VL-8B-Instruct" or model_name == "Qwen/Qwen3-VL-8B-Thinking":
        prompt = """<image>\nPlease write a radiology report for this Chext X-ray.

It should be one unstructured paragraph of findings only: concise, natural clinical language, objective, declarative sentences describing visible features only, suitable for a radiology findings report using standard radiology phrasing."""
        return prompt
    else:
        # RadVLM models
        prompt = prompt_prefix + datapoint["instr"]["question"]

        if r1 and (not remove_think_suffixe):
            prompt = prompt + prompt_suffix

    return prompt

from pprint import pprint

def inference_qwen2vl_vllm(
    data_loader,
    model_name: str,
    process_batch_num: int | None = None,
    r1: bool = False,
    test_dataset_path: str | None = None,
    tensor_parallel_size: int = 4,
    gpu_memory_utilization: float = 0.7,
    temperature: float = 0.0,
    max_input_tokens: int = 32768,
    max_new_tokens: int = 4096,
    remove_think_suffixe: bool = False,
):
    
    params = {"temperature": temperature, "max_new_tokens": max_new_tokens, "remove_think_suffixe": remove_think_suffixe}
    pprint(f"{params=}")
    try:
        from verl.utils import hf_tokenizer, hf_processor
        from verl.utils.dataset.rl_dataset import RLHFDataset
    except ImportError as e:
        raise ImportError(
            "verl is not installed. Please install it by going into RadVLM-GRPO/verl and type:"
            " pip install -e ."
        ) from e

    try:
        from vllm import LLM, SamplingParams
    except ImportError as e:
        raise ImportError(
            "vllm is not installed. Please install it."
        ) from e

    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader
    
    tasks = []
    tasks_for_ds = []

    total_batches = len(data_loader)

    max_pixels = MAX_PIXELS
    min_pixels = MIN_PIXELS

    if model_name == "Qika/DeepMedix-R1":
        # https://huggingface.co/Qika/DeepMedix-R1
        max_pixels = 262144

    for batch_i, batch in enumerate(data_loader):
        if process_batch_num and batch_i >= process_batch_num:
            break
        if batch_i % 10 == 0:
            print(f"Processing batch {batch_i + 1} / {total_batches}")
        datapoint = batch[0]

        image_path = datapoint['img_path']
        prompt = get_prompt(datapoint=datapoint, model_name=model_name, r1=r1, remove_think_suffixe=remove_think_suffixe)
        tasks.append((datapoint, prompt, image_path))

        img_entry = {
            "image": f"file://{image_path}",
            "min_pixels": min_pixels,
            "max_pixels":  max_pixels
        }
        prompt_msg = {
            "role": "user",
            "content": prompt
        }
        item = {
            "data_source": DATA_SOURCE,
            "prompt": [prompt_msg],
            "images": [img_entry],     
            "ability": ABILITY,
            "reward_model": {
                "style": "rule",
                "ground_truth": datapoint["instr"]["answer"]
            },
            "extra_info"  : {
                "split": "test",
                # "index": datapoint["idx"]
            },
        }

        tasks_for_ds.append(item)


    test_ds = datasets.Dataset.from_list(tasks_for_ds)
    test_ds.to_parquet(test_dataset_path)

    # from verl
    tokenizer = hf_tokenizer(model_name, trust_remote_code=False)
    processor = hf_processor(model_name, trust_remote_code=False, use_fast=True)

    data_cfg = OmegaConf.create({
        "prompt_key": "prompt",
        "image_key": "images",
        "max_prompt_length": max_input_tokens,
        "max_response_length": max_new_tokens,
        "train_batch_size": 128,
        "filter_overlong_prompts": False,
        "truncation": "error",
        "return_multi_modal_inputs": True,
    })

    # using rlhf dataset from verl to use the same preprocessing as during training

    dataset = RLHFDataset(test_dataset_path, tokenizer=tokenizer, processor=processor, config=data_cfg)

    # process the vllm inputs in parallel.
    ldr = DataLoader(dataset, batch_size=1, num_workers=os.cpu_count() // 2, shuffle=False, collate_fn=lambda xs: xs)

    vllm_inputs = []
    for batch in ldr:
        s = batch[0]
        vllm_inputs.append({"prompt_token_ids": s["raw_prompt_ids"], "multi_modal_data": s["multi_modal_data"]})

    # print(processor.decode(vllm_inputs[0]["prompt_token_ids"]))

    max_model_len = max_input_tokens + max_new_tokens
    llm = LLM(model=model_name, tensor_parallel_size=tensor_parallel_size, gpu_memory_utilization=gpu_memory_utilization, max_model_len=max_model_len, disable_mm_preprocessor_cache=True)
    params = SamplingParams(max_tokens=max_new_tokens, temperature=temperature)

    outputs = llm.generate(prompts=vllm_inputs, sampling_params=params)

    fail_ix = []
    if r1:
        for i, out in enumerate(outputs):
            if _needs_retry_r1(out):
                fail_ix.append(i)

    tries = 0
    while r1 and fail_ix and tries < 64:
        tries += 1
        retry_inputs = [vllm_inputs[i] for i in fail_ix]

        print(f"Retrying with {len(fail_ix)}")
        retry_outs = llm.generate(prompts=retry_inputs, sampling_params=params)

        new_fail_ix = []
        for j, gi in enumerate(fail_ix):
            outputs[gi] = retry_outs[j]

            if _needs_retry_r1(outputs[gi]):
                new_fail_ix.append(gi)

        fail_ix = new_fail_ix

    ret = []
    count_no_think_token = 0
    for task, output in zip(tasks, outputs):
        datapoint = task[0]
        output_text = output.outputs[0].text

        ans = {}
        
        if r1:
            answer, found = _extract_answer(output_text, r1_enabled=r1)
            if not found:
                count_no_think_token += 1
        else:
            answer = output_text

        # truncate overlong answers (after extracting if thinking model), otherwise downstream eval models might go OOM
        answer = answer[:4000]

        prompt = get_prompt(datapoint=datapoint, model_name=model_name, r1=r1, remove_think_suffixe=remove_think_suffixe)

        ans["full_output"] = output_text
        ans["output"] = answer
        ans["instr"] = prompt
        ans["answer"] = datapoint['instr']["answer"]

        for key in ["id", "idx", "img_path", "img", "labels", "label", "txt", "boxes", "bbox", "answer"]:
            if key in datapoint:
                ans[key] = datapoint[key]

        ret.append(ans)

    print(f"{count_no_think_token=}")
    del llm

    import gc, torch
    gc.collect()
    torch.cuda.empty_cache()

    return ret
