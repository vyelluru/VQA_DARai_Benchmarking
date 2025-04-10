import torch
from torchvision.transforms import ToPILImage
from transformers import BitsAndBytesConfig, LlavaNextVideoForConditionalGeneration, LlavaNextVideoProcessor


def load_model_and_processor(model_name="llava-hf/LLaVA-NeXT-Video-7B-hf"):
    """
    Loads the LlavaNextVideo model and its processor.

    Returns:
        model, processor: The loaded model and processor.
    """
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )

    processor = LlavaNextVideoProcessor.from_pretrained("llava-hf/LLaVA-NeXT-Video-7B-hf")
    model = LlavaNextVideoForConditionalGeneration.from_pretrained(
        "llava-hf/LLaVA-NeXT-Video-7B-hf",
        quantization_config=quantization_config,
        # device_map='auto'
        device_map= {"": "cuda:0"}
    )
    return model, processor


def load_model_and_processor_instruct_blip_video(model_name="Salesforce/instructblip-vicuna-7b"):
    '''
        Loads the InstructBlipVideo model and its processor.
        Returns: model, processor: The loaded model and processor
    '''

    processor = InstructBlipVideoProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b", device_map= {"": "cuda:0"})
    model = InstructBlipVideoForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-7b")

    return model, processor
    

def generate_answer(instance, question, processor, model, max_new_tokens=100):
    """
    Generates an answer from the LlavaNextVideo model based on a given question and a video sample.
    Returns:
        str: The cleaned answer (only the text after 'ASSISTANT:').
    """
    # Unpack the sample
    frames, activity, camera, (subject_id, session_id) = instance

    # Convert each NumPy frame to a PIL Image
    to_pil = ToPILImage()
    frame_images = [to_pil(frame) for frame in frames]

    # Construct the prompt using the provided question
    prompt = (
        f"USER: <video>\n"
        f"Question: {question}\n"
        f"ASSISTANT:"
    )

    # Process the prompt and video frames with the processor.
    # Note: both videos and text must be wrapped in lists.
    inputs = processor(
        videos=[frame_images],
        text=[prompt],
        padding=True,
        return_tensors="pt"
    ).to(model.device)

    # Generate the answer from the model
    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    # Decode the generated tokens into a string
    full_answer = processor.batch_decode(output_ids, skip_special_tokens=True)[0]

    # Keep only the content after 'ASSISTANT:'
    marker = "ASSISTANT:"
    idx = full_answer.find(marker)
    cleaned_answer = full_answer if idx == -1 else full_answer[idx + len(marker):].strip()

    return cleaned_answer



def generate_answer_instruct_blip_video(instance, question, processor, model, max_new_tokens=100):
    # Unpack the sample
    frames, activity, camera, (subject_id, session_id) = instance

    # Convert each NumPy frame to a PIL Image
    to_pil = ToPILImage()
    frame_images = [to_pil(frame) for frame in frames]

    frame_images = random.sample(frame_images, 4)

    prompt = question
    inputs = processor(text=prompt, images=video_frames, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        do_sample=False,
        num_beams=5,
        max_length=max_new_tokens,
        repetition_penalty=1.5,
        length_penalty=1.0,
    )
    cleaned_answer = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
    return cleaned_answer
