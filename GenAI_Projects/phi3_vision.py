# Requirements:
# flash_attn
# numpy
# Pillow
# Requests
# torch
# torchvision
# transformers



import warnings
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
import os


def load_model_and_processor(model_id="microsoft/Phi-3.5-vision-instruct"):
    """
    Load the model and processor. This should be done once and the results reused.
    
    Args:
    model_id (str): The ID of the model to load.
    
    Returns:
    tuple: The loaded model and processor.
    """
    warnings.filterwarnings("ignore")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="cpu", #change to cuda when using gpu
        trust_remote_code=True,
        torch_dtype="auto",
        _attn_implementation='eager' # set as "flash_attention_2" when using gpu
    )
    
    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True,
        num_crops=4
    )

    print("Loading the model is completed and retuned model and processor")

    return model, processor



def load_image(image_source):
    """
    Load an image from a file path or use a direct image object.
    
    Args:
    image_source (str or PIL.Image): The path to the image file or a PIL Image object.
    
    Returns:
    PIL.Image: The loaded or provided image.
    """
    if isinstance(image_source, str):
        if os.path.isfile(image_source):
            return Image.open(image_source)
        else:
            raise FileNotFoundError(f"The file {image_source} does not exist.")
    elif isinstance(image_source, Image.Image):
        return image_source
    else:
        raise ValueError("image_source must be either a file path or a PIL Image object.")




def process_image_and_prompt(processor, image, prompt):
    """
    Process the image and prompt.
    
    Args:
    processor: The loaded processor.
    image (PIL.Image): The image to process.
    prompt (str): The prompt to use.
    
    Returns:
    dict: The processed inputs.
    """
    messages = [
        {"role": "user", "content": "<|image_1|>\n" + prompt},
    ]

    chat_prompt = processor.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = processor(chat_prompt, [image], return_tensors="pt").to("cpu")  #.to(cuda:0) when using gpu
    return inputs


def generate_response(model, processor, inputs, max_new_tokens=1000, temperature=0.0):
    """
    Generate a response based on the inputs.
    
    Args:
    model: The loaded model.
    processor: The loaded processor.
    inputs (dict): The processed inputs.
    max_new_tokens (int): The maximum number of new tokens to generate.
    temperature (float): The temperature for generation.
    
    Returns:
    str: The generated response.
    """

    print("generating the response started")


    generation_args = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "do_sample": False,
    }

    generate_ids = model.generate(
        **inputs,
        eos_token_id=processor.tokenizer.eos_token_id,
        **generation_args
    )

    generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
    response = processor.batch_decode(
        generate_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    return response




class ImageAnalyzer:
    def __init__(self, model_id="microsoft/Phi-3.5-vision-instruct"):
        """
        Initialize the ImageAnalyzer with a model and processor.
        
        Args:
        model_id (str): The ID of the model to load.
        """
        print("loading and processing of model has been completed")
        self.model, self.processor = load_model_and_processor(model_id)
    
    def analyze(self, image_source, prompt, max_new_tokens=1000, temperature=0.0):
        """
        Analyze an image with a custom prompt.
        
        Args:
        image_source (str or PIL.Image): The path to the image file or a PIL Image object.
        prompt (str): The custom prompt to use for analysis.
        max_new_tokens (int): The maximum number of new tokens to generate.
        temperature (float): The temperature for generation.
        
        Returns:
        str: The generated analysis.
        """

        image = load_image(image_source)
        inputs = process_image_and_prompt(self.processor, image, prompt)
        response = generate_response(self.model, self.processor, inputs, max_new_tokens, temperature)
        return response




# # Example usage:
analyzer = ImageAnalyzer()

# # Analyze a local image file
# analysis1 = analyzer.analyze("/path/to/image1.jpg", "Describe the main elements in this image.")
# print(analysis1)

# Analyze a PIL Image object
from PIL import Image
img = Image.open(r"C:\Users\deekshith.p\EYMP_Doc_Rec_Extraction\Datasets\Test\Passport\2.jpg")
img.show()
analysis2 = analyzer.analyze(img, "Issue place of this passport")
print(analysis2)
