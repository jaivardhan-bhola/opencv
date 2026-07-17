# This file is part of OpenCV project.
# It is subject to the license terms in the LICENSE file found in the top-level directory
# of this distribution and at http://opencv.org/license.html.


'''
This is a sample script to run PaddleOCR-VL-1.5 vision-language inference in OpenCV
using ONNX models and OpenCV's new DNN engine (ENGINE_NEW). Given an image and a text
prompt, it generates a text response (e.g. recognized document text).

The model is split into three ONNX files:
    - Vision encoder : image patches -> image embeddings
    - Embedding      : prompt token ids -> text embeddings
    - Decoder        : [image embeddings | text embeddings] -> logits (with KV-cache)

Model: https://huggingface.co/PaddlePaddle/PaddleOCR-VL
ONNX:  https://huggingface.co/onnx-community/PaddleOCR-VL-1.5-ONNX

Run the script:
1. Download the plain (non-quantized) fp32 ONNX export into <model_dir>, keeping the
   upstream layout (config.json, tokenizer.json, processor_config.json at the root;
   onnx/vision_encoder.onnx, onnx/embedding.onnx, onnx/decoder.onnx under onnx/):

2. Run the script:

    python paddleocr_vl_inference.py --model_dir=<model_dir> \
                                     --input=<path-to-image>
'''

import json
import argparse
import numpy as np
import cv2 as cv

def parse_args():
    parser = argparse.ArgumentParser(description='Use this script to run PaddleOCR-VL-1.5 vision-language inference in OpenCV',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Path to the local onnx-community/PaddleOCR-VL-1.5-ONNX export '
                             '(config.json, processor_config.json, '
                             'onnx/{vision_encoder,embedding,decoder}.onnx).')
    parser.add_argument('--input', '-i', type=str, required=True, help='Path to the input image.')
    parser.add_argument('--prompt', type=str, default='OCR', help='Task prompt.')
    parser.add_argument('--max_new_tokens', type=int, default=512, help='Maximum number of new tokens to generate.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    return parser.parse_args()

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def config_value(config, name, default=None):
    value = config.get(name)
    if value is not None:
        return value
    return (config.get('text_config') or {}).get(name, default)

def smart_resize(height, width, factor, min_pixels, max_pixels):
    '''Qwen2VL-style resize: round to a multiple of factor while keeping the pixel budget
    within [min_pixels, max_pixels].'''
    if height < factor:
        width = round((width * factor) / height)
        height = factor
    if width < factor:
        height = round((height * factor) / width)
        width = factor
    if max(height, width) / min(height, width) > 200:
        raise ValueError(f'absolute aspect ratio is too large: {height}x{width}')

    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = np.sqrt((height * width) / max_pixels)
        h_bar = int(np.floor(height / beta / factor) * factor)
        w_bar = int(np.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = np.sqrt(min_pixels / (height * width))
        h_bar = int(np.ceil(height * beta / factor) * factor)
        w_bar = int(np.ceil(width * beta / factor) * factor)
    return h_bar, w_bar

def preprocess_image(image_path, preprocessor):
    '''Smart-resize then flatten into merge_size x merge_size patches, matching the
    Qwen2VL-style image processor this vision encoder was exported with.'''
    patch_size = int(preprocessor.get('patch_size', 14))
    merge_size = int(preprocessor.get('merge_size', 2))
    temporal_patch_size = int(preprocessor.get('temporal_patch_size', 1))
    min_pixels = int(preprocessor.get('min_pixels', 28 * 28 * 130))
    max_pixels = int(preprocessor.get('max_pixels', 28 * 28 * 1280))
    image_mean = np.asarray(preprocessor.get('image_mean', [0.5, 0.5, 0.5]), dtype=np.float32)
    image_std = np.asarray(preprocessor.get('image_std', [0.5, 0.5, 0.5]), dtype=np.float32)
    rescale_factor = float(preprocessor.get('rescale_factor', 1 / 255))

    image = cv.imread(image_path)
    if image is None:
        raise IOError('Could not read image: ' + image_path)
    height, width = image.shape[:2]
    resized_height, resized_width = smart_resize(height, width, patch_size * merge_size, min_pixels, max_pixels)
    image = cv.resize(image, (resized_width, resized_height), interpolation=cv.INTER_CUBIC)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB).astype(np.float32) * rescale_factor
    image = (image - image_mean.reshape(1, 1, 3)) / image_std.reshape(1, 1, 3)

    patches = image.transpose(2, 0, 1)[np.newaxis]
    if patches.shape[0] == 1:
        patches = np.tile(patches, (temporal_patch_size, 1, 1, 1))

    channel = patches.shape[1]
    grid_t = patches.shape[0] // temporal_patch_size
    grid_h = resized_height // patch_size
    grid_w = resized_width // patch_size
    patches = patches.reshape(grid_t, temporal_patch_size, channel, grid_h, patch_size, grid_w, patch_size)
    patches = patches.transpose(0, 3, 5, 2, 1, 4, 6)
    flatten_patches = patches.reshape(grid_t * grid_h * grid_w, channel, patch_size, patch_size).astype(np.float32, copy=False)
    pixel_values = flatten_patches[np.newaxis]
    image_grid_thw = np.asarray([[grid_t, grid_h, grid_w]], dtype=np.int64)
    return pixel_values, image_grid_thw

def build_prompt(prompt, image_token_repeats):
    return ('<|begin_of_sentence|>User: <|IMAGE_START|>'
            + '<|IMAGE_PLACEHOLDER|>' * image_token_repeats
            + '<|IMAGE_END|>' + prompt + '\nAssistant:\n')

def paddleocr_vl_inference(vision_net, embed_net, decoder_net, pixel_values, image_grid_thw, prompt,
                           max_new_tokens, tokenizer, image_token_id, eos_token_id, merge_size):

    print('Inferencing PaddleOCR-VL-1.5 model...')

    image_token_repeats = int(np.prod(image_grid_thw[0]) // merge_size // merge_size)
    input_ids = np.array([tokenizer.encode(build_prompt(prompt, image_token_repeats))], dtype=np.int64)
    prompt_len = input_ids.shape[1]

    vision_net.setInput(pixel_values, 'pixel_values')
    vision_net.setInput(image_grid_thw, 'image_grid_thw')
    image_embeds = vision_net.forward()

    embed_net.setInput(input_ids, 'input_ids')
    inputs_embeds = embed_net.forward()
    image_positions = input_ids == image_token_id
    inputs_embeds[image_positions] = image_embeds.reshape(-1, image_embeds.shape[-1])

    decoder_net.enableKVCache()
    attention_mask = np.ones((1, prompt_len), dtype=np.int64)

    decoder_net.setInput(inputs_embeds, 'inputs_embeds')
    decoder_net.setInput(attention_mask, 'attention_mask')
    logits = decoder_net.forward()
    new_id = int(np.argmax(logits[:, -1, :].reshape(-1)))
    generated = [new_id]

    for _ in range(max_new_tokens - 1):
        if new_id == eos_token_id:
            break
        embed_net.setInput(np.array([[new_id]], dtype=np.int64), 'input_ids')
        new_embed = embed_net.forward()
        attention_mask = np.concatenate([attention_mask, [[1]]], axis=-1)
        decoder_net.setInput(new_embed, 'inputs_embeds')
        decoder_net.setInput(attention_mask, 'attention_mask')
        logits = decoder_net.forward()
        new_id = int(np.argmax(logits[:, -1, :].reshape(-1)))
        generated.append(new_id)

    if generated and generated[-1] == eos_token_id:
        generated.pop()

    return generated

if __name__ == '__main__':

    args = parse_args()
    np.random.seed(args.seed)

    print('Preparing PaddleOCR-VL-1.5 model...')
    tokenizer = cv.dnn.Tokenizer.loadVLM(args.model_dir + '/', 'paddleocr-vl')

    config = load_json(args.model_dir + '/config.json')
    processor = load_json(args.model_dir + '/processor_config.json')['image_processor']
    image_token_id = int(config_value(config, 'image_token_id'))
    eos_token_id = int(config_value(config, 'eos_token_id', 2))
    merge_size = int(processor.get('merge_size', 2))

    vision_net  = cv.dnn.readNetFromONNX(args.model_dir + '/onnx/vision_encoder.onnx', cv.dnn.ENGINE_NEW)
    embed_net   = cv.dnn.readNetFromONNX(args.model_dir + '/onnx/embedding.onnx', cv.dnn.ENGINE_NEW)
    decoder_net = cv.dnn.readNetFromONNX(args.model_dir + '/onnx/decoder.onnx', cv.dnn.ENGINE_NEW)

    pixel_values, image_grid_thw = preprocess_image(args.input, processor)
    print(f'Prompt:\n{args.prompt}')

    generated = paddleocr_vl_inference(vision_net, embed_net, decoder_net, pixel_values, image_grid_thw,
                                       args.prompt, args.max_new_tokens, tokenizer, image_token_id,
                                       eos_token_id, merge_size)
    response = tokenizer.decode(generated)
    print(f'Response:\n{response}')
