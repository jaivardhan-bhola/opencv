# This file is part of OpenCV project.
# It is subject to the license terms in the LICENSE file found in the top-level directory
# of this distribution and at http://opencv.org/license.html.
# Copyright (C) 2026, BigVision LLC, all rights reserved.
# Third party copyrights are property of their respective owners.

'''
This is a sample script to run Granite-Docling-258M vision-language inference in OpenCV
using ONNX models and OpenCV's new DNN engine (ENGINE_NEW). Given a page image and a
text prompt, it generates a "doctag" style text response describing the page (OCR text,
tables, form fields, and layout/section structure).

The model is split into three ONNX files:
    - Vision encoder : image tiles -> image features
    - Embed tokens    : prompt token ids -> text embeddings
    - Decoder         : [image features | text embeddings] -> logits (with KV-cache)

Model: https://huggingface.co/ibm-granite/granite-docling-258M
ONNX:  https://huggingface.co/onnx-community/granite-docling-258M-ONNX

Run the script:
1. Download the plain (non-quantized) fp32 ONNX export into <model_dir>, keeping the
   upstream layout (config.json, preprocessor_config.json, processor_config.json,
   tokenizer.json at the root; onnx/vision_encoder.onnx, onnx/embed_tokens.onnx,
   onnx/decoder_model_merged.onnx under onnx/):

2. Run the script:

    python granite_docling_inference.py --model_dir=<model_dir> \
                                        --input=<path-to-page-image>

'''

import math
import json
import argparse
import numpy as np
import cv2 as cv

DEFAULT_PROMPT = ('Convert this page to docling. Preserve OCR text, table structure, '
                   'form fields, and layout/section structure.')

def parse_args():
    parser = argparse.ArgumentParser(description='Use this script to run Granite-Docling-258M vision-language inference in OpenCV',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Path to the local onnx-community/granite-docling-258M-ONNX export '
                             '(config.json, preprocessor_config.json, processor_config.json, '
                             'onnx/{vision_encoder,embed_tokens,decoder_model_merged}.onnx).')
    parser.add_argument('--input', '-i', type=str, required=True, help='Path to the input page image.')
    parser.add_argument('--prompt', type=str, default=DEFAULT_PROMPT, help='Task prompt.')
    parser.add_argument('--max_new_tokens', type=int, default=512, help='Maximum number of new tokens to generate.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    return parser.parse_args()

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def resize(image, size):
    dst_w, dst_h = size
    src_h, src_w = image.shape[:2]
    shrinking = dst_w * dst_h < src_w * src_h
    return cv.resize(image, size, interpolation=cv.INTER_AREA if shrinking else cv.INTER_LANCZOS4)

def tile_image(image_bgr, longest_edge, tile_size, mean, std):
    h0, w0 = image_bgr.shape[:2]
    if w0 >= h0:
        new_w, new_h = longest_edge, max(1, round(longest_edge * h0 / w0))
    else:
        new_w, new_h = max(1, round(longest_edge * w0 / h0)), longest_edge
    resized = resize(image_bgr, (new_w, new_h))

    rows, cols = math.ceil(new_h / tile_size), math.ceil(new_w / tile_size)
    mean_arr = np.asarray(mean, dtype=np.float32)
    std_arr = np.asarray(std, dtype=np.float32)

    def normalize(tile_bgr):
        tile_rgb = cv.cvtColor(tile_bgr, cv.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return ((tile_rgb - mean_arr) / std_arr).transpose(2, 0, 1)  # HWC -> CHW

    grid = resize(resized, (cols * tile_size, rows * tile_size))
    pixel_values = np.zeros((rows * cols + 1, 3, tile_size, tile_size), dtype=np.float32)
    idx = 0
    for r in range(rows):
        for c in range(cols):
            y0, x0 = r * tile_size, c * tile_size
            pixel_values[idx] = normalize(grid[y0:y0 + tile_size, x0:x0 + tile_size])
            idx += 1
    thumbnail = resize(resized, (tile_size, tile_size))
    pixel_values[idx] = normalize(thumbnail)
    return pixel_values[np.newaxis], rows, cols

def build_prompt(rows, cols, image_seq_len, user_text):
    image_part = ''
    for h in range(rows):
        for w in range(cols):
            image_part += f'<fake_token_around_image><row_{h + 1}_col_{w + 1}>' + '<image>' * image_seq_len
        image_part += '\n'
    image_part += '\n<fake_token_around_image><global-img>' + '<image>' * image_seq_len + '<fake_token_around_image>'
    return f'<|start_of_role|>user<|end_of_role|>{image_part}{user_text}<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>'

def granite_docling_inference(vision_net, embed_net, decoder_net, pixel_values, prompt, max_new_tokens, tokenizer, image_token_id, eos_token_id):

    print('Inferencing Granite-Docling-258M model...')

    input_ids = np.array([tokenizer.encode(prompt)], dtype=np.int64)
    prompt_len = input_ids.shape[1]

    vision_net.setInput(pixel_values, 'pixel_values')
    vision_net.setInput(np.ones(pixel_values.shape[:2] + pixel_values.shape[3:], dtype=bool), 'pixel_attention_mask')
    image_features = vision_net.forward()

    embed_net.setInput(input_ids, 'input_ids')
    inputs_embeds = embed_net.forward()
    inputs_embeds[input_ids == image_token_id] = image_features.reshape(-1, image_features.shape[-1])

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

    print('Preparing Granite-Docling-258M model...')
    tokenizer = cv.dnn.Tokenizer.loadVLM(args.model_dir + '/', 'granite-docling')

    config = load_json(args.model_dir + '/config.json')
    preprocessor = load_json(args.model_dir + '/preprocessor_config.json')
    image_seq_len = load_json(args.model_dir + '/processor_config.json')['image_seq_len']
    image_token_id = int(config['image_token_id'])
    eos_token_id = int(config.get('text_config', {}).get('eos_token_id', config.get('eos_token_id')))

    vision_net  = cv.dnn.readNetFromONNX(args.model_dir + '/onnx/vision_encoder.onnx', cv.dnn.ENGINE_NEW)
    embed_net   = cv.dnn.readNetFromONNX(args.model_dir + '/onnx/embed_tokens.onnx', cv.dnn.ENGINE_NEW)
    decoder_net = cv.dnn.readNetFromONNX(args.model_dir + '/onnx/decoder_model_merged.onnx', cv.dnn.ENGINE_NEW)

    image = cv.imread(args.input)
    if image is None:
        raise IOError('Could not read image: ' + args.input)
    pixel_values, rows, cols = tile_image(
        image,
        preprocessor['size']['longest_edge'],
        preprocessor['max_image_size']['longest_edge'],
        preprocessor['image_mean'],
        preprocessor['image_std'],
    )
    prompt = build_prompt(rows, cols, image_seq_len, args.prompt)
    print(f'Prompt:\n{args.prompt}')

    generated = granite_docling_inference(vision_net, embed_net, decoder_net, pixel_values, prompt,
                                          args.max_new_tokens, tokenizer, image_token_id, eos_token_id)
    response = tokenizer.decode(generated)
    print(f'Response:\n{response}')
