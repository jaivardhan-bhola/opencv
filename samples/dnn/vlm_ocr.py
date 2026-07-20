# This file is part of OpenCV project.
# It is subject to the license terms in the LICENSE file found in the top-level directory
# of this distribution and at http://opencv.org/license.html.
# Third party copyrights are property of their respective owners.

'''
This is a sample script demonstrating cv2.vlm: a single API for running
vision-language OCR / document-understanding inference with either PaddleOCR-VL-1.5 or
Granite-Docling-258M, given a model type, a local ONNX export directory, and an input image.

Run the script:

    python vlm_ocr.py --model_type=paddleocr-vl --model_dir=<dir> --input=<path-to-image>
'''

import argparse
import cv2 as cv

MODEL_TYPES = {
    'paddleocr-vl': cv.vlm.VLM_MODEL_PADDLEOCR_VL,
    'granite-docling': cv.vlm.VLM_MODEL_GRANITE_DOCLING,
}

def parse_args():
    parser = argparse.ArgumentParser(description='Use this script to run vision-language OCR / '
                                                  'document-understanding inference in OpenCV',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_type', type=str, required=True, choices=sorted(MODEL_TYPES),
                        help='Which VLM to run.')
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Path to the local ONNX export directory for the chosen model_type.')
    parser.add_argument('--input', '-i', type=str, required=True, help='Path to the input image.')
    parser.add_argument('--prompt', type=str, default='', help="Task prompt (default: the model's built-in prompt).")
    parser.add_argument('--max_new_tokens', type=int, default=512, help='Maximum number of new tokens to generate.')
    parser.add_argument('--engine', type=str, default='new', choices=['new', 'ort'],
                        help='dnn engine used to load each ONNX sub-model.')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help='Compute device.')
    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()

    print(f'Preparing {args.model_type} model...')
    model = cv.vlm.create(MODEL_TYPES[args.model_type], args.model_dir,
                          args.engine, args.device)

    print(f'Running inference on {args.input}...')
    results = model.inferDocument(args.input, args.prompt, args.max_new_tokens)
    for i, text in enumerate(results):
        print(f'Page {i + 1}:\n{text}')
