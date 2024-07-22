#!/usr/bin/env python3
# Copyright 2024 Marc-Antoine Ruel. All rights reserved.
# Use of this source code is governed under the Apache License, Version 2.0
# that can be found in the LICENSE file.

"""Run PhotoMakerv2."""

import argparse
import datetime
import os
import sys
import time

import diffusers
import diffusers.utils
import huggingface_hub
import numpy as np
import photomaker
import torch


def load_model(device, use_style):
  print("- load_model(%s)" % device)
  start = time.time()
  torch_dtype = torch.float16
  if device != "mps" and torch.cuda.is_bf16_supported():
    # What about AVX-512...
    torch_dtype = torch.bfloat16
  # https://github.com/TencentARC/PhotoMaker/blob/main/README_pmv2.md
  photomaker_path = huggingface_hub.hf_hub_download(
      repo_id="TencentARC/PhotoMaker-V2", filename="photomaker-v2.bin",
      repo_type="model")
  #root = "SG161222/RealVisXL_V4.0"
  root = "segmind/SSD-1B"
  pipe = photomaker.PhotoMakerStableDiffusionXLPipeline.from_pretrained(
        root, torch_dtype=torch_dtype).to(device)
  if use_style:
    pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models",
                        weight_name="ip-adapter_sdxl.bin")
    pipe.set_ip_adapter_scale(0.7)
  #pipe.load_lora_weights("latent-consistency/lcm-lora-ssd-1b")
  # Load PhotoMaker checkpoint.
  pipe.load_photomaker_adapter(
      os.path.dirname(photomaker_path),
      subfolder="",
      weight_name=os.path.basename(photomaker_path),
      trigger_word="img"  # define the trigger word
  )
  #pipe.set_adapters(["lcm-lora", "photomaker"], adapter_weights=[1.0, 1.0])
  pipe.fuse_lora()
  pipe.scheduler = diffusers.EulerDiscreteScheduler.from_config(pipe.scheduler.config)
  if device == "cuda":
    pipe.enable_model_cpu_offload()
  print(f"  done in {time.time()-start:.1f}s")
  return pipe


def load_images(device, images):
  print("- load_images(%s)" % images)
  input_id_images = [diffusers.utils.load_image(i) for i in images]

  # TODO: Why no auto-detect?
  p = ["CPUExecutionProvider"]
  if device == "mps":
    # https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html
    p = ["CoreMLExecutionProvider"]
  elif device == "cuda":
    # https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html
    p = ["CUDAExecutionProvider"]
  face_detector = photomaker.FaceAnalysis2(
      providers=p, allowed_modules=['detection', 'recognition'])
  face_detector.prepare(ctx_id=0, det_size=(640, 640))

  id_embed_list = []
  for img in input_id_images:
    img = np.array(img)
    img = img[:, :, ::-1]
    faces = photomaker.analyze_faces(face_detector, img)
    if len(faces) > 0:
      id_embed_list.append(torch.from_numpy((faces[0]['embedding'])))
  if not id_embed_list:
      raise ValueError("No face detected in input image pool")
  id_embeds = torch.stack(id_embed_list)
  return input_id_images, id_embeds


def generate(pipe, device, images, style_images, prompt, num_steps, num_images,
             seed):
  start = time.time()
  input_id_images, id_embeds = load_images(device, images)
  style_images = [diffusers.utils.load_image(s) for s in style_images]

  print("- generate")
  negative_prompt = "(asymmetry, worst quality, low quality, illustration, 3d, 2d, painting, cartoons, sketch), open mouth, grayscale"
  generator = torch.Generator(device=device).manual_seed(seed)
  kwargs = {}
  if style_images:
    kwargs["ip_adapter_image"] = [style_images]
  images = pipe(
      prompt=prompt,
      negative_prompt=negative_prompt,
      input_id_images=input_id_images,
      id_embeds=id_embeds,
      num_images_per_prompt=num_images,
      num_inference_steps=num_steps,
      start_merge_step=10,
      generator=generator,
      width=1216,
      height=832,
      **kwargs).images
  print(f"  done in {time.time()-start:.1f}s")
  return images


def get_device():
  if torch.cuda.is_available():
      return "cuda"
  if sys.platform == "darwin" and torch.backends.mps.is_available():
    return "mps"
  return "cpu"


def main():
  parser = argparse.ArgumentParser(description=sys.modules[__name__].__doc__)
  parser.add_argument(
      "-p", "--prompt",
      default="a half-body portrait of a man img in tuxedo, best quality",
      help="Note that the trigger word `img` must follow the class word for personalization")
  group = parser.add_mutually_exclusive_group(required=True)
  group.add_argument(
      "-i", "--input", action="append", default=[], help="Input images")
  group.add_argument(
      "--input-dir", help="Directory containing input images")
  parser.add_argument(
      "-s", "--style", action="append", default=[],
      help="Style images (optional)")
  parser.add_argument(
      "-n", "--num-steps", default=20, type=int,
      help="Number of steps, higher is better, don't use more than 50. 40 is "+
           "recommended but takes a while.")
  parser.add_argument(
      "--num-images", default=4, type=int,
      help="Number of images to generate at a time")
  parser.add_argument(
      "--seed", default=1, type=int,
      help="Seed to use when generating images")
  args = parser.parse_args()
  if args.input_dir:
    args.input = [
        os.path.join(args.input_dir, i) for i in os.listdir(args.input_dir)
        if i.lower().endswith((".jpg", ".png"))
    ]
  device = get_device()
  pipe = load_model(device, bool(args.style))
  images = generate(pipe, device, args.input, args.style, args.prompt,
                    args.num_steps, args.num_images, args.seed)
  now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
  for i, image in enumerate(images):
    name = f"photomaker-{now}-{i}.png"
    print(f"- Saving {name}")
    image.save(name)
  return 0


if __name__ == "__main__":
  sys.exit(main())
