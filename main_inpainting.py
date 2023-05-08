import engine


generator = engine.datasets.inpainting.MaskGenerator(masks_dir="engine/datasets/inpainting/masks")
generator.gen_masks(10, 10, 10)
