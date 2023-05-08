import engine


generator = engine.datasets.mixing.CutmixGen(
    source_name="StyleTransfer3k",
    config_path="/home/dani/repos/content_replication/engine/datasets/mixing/Cutmix/config_CutmixGen_WikiArt.json",
    images_dir="/home/dani/repos/content_replication/datasets/Cutmix_WikiArt",
)
generator.generate_config(num_examples=1000)
generator.generate_images()
