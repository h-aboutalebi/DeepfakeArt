import engine

generator = engine.datasets.mixing.SegmixGen(
    source_name="VOCSegmentation",
    config_path="/home/dani/repos/content_replication/engine/datasets/mixing/Segmix/config_Segmix_VOCSegmentation.json",
    images_dir="/home/dani/repos/content_replication/datasets/Segmix_VOCSegmentation",
)
generator.generate_config(num_examples=1000)
generator.generate_images()
