import importlib.resources
from pathlib import Path


# constants related to data paths
DATA_URL = "https://tufts.box.com/shared/static/fq0awbrahmsr97zetqo1v2uz5rjkvon6.zip"
with importlib.resources.path("polycraft_nov_data", "dataset") as dataset_root:
    DATASET_ROOT = Path(dataset_root)
DATASET_TARGETS = DATASET_ROOT / Path("targets.csv")
# constants related to shape of data
IMAGE_SHAPE = (3, 256, 256)
PATCH_SHAPE = (3, 32, 32)
# constant for (inclusive) minimum percentage of image taken up by novel object
NOV_THRESH = .01
# constants related to classes
NORMAL_CLASSES = [
    "normal",
    "fence",
    "item_bedrock",
    "item_sand",
    "item_coal_block",
]
NOVEL_CLASSES = [
    "tree_easy",
    "item_bookshelf",
    "item_deadbush",
    "item_cauldron",
    "item_netherrack",
    "item_enchanting_table",
    "item_jukebox",
    "item_anvil",
    "item_snow",
    "item_lapis_block",
    "item_stonebrick",
    "item_cake",
    "item_noteblock",
    "item_quartz_block",
    "item_obsidian",
    "item_prismarine",
    "item_tnt",
    "item_sea_lantern",
    "item_wheat",
    "item_mycelium",
    "item_waterlily",
    "item_dropper",
    "item_reeds",
    "item_bed",
    "item_glowstone",
    "item_lever",
    "item_torch",
    "item_cobblestone",
    "item_planks",
    "item_slime",
    "item_beacon",
    "item_iron_block",
    "item_clay",
    "item_ice",
    "item_sponge",
    "item_sandstone",
    "item_dispenser",
    "item_tallgrass",
    "item_cactus",
    "item_gravel",
    "item_piston",
    "item_web",
    "item_hopper",
    "item_stone",
    "item_soul_sand",
    "item_brewing_stand",
    "item_pumpkin",
    "item_wool",
    "item_vine",
    "item_emerald_block",
]
