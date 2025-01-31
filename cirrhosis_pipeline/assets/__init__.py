from dagster import Definitions, load_assets_from_modules
from cirrhosis_pipeline.assets import data_asset, model_asset

all_assets = load_assets_from_modules([data_asset, model_asset])

defs = Definitions(
    assets=all_assets,
)