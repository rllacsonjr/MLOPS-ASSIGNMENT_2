from dagster import Definitions, load_assets_from_modules
from . import assets.data_asset, assets.model_asset

all_assets = load_assets_from_modules([
    assets.data_asset,
    assets.model_asset
])

defs = Definitions(
    assets=all_assets,
)