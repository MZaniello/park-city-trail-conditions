from pathlib import Path
import json
import math
import shutil
import tempfile
import time

import geopandas as gpd
import numpy as np
import rasterio
import requests

from rasterio.merge import merge
from rasterio.mask import mask
from shapely.geometry import box, mapping


# ============================================================
# SETTINGS
# ============================================================

USGS_API_URL = (
    "https://tnmaccess.nationalmap.gov/api/v1/products"
)

DATASET_NAME = (
    "National Elevation Dataset (NED) 1/3 arc-second"
)

BUFFER_DEGREES = 0.01

MAX_RETRIES = 5

REQUEST_TIMEOUT = 120


# ============================================================
# HELPERS
# ============================================================


def get_project_paths():
    """
    Return important project paths.
    """

    project_root = (
        Path(__file__)
        .resolve()
        .parents[2]
    )

    trail_path = (
        project_root
        / "data"
        / "processed"
        / "master_trail_geometries.geojson"
    )

    output_path = (
        project_root
        / "data"
        / "raw"
        / "expanded_park_city_dem_10m.tif"
    )

    return (
        project_root,
        trail_path,
        output_path,
    )


def get_bounds(trails):
    """
    Get buffered geographic bounds from trail geometry.
    """

    trails_wgs84 = trails.to_crs(
        "EPSG:4326"
    )

    minx, miny, maxx, maxy = (
        trails_wgs84.total_bounds
    )

    minx -= BUFFER_DEGREES
    miny -= BUFFER_DEGREES
    maxx += BUFFER_DEGREES
    maxy += BUFFER_DEGREES

    return (
        minx,
        miny,
        maxx,
        maxy,
    )


def request_json(
    url,
    params,
):
    """
    Request JSON with retry / backoff.
    """

    for attempt in range(
        MAX_RETRIES
    ):

        try:

            response = requests.get(
                url,
                params=params,
                timeout=REQUEST_TIMEOUT,
            )

            if response.status_code == 200:

                return response.json()

            if response.status_code in {
                429,
                500,
                502,
                503,
                504,
            }:

                wait_seconds = (
                    2 ** attempt
                ) * 3

                print(
                    f"  Temporary USGS/API error "
                    f"{response.status_code}. "
                    f"Waiting {wait_seconds}s..."
                )

                time.sleep(
                    wait_seconds
                )

                continue

            response.raise_for_status()

        except requests.RequestException:

            if (
                attempt
                == MAX_RETRIES - 1
            ):
                raise

            wait_seconds = (
                2 ** attempt
            ) * 3

            print(
                f"  Request failed. "
                f"Waiting {wait_seconds}s..."
            )

            time.sleep(
                wait_seconds
            )

    raise RuntimeError(
        "USGS request failed after retries."
    )


def query_usgs_products(bounds):
    """
    Query The National Map for 1/3 arc-second DEM products
    intersecting the requested bounding box.
    """

    west, south, east, north = (
        bounds
    )

    bbox_string = (
        f"{west},"
        f"{south},"
        f"{east},"
        f"{north}"
    )

    params = {
        "datasets":
            DATASET_NAME,

        "bbox":
            bbox_string,

        "outputFormat":
            "JSON",

        "max":
            1000,
    }

    print(
        "Querying USGS National Map..."
    )

    data = request_json(
        USGS_API_URL,
        params,
    )

    items = data.get(
        "items",
        []
    )

    return items


def choose_geotiff_products(items):
    """
    Extract GeoTIFF download URLs and remove duplicates.

    Prefer downloadURL, but fall back to URLs supplied by
    the API if necessary.
    """

    products = []

    seen_urls = set()

    for item in items:

        title = str(
            item.get(
                "title",
                ""
            )
        )

        urls = []

        for key in [
            "downloadURL",
            "url",
        ]:

            value = item.get(
                key
            )

            if value:
                urls.append(
                    value
                )

        for url in urls:

            url = str(
                url
            ).strip()

            lower = url.lower()

            if not (
                lower.endswith(
                    ".tif"
                )
                or ".tif?" in lower
                or lower.endswith(
                    ".zip"
                )
            ):
                continue

            if url in seen_urls:
                continue

            seen_urls.add(
                url
            )

            products.append(
                {
                    "title":
                        title,
                    "url":
                        url,
                }
            )

            break

    return products


def download_file(
    url,
    destination,
):
    """
    Download one file with retries.
    """

    for attempt in range(
        MAX_RETRIES
    ):

        try:

            with requests.get(
                url,
                stream=True,
                timeout=REQUEST_TIMEOUT,
            ) as response:

                if (
                    response.status_code
                    == 200
                ):

                    with open(
                        destination,
                        "wb",
                    ) as file:

                        shutil.copyfileobj(
                            response.raw,
                            file,
                        )

                    return

                if (
                    response.status_code
                    in {
                        429,
                        500,
                        502,
                        503,
                        504,
                    }
                ):

                    wait_seconds = (
                        2 ** attempt
                    ) * 3

                    print(
                        "    Temporary error "
                        f"{response.status_code}. "
                        f"Waiting {wait_seconds}s..."
                    )

                    time.sleep(
                        wait_seconds
                    )

                    continue

                response.raise_for_status()

        except requests.RequestException:

            if (
                attempt
                == MAX_RETRIES - 1
            ):
                raise

            wait_seconds = (
                2 ** attempt
            ) * 3

            print(
                f"    Download failed. "
                f"Waiting {wait_seconds}s..."
            )

            time.sleep(
                wait_seconds
            )

    raise RuntimeError(
        f"Could not download {url}"
    )


def inspect_raster(path):
    """
    Validate that a downloaded file is a readable raster.
    """

    try:

        with rasterio.open(
            path
        ) as src:

            if (
                src.width <= 0
                or src.height <= 0
            ):
                return False

            return True

    except Exception:

        return False


def mosaic_tiles(
    raster_paths,
    mosaic_path,
):
    """
    Merge downloaded DEM tiles.
    """

    sources = []

    try:

        for path in raster_paths:

            sources.append(
                rasterio.open(
                    path
                )
            )

        mosaic, transform = merge(
            sources
        )

        metadata = (
            sources[0]
            .meta
            .copy()
        )

        metadata.update(
            {
                "driver":
                    "GTiff",

                "height":
                    mosaic.shape[1],

                "width":
                    mosaic.shape[2],

                "transform":
                    transform,

                "compress":
                    "deflate",
            }
        )

        with rasterio.open(
            mosaic_path,
            "w",
            **metadata,
        ) as destination:

            destination.write(
                mosaic
            )

    finally:

        for source in sources:
            source.close()


def clip_mosaic(
    mosaic_path,
    bounds,
    output_path,
):
    """
    Clip mosaic to buffered study bounds.
    """

    west, south, east, north = (
        bounds
    )

    bounds_geometry = box(
        west,
        south,
        east,
        north,
    )

    with rasterio.open(
        mosaic_path
    ) as source:

        geometry_gdf = (
            gpd.GeoDataFrame(
                geometry=[
                    bounds_geometry
                ],
                crs="EPSG:4326",
            )
        )

        geometry_gdf = (
            geometry_gdf
            .to_crs(
                source.crs
            )
        )

        shapes = [
            mapping(
                geometry
            )
            for geometry
            in geometry_gdf.geometry
        ]

        clipped, transform = mask(
            source,
            shapes,
            crop=True,
        )

        metadata = (
            source.meta
            .copy()
        )

        metadata.update(
            {
                "driver":
                    "GTiff",

                "height":
                    clipped.shape[1],

                "width":
                    clipped.shape[2],

                "transform":
                    transform,

                "compress":
                    "deflate",
            }
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with rasterio.open(
            output_path,
            "w",
            **metadata,
        ) as destination:

            destination.write(
                clipped
            )


# ============================================================
# MAIN
# ============================================================


def main():

    (
        project_root,
        trail_path,
        output_path,
    ) = get_project_paths()

    # ---------------------------------------------------------
    # LOAD TRAILS
    # ---------------------------------------------------------

    print(
        "Loading master trail geometries..."
    )

    trails = gpd.read_file(
        trail_path
    )

    print(
        f"Trails found: "
        f"{len(trails):,}"
    )

    # ---------------------------------------------------------
    # STUDY BOUNDS
    # ---------------------------------------------------------

    bounds = get_bounds(
        trails
    )

    west, south, east, north = (
        bounds
    )

    print()
    print(
        "Buffered DEM bounds:"
    )

    print(
        f"  west:  {west:.6f}"
    )

    print(
        f"  south: {south:.6f}"
    )

    print(
        f"  east:  {east:.6f}"
    )

    print(
        f"  north: {north:.6f}"
    )

    # ---------------------------------------------------------
    # QUERY USGS
    # ---------------------------------------------------------

    items = query_usgs_products(
        bounds
    )

    print()
    print(
        f"USGS products returned: "
        f"{len(items):,}"
    )

    products = (
        choose_geotiff_products(
            items
        )
    )

    print(
        f"Downloadable DEM products: "
        f"{len(products):,}"
    )

    if not products:

        print()
        print(
            "No downloadable GeoTIFF products "
            "were found."
        )

        print(
            "The USGS API response structure "
            "may have changed."
        )

        print()
        print(
            "First returned item:"
        )

        if items:

            print(
                json.dumps(
                    items[0],
                    indent=2,
                    default=str,
                )
            )

        raise RuntimeError(
            "No usable DEM downloads found."
        )

    # ---------------------------------------------------------
    # TEMP DIRECTORY
    # ---------------------------------------------------------

    with tempfile.TemporaryDirectory(
        prefix="park_city_dem_"
    ) as temp_directory:

        temp_directory = Path(
            temp_directory
        )

        downloaded_rasters = []

        # -----------------------------------------------------
        # DOWNLOAD TILES
        # -----------------------------------------------------

        print()
        print(
            "Downloading DEM tiles..."
        )

        for number, product in enumerate(
            products,
            start=1,
        ):

            url = product[
                "url"
            ]

            suffix = (
                ".zip"
                if url.lower()
                .split("?")[0]
                .endswith(".zip")
                else ".tif"
            )

            destination = (
                temp_directory
                / f"dem_{number:03d}{suffix}"
            )

            print(
                f"[{number}/{len(products)}] "
                f"{product['title']}"
            )

            download_file(
                url,
                destination,
            )

            # -------------------------------------------------
            # HANDLE ZIP FILES
            # -------------------------------------------------

            if (
                destination.suffix.lower()
                == ".zip"
            ):

                import zipfile

                extract_folder = (
                    temp_directory
                    / f"tile_{number:03d}"
                )

                extract_folder.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                with zipfile.ZipFile(
                    destination
                ) as archive:

                    archive.extractall(
                        extract_folder
                    )

                tif_files = list(
                    extract_folder.rglob(
                        "*.tif"
                    )
                )

                for tif_path in tif_files:

                    if inspect_raster(
                        tif_path
                    ):

                        downloaded_rasters.append(
                            tif_path
                        )

            else:

                if inspect_raster(
                    destination
                ):

                    downloaded_rasters.append(
                        destination
                    )

        print()
        print(
            f"Readable raster tiles: "
            f"{len(downloaded_rasters):,}"
        )

        if not downloaded_rasters:

            raise RuntimeError(
                "Downloads completed, but no readable "
                "DEM rasters were found."
            )

        # -----------------------------------------------------
        # MOSAIC
        # -----------------------------------------------------

        mosaic_path = (
            temp_directory
            / "expanded_dem_mosaic.tif"
        )

        print()
        print(
            "Mosaicking DEM tiles..."
        )

        mosaic_tiles(
            downloaded_rasters,
            mosaic_path,
        )

        # -----------------------------------------------------
        # CLIP
        # -----------------------------------------------------

        print(
            "Clipping DEM to trail study area..."
        )

        clip_mosaic(
            mosaic_path,
            bounds,
            output_path,
        )

    # ---------------------------------------------------------
    # FINAL VALIDATION
    # ---------------------------------------------------------

    print()
    print(
        "Validating final DEM..."
    )

    with rasterio.open(
        output_path
    ) as dem:

        dem_bounds = dem.bounds

        data = dem.read(
            1,
            masked=True,
        )

        valid = data.compressed()

        if valid.size == 0:

            raise RuntimeError(
                "Final DEM contains no valid elevation data."
            )

        minimum = float(
            np.min(
                valid
            )
        )

        maximum = float(
            np.max(
                valid
            )
        )

        print()
        print("=" * 72)
        print(
            "EXPANDED DEM DOWNLOAD COMPLETE"
        )
        print("=" * 72)

        print(
            f"CRS: "
            f"{dem.crs}"
        )

        print(
            f"Dimensions: "
            f"{dem.width:,} x {dem.height:,}"
        )

        print(
            f"Resolution: "
            f"{dem.res}"
        )

        print(
            f"Elevation range: "
            f"{minimum:.1f} m "
            f"to {maximum:.1f} m"
        )

        print()
        print(
            "Raster bounds:"
        )

        print(
            f"  left:   "
            f"{dem_bounds.left}"
        )

        print(
            f"  bottom: "
            f"{dem_bounds.bottom}"
        )

        print(
            f"  right:  "
            f"{dem_bounds.right}"
        )

        print(
            f"  top:    "
            f"{dem_bounds.top}"
        )

        print()
        print(
            f"Saved to:"
            f"\n  {output_path}"
        )


if __name__ == "__main__":
    main()