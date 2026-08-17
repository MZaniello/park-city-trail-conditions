from pathlib import Path

import rioxarray


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]

    dem_path = (
        project_root
        / "data"
        / "raw"
        / "park_city_dem_10m.tif"
    )

    dem = rioxarray.open_rasterio(
        dem_path,
        masked=True,
    )

    try:
        print("DEM loaded successfully!")
        print(f"Shape: {dem.shape}")
        print(f"CRS: {dem.rio.crs}")
        print(f"Minimum elevation: {float(dem.min()):.1f} meters")
        print(f"Maximum elevation: {float(dem.max()):.1f} meters")
    finally:
        dem.close()

    print("DEM closed successfully.")


if __name__ == "__main__":
    main()