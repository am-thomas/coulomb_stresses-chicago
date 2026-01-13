
# Utility file to process DEM files (reprojection, clipping, etc. )

import rasterio
from rasterio.enums import Resampling
from rasterio.plot import show
from rasterio.windows import from_bounds
import matplotlib.pyplot as plt
import numpy as np
from rasterio.warp import calculate_default_transform, reproject, Resampling

def reproject_DEM(filename, newfilename, target_crs= 'EPSG:26916'):
    # reproject DEM to a target coordinate reference system
    with rasterio.open(f'data/{filename}') as src:
        transform, width, height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': target_crs,
            'transform': transform,
            'width': width,
            'height': height
        })
        with rasterio.open(f'data/{newfilename}', 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    resampling=Resampling.nearest)



def resample(filename, newfilename, xres, yres):
    # resample DEM to desired reoslution in x and y (units based on CRS)
    with rasterio.open(f"data/{filename}") as dataset:
        scale_factor_x = dataset.res[0]/xres
        scale_factor_y = dataset.res[1]/yres

        profile = dataset.profile.copy()
        # resample data to target shape
        data = dataset.read(
            out_shape=(
                dataset.count,
                int(dataset.height * scale_factor_y),
                int(dataset.width * scale_factor_x)
            ),
            resampling=Resampling.bilinear)

        # scale image transform
        transform = dataset.transform * dataset.transform.scale(
            (1 / scale_factor_x),
            (1 / scale_factor_y)
        )
        profile.update({"height": data.shape[-2],
                        "width": data.shape[-1],
                    "transform": transform})

    with rasterio.open(f"data/{newfilename}", "w", **profile) as dataset:
        dataset.write(data)



def crop(filename, newfilename, minx=428200, miny=4624000, maxx=431700, maxy=4629000):
    # crop DEM to a specific bounding box

    with rasterio.open(f"data/{filename}") as src:
        # Create a window based on bounds
        window = from_bounds(minx, miny, maxx, maxy, src.transform)
        # Read the data in the window
        cropped_data = src.read(window=window)
        # Update metadata
        out_meta = src.meta.copy()
        out_meta.update({
            "height": window.height,
            "width": window.width,
            "transform": src.window_transform(window)})

        show(cropped_data, transform=src.window_transform(window))
    # Save the cropped raster
    with rasterio.open(f"data/{newfilename}", "w", **out_meta) as dest:
        dest.write(cropped_data)

# reproject_DEM("original_topos/output_GMRT.tif", "1992_GMRT_reproj.tif")
# resample("1992_GMRT_reproj.tif", "1992_GMRT_100m_reproj.tif", 100, 100)
# crop("1992_GMRT_100m_reproj.tif", "1992_GMRT_100m_reproj_resamp_crop.tif")
# raster = rasterio.open("data/original_topos/output_GMRT.tif")

# open desired raster and print relevant properties
raster = rasterio.open("data/original_topos/cook_2017_clipped.tif")
# show(raster)
print('resolution [x]:', raster.res[0])
print('resolution [y]:', raster.res[1])
print('CRS:', raster.crs)
print('raster width & height', raster.width, raster.height)

# scale coordinates so that (0,0) is at the center of figire
cols, rows = np.meshgrid(np.arange(raster.width), np.arange(raster.height))
xs, ys = rasterio.transform.xy(raster.transform, rows, cols)
xcenter, ycenter = raster.xy(raster.height // 2, raster.width // 2)
xs = xs - xcenter
ys = ys - ycenter

# visualize a "demeaned" figure of elevation data (original elevations - 180 m)
elevs = raster.read(1)
adjusted_elevs = np.zeros((raster.height, raster.width))
print(np.shape(elevs), np.shape(adjusted_elevs))
for i in range(raster.height):
    for j in range(raster.width):
        val = elevs[i,j]
        #print(val)
        if val > 180:
            adjusted_elevs[i,j] = 0
        else:
            adjusted_elevs[i,j] = 180 - val
plt.scatter(xs, ys, c=adjusted_elevs)
plt.colorbar()
plt.show()

