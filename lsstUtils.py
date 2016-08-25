import numpy as np
import lsst.pex.logging as pexLog
import lsst.afw.detection as afwDet
import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage

# ######## UTILITIES FUNCTIONS FOR PLOTTING  ####

# Utility class containing methods for displaying dipoles/footprints
# in difference images, mostly used for debugging.


def importMatplotlib():
    """!Import matplotlib.pyplot when needed, warn if not available.
    @return the imported module if available, else False.

    """
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError as err:
        log = pexLog.Log(pexLog.getDefaultLog(),
                         'lsst.ip.diffim.utils', pexLog.INFO)
        log.warn('Unable to import matplotlib: %s' % err)
        return False


def zscale(input_img, contrast=0.25):
    """This emulates ds9's zscale feature. Returns the suggested minimum and
    maximum values to display. Borrowed from ctslater."""

    samples = input_img.flatten()
    samples = samples[~np.isnan(samples)]
    samples.sort()
    chop_size = int(0.10*len(samples))
    subset = samples[chop_size:-chop_size]

    i_midpoint = int(len(subset)/2)
    I_mid = subset[i_midpoint]

    fit = np.polyfit(np.arange(len(subset)) - i_midpoint, subset, 1)
    # fit = [ slope, intercept]

    z1 = I_mid + fit[0]/contrast * (1-i_midpoint)/1.0
    z2 = I_mid + fit[0]/contrast * (len(subset)-i_midpoint)/1.0
    return z1, z2


def display2dArray(arr, title='Data', showBars=True, extent=None):
    """Use matplotlib.pyplot.imshow() to display a 2-D array.

    @param arr The 2-D array to display
    @param title Optional title to display
    @param showBars Show grey-scale bar alongside
    @param extent If not None, a 4-tuple giving the bounding box coordinates of the array

    @return matplotlib.pyplot.figure dispaying the image
    """
    plt = importMatplotlib()
    if not plt:
        return

    z1, z2 = zscale(arr)
    fig = plt.imshow(arr, origin='lower', interpolation='none', cmap='gray',
                     extent=extent, vmin=z1, vmax=z2)
    plt.title(title)
    if showBars:
        plt.colorbar(fig, cmap='gray')
    return fig


def displayImage(image, showBars=True, width=8, height=2.5):
    """Use matplotlib.pyplot.imshow() to display an afw.image.Image within
    its bounding box.

    @see displayImages()
    """
    return displayImages((image), showBars=True, width=8, height=2.5)


def displayImages(images, showBars=True, width=8, height=2.5):
    """Use matplotlib.pyplot.imshow() to display up to three
    afw.image.Images alongside each other, each within its
    bounding box.

    @see display2dArray() for params not listed here.
    @param images tuple of up to three images to display
    @param image The image to display
    @param width The width of the display (inches)
    @param height The height of the display (inches)

    @return matplotlib.pyplot.figure dispaying the image

    """
    plt = importMatplotlib()
    if not plt:
        return

    fig = plt.figure(figsize=(width, height))
    for i, image in enumerate(images):
        bbox = image.getBBox()
        extent = (bbox.getBeginX(), bbox.getEndX(), bbox.getBeginY(), bbox.getEndY())
        plt.subplot(1, len(images), i+1)
        ma = image.getArray()
        display2dArray(ma, title='Data', showBars=showBars, extent=extent)
    return fig


def displayMaskedImage(maskedImage, showMasks=True, showVariance=False, showBars=True, width=8,
                       height=2.5):
    """Use matplotlib.pyplot.imshow() to display a afwImage.MaskedImageF,
    alongside its masks and variance plane

    @see displayImage() for params not listed here

    @param maskedImage MaskedImage to display
    @param showMasks Display the MaskedImage's masks
    @param showVariance Display the MaskedImage's variance plane

    @return matplotlib.pyplot.figure dispaying the image

    """
    plt = importMatplotlib()
    if not plt:
        return

    fig = plt.figure(figsize=(width, height))
    bbox = maskedImage.getBBox()
    extent = (bbox.getBeginX(), bbox.getEndX(), bbox.getBeginY(), bbox.getEndY())
    plt.subplot(1, 3, 1)
    ma = maskedImage.getArrays()
    display2dArray(ma[0], title='Data', showBars=showBars, extent=extent)
    if showMasks:
        plt.subplot(1, 3, 2)
        display2dArray(ma[1], title='Masks', showBars=showBars, extent=extent)
    if showVariance:
        plt.subplot(1, 3, 3)
        display2dArray(ma[2], title='Variance', showBars=showBars, extent=extent)
    return fig


def displayExposure(exposure, showMasks=True, showVariance=False, showPsf=False, showBars=True,
                    width=8, height=2.5):
    """Use matplotlib.pyplot.imshow() to display a afw.image.Exposure,
    including its masks and variance plane and optionally its Psf.

    @see displayMaskedImage() for params not listed here

    @param exposure Exposure to display
    @param showPsf Display the exposure's Psf

    @return matplotlib.pyplot.figure dispaying the image

    """
    plt = importMatplotlib()
    if not plt:
        return

    fig = displayMaskedImage(exposure.getMaskedImage(), showMasks,
                             showVariance=not showPsf,
                             showBars=showBars, width=width, height=height)
    if showPsf:
        plt.subplot(1, 3, 3)
        psfIm = exposure.getPsf().computeImage()
        bbox = psfIm.getBBox()
        extent = (bbox.getBeginX(), bbox.getEndX(), bbox.getBeginY(), bbox.getEndY())
        display2dArray(psfIm.getArray(), title='PSF', showBars=showBars, extent=extent)
    return fig


def displayCutouts(source, exposure, posImage=None, negImage=None, asHeavyFootprint=False, title='',
                   figsize=(8, 2.5)):
    """Use matplotlib.pyplot.imshow() to display cutouts within up to
    three afw.image.Exposure's, given by an input SourceRecord.

    @param source afw.table.SourceRecord defining the footprint to extract and display
    @param exposure afw.image.Exposure from which to extract the cutout to display
    @param posImage Second exposure from which to extract the cutout to display
    @param negImage Third exposure from which to extract the cutout to display
    @param asHeavyFootprint Display the cutouts as
    afw.detection.HeavyFootprint, with regions outside the
    footprint removed

    @return matplotlib.pyplot.figure dispaying the image

    """
    plt = importMatplotlib()
    if not plt:
        return

    fp = source.getFootprint()
    bbox = fp.getBBox()
    extent = (bbox.getBeginX(), bbox.getEndX(), bbox.getBeginY(), bbox.getEndY())

    fig = plt.figure(figsize=figsize)
    if not asHeavyFootprint:
        subexp = afwImage.ImageF(exposure.getMaskedImage().getImage(), bbox, afwImage.PARENT)
    else:
        hfp = afwDet.HeavyFootprintF(fp, exposure.getMaskedImage())
        subexp = getHeavyFootprintSubimage(hfp)
    plt.subplot(1, 3, 1)
    display2dArray(subexp.getArray(), title=title+' Diffim', extent=extent)
    if posImage is not None:
        if not asHeavyFootprint:
            subexp = afwImage.ImageF(posImage.getMaskedImage().getImage(), bbox, afwImage.PARENT)
        else:
            hfp = afwDet.HeavyFootprintF(fp, posImage.getMaskedImage())
            subexp = getHeavyFootprintSubimage(hfp)
        plt.subplot(1, 3, 2)
        display2dArray(subexp.getArray(), title=title+' Pos', extent=extent)
    if negImage is not None:
        if not asHeavyFootprint:
            subexp = afwImage.ImageF(negImage.getMaskedImage().getImage(), bbox, afwImage.PARENT)
        else:
            hfp = afwDet.HeavyFootprintF(fp, negImage.getMaskedImage())
            subexp = getHeavyFootprintSubimage(hfp)
        plt.subplot(1, 3, 3)
        display2dArray(subexp.getArray(), title=title+' Neg', extent=extent)
    return fig


def makeHeavyCatalog(catalog, exposure, verbose=False):
    """Turn all footprints in a catalog into heavy footprints."""
    for i, source in enumerate(catalog):
        fp = source.getFootprint()
        if not fp.isHeavy():
            if verbose:
                print(i, 'not heavy => heavy')
            hfp = afwDet.HeavyFootprintF(fp, exposure.getMaskedImage())
            source.setFootprint(hfp)

    return catalog


def getHeavyFootprintSubimage(fp, badfill=np.nan, grow=0):
    """Extract the image from a heavyFootprint as an ImageF."""
    hfp = afwDet.HeavyFootprintF_cast(fp)
    bbox = hfp.getBBox()
    if grow > 0:
        bbox.grow(grow)

    subim2 = afwImage.ImageF(bbox, badfill)  # set the mask to NA (can use 0. if desired)
    afwDet.expandArray(hfp, hfp.getImageArray(), subim2.getArray(), bbox.getCorners()[0])
    return subim2


def searchCatalog(catalog, x, y):
    """Search a catalog for a source whose footprint contains the given x, y pixel coordinates."""
    for i, s in enumerate(catalog):
        bbox = s.getFootprint().getBBox()
        try:
            if bbox.contains(afwGeom.Point2D(x, y)):
                print(i)
                return s
        except:
            if bbox.contains(afwGeom.Point2I(int(x), int(y))):
                print(i)
                return s


def dpDisplayImages(dipoleImage):
    # dipoleImage is a ip_diffim.utils.DipoleTestImage
    displayExposure(dipoleImage.diffim)
    displayExposure(dipoleImage.posImage)
    displayExposure(dipoleImage.negImage)


def dpDisplayCutouts(dipoleImage, source, asHeavyFootprint=True):
    # dipoleImage is a ip_diffim.utils.DipoleTestImage
    displayCutouts(source, dipoleImage.diffim, dipoleImage.posImage, dipoleImage.negImage, asHeavyFootprint)

