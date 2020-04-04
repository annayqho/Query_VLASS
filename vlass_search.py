""" Search VLASS for a given RA and Dec """

## TO DO
# figure out the weird 1 pixel offset


import numpy as np
import subprocess
import os
import sys
import argparse
import glob
from astropy.io import fits as pyfits
import matplotlib.pyplot as plt
from urllib.request import urlopen
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.time import Time


def get_tiles():
    """ Get tiles 
    I ran wget https://archive-new.nrao.edu/vlass/VLASS_dyn_summary.php
    """

    inputf = open(
            "/Users/annaho/Github/Query_VLASS/VLASS_dyn_summary.php", "r")
    lines = inputf.readlines()
    inputf.close()

    header = list(filter(None, lines[0].split("  ")))
    # get rid of white spaces
    header = np.array([val.strip() for val in header])

    names = []
    dec_min = []
    dec_max = []
    ra_min = []
    ra_max = []
    obsdate = []
    epoch = []

    # Starting at lines[3], read in values
    for line in lines[3:]:
        dat = list(filter(None, line.split("  "))) 
        dat = np.array([val.strip() for val in dat]) 
        names.append(dat[0])
        dec_min.append(float(dat[1]))
        dec_max.append(float(dat[2]))
        ra_min.append(float(dat[3]))
        ra_max.append(float(dat[4]))
        obsdate.append(dat[6])
        epoch.append(dat[5])

    names = np.array(names)
    dec_min = np.array(dec_min)
    dec_max = np.array(dec_max)
    ra_min = np.array(ra_min)
    ra_max = np.array(ra_max)
    obsdate = np.array(obsdate)
    epoch = np.array(epoch)

    return (names, dec_min, dec_max, ra_min, ra_max, epoch, obsdate)


def search_tiles(tiles, c):
    """ Now that you've processed the file, search for the given RA and Dec
    
    Parameters
    ----------
    c: SkyCoord object
    """
    ra_h = c.ra.hour
    dec_d = c.dec.deg
    names, dec_min, dec_max, ra_min, ra_max, epochs, obsdate = tiles
    has_dec = np.logical_and(dec_d > dec_min, dec_d < dec_max)
    has_ra = np.logical_and(ra_h > ra_min, ra_h < ra_max)
    in_tile = np.logical_and(has_ra, has_dec)
    name = names[in_tile]
    epoch = epochs[in_tile]
    date = obsdate[in_tile]
    if len(name) > 1:
        print("Error: this source is in more than one tile")
    elif len(name) == 0:
        print("Sorry, no tile found.")
        return None, None, None
    else:
        return name[0], epoch[0], date[0]


def get_subtiles(tilename, epoch):
    """ For a given tile name, get the filenames in the VLASS directory.
    Parse those filenames and return a list of subtile RA and Dec.
    RA and Dec returned as a SkyCoord object
    """
    urlpath = urlopen(
        'https://archive-new.nrao.edu/vlass/quicklook/%s/%s/' %(epoch,tilename))
    string = (urlpath.read().decode('utf-8')).split("\n")
    vals = np.array([val.strip() for val in string])
    keep_link = np.array(["href" in val.strip() for val in string])
    keep_name = np.array([tilename in val.strip() for val in string])
    string_keep = vals[np.logical_and(keep_link, keep_name)]
    fname = np.array([val.split("\"")[1] for val in string_keep])
    pos_raw = np.array([val.split(".")[4] for val in fname])
    if '-' in pos_raw[0]:
        # dec < 0
        ra_raw = np.array([val.split("-")[0] for val in pos_raw])
        dec_raw = np.array([val.split("-")[1] for val in pos_raw])
    else:
        # dec > 0
        ra_raw = np.array([val.split("+")[0] for val in pos_raw])
        dec_raw = np.array([val.split("+")[1] for val in pos_raw])
    ra = []
    dec = []
    for ii,val in enumerate(ra_raw):
        if val[1:3] == '24':
            rah = '00'
        else:
            rah = val[1:3]
        hms = "%sh%sm%ss" %(rah, val[3:5], val[5:])
        ra.append(hms)
        dms = "%sd%sm%ss" %(
                dec_raw[ii][0:2], dec_raw[ii][2:4], dec_raw[ii][4:])
        dec.append(dms)
    ra = np.array(ra)
    dec = np.array(dec)
    c = SkyCoord(ra, dec, frame='icrs')
    return fname, c


def get_cutout(imname, name, c):
    print("Generating cutout")
    # Position of source
    ra_deg = c.ra.deg
    dec_deg = c.dec.deg

    print("Cutout centered at position %s,%s" %(ra_deg, dec_deg))

    # Open image and establish coordinate system
    im = pyfits.open(imname)[0].data[0,0]
    w = WCS(imname)

    # Find the source position in pixels.
    # This will be the center of our image.
    src_pix = w.wcs_world2pix([[ra_deg, dec_deg, 0, 0]], 0)
    x = src_pix[0,0]
    y = src_pix[0,1]

    # Check if the source is actually in the image
    pix1 = pyfits.open(imname)[0].header['CRPIX1']
    pix2 = pyfits.open(imname)[0].header['CRPIX2']
    badx = np.logical_or(x < 0, x > 2 * pix1)
    bady = np.logical_or(y < 0, y > 2 * pix1)
    if np.logical_or(badx, bady):
        print("Tile has not been imaged at the position of the source")
        return None

    else:
        # Set the dimensions of the image
        # Say we want it to be 13.5 arcseconds on a side,
        # to match the DES images
        delt1 = pyfits.open(imname)[0].header['CDELT1']
        delt2 = pyfits.open(imname)[0].header['CDELT2']
        cutout_size = 12 / 3600 # in degrees
        dside1 = -cutout_size/2./delt1
        dside2 = cutout_size/2./delt2

        vmin = -1e-4
        vmax = 1e-3

        im_plot_raw = im[int(y-dside1):int(y+dside1),int(x-dside2):int(x+dside2)]
        im_plot = np.ma.masked_invalid(im_plot_raw)

        # 3-sigma clipping
        rms_temp = np.ma.std(im_plot)
        keep = np.ma.abs(im_plot) <= 3*rms_temp
        rms = np.ma.std(im_plot[keep])

        peak_flux = np.ma.max(im.flatten())

        plt.imshow(
                np.flipud(im_plot),
                extent=[-0.5*cutout_size*3600.,0.5*cutout_size*3600.,
                        -0.5*cutout_size*3600.,0.5*cutout_size*3600],
                vmin=vmin,vmax=vmax,cmap='YlOrRd')

        peakstr = "Peak Flux %s mJy" %(np.round(peak_flux*1e3, 3))
        rmsstr = "RMS Flux %s mJy" %(np.round(rms*1e3, 3))
        plt.title(name + ": %s; %s" %(peakstr, rmsstr))
        plt.xlabel("Offset in RA (arcsec)")
        plt.ylabel("Offset in Dec (arcsec)")

        plt.savefig(name + ".png") 
        plt.close()

        return peak_flux,rms


def search_vlass(name, c, date=None):
    """ 
    Searches the VLASS catalog for a source

    Parameters
    ----------
    names: name of the sources
    c: coordinates as SkyCoord object
    date: date in astropy Time format
    """
    print("Running for %s" %name)
    print("Coordinates %s" %c)
    print("Date %s" %date)

    # Find the VLASS tile
    tiles = get_tiles()
    tilename, epoch, obsdate = search_tiles(tiles, c)

    if tilename is None:
        print("There is no VLASS tile at this location")

    else:
        # The VLASS quicklook site only has 1.1, unfortunately
        if np.logical_and(epoch != "VLASS1.1", epoch != "VLASS1.2"):
            print("Sorry, tile will be observed in a later epoch")
        else:
            print("Tile found:")
            print(tilename, epoch)
            subtiles, c_tiles = get_subtiles(tilename, epoch)
            dist = c.separation(c_tiles)
            subtile = subtiles[np.argmin(dist)]

            url_get = "https://archive-new.nrao.edu/vlass/quicklook/%s/%s/%s" %(
                    epoch, tilename, subtile)
            imname = "%s.I.iter1.image.pbcor.tt0.subim.fits" %subtile[0:-1]
            print(imname)
            if len(glob.glob(imname)) == 0:
                fname = url_get + imname
                subprocess.run(["wget", fname])
            out = get_cutout(imname, name, c)
            if out is not None:
                peak, rms = out
                print("Peak flux is %s uJy" %(peak*1e6))
                print("RMS is %s uJy" %(rms*1e6))
                limit = rms*1e6
                obsdate = Time(obsdate, format='iso').mjd
                print("Tile observed on %s" %obsdate)
                return limit,obsdate
    return None


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=\
        '''
        Searches VLASS for a source.
        User needs to supply name, RA (in decimal degrees),
        Dec (in decimal degrees), and (optionally) date (in mjd).
        If there is a date, then will only return VLASS images taken after that date
        (useful for transients with known explosion dates).
        
        Usage: vlass_search.py <Name> <RA [deg]> <Dec [deg]> <(optional) Date [mjd]>
        ''', formatter_class=argparse.RawTextHelpFormatter)
        
    #Check if correct number of arguments are given
    if len(sys.argv) < 3:
        print("Usage: vlass_search.py <Name> <RA [deg]> <Dec [deg]> <(optional) Date [astropy Time]>")
        sys.exit()
     
    name = str(sys.argv[1])
    ra = float(sys.argv[2])
    dec = float(sys.argv[3])
    c = SkyCoord(ra, dec, unit='deg')

    if glob.glob("/Users/annaho/Github/Query_VLASS/VLASS_dyn_summary.php"):
        pass
    else:
        print("Tile summary file is not here. Download it using wget:\
               wget https://archive-new.nrao.edu/vlass/VLASS_dyn_summary.php")

    if (len(sys.argv) > 4):
        date = Time(float(sys.argv[4]), format='mjd')
        print ('Searching for observations after %s' %date)
        search_vlass(name, c, date) 
    else:
        print ('Searching all obs dates')
        search_vlass(name, c) 
